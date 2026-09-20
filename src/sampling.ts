import { type MtpDraftBatch, type MtpProposal, type SamplingParams, type TokenSelector } from "./chat_model";
import { DeviceOps } from "./device_ops";
import { MemcpyKind } from "./enums";
import { Tensor } from "./tensor";
import { WorkspaceBase } from "./workspace";
import { CaptureManager } from "./capture-manager";
export { MemcpyKind };

interface ParameterBank {
  params: SamplingParams[];
  readonly maxWindow: number;
  temperatures: Tensor;
  repPenalties: Tensor;
  presPenalties: Tensor;
  topKs: Tensor;
  topPs: Tensor;
}

export interface MtpSamplingConfig {
  maxBatchSize: number;
  depth: number;
  retainProposalsOnGpu?: boolean;
}

export class SamplingWorkspace extends WorkspaceBase implements TokenSelector {
  readonly vocabSize: number;
  readonly maxBatchSize: number;
  readonly maxWindow: number;
  batchSize: number;
  params: SamplingParams[] = [];
  captureKey = 0;

  readonly penaltyTokens: Tensor;
  readonly penaltyCount: Tensor;
  readonly stepCounter: Tensor;
  readonly temperatures: Tensor;
  readonly repPenalties: Tensor;
  readonly presPenalties: Tensor;
  readonly topKs: Tensor;
  readonly topPs: Tensor;
  readonly mtpEnabled: boolean;
  readonly capacity = 256;
  private readonly draftParams!: ParameterBank;
  private readonly verificationParams!: ParameterBank;
  readonly draftStepCounter!: Tensor;
  mtpCaptureKey = "linear:0";
  private mtpBatchSize = 0;
  private draftBatchSize = 0;
  private generation = 0;
  private readonly proposalOwner = {};
  private candidateCount = 0;
  private readonly qDraftProbs!: Tensor;
  private readonly qDraftIds!: Tensor;
  private readonly qHostProbs!: Tensor;
  private readonly qHostIds!: Tensor;
  private readonly qInputProbs!: Tensor;
  private readonly qInputIds!: Tensor;
  private readonly draftTokens!: Tensor;
  private readonly rejectionStepCounter!: Tensor;
  private readonly mtpMaxBatchSize: number;
  readonly depth: number;
  readonly retainProposalsOnGpu: boolean;

  constructor(glm: DeviceOps, maxBatchSize: number, vocabSize: number, maxWindow: number, mtp?: MtpSamplingConfig) {
    super(glm);
    if (mtp && (!Number.isInteger(mtp.maxBatchSize) || mtp.maxBatchSize < 1
      || !Number.isInteger(mtp.depth) || mtp.depth < 1 || maxBatchSize < mtp.maxBatchSize * (mtp.depth + 1))) {
      throw new Error("Linear MTP requires valid batch/depth and a compatible target sampling workspace");
    }
    this.vocabSize = vocabSize;
    this.maxBatchSize = maxBatchSize;
    this.maxWindow = maxWindow;
    this.batchSize = 0;

    this.penaltyTokens = this.alloc([maxWindow > 0 ? maxBatchSize * maxWindow : maxBatchSize], "I32", "penaltyTokens");
    this.penaltyCount = this.alloc([maxBatchSize], "I32", "penaltyCount");
    this.stepCounter = this.alloc([1], "U32", "stepCounter");
    this.temperatures = this.alloc([maxBatchSize], "F32", "temperatures");
    this.repPenalties = this.alloc([maxBatchSize], "F32", "repetitionPenalties");
    this.presPenalties = this.alloc([maxBatchSize], "F32", "presencePenalties");
    this.topKs = this.alloc([maxBatchSize], "I32", "topKs");
    this.topPs = this.alloc([maxBatchSize], "F32", "topPs");

    const seedBuf = Buffer.alloc(4);
    seedBuf.writeUInt32LE(Math.floor(Math.random() * 0xFFFFFFFF) >>> 0, 0);
    this.stepCounter.h2d(seedBuf);
    this.mtpEnabled = mtp !== undefined;
    this.mtpMaxBatchSize = mtp?.maxBatchSize ?? 0;
    this.depth = mtp?.depth ?? 0;
    this.retainProposalsOnGpu = mtp?.retainProposalsOnGpu ?? false;
    if (mtp) {
      const { maxBatchSize, depth } = mtp;
      if (this.retainProposalsOnGpu) this.mtpCaptureKey += ":gpu";
      this.draftParams = this.createParameterBank(maxBatchSize, "draft");
      this.verificationParams = this.createParameterBank(maxBatchSize * (depth + 1), "verification");
      this.draftStepCounter = this.alloc([1], "U32", "draftStepCounter");
      const C = this.capacity;
      this.qDraftProbs = this.alloc([depth, maxBatchSize, C], "F32", "qDraftProbs");
      this.qDraftIds = this.alloc([depth, maxBatchSize, C], "I32", "qDraftIds");
      if (!this.retainProposalsOnGpu) {
        this.qHostProbs = this.allocPinned([depth, maxBatchSize, C], "F32", "qHostProbs");
        this.qHostIds = this.allocPinned([depth, maxBatchSize, C], "I32", "qHostIds");
      }
      this.qInputProbs = this.alloc([maxBatchSize, depth, C], "F32", "qInputProbs");
      this.qInputIds = this.alloc([maxBatchSize, depth, C], "I32", "qInputIds");
      this.draftTokens = this.alloc([maxBatchSize, depth], "I32", "draftTokens");
      this.rejectionStepCounter = this.alloc([1], "U32", "rejectionStepCounter");
      seedBuf.writeUInt32LE(Math.floor(Math.random() * 0xFFFFFFFF) >>> 0);
      this.rejectionStepCounter.h2d(seedBuf);
      seedBuf.writeUInt32LE(Math.floor(Math.random() * 0xFFFFFFFF) >>> 0);
      this.draftStepCounter.h2d(seedBuf);
    }
  }

  private createParameterBank(rows: number, name: string): ParameterBank {
    return {
      params: [],
      maxWindow: 0,
      temperatures: this.alloc([rows], "F32", `${name}Temperatures`),
      repPenalties: this.alloc([rows], "F32", `${name}RepPenalties`),
      presPenalties: this.alloc([rows], "F32", `${name}PresPenalties`),
      topKs: this.alloc([rows], "I32", `${name}TopKs`),
      topPs: this.alloc([rows], "F32", `${name}TopPs`),
    };
  }

  private uploadParameters(bank: ParameterBank, params: SamplingParams[]): void {
    if (CaptureManager.capturing) throw new Error("Sampling parameters must be uploaded outside capture");
    if (params.length > bank.temperatures.numElements) throw new Error("Sampling params exceeds maxBatchSize");
    const rows = params.length;
    if (rows === 0) { bank.params = []; return; }
    // Disposed pinned staging cannot be reused until the asynchronous upload completes.
    using staging = this.allocPinned([5, rows], "F32");
    staging.withPinnedBuffer(buf => params.forEach((p, i) => {
      buf.writeFloatLE(p.temperature > 0 ? p.temperature : 0, i * 4);
      buf.writeFloatLE(p.repetitionPenalty, (rows + i) * 4);
      buf.writeFloatLE(p.presencePenalty, (2 * rows + i) * 4);
      buf.writeInt32LE(p.topK > 0 ? p.topK : 0, (3 * rows + i) * 4);
      buf.writeFloatLE(p.topP, (4 * rows + i) * 4);
    }));
    [bank.temperatures, bank.repPenalties, bank.presPenalties, bank.topKs, bank.topPs].forEach((device, i) => {
      using host = staging.narrow(i, 1);
      device.memcpy(host, rows * 4, MemcpyKind.HostToDevice);
    });
    bank.params = params.map(p => ({ ...p }));
  }

  selectTarget(logits: Tensor): Tensor { return this.sample(logits); }

  initPenaltyState(params: SamplingParams[], tokenHistories: number[][]): void {
    const I32 = 4;
    const batchSize = this.batchSize;
    const vs = this.vocabSize;
    const maxWindow = this.maxWindow;

    const penaltyBufSize = maxWindow > 0 ? batchSize * maxWindow * I32 : batchSize * I32;
    const penaltyBuf = Buffer.alloc(penaltyBufSize);
    const countBuf = Buffer.alloc(batchSize * I32);

    for (let i = 0; i < batchSize; i++) {
      const p = params[i];
      const hasPenalty = p.repetitionPenalty !== 1.0 || p.presencePenalty !== 0;
      let numTokens = 0;
      if (hasPenalty && maxWindow > 0) {
        const history = tokenHistories[i];
        const seen = new Set<number>();
        const start = Math.max(0, history.length - maxWindow);
        for (let j = start; j < history.length; j++) seen.add(history[j]);
        for (const tid of seen) {
          if (tid < vs) {
            penaltyBuf.writeInt32LE(tid, (i * maxWindow + numTokens) * I32);
            numTokens++;
          }
        }
      }
      countBuf.writeInt32LE(numTokens, i * I32);
    }

    if (maxWindow > 0) {
      this.penaltyTokens.h2d(penaltyBuf, penaltyBufSize);
    }
    this.penaltyCount.h2d(countBuf);
  }

  updateSampler(params: SamplingParams[], tokenHistories?: number[][]): void {
    if (params.length > this.maxBatchSize) {
      throw new Error(`updateSampler: ${params.length} params exceeds maxBatchSize ${this.maxBatchSize}`);
    }
    this.uploadParameters(this, params);
    this.batchSize = params.length;
    this.captureKey = Math.max(0, ...params.map(p => p.temperature <= 0 ? 1 : p.topK > 0 ? Math.min(p.topK, this.vocabSize) : 32));

    if (tokenHistories !== undefined) {
      this.initPenaltyState(params, tokenHistories);
    }
  }

  /** Caller owns the output; return it from captured callbacks to retain replay results. */
  sample(logits: Tensor, outProbs?: Tensor, outIds?: Tensor, supportCapacity = 0, bank: ParameterBank = this): Tensor {
    const batchSize = logits.shape[0];
    const vs = this.vocabSize;
    if (batchSize < 1 || logits.type !== "BF16" || logits.shape.length !== 2 || logits.shape[1] !== vs
      || bank.params.length !== batchSize || logits.numElements !== batchSize * vs) {
      throw new Error("sample: invalid parameter or logits row capacities");
    }

    let maxEffectiveK = 0;
    for (let i = 0; i < batchSize; i++) {
      const p = bank.params[i];
      const topK = p.topK > 0 ? p.topK : 0;
      const temperature = p.temperature > 0 ? p.temperature : 0;
      let effectiveK: number;
      if (temperature <= 0) {
        effectiveK = 1;
      } else if (topK > 0) {
        effectiveK = topK < vs ? topK : vs;
      } else {
        effectiveK = 32;
      }
      if (effectiveK > maxEffectiveK) maxEffectiveK = effectiveK;
    }

    if (outProbs || outIds) {
      if (!outProbs || !outIds || !Number.isInteger(supportCapacity) || supportCapacity <= 0 || maxEffectiveK > supportCapacity) {
        throw new Error(`sample: distribution outputs require capacity >= maxEffectiveK (${maxEffectiveK})`);
      }
      if (outProbs.numElements < batchSize * supportCapacity || outIds.numElements < batchSize * supportCapacity
        || outProbs.type !== "F32" || outIds.type !== "I32") {
        throw new Error("sample: invalid distribution output types or row capacities");
      }
    }

    const outToken = this.alloc([batchSize], "I32");
    using topkVals = this.alloc([batchSize * 256 * 256], "F32");
    using topkIdxs = this.alloc([batchSize * 256 * 256], "I32");
    using scratch = this.alloc([batchSize * vs], "F32");
    logits.workspace.glm.sampleBatch(
      outToken,
      topkVals,
      topkIdxs,
      scratch,
      logits,
      this.penaltyTokens,
      this.penaltyCount,
      bank.maxWindow,
      vs,
      batchSize,
      bank.temperatures,
      bank.repPenalties,
      bank.presPenalties,
      bank.topKs,
      bank.topPs,
      this.stepCounter,
      maxEffectiveK,
      outProbs,
      outIds,
      supportCapacity,
    );

    return outToken;
  }

  updateMtpSampler(params: SamplingParams[]): void {
    if (!this.mtpEnabled) throw new Error("MTP sampling is not enabled");
    let maxEffectiveK = 0;
    for (const param of params) {
      const effectiveK = param.temperature <= 0 ? 1 : param.topK > 0 ? Math.min(param.topK, this.vocabSize) : 32;
      if (![param.temperature, param.topP, param.topK].every(Number.isFinite) || !Number.isInteger(param.topK)
        || !Number.isInteger(effectiveK) || effectiveK > this.capacity
        || param.repetitionPenalty !== 1 || param.presencePenalty !== 0) {
        throw new Error("Linear MTP requires finite sampling parameters, integer effective top_k <= 256, and no penalties");
      }
      maxEffectiveK = Math.max(maxEffectiveK, effectiveK);
    }
    // Draft rows are [B]; verification is sequence-major [B, D+1].
    // Ordinary/initial target rows and captureKey are managed only by updateSampler.
    this.uploadParameters(this.draftParams, params);
    this.uploadParameters(this.verificationParams, params.flatMap(p => Array.from({ length: this.depth + 1 }, () => p)));
    this.candidateCount = Math.min(maxEffectiveK, this.vocabSize);
    this.mtpCaptureKey = `linear:${maxEffectiveK}${this.retainProposalsOnGpu ? ":gpu" : ""}`;
  }

  prepareDraft(batchSize: number, depth: number): void {
    this.prepareBatch(batchSize, depth);
    this.draftBatchSize = batchSize;
    this.generation++;
  }

  private prepareBatch(batchSize: number, depth: number): void {
    if (!this.mtpEnabled) throw new Error("MTP sampling is not enabled");
    if (!Number.isInteger(batchSize) || batchSize < 1 || batchSize > this.mtpMaxBatchSize
      || depth !== this.depth || this.draftParams.params.length !== batchSize) {
      throw new Error("Linear MTP draft batch/depth does not match sampling workspace");
    }
    this.mtpBatchSize = batchSize;
  }

  sampleDraft(logits: Tensor, depth: number): Tensor {
    if (!this.mtpEnabled) throw new Error("MTP sampling is not enabled");
    if (!Number.isInteger(depth) || depth < 0 || depth >= this.depth || this.mtpBatchSize < 1) {
      throw new Error("Invalid linear MTP draft depth or unprepared batch");
    }
    if (logits.type !== "BF16" || logits.shape.length !== 2 || logits.shape[0] !== this.mtpBatchSize
      || logits.shape[1] !== this.vocabSize) {
      throw new Error("sampleDraft: invalid logits type or row capacities");
    }
    using probs = this.qDraftProbs.narrow(depth, 1);
    using ids = this.qDraftIds.narrow(depth, 1);
    // Distributed top-k exchanges candidates, not the full vocabulary logits.
    const candidates = logits.topk(this.candidateCount, this.vocabSize);
    using values = candidates.values;
    using globalIds = candidates.indices;
    const output = this.alloc([this.mtpBatchSize], "I32");
    this.glm.sampleCandidates(output, probs, ids, values, globalIds,
      this.draftParams.temperatures, this.draftParams.topKs, this.draftParams.topPs,
      this.draftStepCounter, this.mtpBatchSize, this.candidateCount, this.capacity);
    if (!this.retainProposalsOnGpu) {
      using probsH = this.qHostProbs.narrow(depth, 1);
      using idsH = this.qHostIds.narrow(depth, 1);
      const bytes = this.mtpBatchSize * this.capacity * 4;
      probsH.memcpy(probs, bytes, MemcpyKind.DeviceToHost);
      idsH.memcpy(ids, bytes, MemcpyKind.DeviceToHost);
    }
    return output;
  }

  finishDraft(): MtpProposal {
    if (!this.mtpEnabled) throw new Error("MTP sampling is not enabled");
    if (this.mtpBatchSize < 1) throw new Error("Linear MTP draft is not prepared");
    if (this.retainProposalsOnGpu) {
      return {
        probabilities: [], tokenIds: [], capacity: this.capacity,
        device: { owner: this.proposalOwner, generation: this.generation, rows: Array.from({ length: this.draftBatchSize }, (_, b) => b) },
      };
    }
    const probabilities: Buffer[] = [];
    const tokenIds: Buffer[] = [];
    const probs = this.qHostProbs.readPinnedBuffer();
    const ids = this.qHostIds.readPinnedBuffer();
    const rowBytes = this.capacity * 4;
    for (let batch = 0; batch < this.mtpBatchSize; batch++) {
      const p = Buffer.alloc(this.depth * rowBytes);
      const t = Buffer.alloc(this.depth * rowBytes);
      // Each depth packs only the active B rows into its fixed maxB allocation.
      for (let depth = 0; depth < this.depth; depth++) {
        const offset = (depth * this.mtpMaxBatchSize + batch) * rowBytes;
        probs.copy(p, depth * rowBytes, offset, offset + rowBytes);
        ids.copy(t, depth * rowBytes, offset, offset + rowBytes);
      }
      probabilities.push(p);
      tokenIds.push(t);
    }
    return { probabilities, tokenIds, capacity: this.capacity };
  }

  prepareVerification(draft: MtpDraftBatch): void {
    if (CaptureManager.capturing) throw new Error("MTP verification must be prepared outside capture");
    const B = draft.targetTokens.length;
    const D = this.depth;
    const proposal = draft.proposal;
    const rowBytes = D * this.capacity * 4;
    this.prepareBatch(B, draft.topks.length);
    if (draft.topks.some(k => k !== 1) || draft.treeTokens.length !== B
      || draft.treeTokens.some(tokens => tokens.length !== D || tokens.some(t => !Number.isInteger(t) || t < 0 || t >= this.vocabSize))
      || !proposal || proposal.capacity !== this.capacity) {
      throw new Error("Invalid linear MTP proposal batch, depth, capacity, or buffers");
    }
    if (this.retainProposalsOnGpu) {
      const device = proposal.device;
      if (!device || device.owner !== this.proposalOwner || device.generation !== this.generation
        || device.rows.length !== B || proposal.probabilities.length !== 0 || proposal.tokenIds.length !== 0
        || device.rows.some(row => !Number.isInteger(row) || row < 0 || row >= this.draftBatchSize)) {
        throw new Error("Invalid, stale, or foreign GPU linear MTP proposal");
      }
      // Transpose [D, maxB, C] into [B, D, C], retaining the original row mapping after compaction.
      const width = this.capacity * 4;
      device.rows.forEach((row, b) => {
        this.qInputProbs.memcpy2d(b * rowBytes, width, this.qDraftProbs, row * width, this.mtpMaxBatchSize * width, width, D, MemcpyKind.DeviceToDevice);
        this.qInputIds.memcpy2d(b * rowBytes, width, this.qDraftIds, row * width, this.mtpMaxBatchSize * width, width, D, MemcpyKind.DeviceToDevice);
      });
    } else {
      if (proposal.device || proposal.probabilities.length !== B || proposal.tokenIds.length !== B
        || proposal.probabilities.some(buf => !Buffer.isBuffer(buf) || buf.length !== rowBytes)
        || proposal.tokenIds.some(buf => !Buffer.isBuffer(buf) || buf.length !== rowBytes)) {
        throw new Error("Invalid host linear MTP proposal buffers or mode");
      }
      using probsH = this.allocPinned([B, D, this.capacity], "F32");
      using idsH = this.allocPinned([B, D, this.capacity], "I32");
      probsH.withPinnedBuffer(buf => proposal.probabilities.forEach((row, i) => row.copy(buf, i * rowBytes)));
      idsH.withPinnedBuffer(buf => proposal.tokenIds.forEach((row, i) => row.copy(buf, i * rowBytes)));
      this.qInputProbs.memcpy(probsH, B * rowBytes, MemcpyKind.HostToDevice);
      this.qInputIds.memcpy(idsH, B * rowBytes, MemcpyKind.HostToDevice);
    }
    using draftTokensH = this.allocPinned([B, D], "I32");
    draftTokensH.withPinnedBuffer(buf => {
      draft.treeTokens.forEach((tokens, b) => tokens.forEach((token, d) => buf.writeInt32LE(token, (b * D + d) * 4)));
    });
    this.draftTokens.memcpy(draftTokensH, B * D * 4, MemcpyKind.HostToDevice);
  }

  verify(logits: Tensor): { tokens: Tensor; numAccepted: Tensor } {
    if (!this.mtpEnabled) throw new Error("MTP sampling is not enabled");
    const rows = this.mtpBatchSize * (this.depth + 1);
    if (this.mtpBatchSize < 1 || this.verificationParams.params.length !== rows) {
      throw new Error("Linear MTP target sampler must contain B * (D + 1) rows");
    }
    using pProbs = this.alloc([rows, this.capacity], "F32");
    using pIds = this.alloc([rows, this.capacity], "I32");
    using _sampled = this.sample(logits, pProbs, pIds, this.capacity, this.verificationParams);
    const tokens = this.alloc([rows], "I32");
    const acceptedCounts = this.alloc([this.mtpBatchSize], "I32");
    this.glm.specRejectLinear(tokens, acceptedCounts, this.draftTokens,
      this.qInputProbs, this.qInputIds, pProbs, pIds, this.rejectionStepCounter,
      this.mtpBatchSize, this.depth, this.capacity);
    return { tokens, numAccepted: acceptedCounts };
  }

  prepareVerificationFromDevice(draftTokens: Tensor, batchSize: number): void {
    this.prepareBatch(batchSize, this.depth);
    const width = this.capacity * 4;
    const rowBytes = this.depth * width;
    for (let batch = 0; batch < batchSize; batch++) {
      this.qInputProbs.memcpy2d(batch * rowBytes, width, this.qDraftProbs,
        batch * width, this.mtpMaxBatchSize * width, width, this.depth, MemcpyKind.DeviceToDevice);
      this.qInputIds.memcpy2d(batch * rowBytes, width, this.qDraftIds,
        batch * width, this.mtpMaxBatchSize * width, width, this.depth, MemcpyKind.DeviceToDevice);
    }
    this.draftTokens.memcpy(draftTokens, batchSize * this.depth * 4, MemcpyKind.DeviceToDevice);
  }
}
