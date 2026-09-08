import { type LinearMtpSampler, type MtpDraftBatch, type MtpProposal, type SamplingParams } from "./chat_model";
import { DeviceOps } from "./device_ops";
import { MemcpyKind } from "./enums";
import { Tensor } from "./tensor";
import { WorkspaceBase } from "./workspace";
export { MemcpyKind };


export class SamplingWorkspace extends WorkspaceBase {
  readonly vocabSize: number;
  readonly maxBatchSize: number;
  readonly maxWindow: number;
  batchSize: number;
  params!: SamplingParams[];

  readonly penaltyTokens: Tensor;
  readonly penaltyCount: Tensor;
  readonly stepCounter: Tensor;
  readonly temperatures: Tensor;
  readonly temperaturesH: Tensor;
  readonly repPenalties: Tensor;
  readonly repPenaltiesH: Tensor;
  readonly presPenalties: Tensor;
  readonly presPenaltiesH: Tensor;
  readonly topKs: Tensor;
  readonly topKsH: Tensor;
  readonly topPs: Tensor;
  readonly topPsH: Tensor;
  readonly outToken: Tensor;

  private readonly topkVals: Tensor;
  private readonly topkIdxs: Tensor;
  private readonly sampleWorkspaceBuf: Tensor;

  constructor(glm: DeviceOps, maxBatchSize: number, vocabSize: number, maxWindow: number) {
    super(glm);
    this.vocabSize = vocabSize;
    this.maxBatchSize = maxBatchSize;
    this.maxWindow = maxWindow;
    this.batchSize = 0;

    this.penaltyTokens = this.alloc([maxWindow > 0 ? maxBatchSize * maxWindow : maxBatchSize], "I32", "penaltyTokens");
    this.penaltyCount = this.alloc([maxBatchSize], "I32", "penaltyCount");
    this.stepCounter = this.alloc([1], "U32", "stepCounter");
    this.temperatures = this.alloc([maxBatchSize], "F32", "temperatures");
    this.temperaturesH = this.allocPinned([maxBatchSize], "F32", "temperaturesHost");
    this.repPenalties = this.alloc([maxBatchSize], "F32", "repetitionPenalties");
    this.repPenaltiesH = this.allocPinned([maxBatchSize], "F32", "repetitionPenaltiesHost");
    this.presPenalties = this.alloc([maxBatchSize], "F32", "presencePenalties");
    this.presPenaltiesH = this.allocPinned([maxBatchSize], "F32", "presencePenaltiesHost");
    this.topKs = this.alloc([maxBatchSize], "I32", "topKs");
    this.topKsH = this.allocPinned([maxBatchSize], "I32", "topKsHost");
    this.topPs = this.alloc([maxBatchSize], "F32", "topPs");
    this.topPsH = this.allocPinned([maxBatchSize], "F32", "topPsHost");
    this.outToken = this.alloc([maxBatchSize], "I32", "outToken");

    const SAMPLING_MAX_TOPK = 256;
    const SAMPLING_BLOCK_SIZE = 256;
    this.topkVals = this.alloc([maxBatchSize * SAMPLING_MAX_TOPK * SAMPLING_BLOCK_SIZE], "F32", "topkValues");
    this.topkIdxs = this.alloc([maxBatchSize * SAMPLING_MAX_TOPK * SAMPLING_BLOCK_SIZE], "I32", "topkIndices");
    this.sampleWorkspaceBuf = this.alloc([maxBatchSize * vocabSize], "F32", "sampleWorkspace");

    const seedBuf = Buffer.alloc(4);
    seedBuf.writeUInt32LE(Math.floor(Math.random() * 0xFFFFFFFF) >>> 0, 0);
    this.stepCounter.h2d(seedBuf);
  }

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
    this.batchSize = params.length;
    this.params = params;

    const I32 = 4;
    const batchSize = this.batchSize;

    for (let i = 0; i < batchSize; i++) {
      const p = params[i];
      const topK = p.topK > 0 ? p.topK : 0;
      const temperature = p.temperature > 0 ? p.temperature : 0;

      this.temperaturesH.withPinnedBuffer(buf => {
        buf.writeFloatLE(temperature, i * 4);
      });
      this.repPenaltiesH.withPinnedBuffer(buf => {
        buf.writeFloatLE(p.repetitionPenalty, i * 4);
      });
      this.presPenaltiesH.withPinnedBuffer(buf => {
        buf.writeFloatLE(p.presencePenalty, i * 4);
      });
      this.topKsH.withPinnedBuffer(buf => {
        buf.writeInt32LE(topK, i * I32);
      });
      this.topPsH.withPinnedBuffer(buf => {
        buf.writeFloatLE(p.topP, i * 4);
      });
    }

    this.temperatures.memcpy(this.temperaturesH, batchSize * 4, MemcpyKind.HostToDevice);
    this.repPenalties.memcpy(this.repPenaltiesH, batchSize * 4, MemcpyKind.HostToDevice);
    this.presPenalties.memcpy(this.presPenaltiesH, batchSize * 4, MemcpyKind.HostToDevice);
    this.topKs.memcpy(this.topKsH, batchSize * I32, MemcpyKind.HostToDevice);
    this.topPs.memcpy(this.topPsH, batchSize * 4, MemcpyKind.HostToDevice);

    if (tokenHistories !== undefined) {
      this.initPenaltyState(params, tokenHistories);
    }
  }

  /** Returns a caller-owned view of the workspace's reusable output buffer. */
  sample(logits: Tensor): Tensor {
    return this.sampleInto(logits, this.outToken).viewClone();
  }

  sampleInto(logits: Tensor, outToken: Tensor, outProbs?: Tensor, outIds?: Tensor, supportCapacity = 0): Tensor {
    const batchSize = this.batchSize;
    const vs = this.vocabSize;

    let maxEffectiveK = 0;
    for (let i = 0; i < batchSize; i++) {
      const p = this.params[i];
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
        throw new Error(`sampleInto: distribution outputs require capacity >= maxEffectiveK (${maxEffectiveK})`);
      }
      if (logits.numElements !== batchSize * vs || outToken.numElements < batchSize
        || outProbs.numElements < batchSize * supportCapacity || outIds.numElements < batchSize * supportCapacity
        || outProbs.type !== "F32" || outIds.type !== "I32") {
        throw new Error("sampleInto: invalid distribution output types or row capacities");
      }
    }

    logits.workspace.glm.sampleBatch(
      outToken,
      this.topkVals,
      this.topkIdxs,
      this.sampleWorkspaceBuf,
      logits,
      this.penaltyTokens,
      this.penaltyCount,
      this.maxWindow,
      vs,
      batchSize,
      this.temperatures,
      this.repPenalties,
      this.presPenalties,
      this.topKs,
      this.topPs,
      this.stepCounter,
      maxEffectiveK,
      outProbs,
      outIds,
      supportCapacity,
    );

    return outToken;
  }
}

export class LinearMtpSamplingWorkspace extends WorkspaceBase implements LinearMtpSampler {
  readonly capacity = 256;
  readonly draftSampler: SamplingWorkspace;
  captureKey = "linear:0";
  private batchSize = 0;
  private draftBatchSize = 0;
  private generation = 0;
  private readonly proposalOwner = {};
  private candidateCount = 0;
  private readonly qDraftProbs: Tensor;
  private readonly qDraftIds: Tensor;
  private readonly qHostProbs: Tensor;
  private readonly qHostIds: Tensor;
  private readonly qInputProbs: Tensor;
  private readonly qInputIds: Tensor;
  private readonly qInputProbsH: Tensor;
  private readonly qInputIdsH: Tensor;
  private readonly draftTokens: Tensor;
  private readonly draftTokensH: Tensor;
  private readonly pProbs: Tensor;
  private readonly pIds: Tensor;
  private readonly tokens: Tensor;
  private readonly acceptedCounts: Tensor;
  private readonly stepCounter: Tensor;

  constructor(glm: DeviceOps, readonly maxBatchSize: number, readonly depth: number, readonly targetSampler: SamplingWorkspace, readonly retainProposalsOnGpu = false) {
    super(glm);
    if (retainProposalsOnGpu) this.captureKey += ":gpu";
    if (!Number.isInteger(maxBatchSize) || maxBatchSize < 1 || !Number.isInteger(depth) || depth < 1
      || targetSampler.maxBatchSize < maxBatchSize * (depth + 1) || targetSampler.glm !== glm) {
      throw new Error("Linear MTP requires valid batch/depth and a compatible target sampling workspace");
    }
    this.draftSampler = new SamplingWorkspace(glm, maxBatchSize, targetSampler.vocabSize, 0);
    const C = this.capacity;
    this.qDraftProbs = this.alloc([depth, maxBatchSize, C], "F32", "qDraftProbs");
    this.qDraftIds = this.alloc([depth, maxBatchSize, C], "I32", "qDraftIds");
    this.qHostProbs = this.allocPinned([depth, maxBatchSize, C], "F32", "qHostProbs");
    this.qHostIds = this.allocPinned([depth, maxBatchSize, C], "I32", "qHostIds");
    this.qInputProbs = this.alloc([maxBatchSize, depth, C], "F32", "qInputProbs");
    this.qInputIds = this.alloc([maxBatchSize, depth, C], "I32", "qInputIds");
    this.qInputProbsH = this.allocPinned([maxBatchSize, depth, C], "F32", "qInputProbsH");
    this.qInputIdsH = this.allocPinned([maxBatchSize, depth, C], "I32", "qInputIdsH");
    this.draftTokens = this.alloc([maxBatchSize, depth], "I32", "draftTokens");
    this.draftTokensH = this.allocPinned([maxBatchSize, depth], "I32", "draftTokensH");
    this.pProbs = this.alloc([maxBatchSize * (depth + 1), C], "F32", "pProbs");
    this.pIds = this.alloc([maxBatchSize * (depth + 1), C], "I32", "pIds");
    this.tokens = this.alloc([maxBatchSize * (depth + 1)], "I32", "tokens");
    this.acceptedCounts = this.alloc([maxBatchSize], "I32", "acceptedCounts");
    this.stepCounter = this.alloc([1], "U32", "rejectionStepCounter");
    const seed = Buffer.alloc(4);
    seed.writeUInt32LE(Math.floor(Math.random() * 0xFFFFFFFF) >>> 0);
    this.stepCounter.h2d(seed);
  }

  updateSampler(params: SamplingParams[]): void {
    let maxEffectiveK = 0;
    for (const param of params) {
      const effectiveK = param.temperature <= 0 ? 1 : param.topK > 0 ? Math.min(param.topK, this.targetSampler.vocabSize) : 32;
      if (![param.temperature, param.topP, param.topK].every(Number.isFinite) || !Number.isInteger(param.topK)
        || !Number.isInteger(effectiveK) || effectiveK > this.capacity
        || param.repetitionPenalty !== 1 || param.presencePenalty !== 0) {
        throw new Error("Linear MTP requires finite sampling parameters, integer effective top_k <= 256, and no penalties");
      }
      maxEffectiveK = Math.max(maxEffectiveK, effectiveK);
    }
    this.draftSampler.updateSampler(params, params.map(() => []));
    this.candidateCount = Math.min(maxEffectiveK, this.targetSampler.vocabSize);
    this.captureKey = `linear:${maxEffectiveK}${this.retainProposalsOnGpu ? ":gpu" : ""}`;
  }

  prepareDraft(batchSize: number, depth: number): void {
    this.prepareBatch(batchSize, depth);
    this.draftBatchSize = batchSize;
    this.generation++;
  }

  private prepareBatch(batchSize: number, depth: number): void {
    if (!Number.isInteger(batchSize) || batchSize < 1 || batchSize > this.maxBatchSize
      || depth !== this.depth || this.draftSampler.batchSize !== batchSize) {
      throw new Error("Linear MTP draft batch/depth does not match sampling workspace");
    }
    this.batchSize = batchSize;
  }

  sampleDraft(logits: Tensor, depth: number): Tensor {
    if (!Number.isInteger(depth) || depth < 0 || depth >= this.depth || this.batchSize < 1) {
      throw new Error("Invalid linear MTP draft depth or unprepared batch");
    }
    if (logits.type !== "BF16" || logits.shape.length !== 2 || logits.shape[0] !== this.batchSize
      || logits.shape[1] !== this.targetSampler.vocabSize) {
      throw new Error("sampleDraft: invalid logits type or row capacities");
    }
    using probs = this.qDraftProbs.narrow(depth, 1);
    using ids = this.qDraftIds.narrow(depth, 1);
    // Distributed top-k exchanges candidates, not the full vocabulary logits.
    const candidates = logits.topk(this.candidateCount, this.targetSampler.vocabSize);
    using values = candidates.values;
    using globalIds = candidates.indices;
    this.glm.sampleCandidates(this.draftSampler.outToken, probs, ids, values, globalIds,
      this.draftSampler.temperatures, this.draftSampler.topKs, this.draftSampler.topPs,
      this.draftSampler.stepCounter, this.batchSize, this.candidateCount, this.capacity);
    if (!this.retainProposalsOnGpu) {
      using probsH = this.qHostProbs.narrow(depth, 1);
      using idsH = this.qHostIds.narrow(depth, 1);
      const bytes = this.batchSize * this.capacity * 4;
      probsH.memcpy(probs, bytes, MemcpyKind.DeviceToHost);
      idsH.memcpy(ids, bytes, MemcpyKind.DeviceToHost);
    }
    return this.draftSampler.outToken.narrow(0, this.batchSize);
  }

  finishDraft(): MtpProposal {
    if (this.batchSize < 1) throw new Error("Linear MTP draft is not prepared");
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
    for (let batch = 0; batch < this.batchSize; batch++) {
      const p = Buffer.alloc(this.depth * rowBytes);
      const t = Buffer.alloc(this.depth * rowBytes);
      // Each depth packs only the active B rows into its fixed maxB allocation.
      for (let depth = 0; depth < this.depth; depth++) {
        const offset = (depth * this.maxBatchSize + batch) * rowBytes;
        probs.copy(p, depth * rowBytes, offset, offset + rowBytes);
        ids.copy(t, depth * rowBytes, offset, offset + rowBytes);
      }
      probabilities.push(p);
      tokenIds.push(t);
    }
    return { probabilities, tokenIds, capacity: this.capacity };
  }

  prepareVerification(draft: MtpDraftBatch): void {
    const B = draft.targetTokens.length;
    const D = this.depth;
    const proposal = draft.proposal;
    const rowBytes = D * this.capacity * 4;
    this.prepareBatch(B, draft.topks.length);
    if (draft.topks.some(k => k !== 1) || draft.treeTokens.length !== B
      || draft.treeTokens.some(tokens => tokens.length !== D || tokens.some(t => !Number.isInteger(t) || t < 0 || t >= this.targetSampler.vocabSize))
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
        this.qInputProbs.memcpy2d(b * rowBytes, width, this.qDraftProbs, row * width, this.maxBatchSize * width, width, D, MemcpyKind.DeviceToDevice);
        this.qInputIds.memcpy2d(b * rowBytes, width, this.qDraftIds, row * width, this.maxBatchSize * width, width, D, MemcpyKind.DeviceToDevice);
      });
    } else {
      if (proposal.device || proposal.probabilities.length !== B || proposal.tokenIds.length !== B
        || proposal.probabilities.some(buf => !Buffer.isBuffer(buf) || buf.length !== rowBytes)
        || proposal.tokenIds.some(buf => !Buffer.isBuffer(buf) || buf.length !== rowBytes)) {
        throw new Error("Invalid host linear MTP proposal buffers or mode");
      }
      this.qInputProbsH.withPinnedBuffer(buf => proposal.probabilities.forEach((row, i) => row.copy(buf, i * rowBytes)));
      this.qInputIdsH.withPinnedBuffer(buf => proposal.tokenIds.forEach((row, i) => row.copy(buf, i * rowBytes)));
      this.qInputProbs.memcpy(this.qInputProbsH, B * rowBytes, MemcpyKind.HostToDevice);
      this.qInputIds.memcpy(this.qInputIdsH, B * rowBytes, MemcpyKind.HostToDevice);
    }
    this.draftTokensH.withPinnedBuffer(buf => {
      draft.treeTokens.forEach((tokens, b) => tokens.forEach((token, d) => buf.writeInt32LE(token, (b * D + d) * 4)));
    });
    this.draftTokens.memcpy(this.draftTokensH, B * D * 4, MemcpyKind.HostToDevice);
  }

  verify(logits: Tensor): { tokens: Tensor; numAccepted: Tensor } {
    const rows = this.batchSize * (this.depth + 1);
    if (this.batchSize < 1 || this.targetSampler.batchSize !== rows) {
      throw new Error("Linear MTP target sampler must contain B * (D + 1) rows");
    }
    this.targetSampler.sampleInto(logits, this.targetSampler.outToken, this.pProbs, this.pIds, this.capacity);
    this.glm.specRejectLinear(this.tokens, this.acceptedCounts, this.draftTokens,
      this.qInputProbs, this.qInputIds, this.pProbs, this.pIds, this.stepCounter,
      this.batchSize, this.depth, this.capacity);
    return { tokens: this.tokens.narrow(0, rows), numAccepted: this.acceptedCounts.narrow(0, this.batchSize) };
  }

  override free(): void {
    this.draftSampler.free();
    super.free();
  }
}
