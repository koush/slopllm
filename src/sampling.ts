import { type SamplingParams } from "./chat_model";
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

  sample(logits: Tensor): Tensor {
    return this.sampleInto(logits, this.outToken);
  }

  sampleInto(logits: Tensor, outToken: Tensor): Tensor {
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
    );

    return outToken;
  }
}
