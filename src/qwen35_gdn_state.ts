import { GlmOps, BF16, I32 } from "./glm_ops";
import { Tensor } from "./tensor";
import { Qwen35Config } from "./qwen35_model";

export class Qwen35GdnState {
  convState: Tensor[];
  recurrentState: Tensor[];
  cuSeqlens: Tensor;
  private glm: GlmOps;
  private cfg: Qwen35Config;
  readonly batchSize: number;

  readonly convStateSize: number;
  readonly recurrentStateSize: number;
  readonly convStateStride: number;
  readonly recurrentStateStride: number;

  constructor(glm: GlmOps, cfg: Qwen35Config, batchSize = 1) {
    this.glm = glm;
    this.cfg = cfg;
    this.batchSize = batchSize;
    const linHeads = cfg.linearNumKeyHeads;
    const linKDim = cfg.linearKeyHeadDim;
    const linVDim = cfg.linearValueHeadDim;
    const convDim = linHeads * (linKDim * 2 + linVDim);
    const kernelSize = cfg.linearConvKernelDim;
    this.convStateSize = convDim * (kernelSize - 1);
    this.recurrentStateSize = linHeads * linKDim * linVDim;
    this.convStateStride = this.convStateSize;
    this.recurrentStateStride = this.recurrentStateSize;
    this.convState = [];
    this.recurrentState = [];
    for (let i = 0; i < cfg.numHiddenLayers; i++) {
      if (cfg.layerTypes[i] === "linear_attention") {
        this.convState.push(Tensor.alloc(glm, [batchSize * this.convStateSize], "BF16"));
        this.recurrentState.push(Tensor.alloc(glm, [batchSize * this.recurrentStateSize], "F32"));
      } else {
        this.convState.push(null!);
        this.recurrentState.push(null!);
      }
    }
    this.cuSeqlens = Tensor.alloc(glm, [batchSize + 1], "I32");
    this.zeroStates();
  }

  private zeroStates(): void {
    for (let i = 0; i < this.cfg.numHiddenLayers; i++) {
      if (this.cfg.layerTypes[i] === "linear_attention") {
        this.glm.fill(this.recurrentState[i].data, 0, 2 * this.batchSize * this.recurrentStateSize);
        this.glm.fill(this.convState[i].data, 0, this.batchSize * this.convStateSize);
      }
    }
  }

  uploadCuSeqlens(seqLens: number[]): void {
    const cu = new Int32Array(this.batchSize + 1);
    cu[0] = 0;
    for (let i = 0; i < this.batchSize; i++) {
      cu[i + 1] = cu[i] + seqLens[i];
    }
    this.cuSeqlens.h2d(Buffer.from(cu.buffer, cu.byteOffset, cu.byteLength));
  }

  reset(): void {
    this.zeroStates();
  }

  free(): void {
    for (const t of this.convState) { if (t) t.free(); }
    for (const t of this.recurrentState) { if (t) t.free(); }
    this.convState = [];
    this.recurrentState = [];
    this.cuSeqlens.free();
  }
}
