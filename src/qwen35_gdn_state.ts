import { DeviceOps } from "./device_ops";
import { Qwen35Config } from "./qwen35_model";
import { Tensor } from "./tensor";
import { WorkspaceBase } from "./workspace";

export class Qwen35GdnState extends WorkspaceBase {
  convState: Tensor[];
  recurrentState: Tensor[];
  cuSeqlens: Tensor;
  private cfg: Qwen35Config;
  readonly batchSize: number;
  private readonly convStateSize: number;
  private readonly recurrentStateSize: number;

  constructor(glm: DeviceOps, cfg: Qwen35Config, batchSize = 1) {
    super(glm);
    this.cfg = cfg;
    this.batchSize = batchSize;
    const fullLinHeads = cfg.linearNumKeyHeads;
    const linKDim = cfg.linearKeyHeadDim;
    const linVDim = cfg.linearValueHeadDim;
    const linHeads = fullLinHeads / glm.worldSize;
    const convDim = linHeads * (linKDim * 2 + linVDim);
    this.convStateSize = convDim * (cfg.linearConvKernelDim - 1);
    this.recurrentStateSize = linHeads * linKDim * linVDim;
    this.convState = [];
    this.recurrentState = [];
    for (let i = 0; i < cfg.numHiddenLayers; i++) {
      if (cfg.layerTypes[i] === "linear_attention") {
        this.convState.push(this.alloc([batchSize * this.convStateSize], "BF16"));
        this.recurrentState.push(this.alloc([batchSize * this.recurrentStateSize], "F32"));
      } else {
        this.convState.push(null!);
        this.recurrentState.push(null!);
      }
    }
    this.cuSeqlens = this.alloc([batchSize + 1], "I32", "cuSeqlens");
    this.zeroStates();
  }

  private zeroStates(): void {
    for (let i = 0; i < this.cfg.numHiddenLayers; i++) {
      if (this.cfg.layerTypes[i] === "linear_attention") {
        this.recurrentState[i].fill(0, 2 * this.batchSize * this.recurrentStateSize);
        this.convState[i].fill(0, this.batchSize * this.convStateSize);
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
}
