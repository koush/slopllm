import { DeviceOps, TensorParallelism } from "./device_ops";
import { Qwen35Config } from "./qwen35_model";
import { Tensor } from "./tensor";
import { WorkspaceBase } from "./workspace";

export class Qwen35GdnState extends WorkspaceBase {
  convState: Tensor[];
  recurrentState: Tensor[];
  private cfg: Qwen35Config;
  readonly batchSize: number;

  constructor(ops: DeviceOps, cfg: Qwen35Config, batchSize = 1) {
    super(ops);
    this.cfg = cfg;
    this.batchSize = batchSize;
    const fullLinHeads = cfg.linearNumKeyHeads;
    const linKDim = cfg.linearKeyHeadDim;
    const linVDim = cfg.linearValueHeadDim;
    const fullConvDim = fullLinHeads * (linKDim * 2 + linVDim);
    const fullConvStateSize = fullConvDim * (cfg.linearConvKernelDim - 1);
    const fullRecurrentStateSize = fullLinHeads * linKDim * linVDim;
    this.convState = [];
    this.recurrentState = [];
    for (let i = 0; i < cfg.numHiddenLayers; i++) {
      if (cfg.layerTypes[i] === "linear_attention") {
        this.convState.push(this.alloc([batchSize, fullConvStateSize], "BF16", undefined, TensorParallelism.Row));
        this.recurrentState.push(this.alloc([batchSize, fullRecurrentStateSize], "F32", undefined, TensorParallelism.Row));
      } else {
        this.convState.push(null!);
        this.recurrentState.push(null!);
      }
    }
    this.zeroStates();
  }

  private zeroStates(): void {
    const fullLinHeads = this.cfg.linearNumKeyHeads;
    const linKDim = this.cfg.linearKeyHeadDim;
    const linVDim = this.cfg.linearValueHeadDim;
    const fullConvDim = fullLinHeads * (linKDim * 2 + linVDim);
    const fullConvStateSize = fullConvDim * (this.cfg.linearConvKernelDim - 1);
    const fullRecurrentStateSize = fullLinHeads * linKDim * linVDim;
    for (let i = 0; i < this.cfg.numHiddenLayers; i++) {
      if (this.cfg.layerTypes[i] === "linear_attention") {
        this.recurrentState[i].fill(0, this.batchSize * fullRecurrentStateSize);
        this.convState[i].fill(0, this.batchSize * fullConvStateSize);
      }
    }
  }

  reset(): void {
    this.zeroStates();
  }
}
