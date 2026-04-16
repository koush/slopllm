import { GlmOps } from "./glm_ops";
import { Tensor } from "./tensor";
import { Qwen35Config } from "./qwen35_model";

export class Qwen35GdnState {
  convState: Tensor[];
  recurrentState: Tensor[];
  private glm: GlmOps;
  private cfg: Qwen35Config;

  constructor(glm: GlmOps, cfg: Qwen35Config) {
    this.glm = glm;
    this.cfg = cfg;
    const linHeads = cfg.linearNumKeyHeads;
    const linKDim = cfg.linearKeyHeadDim;
    const linVDim = cfg.linearValueHeadDim;
    const convDim = linHeads * (linKDim * 2 + linVDim);
    const kernelSize = cfg.linearConvKernelDim;
    this.convState = [];
    this.recurrentState = [];
    for (let i = 0; i < cfg.numHiddenLayers; i++) {
      if (cfg.layerTypes[i] === "linear_attention") {
        this.convState.push(Tensor.alloc(glm, [convDim * (kernelSize - 1)], "BF16"));
        this.recurrentState.push(Tensor.alloc(glm, [linHeads * linKDim * linVDim], "F32"));
      } else {
        this.convState.push(null!);
        this.recurrentState.push(null!);
      }
    }
    this.zeroStates();
  }

  private zeroStates(): void {
    const linHeads = this.cfg.linearNumKeyHeads;
    const linKDim = this.cfg.linearKeyHeadDim;
    const linVDim = this.cfg.linearValueHeadDim;
    const convDim = linHeads * (linKDim * 2 + linVDim);
    const kernelSize = this.cfg.linearConvKernelDim;
    for (let i = 0; i < this.cfg.numHiddenLayers; i++) {
      if (this.cfg.layerTypes[i] === "linear_attention") {
        this.glm.fill(this.recurrentState[i].data, 0, 2 * linHeads * linKDim * linVDim);
        this.glm.fill(this.convState[i].data, 0, convDim * (kernelSize - 1));
      }
    }
  }

  reset(): void {
    this.zeroStates();
  }

  free(): void {
    for (const t of this.convState) { if (t) t.free(); }
    for (const t of this.recurrentState) { if (t) t.free(); }
    this.convState = [];
    this.recurrentState = [];
  }
}
