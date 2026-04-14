import { GlmOps, BF16 } from "./glm_ops";

export class FlatKVCache {
  private glm: GlmOps;
  readonly nKv: number;
  readonly hd: number;
  readonly nLayers: number;
  readonly maxBatch: number;
  readonly maxSeqLen: number;
  kData: number[];
  vData: number[];
  cachePos: number;

  constructor(glm: GlmOps, nKv: number, hd: number, nLayers: number, maxBatch: number, maxSeqLen: number) {
    this.glm = glm;
    this.nKv = nKv;
    this.hd = hd;
    this.nLayers = nLayers;
    this.maxBatch = maxBatch;
    this.maxSeqLen = maxSeqLen;
    this.cachePos = 0;
    this.kData = [];
    this.vData = [];
    for (let i = 0; i < nLayers; i++) {
      this.kData.push(glm.alloc(maxBatch * nKv * maxSeqLen * hd * BF16));
      this.vData.push(glm.alloc(maxBatch * nKv * maxSeqLen * hd * BF16));
    }
  }

  free(): void {
    const glm = this.glm;
    for (const ptr of this.kData) glm.freeBuf(ptr);
    for (const ptr of this.vData) glm.freeBuf(ptr);
    this.kData = [];
    this.vData = [];
  }

  reset(): void {
    this.cachePos = 0;
  }
}
