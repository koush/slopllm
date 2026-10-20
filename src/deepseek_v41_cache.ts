import { PagedKVCache } from "./paged_kv";
import type { DeepseekV41Model } from "./deepseek_v41_model";
import type { Tensor } from "./tensor";

export const DSV41_KV_SOURCES = [2, 8, 14, 20];
export const DSV41_INDEX_SOURCES = [2, 8, 14, 20, 24, 28, 32, 36];

/** Ordinary page ownership for 40 local caches and four shared compressed caches.
 * Recurrent compressor carry belongs to execution-scoped LayerHolders and is
 * reconstructed on resumed execution, rather than stored in the reusable prefix.
 */
export class DeepseekV41Cache extends PagedKVCache {
  readonly draft?: PagedKVCache;

  constructor(readonly model: DeepseekV41Model, maxPages: number, maxBatch: number, readonly maxContext: number) {
    super(model.ops, 1, 512, 44, maxPages, maxBatch, 64, 512, 64, false, 128,
      Array.from({ length: 44 }, (_, i) => i < 40), false, [], "dsv4");
    if (model.mtp) this.draft = new PagedKVCache(model.ops, 1, 512, 3, maxBatch * 5, maxBatch,
      64, 512, 64, false, 128, [true, true, true], false, [], "dsv4");
  }

  override reset(batchSize: number): void {
    this.draft?.reset(0);
    super.reset(batchSize);
  }

  compressed(layer: number): Tensor { return this.ckvData[40 + DSV41_KV_SOURCES.indexOf(layer)]; }
  indexKeys(layer: number): Tensor { return this.kData[40 + DSV41_KV_SOURCES.indexOf(layer)]; }

  override free(): void {
    this.draft?.free();
    super.free();
  }
}
