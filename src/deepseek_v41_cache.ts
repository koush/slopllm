import { PagedKVCache } from "./paged_kv";
import type { DeepseekV41Model } from "./deepseek_v41_model";
import type { Tensor } from "./tensor";
import { WorkspaceBase } from "./workspace";

export const DSV41_KV_SOURCES = [2, 8, 14, 20];
export const DSV41_INDEX_SOURCES = [2, 8, 14, 20, 24, 28, 32, 36];

/** Ordinary page ownership for 40 local caches and four shared compressed caches.
 * Unfinished compressor pairs are live batch-slot scratch, restored by a one-token
 * overlap on resumed execution. They are not part of the reusable prefix.
 */
export class DeepseekV41Cache extends PagedKVCache {
  readonly compressorPending = new Map<number, Tensor>();
  readonly compressorPosition = new Map<number, Tensor>();
  private readonly compressorWorkspace: WorkspaceBase;

  constructor(readonly model: DeepseekV41Model, maxPages: number, maxBatch: number, readonly maxContext: number) {
    super(model.ops, 1, 512, 44, maxPages, maxBatch, 64, 512, 64, false, 128,
      Array.from({ length: 44 }, (_, i) => i < 40), false, [], "dsv4");
    this.compressorWorkspace = new WorkspaceBase(model.ops);
    for (const source of DSV41_KV_SOURCES) {
      this.compressorPending.set(source, this.compressorWorkspace.alloc([maxBatch, 1024], "F32", `compressor.pending.${source}`));
      const position = this.compressorWorkspace.alloc([maxBatch], "I32", `compressor.position.${source}`);
      position.fill(-1, maxBatch);
      this.compressorPosition.set(source, position);
    }
  }

  compressed(layer: number): Tensor { return this.ckvData[40 + DSV41_KV_SOURCES.indexOf(layer)]; }
  indexKeys(layer: number): Tensor { return this.kData[40 + DSV41_KV_SOURCES.indexOf(layer)]; }

  override free(): void {
    this.compressorWorkspace?.free();
    super.free();
  }
}
