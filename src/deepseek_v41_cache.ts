import { PagedKVCache } from "./paged_kv";
import type { DeepseekV41Model, LayerHolders } from "./deepseek_v41_model";
import type { Tensor } from "./tensor";
import { WorkspaceBase } from "./workspace";
import { UsingHolder } from "./using-holder";
import type { ExecutionState } from "./execution-workspace";

export const DSV41_KV_SOURCES = [2, 8, 14, 20];
export const DSV41_INDEX_SOURCES = [2, 8, 14, 20, 24, 28, 32, 36];

/** Immutable once its producing forward has completed. Views retain both allocations. */
export class CompressorCarry implements Disposable {
  constructor(readonly pending: Tensor, readonly pendingPosition: Tensor) {}
  viewClone(): CompressorCarry { return new CompressorCarry(this.pending.viewClone(), this.pendingPosition.viewClone()); }
  [Symbol.dispose](): void { this.pending[Symbol.dispose](); this.pendingPosition[Symbol.dispose](); }
}

/** Ordinary page ownership for 40 local caches and four shared compressed caches.
 * Unfinished compressor pairs are owned immutable values, restored by a one-token
 * overlap on resumed execution. They are not part of the reusable prefix.
 */
export class DeepseekV41Cache extends PagedKVCache {
  readonly draft?: PagedKVCache;
  readonly compressorCarry = new Map<number, UsingHolder<CompressorCarry>>();
  readonly compressorWorkspace: WorkspaceBase;

  constructor(readonly model: DeepseekV41Model, maxPages: number, maxBatch: number, readonly maxContext: number) {
    super(model.ops, 1, 512, 44, maxPages, maxBatch, 64, 512, 64, false, 128,
      Array.from({ length: 44 }, (_, i) => i < 40), false, [], "dsv4");
    this.compressorWorkspace = new WorkspaceBase(model.ops);
    if (model.mtp) this.draft = new PagedKVCache(model.ops, 1, 512, 3, maxBatch * 5, maxBatch,
      64, 512, 64, false, 128, [true, true, true], false, [], "dsv4");
    for (const source of DSV41_KV_SOURCES.filter(source => source !== 20)) this.compressorCarry.set(source, new UsingHolder<CompressorCarry>(undefined!));
  }

  /** Select an immutable first-half row at each accepted verification boundary. */
  commitCompressor(state: ExecutionState, holders: LayerHolders, accepted: readonly number[]): void {
    if (accepted.length !== state.batchSize || accepted.some((count, b) => !Number.isInteger(count)
      || count < 0 || count >= state.seqLens[b])) throw new Error("Invalid compressor acceptance boundary");
    const ws = this.compressorWorkspace;
    const indices = new Int32Array(this.maxBatch), positions = new Int32Array(this.maxBatch).fill(-1);
    let offset = 0;
    for (let b = 0; b < this.maxBatch; b++) {
      indices[b] = b;
      if (b >= state.batchSize) continue;
      const start = state.seqKvLens[b] - state.seqLens[b];
      const even = (start + accepted[b]) & ~1;
      positions[b] = even;
      if (even >= start) indices[b] = this.maxBatch + offset + even - start;
      offset += state.seqLens[b];
    }
    using selection = ws.alloc([this.maxBatch], "I32"); selection.h2d(Buffer.from(indices.buffer));
    for (const [layer, holder] of this.compressorCarry) {
      const previous = holders.compressorInputs.get(layer);
      const rows = holders.compressorRows.get(layer)?.value;
      if (!previous || !rows) throw new Error("Compressor verification requires retained inputs and projection rows");
      using joined = previous.pending.cat([rows], 0);
      const pending = joined.indexSelect(selection);
      const position = ws.alloc([this.maxBatch], "I32"); position.h2d(Buffer.from(positions.buffer));
      holder.replace(new CompressorCarry(pending, position));
    }
  }

  private releaseCompressor(): void {
    for (const holder of this.compressorCarry.values()) holder.release();
  }

  override reset(batchSize: number): void {
    this.releaseCompressor();
    this.draft?.reset(0);
    super.reset(batchSize);
  }

  compressed(layer: number): Tensor { return this.ckvData[40 + DSV41_KV_SOURCES.indexOf(layer)]; }
  indexKeys(layer: number): Tensor { return this.kData[40 + DSV41_KV_SOURCES.indexOf(layer)]; }

  override free(): void {
    this.releaseCompressor();
    this.draft?.free();
    this.compressorWorkspace?.free();
    super.free();
  }
}
