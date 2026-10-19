import { PagedKVCache } from "./paged_kv";
import type { DeepseekV41Model } from "./deepseek_v41_model";
import type { Tensor } from "./tensor";
import { WorkspaceBase } from "./workspace";
import { UsingHolder } from "./using-holder";
import type { ExecutionState, ExecutionWorkspace } from "./execution-workspace";

export const DSV41_KV_SOURCES = [2, 8, 14, 20];
export const DSV41_INDEX_SOURCES = [2, 8, 14, 20, 24, 28, 32, 36];

/** Immutable once its producing forward has completed. Views retain both allocations. */
export class CompressorCarry implements Disposable {
  constructor(readonly pending: Tensor, readonly pendingPosition: Tensor) {}
  viewClone(): CompressorCarry { return new CompressorCarry(this.pending.viewClone(), this.pendingPosition.viewClone()); }
  [Symbol.dispose](): void { this.pending[Symbol.dispose](); this.pendingPosition[Symbol.dispose](); }
}

class CompressorPlan implements Disposable {
  readonly inputs = new Map<number, CompressorCarry>();
  readonly outputs = new Map<number, CompressorCarry>();
  [Symbol.dispose](): void {
    for (const carry of [...this.inputs.values(), ...this.outputs.values()]) carry[Symbol.dispose]();
    this.inputs.clear(); this.outputs.clear();
  }
}

/** Ordinary page ownership for 40 local caches and four shared compressed caches.
 * Unfinished compressor pairs are owned immutable values, restored by a one-token
 * overlap on resumed execution. They are not part of the reusable prefix.
 */
export class DeepseekV41Cache extends PagedKVCache {
  readonly compressorCarry = new Map<number, UsingHolder<CompressorCarry>>();
  private readonly compressorWorkspace: WorkspaceBase;
  private readonly compressorPlans = new Map<ExecutionWorkspace, Map<number, CompressorPlan>>();

  constructor(readonly model: DeepseekV41Model, maxPages: number, maxBatch: number, readonly maxContext: number) {
    super(model.ops, 1, 512, 44, maxPages, maxBatch, 64, 512, 64, false, 128,
      Array.from({ length: 44 }, (_, i) => i < 40), false, [], "dsv4");
    this.compressorWorkspace = new WorkspaceBase(model.ops);
    for (const source of DSV41_KV_SOURCES.filter(source => source !== 20)) this.compressorCarry.set(source, new UsingHolder<CompressorCarry>(undefined!));
  }

  /** Reserve fresh outputs before graph capture/replay. A plan retains its inputs
   * even when a subsequent plan replaces the current carry (plan+plan+run+run).
   * Reusing a workspace plan slot releases the preceding plan's references. */
  planCompressor(state: ExecutionState): CompressorPlan {
    const existing = state.extras.get("dsv41.compressor") as CompressorPlan | undefined;
    if (existing) return existing;
    let plans = this.compressorPlans.get(state.ws);
    if (!plans) this.compressorPlans.set(state.ws, plans = new Map());
    plans.get(state.slot)?.[Symbol.dispose]();
    const plan = new CompressorPlan();
    plans.set(state.slot, plan);
    for (const [source, holder] of this.compressorCarry) {
      if (holder.value) plan.inputs.set(source, holder.value.viewClone());
      const pending = this.compressorWorkspace.alloc([this.maxBatch, 1024], "F32");
      const position = this.compressorWorkspace.alloc([this.maxBatch], "I32");
      // Unused batch slots must remain invalid if the next plan expands the batch.
      if (state.batchSize < this.maxBatch) {
        using unused = position.narrow(state.batchSize, this.maxBatch - state.batchSize);
        unused.fill(-1, unused.numElements);
      }
      holder.replace(new CompressorCarry(pending, position));
      plan.outputs.set(source, holder.value.viewClone());
    }
    state.extras.set("dsv41.compressor", plan);
    return plan;
  }

  compressorPlan(state: ExecutionState): CompressorPlan {
    const plan = state.extras.get("dsv41.compressor") as CompressorPlan | undefined;
    if (!plan) throw new Error("V4.1 compressor must be planned before forward/capture");
    return plan;
  }

  private releaseCompressor(): void {
    for (const plans of this.compressorPlans.values()) for (const plan of plans.values()) plan[Symbol.dispose]();
    this.compressorPlans.clear();
    for (const holder of this.compressorCarry.values()) holder.release();
  }

  override reset(batchSize: number): void {
    this.releaseCompressor();
    super.reset(batchSize);
  }

  compressed(layer: number): Tensor { return this.ckvData[40 + DSV41_KV_SOURCES.indexOf(layer)]; }
  indexKeys(layer: number): Tensor { return this.kData[40 + DSV41_KV_SOURCES.indexOf(layer)]; }

  override free(): void {
    this.releaseCompressor();
    this.compressorWorkspace?.free();
    super.free();
  }
}
