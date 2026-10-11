import { type DeviceOps, TensorParallelism } from "./device_ops";
import type { Tensor } from "./tensor";
import { WorkspaceBase } from "./workspace";

/** State for one KDA layer, head-sharded under TP. Caller owns sequence slots. */
export class KdaState extends WorkspaceBase {
  readonly qConv: Tensor;
  readonly kConv: Tensor;
  readonly vConv: Tensor;
  readonly recurrent: Tensor;

  constructor(ops: DeviceOps, readonly batchSize: number, readonly numHeads: number) {
    super(ops);
    if (!Number.isInteger(batchSize) || batchSize < 1 || !Number.isInteger(numHeads) || numHeads < 1 || numHeads % ops.worldSize) {
      throw new Error("KdaState requires a positive batch and heads divisible by TP size");
    }
    const convShape = [batchSize, numHeads * 128 * 3];
    this.qConv = this.alloc(convShape, "BF16", "q_conv", TensorParallelism.Row);
    this.kConv = this.alloc(convShape, "BF16", "k_conv", TensorParallelism.Row);
    this.vConv = this.alloc(convShape, "BF16", "v_conv", TensorParallelism.Row);
    this.recurrent = this.alloc([batchSize, numHeads * 128 * 128], "F32", "recurrent", TensorParallelism.Row);
    this.reset();
  }

  reset(): void {
    for (const t of [this.qConv, this.kConv, this.vConv, this.recurrent]) t.fill(0, t.numElements);
  }
}

export interface KdaWeights {
  q: Tensor; k: Tensor; v: Tensor;
  qConv: Tensor; kConv: Tensor; vConv: Tensor;
  fA: Tensor; fB: Tensor; beta: Tensor;
  gA: Tensor; gB: Tensor; norm: Tensor; out: Tensor;
  aLog: Tensor; dtBias: Tensor;
}

/** Complete KDA attention sublayer on already-normalized packed hidden states.
 * cuSeqlens partitions tokens by state slot; supports decode and resumed prefill.
 * Allocation tracking belongs to the caller. Output is PartialSum under TP.
 */
export function kdaAttention(hidden: Tensor, w: KdaWeights, state: KdaState, cuSeqlens: Tensor, eps = 1e-5): Tensor {
  const width = state.numHeads * 128, tokens = hidden.shape[0];
  if (hidden.type !== "BF16" || hidden.shape.length !== 2 || hidden.workspace.ops !== state.ops ||
      cuSeqlens.type !== "I32" || cuSeqlens.numElements !== state.batchSize + 1) {
    throw new Error("kdaAttention: incompatible hidden states or sequence metadata");
  }
  for (const weight of [w.q, w.k, w.v, w.fB, w.gB]) {
    if (weight.type !== "BF16" || weight.shape[0] !== width) throw new Error("kdaAttention: projection width must match state heads");
  }
  for (const weight of [w.qConv, w.kConv, w.vConv]) {
    if (weight.type !== "BF16" || weight.shape.join(",") !== `${width},1,4`) throw new Error("kdaAttention: expected [heads * 128, 1, 4] convolution weights");
  }
  const conv = (projection: Tensor, weight: Tensor, history: Tensor): Tensor => {
    using input = hidden.linear(projection);
    const output = hidden.workspace.alloc([tokens, width], "BF16", undefined, TensorParallelism.Row);
    output.causalConv1d(history, input, weight, cuSeqlens, width, tokens, 4, state.batchSize, width * 3, 1, width);
    return output;
  };
  using q = conv(w.q, w.qConv, state.qConv);
  using k = conv(w.k, w.kConv, state.kConv);
  using v = conv(w.v, w.vConv, state.vConv);
  using fA = hidden.linear(w.fA);
  using forget = fA.linear(w.fB);
  using beta = hidden.linear(w.beta);
  using core = hidden.workspace.ops.kdaScan(q, state.recurrent, k, v, forget, beta, w.aLog, w.dtBias, cuSeqlens);
  using gA = hidden.linear(w.gA);
  using gate = gA.linear(w.gB);
  using gated = hidden.workspace.ops.kdaNorm(core, gate, w.norm, eps);
  return gated.linear(w.out);
}
