import type { DeviceOps } from "./device_ops";
import type { Tensor } from "./tensor";

function checkTensorLayout(ops: DeviceOps, t: Tensor, type: string, shape: number[], op: string): void {
  if (t.type !== type || t.shape.length !== shape.length || t.shape.some((n, i) => n !== shape[i])) {
    throw new Error(`${op}: expected ${type} [${shape}], got ${t.type} [${t.shape}]`);
  }
  if (t.workspace.ops !== ops) throw new Error(`${op}: tensors must use the executing backend`);
}

export function validateKdaScan(ops: DeviceOps, query: Tensor, state: Tensor, key: Tensor, value: Tensor, forget: Tensor, beta: Tensor, aLog: Tensor, dtBias: Tensor, cuSeqlens: Tensor, lowerBound: number): void {
  const [tokens, width] = query.shape;
  if (query.shape.length !== 2 || !width || width % 128 || !Number.isFinite(lowerBound) || lowerBound >= 0) {
    throw new Error("kdaScan: expected [tokens, heads * 128] and a negative finite gate lower bound");
  }
  const heads = width / 128;
  for (const t of [query, key, value, forget]) checkTensorLayout(ops, t, "BF16", [tokens, width], "kdaScan");
  checkTensorLayout(ops, beta, "BF16", [tokens, heads], "kdaScan beta");
  checkTensorLayout(ops, aLog, "F32", [heads], "kdaScan A_log");
  checkTensorLayout(ops, dtBias, "F32", [width], "kdaScan dt_bias");
  const batch = state.shape[0];
  if (!Number.isInteger(batch) || batch < 1) throw new Error("kdaScan: empty state batch");
  checkTensorLayout(ops, state, "F32", [batch, heads * 128 * 128], "kdaScan state");
  checkTensorLayout(ops, cuSeqlens, "I32", [batch + 1], "kdaScan cuSeqlens");
}

export function validateKdaNorm(ops: DeviceOps, input: Tensor, gate: Tensor, weight: Tensor, eps: number): void {
  if (input.shape.length !== 2 || !input.shape[1] || input.shape[1] % 128 || !Number.isFinite(eps) || eps <= 0) {
    throw new Error("kdaNorm: expected [tokens, heads * 128] and positive epsilon");
  }
  checkTensorLayout(ops, input, "BF16", input.shape, "kdaNorm");
  checkTensorLayout(ops, gate, "BF16", input.shape, "kdaNorm gate");
  checkTensorLayout(ops, weight, "BF16", [128], "kdaNorm weight");
}

export function validateMhcPre(ops: DeviceOps, input: Tensor, fn: Tensor, base: Tensor, scale: Tensor, normEps: number, hcEps: number, iters: number): void {
  if (input.shape.length !== 3 || input.shape[1] !== 4 || input.shape[2] < 1 ||
      !Number.isFinite(normEps) || normEps <= 0 || !Number.isFinite(hcEps) || hcEps <= 0 || !Number.isInteger(iters) || iters < 1) {
    throw new Error("mhcPre: expected [tokens, 4, hidden] and positive epsilons/iteration count");
  }
  checkTensorLayout(ops, input, "BF16", input.shape, "mhcPre");
  checkTensorLayout(ops, fn, "BF16", [24, 4 * input.shape[2]], "mhcPre fn");
  checkTensorLayout(ops, base, "F32", [24], "mhcPre base");
  checkTensorLayout(ops, scale, "F32", [3], "mhcPre scale");
}

export function validateMhcPost(ops: DeviceOps, residual: Tensor, block: Tensor, post: Tensor, comb: Tensor): void {
  if (residual.shape.length !== 3 || residual.shape[1] !== 4) throw new Error("mhcPost: expected [tokens, 4, hidden]");
  checkTensorLayout(ops, residual, "BF16", residual.shape, "mhcPost");
  checkTensorLayout(ops, block, "BF16", [residual.shape[0], residual.shape[2]], "mhcPost block");
  checkTensorLayout(ops, post, "F32", [residual.shape[0], 4], "mhcPost post");
  checkTensorLayout(ops, comb, "F32", [residual.shape[0], 4, 4], "mhcPost comb");
}

export function validateMhcStreams(ops: DeviceOps, input: Tensor, expand: boolean): void {
  if (input.type !== "BF16" || (expand ? input.shape.length !== 2 : input.shape.length !== 3 || input.shape[1] !== 4)) {
    throw new Error("mhcStreams: expected BF16 [tokens, hidden] for expand or [tokens, 4, hidden] for collapse");
  }
  checkTensorLayout(ops, input, "BF16", input.shape, "mhcStreams");
}
