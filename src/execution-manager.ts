import type { ExecutionState } from "./execution-workspace";
import type { Tensor } from "./tensor";

export interface ExecutionOptions<I extends Record<string, Tensor>> {
  states: readonly ExecutionState[];
  inputs: I;
  key: readonly (string | number)[];
}

export interface ExecutionResult<T> {
  warmup: boolean;
  result: T;
}

export interface ExecutionManager {
  readonly captureEnabled: boolean;
  execute<T, I extends Record<string, Tensor>>(
    options: ExecutionOptions<I>, fn: (inputs: I) => T,
  ): ExecutionResult<T>;
}

/** Submits work immediately without capturing or synchronizing the device. */
export class EagerExecution implements ExecutionManager {
  readonly captureEnabled = false;

  execute<T, I extends Record<string, Tensor>>(
    options: ExecutionOptions<I>, fn: (inputs: I) => T,
  ): ExecutionResult<T> {
    return { warmup: false, result: fn(options.inputs) };
  }
}
