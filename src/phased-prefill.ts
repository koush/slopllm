import { ChatModel } from "./chat_model";
import { DeviceOps } from "./device_ops";
import { ExecutionState } from "./execution-workspace";
import { Tensor } from "./tensor";

interface PendingForward {
  state: ExecutionState;
  generator: Generator<void, Tensor, void>;
  result: IteratorResult<void, Tensor>;
  owner?: Disposable;
}

type PairConsumer = (stateA: ExecutionState, hiddenA: Tensor, stateB: ExecutionState, hiddenB: Tensor) => void;

function closeGenerator(generator: Generator<void, Tensor, void>): void {
  generator.return(undefined as never);
}

export class PhasedPrefillRunner implements Disposable {
  private pending?: PendingForward;

  constructor(private readonly model: ChatModel, private readonly glm: DeviceOps) {}

  get hasPending(): boolean {
    return this.pending !== undefined;
  }

  enqueue(state: ExecutionState, consume?: PairConsumer): boolean {
    return this.enqueueGenerator(state, this.model.forwardPhased(state), consume);
  }

  enqueueGenerator(state: ExecutionState, generator: Generator<void, Tensor, void>, consume?: PairConsumer, owner?: Disposable): boolean {
    if (!this.pending) {
      try {
        const result = generator.next();
        if (result.done) {
          result.value[Symbol.dispose]();
          throw new Error("Phased prefill completed before its first phase boundary");
        }
        this.pending = { state, generator, result, owner };
        return false;
      } catch (err) {
        owner?.[Symbol.dispose]();
        throw err;
      }
    }

    const pending = this.pending;
    this.pending = undefined;
    const generatorB = generator;
    let resultA = pending.result;
    let resultB: IteratorResult<void, Tensor> | undefined;

    try {
      while (!resultA.done) {
        using streamB = this.glm.withStream(() => generatorB.next());
        try {
          resultA = pending.generator.next();
        } finally {
          streamB.streamWaitEvent();
        }
        resultB = streamB.result;
        if (resultB.done) throw new Error("Phased prefill B completed before A");
      }

      resultB = generatorB.next();
      if (!resultB.done) throw new Error("Phased prefill B did not complete one phase after A");
      using hiddenA = resultA.value;
      using hiddenB = resultB.value;
      consume?.(pending.state, hiddenA, state, hiddenB);
      return true;
    } finally {
      try {
        if (!resultA.done) closeGenerator(pending.generator);
        if (!resultB?.done) closeGenerator(generatorB);
      } finally {
        pending.owner?.[Symbol.dispose]();
        owner?.[Symbol.dispose]();
      }
    }
  }

  flush(): boolean {
    if (!this.pending) return false;
    const pending = this.pending;
    this.pending = undefined;
    let result = pending.result;
    try {
      while (!result.done) result = pending.generator.next();
      using hidden = result.value;
      return true;
    } finally {
      try {
        if (!result.done) closeGenerator(pending.generator);
      } finally {
        pending.owner?.[Symbol.dispose]();
      }
    }
  }

  [Symbol.dispose](): void {
    if (!this.pending) return;
    const { generator, owner } = this.pending;
    this.pending = undefined;
    try {
      closeGenerator(generator);
    } finally {
      owner?.[Symbol.dispose]();
    }
  }
}
