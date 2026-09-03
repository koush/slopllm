import { ChatModel, PhasedPrefillPlan } from "./chat_model";
import { DeviceOps } from "./device_ops";
import { ExecutionState } from "./execution-workspace";
import { Tensor } from "./tensor";

interface PhasedForward {
  state: ExecutionState;
  generator: Generator<void, Tensor, void>;
  owner?: Disposable;
}

type PairConsumer = (stateA: ExecutionState, hiddenA: Tensor, stateB: ExecutionState, hiddenB: Tensor) => void;

export interface RaggedInputSplit {
  inputA: number[][];
  inputB: number[][];
  nextA: number[];
}

export function splitRaggedInput(inputIds: number[][]): RaggedInputSplit | undefined {
  if (inputIds.length === 0 || inputIds.some(ids => ids.length < 2)) return undefined;

  const totalTokens = inputIds.reduce((sum, ids) => sum + ids.length, 0);
  const targetA = Math.floor(totalTokens / 2);
  const cuts = inputIds.map(() => 1);
  let remaining = targetA - cuts.length;

  for (let i = 0; i < inputIds.length && remaining > 0; i++) {
    const take = Math.min(inputIds[i].length - 2, remaining);
    cuts[i] += take;
    remaining -= take;
  }
  if (remaining !== 0) return undefined;

  const inputA = inputIds.map((ids, i) => ids.slice(0, cuts[i]));
  const inputB = inputIds.map((ids, i) => ids.slice(cuts[i]));
  return { inputA, inputB, nextA: inputB.map(ids => ids[0]) };
}

function closeGenerator(generator: Generator<void, Tensor, void>): void {
  generator.return(undefined as never);
}

export class PhasedPrefillRunner {
  constructor(private readonly model: ChatModel, private readonly glm: DeviceOps) {}

  runPair(stateA: ExecutionState, stateB: ExecutionState, consume?: PairConsumer): void {
    this.runGeneratorPair(
      { state: stateA, generator: this.model.forwardPhased(stateA) },
      { state: stateB, generator: this.model.forwardPhased(stateB) },
      consume,
    );
  }

  runPlanPair(planA: PhasedPrefillPlan, planB: PhasedPrefillPlan, consume?: PairConsumer): void {
    this.runGeneratorPair(
      { state: planA.state, generator: planA.generator, owner: planA },
      { state: planB.state, generator: planB.generator, owner: planB },
      consume,
    );
  }

  private runGeneratorPair(a: PhasedForward, b: PhasedForward, consume?: PairConsumer): void {
    let resultA: IteratorResult<void, Tensor> | undefined;
    let resultB: IteratorResult<void, Tensor> | undefined;
    if (a.state.batchSize === 1 && b.state.batchSize === 1) {
      const installPrefetch = (cacheIdx: number, field: string, stream: unknown) => {
        const key = `sparseMlaPrefetchLayer_${cacheIdx}`;
        const extra = b.state.extras.get(key) ?? {};
        extra[field] = stream;
        b.state.extras.set(key, extra);
      };
      a.state.extras.set("setCkv", (cacheIdx: number, stream: unknown) => installPrefetch(cacheIdx, "stream", stream));
      a.state.extras.set("setIndexerK", (cacheIdx: number, stream: unknown) => installPrefetch(cacheIdx, "indexerStream", stream));
    }
    try {
      resultA = a.generator.next();
      if (resultA.done) {
        resultA.value[Symbol.dispose]();
        throw new Error("Phased prefill A completed before its first phase boundary");
      }
      while (!resultA.done) {
        using streamB = this.glm.withStream(() => b.generator.next());
        try {
          resultA = a.generator.next();
        } finally {
          streamB.streamWaitEvent();
        }
        resultB = streamB.result;
        if (resultB.done) {
          resultB.value[Symbol.dispose]();
          throw new Error("Phased prefill B completed before A");
        }
      }

      resultB = b.generator.next();
      if (!resultB.done) throw new Error("Phased prefill B did not complete one phase after A");
      using hiddenA = resultA.value;
      using hiddenB = resultB.value;
      consume?.(a.state, hiddenA, b.state, hiddenB);
    } finally {
      a.state.extras.delete("setCkv");
      a.state.extras.delete("setIndexerK");
      try {
        if (!resultB?.done) closeGenerator(b.generator);
        if (!resultA?.done) closeGenerator(a.generator);
      } finally {
        b.owner?.[Symbol.dispose]();
        a.owner?.[Symbol.dispose]();
      }
    }
  }
}
