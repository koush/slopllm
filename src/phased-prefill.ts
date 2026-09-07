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
  if (inputIds.length !== 1 || inputIds[0].length < 2) return undefined;
  const ids = inputIds[0];
  const cut = Math.floor(ids.length / 2);
  return { inputA: [ids.slice(0, cut)], inputB: [ids.slice(cut)], nextA: [ids[cut]] };
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
    try {
      if (a.state.batchSize !== 1 || b.state.batchSize !== 1) {
        throw new Error("Phased prefill requires a single sequence in each plan");
      }
      const installPrefetch = (cacheIdx: number, field: string, stream: unknown) => {
        const key = `sparseMlaPrefetchLayer_${cacheIdx}`;
        const extra = b.state.extras.get(key) ?? {};
        extra[field] = stream;
        b.state.extras.set(key, extra);
      };
      a.state.extras.set("setCkv", (cacheIdx: number, stream: unknown) => installPrefetch(cacheIdx, "stream", stream));
      a.state.extras.set("setIndexerK", (cacheIdx: number, stream: unknown) => installPrefetch(cacheIdx, "indexerStream", stream));
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
