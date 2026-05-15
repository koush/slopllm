import { ChatCache, ChatModel, SamplingParams } from "../src/chat_model";
import { ExecutionWorkspace } from "../src/execution-workspace";

export function generateBatchTokens(
  model: ChatModel, ws: ExecutionWorkspace, cache: ChatCache,
  inputIdsList: number[][], maxNewTokens: number, eosIds: Set<number>,
): number[][] {
  const batchSize = inputIdsList.length;
  cache.reset(batchSize);
  const firstTokens = ws.forwardEagerPrefill(model, inputIdsList, cache);

  const nextTokens = [...firstTokens];
  const generated: number[][] = nextTokens.map(t => [t]);
  const finished = nextTokens.map(t => eosIds.has(t));

  for (let step = 0; step < maxNewTokens - 1; step++) {
    if (finished.every(f => f)) break;

    const newTokens = ws.forwardEagerDecode(model, nextTokens, cache);

    for (let i = 0; i < batchSize; i++) {
      nextTokens[i] = newTokens[i];
      if (!finished[i]) {
        if (eosIds.has(newTokens[i])) {
          finished[i] = true;
        } else {
          generated[i].push(newTokens[i]);
        }
      }
    }
  }

  return generated;
}

export function* generateTokens(
  model: ChatModel, ws: ExecutionWorkspace, cache: ChatCache,
  inputIds: number[], maxNewTokens: number, eosIds: Set<number>,
  sampling?: SamplingParams,
): Generator<number> {
  cache.reset(1);
  const firstTokens = ws.forwardEagerPrefill(model, [inputIds], cache);
  let nextToken = firstTokens[0];
  yield nextToken;

  const tokenHistory = [...inputIds, nextToken];

  for (let i = 0; i < maxNewTokens - 1; i++) {
    if (eosIds.has(nextToken)) break;

    if (sampling) {
      const state = ws.planDecode(model, 1, cache);
      state.prepareInput([[nextToken]]);
      ws.forwardInput(state);
      const hiddenStates = model.forward(state);
      nextToken = state.computeLogits(hiddenStates, model).sampleTokenGPU(sampling, tokenHistory).readInt32LEArray()[0];
    } else {
      const decodeTokens = ws.forwardEagerDecode(model, [nextToken], cache);
      nextToken = decodeTokens[0];
    }

    tokenHistory.push(nextToken);
    yield nextToken;
  }
}
