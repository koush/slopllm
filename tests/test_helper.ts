import { ChatModel, ChatCache, SamplingParams } from "../src/chat_model";
import { ExecutionWorkspace } from "../src/paged_kv";

export function generateBatchTokens(
  model: ChatModel, ws: ExecutionWorkspace, cache: ChatCache,
  inputIdsList: number[][], maxNewTokens: number, eosIds: Set<number>,
): number[][] {
  const batchSize = inputIdsList.length;
  cache.reset(batchSize);
  const firstTokens = model.forwardEager(ws, inputIdsList, cache);

  const nextTokens = [...firstTokens];
  const generated: number[][] = nextTokens.map(t => [t]);
  const finished = nextTokens.map(t => eosIds.has(t));

  for (let step = 0; step < maxNewTokens - 1; step++) {
    if (finished.every(f => f)) break;

    const newTokens = model.forwardEagerDecode(ws, nextTokens, cache);

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
  const firstTokens = model.forwardEager(ws, [inputIds], cache);
  let nextToken = firstTokens[0];
  yield nextToken;

  const tokenHistory = [...inputIds, nextToken];

  for (let i = 0; i < maxNewTokens - 1; i++) {
    if (eosIds.has(nextToken)) break;

    if (sampling) {
      const state = model.planDecode(ws, [nextToken], cache);
      const logits = model.forwardDecode(state);
      nextToken = logits.sampleTokenGPU(sampling, tokenHistory).readInt32LE()[0];
    } else {
      const decodeTokens = model.forwardEagerDecode(ws, [nextToken], cache);
      nextToken = decodeTokens[0];
    }

    tokenHistory.push(nextToken);
    yield nextToken;
  }
}
