import { ChatModel, ChatCache, SamplingParams, needsSampling } from "../src/chat_model";
import { WorkspaceBuffers } from "../src/paged_kv";

export function generateBatchTokens(
  model: ChatModel, ws: WorkspaceBuffers, cache: ChatCache,
  inputIdsList: number[][], maxNewTokens: number, eosIds: Set<number>,
): number[][] {
  const batchSize = inputIdsList.length;
  cache.reset(batchSize);
  const firstTokens = model.forwardEager(inputIdsList, ws, cache);

  const nextTokens = [...firstTokens];
  const generated: number[][] = nextTokens.map(t => [t]);
  const finished = nextTokens.map(t => eosIds.has(t));

  for (let step = 0; step < maxNewTokens - 1; step++) {
    if (finished.every(f => f)) break;

    const newTokens = model.decodeEager(nextTokens, ws, cache);

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
  model: ChatModel, ws: WorkspaceBuffers, cache: ChatCache,
  inputIds: number[], maxNewTokens: number, eosIds: Set<number>,
  sampling?: SamplingParams,
): Generator<number> {
  cache.reset(1);
  const firstTokens = model.forwardEager([inputIds], ws, cache);
  let nextToken = firstTokens[0];
  yield nextToken;

  const tokenHistory = [...inputIds, nextToken];

  for (let i = 0; i < maxNewTokens - 1; i++) {
    if (eosIds.has(nextToken)) break;

    const decodeTokens = model.decodeEager([nextToken], ws, cache);
    nextToken = decodeTokens[0];

    if (sampling && needsSampling(sampling)) {
      nextToken = model.sampleTokenGPU(sampling, tokenHistory);
    }

    tokenHistory.push(nextToken);
    yield nextToken;
  }
}
