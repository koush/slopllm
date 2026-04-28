import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import { GlmOps } from "../src/glm_ops";
import { Qwen3Model } from "../src/qwen3_model";
import { ExecutionWorkspace, PagedKVCache } from "../src/paged_kv";

const QWEN3_REPO = "Qwen/Qwen3-0.6B";
const PROMPT1 = [151643, 151644, 151645, 1, 2, 3];
const PROMPT_GRAPH = [151643, 151644, 151645, 1, 2988, 279, 1716, 364];

describe("decodeStep eager (no graph)", () => {
  let glm: GlmOps;
  let model: Qwen3Model;
  let ws: ExecutionWorkspace;

  before(() => {
    const deviceId = parseInt(process.env.GLM_GPU ?? "0", 10);
    glm = new GlmOps(deviceId);
    model = Qwen3Model.fromPretrained(glm, QWEN3_REPO, 4, 4096);
    ws = new ExecutionWorkspace(glm, 4, 4096);
  });

  after(() => {
    ws[Symbol.dispose]();
    model.free();
    glm.free();
  });

  function makePagedKV(maxBatch = 4, maxPages = 128): PagedKVCache {
    const cfg = model.cfg;
    return new PagedKVCache(glm, cfg.numKeyValueHeads, cfg.headDim, cfg.numHiddenLayers, maxPages, maxBatch);
  }

  function eagerDecodeRef(numSteps: number): number[] {
    using pagedKV = makePagedKV(1);
    pagedKV.reset(1);
    const firstToken = ws.forwardEagerPrefill(model, [PROMPT1], pagedKV)[0];
    const tokens = [firstToken];
    let current = firstToken;
    for (let i = 0; i < numSteps; i++) {
      current = ws.forwardEagerDecode(model, [current], pagedKV)[0];
      tokens.push(current);
    }
    return tokens;
  }

  it("single-sequence decodeStep matches eager decode", () => {
    const numSteps = 5;
    const refTokens = eagerDecodeRef(numSteps);

    using pagedKV = makePagedKV(1);
    pagedKV.reset(1);
    const firstToken = ws.forwardEagerPrefill(model, [PROMPT1], pagedKV)[0];
    assert.equal(firstToken, refTokens[0], `Prefill mismatch: ${firstToken} != ${refTokens[0]}`);

    const state = ws.planDecode(model, 1, pagedKV);
    state.prepareInput([firstToken]);
    ws.forwardInput(state);
    let logits = model.forward(state);
    using argmax0 = logits.argmax();
    let current = argmax0.readInt32LE()[0];
    assert.equal(current, refTokens[1], `Step 0 mismatch: ${current} != ${refTokens[1]}`);

    const dsTokens = [firstToken, current];
    for (let step = 1; step < numSteps; step++) {
      ws.advanceDecode(state, model);
      ws.decodeStep(state);
      state.prepareInput([current]);
      ws.forwardInputDecode(state);
      logits = model.forward(state);
      using argmax = logits.argmax();
      current = argmax.readInt32LE()[0];
      dsTokens.push(current);
      assert.equal(current, refTokens[step + 1],
        `Step ${step} mismatch: ${current} != ${refTokens[step + 1]}`);
    }

    assert.deepEqual(dsTokens, refTokens,
      `Token sequence mismatch:\n  decodeStep=${JSON.stringify(dsTokens)}\n  eager    =${JSON.stringify(refTokens)}`);
  });

  it("decodeStep matches eager decode across page boundary", () => {
    const longPrompt = Array.from({ length: 14 }, (_, i) => i + 1);
    const numSteps = 5;

    using refKV = makePagedKV(1);
    refKV.reset(1);
    const refFirst = ws.forwardEagerPrefill(model, [longPrompt], refKV)[0];
    const refTokens = [refFirst];
    let refCurrent = refFirst;
    for (let i = 0; i < numSteps; i++) {
      refCurrent = ws.forwardEagerDecode(model, [refCurrent], refKV)[0];
      refTokens.push(refCurrent);
    }

    using pagedKV = makePagedKV(1);
    pagedKV.reset(1);
    const firstToken = ws.forwardEagerPrefill(model, [longPrompt], pagedKV)[0];
    assert.equal(firstToken, refTokens[0]);

    const state = ws.planDecode(model, 1, pagedKV);
    state.prepareInput([firstToken]);
    ws.forwardInput(state);
    let logits = model.forward(state);
    using argmax0 = logits.argmax();
    let current = argmax0.readInt32LE()[0];
    const dsTokens = [firstToken, current];
    assert.equal(current, refTokens[1]);

    for (let step = 1; step < numSteps; step++) {
      ws.advanceDecode(state, model);
      ws.decodeStep(state);
      state.prepareInput([current]);
      ws.forwardInputDecode(state);
      logits = model.forward(state);
      using argmax = logits.argmax();
      current = argmax.readInt32LE()[0];
      dsTokens.push(current);
      assert.equal(current, refTokens[step + 1],
        `Step ${step} mismatch: ${current} != ${refTokens[step + 1]}`);
    }

    assert.deepEqual(dsTokens, refTokens,
      `Token sequence mismatch across page boundary`);
  });

  it("pagesChanged skips h2d when no new page allocated", () => {
    using pagedKV = makePagedKV(1);
    pagedKV.reset(1);

    ws.forwardEagerPrefill(model, [PROMPT1], pagedKV);
    const state = ws.planDecode(model, 1, pagedKV);
    state.prepareInput([0]);
    ws.forwardInput(state);
    model.forward(state);

    const pageSize = 16;
    const prefillLen = PROMPT1.length;
    const boundaryStep = pageSize - prefillLen - 1;

    for (let step = 0; step < pageSize + 2; step++) {
      const pagesBefore = pagedKV.seqPages[0].length;
      ws.advanceDecode(state, model);
      const pagesAfter = pagedKV.seqPages[0].length;
      const newPageAllocated = pagesAfter > pagesBefore;

      if (step === boundaryStep) {
        assert.equal(newPageAllocated, true,
          `Step ${step}: expected new page at boundary (kvLen=${prefillLen + step + 1}, page_size=${pageSize})`);
      } else {
        assert.equal(newPageAllocated, false,
          `Step ${step}: unexpected new page (kvLen=${prefillLen + step + 1}, page_size=${pageSize})`);
      }

      ws.decodeStep(state);
      state.prepareInput([0]);
      ws.forwardInputDecode(state);
      model.forward(state);
    }
  });
});

describe("decodeStep CUDA graph", () => {
  let glm: GlmOps;
  let model: Qwen3Model;
  let ws: ExecutionWorkspace;

  before(() => {
    const deviceId = parseInt(process.env.GLM_GPU ?? "0", 10);
    glm = new GlmOps(deviceId);
    model = Qwen3Model.fromPretrained(glm, QWEN3_REPO, 4, 4096);
    ws = new ExecutionWorkspace(glm, 4, 4096);
  });

  after(() => {
    ws[Symbol.dispose]();
    model.free();
    glm.free();
  });

  function makePagedKV(maxBatch = 1, maxPages = 128): PagedKVCache {
    const cfg = model.cfg;
    return new PagedKVCache(glm, cfg.numKeyValueHeads, cfg.headDim, cfg.numHiddenLayers, maxPages, maxBatch);
  }

  it("graph decode with decodeStep matches eager decode", () => {
    const numSteps = 8;
    const prompt = PROMPT_GRAPH;

    using refKV = makePagedKV();
    refKV.reset(1);
    const refFirst = ws.forwardEagerPrefill(model, [prompt], refKV)[0];
    const refTokens = [refFirst];
    let refCurrent = refFirst;
    for (let i = 0; i < numSteps; i++) {
      refCurrent = ws.forwardEagerDecode(model, [refCurrent], refKV)[0];
      refTokens.push(refCurrent);
    }

    using pagedKV = makePagedKV();
    pagedKV.reset(1);
    const firstToken = ws.forwardEagerPrefill(model, [prompt], pagedKV)[0];
    assert.equal(firstToken, refTokens[0]);

    const state = ws.planDecode(model, 1, pagedKV, true);
    state.prepareInput([firstToken]);
    ws.forwardInput(state);
    let logits = model.forward(state);
    using argmax0 = logits.argmax();
    let current = argmax0.readInt32LE()[0];
    assert.equal(current, refTokens[1]);

    const warmupSteps = 3;
    for (let i = 0; i < warmupSteps; i++) {
      ws.advanceDecode(state, model, true);
      ws.decodeStep(state);
      state.prepareInput([current]);
      ws.forwardInputDecode(state);
      logits = model.forward(state);
      using argmax = logits.argmax();
      current = argmax.readInt32LE()[0];
    }
    assert.equal(current, refTokens[1 + warmupSteps],
      `Warmup step ${warmupSteps} mismatch: ${current} != ${refTokens[1 + warmupSteps]}`);

    ws.advanceDecode(state, model, true);
    state.prepareInput([current]);
    glm.graphBeginCapture();
    ws.decodeStep(state);
    ws.forwardInputDecode(state);
    const captureLogits = model.forward(state);
    const captureArgmax = captureLogits.argmax();
    const graph = glm.graphEndCapture();
    ws.freeze();
    const graphExec = glm.graphInstantiate(graph);
    glm.graphDestroy(graph);

    glm.graphLaunch(graphExec);
    glm.synchronize();
    current = captureArgmax.readInt32LE()[0];
    const expectedStep = 1 + warmupSteps + 1;
    assert.equal(current, refTokens[expectedStep],
      `Capture step mismatch: ${current} != ${refTokens[expectedStep]}`);

    for (let step = expectedStep; step < numSteps; step++) {
      pagedKV.allocDecodeToken(0);
      state.prepareInput([current]);
      glm.graphLaunch(graphExec);
      glm.synchronize();
      current = captureArgmax.readInt32LE()[0];
      assert.equal(current, refTokens[step + 1],
        `Replay step ${step} mismatch: ${current} != ${refTokens[step + 1]}`);
    }

    glm.graphExecDestroy(graphExec);
  });

  it("graph decode with decodeStep matches planDecode graph decode", () => {
    const numSteps = 6;
    const prompt = PROMPT_GRAPH;

    using gws = new ExecutionWorkspace(glm, 4, 4096);

    using pagedKV = makePagedKV();
    pagedKV.reset(1);
    const firstToken = gws.forwardEagerPrefill(model, [prompt], pagedKV)[0];
    pagedKV.updateIndptr(gws);

    const refTokens: number[] = [firstToken];
    let current = firstToken;
    for (let step = 0; step < numSteps; step++) {
      const refState = gws.planDecode(model, 1, pagedKV, true);
      refState.prepareInput([current]);
      gws.forwardInput(refState);
      const refLogits = model.forward(refState);
      using refArgmax = refLogits.argmax();
      current = refArgmax.readInt32LE()[0];
      refTokens.push(current);
    }

    using pagedKV2 = makePagedKV();
    pagedKV2.reset(1);
    const firstToken2 = gws.forwardEagerPrefill(model, [prompt], pagedKV2)[0];
    assert.equal(firstToken2, refTokens[0]);

    const state = gws.planDecode(model, 1, pagedKV2, true);
    state.prepareInput([firstToken2]);
    gws.forwardInput(state);
    let logits = model.forward(state);
    using argmax0 = logits.argmax();
    current = argmax0.readInt32LE()[0];
    assert.equal(current, refTokens[1]);

    for (let i = 0; i < 2; i++) {
      gws.advanceDecode(state, model, true);
      gws.decodeStep(state);
      state.prepareInput([current]);
      gws.forwardInputDecode(state);
      logits = model.forward(state);
      using a = logits.argmax();
      current = a.readInt32LE()[0];
    }

    gws.advanceDecode(state, model, true);
    state.prepareInput([current]);
    glm.graphBeginCapture();
    gws.decodeStep(state);
    gws.forwardInputDecode(state);
    const captureLogits = model.forward(state);
    const captureArgmax = captureLogits.argmax();
    const graph = glm.graphEndCapture();
    gws.freeze();
    const graphExec = glm.graphInstantiate(graph);
    glm.graphDestroy(graph);

    glm.graphLaunch(graphExec);
    glm.synchronize();
    current = captureArgmax.readInt32LE()[0];

    assert.equal(current, refTokens[4],
      `Capture mismatch: ${current} != ${refTokens[4]}`);

    for (let step = 4; step < numSteps; step++) {
      pagedKV2.allocDecodeToken(0);
      state.prepareInput([current]);
      glm.graphLaunch(graphExec);
      glm.synchronize();
      current = captureArgmax.readInt32LE()[0];
      assert.equal(current, refTokens[step + 1],
        `Replay step ${step} mismatch: ${current} != ${refTokens[step + 1]}`);
    }

    glm.graphExecDestroy(graphExec);
  });
});
