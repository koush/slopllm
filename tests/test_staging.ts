import assert from "node:assert/strict";
import { after, before, describe, it } from "node:test";
import { ExecutionWorkspace } from "../src/execution-workspace";
import { GlmOps } from "../src/glm_ops";
import { PagedKVCache } from "../src/paged_kv";
import { Qwen3Model } from "../src/qwen3_model";


const QWEN3_REPO = "Qwen/Qwen3-0.6B";
const PROMPT1 = [151643, 151644, 151645, 1, 2, 3];
const PROMPT2 = [151643, 151644, 1, 2, 3, 4, 5];

describe("PagedKVCache staging", () => {
  let glm: GlmOps;
  let model: Qwen3Model;
  let ws: ExecutionWorkspace;

  before(async () => {
    const deviceId = parseInt(process.env.GLM_GPU ?? "0", 10);
    glm = new GlmOps(deviceId);
    model = await Qwen3Model.fromPretrained(glm, QWEN3_REPO);
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

  it("stageSequence moves sequence to staging and removes from sequences", () => {
    using pagedKV = makePagedKV(2, 256);
    pagedKV.reset(2);

    pagedKV.allocAppendPages(0, 10);
    pagedKV.reportTokens(0, Array.from({ length: 10 }, (_, i) => i));
    pagedKV.allocAppendPages(1, 5);
    pagedKV.reportTokens(1, Array.from({ length: 5 }, (_, i) => 100 + i));

    const seq0Pages = pagedKV.sequences[0].pages.length;
    const seq1Pages = pagedKV.sequences[1].pages.length;

    pagedKV.stageSequence(1, 1);

    assert.equal(pagedKV.staging.size, 1, "staging should have 1 entry");
    assert.equal(pagedKV.sequences.length, 1, "sequences should have 1 entry after staging 1");
    assert.equal(pagedKV.staging.get(1)!.pages.length, seq1Pages, "staged sequence should keep its pages");
    assert.equal(pagedKV.sequences[0].pages.length, seq0Pages, "remaining sequence should keep its pages");
  });

  it("stageSequence with splice shifts indices correctly", () => {
    using pagedKV = makePagedKV(3, 256);
    pagedKV.reset(3);

    pagedKV.allocAppendPages(0, 10);
    pagedKV.reportTokens(0, Array.from({ length: 10 }, (_, i) => i));
    pagedKV.allocAppendPages(1, 20);
    pagedKV.reportTokens(1, Array.from({ length: 20 }, (_, i) => 100 + i));
    pagedKV.allocAppendPages(2, 30);
    pagedKV.reportTokens(2, Array.from({ length: 30 }, (_, i) => 200 + i));

    const seq1Pages = pagedKV.sequences[1].pages.length;

    pagedKV.stageSequence(1, 1);

    assert.equal(pagedKV.sequences.length, 2, "sequences should have 2 entries after staging 1");
    assert.equal(pagedKV.sequences[0].reportedTokenCount(), 10, "seq0 should still have 10 tokens");
    assert.equal(pagedKV.sequences[1].reportedTokenCount(), 30, "former seq2 is now at index 1");
    assert.equal(pagedKV.staging.get(1)!.reportedTokenCount(), 20, "staged seq1 should have 20 tokens");
    assert.equal(pagedKV.staging.get(1)!.pages.length, seq1Pages, "staged seq1 should keep its pages");
  });

  it("unstageSequence restores sequence to end of sequences", () => {
    using pagedKV = makePagedKV(3, 256);
    pagedKV.reset(2);

    pagedKV.allocAppendPages(0, 10);
    pagedKV.reportTokens(0, Array.from({ length: 10 }, (_, i) => i));
    pagedKV.allocAppendPages(1, 20);
    pagedKV.reportTokens(1, Array.from({ length: 20 }, (_, i) => 100 + i));

    pagedKV.stageSequence(0, 0);
    pagedKV.stageSequence(0, 1);

    assert.equal(pagedKV.sequences.length, 0, "all sequences staged");
    assert.equal(pagedKV.staging.size, 2, "staging should have 2 entries");

    const seq0 = pagedKV.unstageSequence(0);
    assert.equal(pagedKV.sequences.length, 1, "1 sequence after unstage");
    assert.equal(pagedKV.sequences[0], seq0, "unstaged sequence at end of array");
    assert.equal(seq0.reportedTokenCount(), 10, "unstaged seq0 should have 10 tokens");

    const seq1 = pagedKV.unstageSequence(1);
    assert.equal(pagedKV.sequences.length, 2, "2 sequences after second unstage");
    assert.equal(pagedKV.sequences[1], seq1, "second unstaged sequence at end");
    assert.equal(seq1.reportedTokenCount(), 20, "unstaged seq1 should have 20 tokens");
    assert.equal(pagedKV.staging.size, 0, "staging should be empty");
  });

  it("unstageAll restores all staged sequences in insertion order", () => {
    using pagedKV = makePagedKV(3, 256);
    pagedKV.reset(3);

    pagedKV.allocAppendPages(0, 10);
    pagedKV.reportTokens(0, Array.from({ length: 10 }, (_, i) => i));
    pagedKV.allocAppendPages(1, 20);
    pagedKV.reportTokens(1, Array.from({ length: 20 }, (_, i) => 100 + i));
    pagedKV.allocAppendPages(2, 30);
    pagedKV.reportTokens(2, Array.from({ length: 30 }, (_, i) => 200 + i));

    pagedKV.stageSequence(2, 2);
    pagedKV.stageSequence(0, 0);

    assert.equal(pagedKV.sequences.length, 1, "1 sequence remaining after staging 2");
    assert.equal(pagedKV.staging.size, 2, "staging should have 2 entries");

    pagedKV.unstageAll();

    assert.equal(pagedKV.sequences.length, 3, "3 sequences after unstageAll");
    assert.equal(pagedKV.staging.size, 0, "staging should be empty after unstageAll");
    assert.equal(pagedKV.sequences[0].reportedTokenCount(), 20, "remaining seq at index 0 (original seq1)");
    assert.equal(pagedKV.sequences[1].reportedTokenCount(), 30, "key 2 unstaged first (insertion order)");
    assert.equal(pagedKV.sequences[2].reportedTokenCount(), 10, "key 0 unstaged second");
  });

  it("reset preserves staged sequence pages", () => {
    using pagedKV = makePagedKV(4, 256);
    pagedKV.reset(2);

    pagedKV.allocAppendPages(0, 32);
    pagedKV.reportTokens(0, Array.from({ length: 32 }, (_, i) => i));
    pagedKV.allocAppendPages(1, 16);
    pagedKV.reportTokens(1, Array.from({ length: 16 }, (_, i) => 100 + i));

    const seq0PageCount = pagedKV.sequences[0].pages.length;
    const seq0PageIds = pagedKV.sequences[0].pages.map(p => p.id);
    const seq1PageCount = pagedKV.sequences[1].pages.length;
    const seq1PageIds = pagedKV.sequences[1].pages.map(p => p.id);

    pagedKV.stageSequence(0, 0);
    pagedKV.stageSequence(0, 1);

    const totalPagesBefore = pagedKV.availablePages.length + seq0PageCount + seq1PageCount;

    pagedKV.reset(1);

    assert.equal(pagedKV.sequences.length, 1, "reset should create 1 new sequence");
    assert.equal(pagedKV.availablePages.length, totalPagesBefore - seq0PageCount - seq1PageCount,
      "availablePages should exclude staged pages");

    assert.equal(pagedKV.staging.get(0)!.pages.length, seq0PageCount, "staged seq0 should keep pages");
    assert.equal(pagedKV.staging.get(1)!.pages.length, seq1PageCount, "staged seq1 should keep pages");
    for (let i = 0; i < seq0PageCount; i++) {
      assert.equal(pagedKV.staging.get(0)!.pages[i].id, seq0PageIds[i], `staged seq0 page ${i} preserved`);
    }
    for (let i = 0; i < seq1PageCount; i++) {
      assert.equal(pagedKV.staging.get(1)!.pages[i].id, seq1PageIds[i], `staged seq1 page ${i} preserved`);
    }

    for (const id of seq0PageIds.concat(seq1PageIds)) {
      assert.ok(!pagedKV.availablePages.includes(id), `staged page ${id} should not be in availablePages`);
    }
  });

  it("clearStaging frees staged sequence pages", () => {
    using pagedKV = makePagedKV(2, 256);
    pagedKV.reset(2);

    pagedKV.allocAppendPages(0, 16);
    pagedKV.reportTokens(0, Array.from({ length: 16 }, (_, i) => i));
    pagedKV.allocAppendPages(1, 16);
    pagedKV.reportTokens(1, Array.from({ length: 16 }, (_, i) => 100 + i));

    pagedKV.stageSequence(0, 0);
    pagedKV.stageSequence(0, 1);

    const totalPages = pagedKV.maxPages;
    const stagedPageCount = pagedKV.staging.get(0)!.pages.length + pagedKV.staging.get(1)!.pages.length;

    pagedKV.clearStaging();

    assert.equal(pagedKV.staging.size, 0, "staging should be empty after clearStaging");
    assert.equal(pagedKV.availablePages.length, totalPages, "all pages should be available after clearStaging");
  });

  it("stageSequence throws on invalid index and duplicate key", () => {
    using pagedKV = makePagedKV(3, 256);
    pagedKV.reset(2);

    assert.throws(() => pagedKV.stageSequence(2, 0), /out of range/);
    assert.throws(() => pagedKV.stageSequence(-1, 0), /out of range/);

    pagedKV.stageSequence(0, 0);
    assert.throws(() => pagedKV.stageSequence(0, 0), /already in use/);
  });

  it("unstageSequence throws on missing key", () => {
    using pagedKV = makePagedKV(2, 256);
    pagedKV.reset(1);

    assert.throws(() => pagedKV.unstageSequence(99), /no staged sequence/);
  });

  it("staging preserves KV cache: decode → stage → prefill → unstage → decode", () => {
    using pagedKV = makePagedKV(4, 256);
    using refKV = makePagedKV(2, 256);

    pagedKV.reset(2);
    refKV.reset(2);
    const batchTokens = ws.forwardEagerPrefill(model, [PROMPT1, PROMPT2], pagedKV);
    ws.forwardEagerPrefill(model, [PROMPT1, PROMPT2], refKV);
    pagedKV.reportTokens(0, PROMPT1);
    pagedKV.reportTokens(1, PROMPT2);
    pagedKV.updateIndptr(ws);
    refKV.reportTokens(0, PROMPT1);
    refKV.reportTokens(1, PROMPT2);
    refKV.updateIndptr(ws);

    const decode1 = ws.forwardEagerDecode(model, batchTokens, pagedKV);
    const refDecode1 = ws.forwardEagerDecode(model, batchTokens, refKV);
    pagedKV.reportTokens(0, [decode1[0]]);
    pagedKV.reportTokens(1, [decode1[1]]);
    pagedKV.updateIndptr(ws);
    refKV.reportTokens(0, [refDecode1[0]]);
    refKV.reportTokens(1, [refDecode1[1]]);
    refKV.updateIndptr(ws);

    const decode2 = ws.forwardEagerDecode(model, decode1, pagedKV);
    const refDecode2 = ws.forwardEagerDecode(model, refDecode1, refKV);
    pagedKV.reportTokens(0, [decode2[0]]);
    pagedKV.reportTokens(1, [decode2[1]]);
    pagedKV.updateIndptr(ws);
    refKV.reportTokens(0, [refDecode2[0]]);
    refKV.reportTokens(1, [refDecode2[1]]);
    refKV.updateIndptr(ws);

    assert.equal(decode1[0], refDecode1[0], "decode1 seq0 matches reference");
    assert.equal(decode1[1], refDecode1[1], "decode1 seq1 matches reference");
    assert.equal(decode2[0], refDecode2[0], "decode2 seq0 matches reference");
    assert.equal(decode2[1], refDecode2[1], "decode2 seq1 matches reference");

    pagedKV.stageSequence(1, 1);
    pagedKV.stageSequence(0, 0);

    assert.equal(pagedKV.sequences.length, 0, "all sequences staged");
    assert.equal(pagedKV.staging.size, 2, "staging has 2 entries");

    pagedKV.reset(1);

    const prompt3 = [151643, 151644, 10, 20, 30];
    const tokens3 = ws.forwardEagerPrefill(model, [prompt3], pagedKV);
    pagedKV.reportTokens(0, prompt3);
    pagedKV.updateIndptr(ws);

    pagedKV.unstageAll();

    assert.equal(pagedKV.sequences.length, 3, "3 sequences after unstageAll");
    assert.equal(pagedKV.staging.size, 0, "staging is empty");

    pagedKV.updateIndptr(ws);

    // unstageAll restores in Map insertion order: key 1 first, then key 0
    // sequences = [newSeq, seq1, seq0]
    // Input tokens must match sequence order
    const resumedDecode = ws.forwardEagerDecode(model, [tokens3[0], decode2[1], decode2[0]], pagedKV);

    // Reference 3rd decode (continuing from refKV)
    const refDecode3 = ws.forwardEagerDecode(model, refDecode2, refKV);

    // resumedDecode[2] = seq0 output, resumedDecode[1] = seq1 output
    assert.equal(resumedDecode[2], refDecode3[0],
      `seq0 decode after unstage should match: got ${resumedDecode[2]}, expected ${refDecode3[0]}`);
    assert.equal(resumedDecode[1], refDecode3[1],
      `seq1 decode after unstage should match: got ${resumedDecode[1]}, expected ${refDecode3[1]}`);
  });

  it("staging with reset allows prefill of new sequence while preserving staged KV", () => {
    using pagedKV = makePagedKV(4, 256);
    using refKV = makePagedKV(1, 256);

    pagedKV.reset(1);
    const tokens1 = ws.forwardEagerPrefill(model, [PROMPT1], pagedKV);
    pagedKV.reportTokens(0, PROMPT1);
    pagedKV.updateIndptr(ws);

    for (let step = 0; step < 3; step++) {
      const decodeTokens = ws.forwardEagerDecode(model, [tokens1[0]], pagedKV);
      pagedKV.reportTokens(0, [decodeTokens[0]]);
      pagedKV.updateIndptr(ws);
      tokens1[0] = decodeTokens[0];
    }

    const seq0PageCount = pagedKV.sequences[0].pages.length;
    const seq0TokenCount = pagedKV.sequences[0].reportedTokenCount();

    pagedKV.stageSequence(0, 0);

    pagedKV.reset(1);
    const prompt2 = [151643, 151644, 10, 20, 30, 40, 50];
    const tokens2 = ws.forwardEagerPrefill(model, [prompt2], pagedKV);
    pagedKV.reportTokens(0, prompt2);
    pagedKV.updateIndptr(ws);

    assert.equal(pagedKV.sequences.length, 1, "1 active sequence after prefill");
    assert.equal(pagedKV.staging.size, 1, "staging still has 1 entry");
    assert.equal(pagedKV.staging.get(0)!.pages.length, seq0PageCount, "staged seq0 pages preserved");
    assert.equal(pagedKV.staging.get(0)!.reportedTokenCount(), seq0TokenCount, "staged seq0 tokens preserved");

    refKV.reset(1);
    const refTokens = ws.forwardEagerPrefill(model, [PROMPT1], refKV);
    refKV.reportTokens(0, PROMPT1);
    refKV.updateIndptr(ws);
    let refDecoded: number[] = [];
    let lastRef = refTokens[0];
    for (let step = 0; step < 3; step++) {
      const stepTokens = ws.forwardEagerDecode(model, [lastRef], refKV);
      refDecoded.push(stepTokens[0]);
      refKV.reportTokens(0, [stepTokens[0]]);
      refKV.updateIndptr(ws);
      lastRef = stepTokens[0];
    }
    const refNext = ws.forwardEagerDecode(model, [lastRef], refKV);

    pagedKV.unstageSequence(0);
    assert.equal(pagedKV.sequences.length, 2, "2 sequences after unstage");
    // sequences = [newSeq, seq0] — unstageSequence appends to end

    pagedKV.updateIndptr(ws);

    // Input tokens must match sequence order: [newSeq, seq0]
    const resumedDecode = ws.forwardEagerDecode(model, [tokens2[0], lastRef], pagedKV);

    // resumedDecode[1] corresponds to seq0
    assert.equal(resumedDecode[1], refNext[0],
      `resumed seq0 decode matches reference: got ${resumedDecode[1]}, expected ${refNext[0]}`);
  });

  it("clearStaging after reset makes staged pages available again", () => {
    using pagedKV = makePagedKV(2, 32);
    pagedKV.reset(2);

    pagedKV.allocAppendPages(0, 32);
    pagedKV.reportTokens(0, Array.from({ length: 32 }, (_, i) => i));
    pagedKV.allocAppendPages(1, 32);
    pagedKV.reportTokens(1, Array.from({ length: 32 }, (_, i) => 100 + i));

    pagedKV.stageSequence(0, 0);
    pagedKV.stageSequence(0, 1);

    const availableAfterStage = pagedKV.availablePages.length;
    pagedKV.reset(1);
    const availableAfterReset = pagedKV.availablePages.length;
    assert.equal(availableAfterReset, availableAfterStage,
      "reset should not free staged pages");

    pagedKV.clearStaging();
    assert.equal(pagedKV.availablePages.length, pagedKV.maxPages,
      "clearStaging should free all staged pages (new sequence has 0 pages)");

    assert.ok(pagedKV.availablePages.length > availableAfterReset,
      "clearStaging should increase availablePages");
  });

  it("removeSequence splices out sequence and frees its pages", () => {
    using pagedKV = makePagedKV(3, 32);
    pagedKV.reset(3);

    pagedKV.allocAppendPages(0, 16);
    pagedKV.allocAppendPages(1, 16);
    pagedKV.allocAppendPages(2, 16);

    const seq0Pages = pagedKV.sequences[0].pages.length;
    const seq1Pages = pagedKV.sequences[1].pages.length;
    const seq2Pages = pagedKV.sequences[2].pages.length;

    pagedKV.removeSequence(1);

    assert.equal(pagedKV.sequences.length, 2, "should have 2 sequences after removal");
    assert.equal(pagedKV.availablePages.length, pagedKV.maxPages - seq0Pages - seq2Pages,
      "removed sequence's pages should be freed");
    assert.equal(pagedKV.sequences[0].pages.length, seq0Pages, "seq0 unchanged");
    assert.equal(pagedKV.sequences[1].pages.length, seq2Pages, "former seq2 is now at index 1");
  });

  it("removeSequence throws on invalid index", () => {
    using pagedKV = makePagedKV(2, 32);
    pagedKV.reset(2);
    assert.throws(() => pagedKV.removeSequence(2), /out of range/);
    assert.throws(() => pagedKV.removeSequence(-1), /out of range/);
  });

  it("removeSequence sets dirty flags", () => {
    using pagedKV = makePagedKV(2, 32);
    pagedKV.reset(2);
    pagedKV.pagesDirtyHost = false;
    pagedKV.pagesDirtyDevice = false;
    pagedKV.positionIdsDirty = false;

    pagedKV.removeSequence(0);

    assert.equal(pagedKV.pagesDirtyHost, true, "pagesDirtyHost should be set");
    assert.equal(pagedKV.pagesDirtyDevice, true, "pagesDirtyDevice should be set");
    assert.equal(pagedKV.positionIdsDirty, true, "positionIdsDirty should be set");
  });
});
