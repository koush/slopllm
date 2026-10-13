// Fix A (tail-chunk merge) equivalence fixture — tests/test_tail_merge_equivalence.ts.
// Provenance: lowctx-20261013/fixA/equivalence-fixture-draft.ts + fixture-notes.md,
// CORRECTED per Codex consult #8 (codex-consult/verdict-lowctx-20261013.md,
// Q4 + "Required pre-adoption evidence" 1-5), applied beside fixA.patch.
//
// The verdict's Q4 five-point oracle design is the contract; this file is
// organized by it:
//
//   (1) Structural token/replay/history/COW/deferred-row oracle — CPU block
//       below runs the REAL selectPrefillChunk and the REAL scheduler mirror
//       (estimateRemainingPrefillChunks) over the slicing level; the GPU
//       block extends it to committed histories, reported/alloc lengths,
//       page-sharing/COW isolation and untouched deferred rows.
//   (2) Numerical state oracle at matching LOGICAL positions — base CKV/KPE
//       and indexer K/scales per layer/rank, stable shifted MTP prefix KV,
//       final hidden/logits, and the decoded MTP startup seed AFTER decode's
//       mandatory rewind installs the real pending target. Gathered by
//       logical token position (never physical page IDs / unused allocation
//       bytes); all values finite. The prefill-terminal MTP row is a
//       PLACEHOLDER by design (glm51_model.ts planPhasedPrefill boundary
//       fill); its placeholder state and the post-rewind valid seed are
//       compared as different objects. Inspect the former eliminated boundary
//       and the moved A/B crossing (8,198-row merge -> halves 4,099/4,099,
//       crossing x4099/x4100) specifically.
//   (3) Correct comparison domain — NO logits requirement at the eliminated
//       outer boundary (it was never published to the client); replay counts
//       intentionally differ (cold 8,198: split [8192,6] replays [0,1] ->
//       merged [8198] replay [0]). Interior comparisons are TEACHER-FORCED at
//       the same global prefix position; final pending target and committed
//       state compared at equal prompt length. Never compare two "chunk 1"
//       endpoints that now represent different prefixes.
//   (4) Output/decode evidence — exact greedy token sequences on fixed
//       fixtures AND held-out production prompts, multiple decode/MTP steps,
//       page crossing, generator closure/reuse, graph replay; MTP accepted
//       lengths/seed and repeatability; decode isolation against identical
//       canonical input/KV; end-to-end decode after EACH prefill arm.
//   (5) Byte-identity-first acceptance ladder with frozen tolerances and
//       README owner sign-off (fixture-notes.md §3): request bit identity
//       first where baseline repeatability supports it; any tolerance is
//       frozen BEFORE candidate scoring and owner-approved — never loosened
//       after a failing candidate. Unexplained fixed-fixture token, state,
//       routing, or MTP divergence BLOCKS adoption (D/README.md:16,85).
//
// CPU block: runnable offline (no native addon, no GPU):
//   npx tsx --test tests/test_tail_merge_equivalence.ts
// GPU block: consult-#7 disposition gate (SLOPLLM_CSF_GPU_TESTS=0 pins off /
// =1 requires; the addon must load in THIS process), plus the MTP-capable
// tiny model (tests/python/test_models/glm51_small/glm51_small_bf16 — a
// genuine MTP head + sparse/MoE coverage; a model without MTP cannot prove
// its replay contract, fixture-notes.md §4). A clean skip under the
// disposition gate is NOT GPU equivalence evidence — the block logs what
// actually executed so the runbook can record it.
//
// Arms switch WITHOUT rebuilds via the serving-path override
// SLOPLLM_TAIL_MERGE_MAX (src/chat_model.ts prefillTailMergeMaxSetting):
//   0   = split arm / pre-Fix-A reproduction (arm 0 of the window plan),
//   256 = merged arm (inclusive endpoint; arm 1 default).

import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { after, before, describe, it } from "node:test";

import {
  ChatModel,
  PHASED_PREFILL_MIN_ROWS,
  PREFILL_TAIL_MERGE_MAX,
  PREFILL_TAIL_MERGE_QUALIFIED_CHUNK_SIZE,
  prefillTailMergeOverflowEligible,
  prefillTailMergeMaxSetting,
  selectPrefillTakes,
  type ChatCache,
  type PrefillTailMergeContext,
  type PrefillTailMergeOptions,
} from "../src/chat_model";
import { GenerationScheduler } from "../src/generation-scheduler";
import { CaptureManager } from "../src/capture-manager";
import { ExecutionWorkspace } from "../src/execution-workspace";
import { Glm51Model } from "../src/glm51_model";
import { GlmOps } from "../src/glm_ops";
import type { PagedKVCache } from "../src/paged_kv";
import type { Tensor } from "../src/tensor";

// ---------------------------------------------------------------------------
// CPU section: the REAL selectPrefillChunk (prototype-bound; the method is
// `protected` and uses no `this`) and the REAL GenerationScheduler.
// estimateRemainingPrefillChunks (private; bound against a minimal fake
// scheduler carrying only `active` rows and `options.ws.maxSeqLen`).
// ---------------------------------------------------------------------------

interface ChunkResult {
  prefillInputIdsList: number[][];
  remainingInputIdsList: number[][];
}

const chunker = Object.create(ChatModel.prototype) as {
  selectPrefillChunk(inputIdsList: readonly number[][], chunkSize: number,
    overlapTokens?: readonly (number | undefined)[], tailMerge?: PrefillTailMergeOptions): ChunkResult;
};

const estimateRemaining = (GenerationScheduler.prototype as unknown as {
  estimateRemainingPrefillChunks(this: unknown, mtp: boolean, chunkSize: number): number;
}).estimateRemainingPrefillChunks;

interface FakeRow {
  inputIds: number[];
  cachedTokenCount: number;
  prefillTokenCount: number;
}

/** Drive the REAL scheduler mirror over a synthetic batch. batchLength for
 *  eligibility is the ORIGINAL row count (empty rows count) — exactly what
 *  the mirror does with this.active.length. */
const mirrorTotal = (rows: { length: number; cached?: boolean }[],
  mtp: boolean, chunkSize: number, maxSeqLen = chunkSize): number => {
  const active: FakeRow[] = rows.map(row => ({
    inputIds: new Array(row.length).fill(0),
    cachedTokenCount: row.cached ? 1 : 0,
    prefillTokenCount: 0,
  }));
  return estimateRemaining.call({ active, options: { ws: { maxSeqLen } } }, mtp, chunkSize);
};

/** Faithful per-chunk port of the mirror's loop (same selectPrefillTakes,
 *  same filtered rows / replay transitions / original batch length) returning
 *  per-chunk takes, for the selector-vs-mirror differential. */
const mirrorChunkTakes = (rows: { length: number; cached?: boolean }[],
  mtp: boolean, chunkSize: number, maxSeqLen = chunkSize): number[][] => {
  const ctx: PrefillTailMergeContext = {
    chunkSize,
    tailMergeMax: PREFILL_TAIL_MERGE_MAX,
    maxSeqLen,
    batchLength: rows.length,
    mtp,
  };
  const lens = rows.map(row => row.length);
  const replay = rows.map(row => mtp && !!row.cached);
  const out: number[][] = [];
  while (lens.some(length => length > 0)) {
    const takes = selectPrefillTakes(lens,
      replay.map((active, i) => active && lens[i] ? 1 : 0), ctx);
    for (let i = 0; i < lens.length; i++) {
      if (takes[i]) {
        lens[i] -= takes[i];
        replay[i] = mtp;
      }
    }
    out.push(takes);
    assert.ok(out.length <= 10000, "mirror loop must terminate");
  }
  return out;
};

const idsOf = (n: number): number[] => Array.from({ length: n }, (_, i) => i);

const MTP_ARM: PrefillTailMergeOptions = {
  // Pinned to the shipped default so the CPU block is deterministic even when
  // SLOPLLM_TAIL_MERGE_MAX is set in the environment (the env override is a
  // CALL-time default for callers that pass nothing — see the GPU block).
  tailMergeMax: PREFILL_TAIL_MERGE_MAX,
  maxSeqLen: PREFILL_TAIL_MERGE_QUALIFIED_CHUNK_SIZE,
  batchLength: 1,
  mtp: true,
};
const SPLIT_ARM: PrefillTailMergeOptions = { tailMergeMax: 0 };

interface RowTrace { newTokens: number[]; replayTokens: number[]; }

/** Replay the scheduler chunk loop with the REAL slice selection for one row.
 *  MTP semantics (glm51 planChunkedPrefill): a fresh row's first chunk has no
 *  replayed overlap; every later chunk of a row prepends one replayed terminal
 *  token; a cached-prefix row carries its entry overlap from chunk 1. */
const traceRow = (ids: number[], mtp: boolean, chunkSize: number,
  tailMerge: PrefillTailMergeOptions = MTP_ARM, cached = false): RowTrace[] => {
  const out: RowTrace[] = [];
  let rest = ids.slice();
  let committed = cached;
  while (rest.length) {
    const overlap = mtp && committed ? -1 : undefined; // -1: sentinel, never a real id
    const { prefillInputIdsList, remainingInputIdsList } =
      chunker.selectPrefillChunk([rest], chunkSize, [overlap], tailMerge);
    const input = prefillInputIdsList[0];
    const replayTokens = input.length && input[0] === -1 ? 1 : 0;
    out.push({ newTokens: [input.length - replayTokens], replayTokens: [replayTokens] });
    if (input.length) committed = true;
    rest = remainingInputIdsList[0];
    assert.ok(out.length <= 10000, "chunk loop must terminate");
  }
  return out;
};

/** Batch version (rows may defer whole across chunks; empty rows allowed). */
const traceBatch = (rows: { length: number; cached?: boolean }[],
  mtp: boolean, chunkSize: number,
  tailMerge: PrefillTailMergeOptions = SPLIT_ARM, maxSeqLen = chunkSize): RowTrace[] => {
  const state = rows.map(row => ({ rest: idsOf(row.length), committed: !!row.cached }));
  const out: RowTrace[] = [];
  while (state.some(s => s.rest.length)) {
    const overlaps = state.map(s => mtp && s.committed && s.rest.length ? -1 : undefined);
    const { prefillInputIdsList, remainingInputIdsList } = chunker.selectPrefillChunk(
      state.map(s => s.rest), chunkSize, overlaps,
      { ...tailMerge, maxSeqLen, batchLength: rows.length, mtp });
    const newTokens: number[] = [];
    const replayTokens: number[] = [];
    for (let i = 0; i < state.length; i++) {
      const input = prefillInputIdsList[i];
      const replay = input.length && input[0] === -1 ? 1 : 0;
      newTokens.push(input.length - replay);
      replayTokens.push(replay);
      if (input.length) state[i].committed = true;
      state[i].rest = remainingInputIdsList[i];
    }
    out.push({ newTokens, replayTokens });
    assert.ok(out.length <= 10000, "chunk loop must terminate");
  }
  return out;
};

const chunkShape = (trace: RowTrace[], row = 0): number[] =>
  trace.map(t => t.newTokens[row]);
const replayShape = (trace: RowTrace[], row = 0): number[] =>
  trace.map(t => t.replayTokens[row]);

describe("fix-a corrected bounded merge — structural oracle (cpu)", () => {
  let savedTailEnv: string | undefined;
  before(() => {
    // Determinism: the CPU block drives explicit tailMerge options; clear the
    // serving-path override so the REAL mirror (which resolves its default at
    // call time) behaves as the shipped default.
    savedTailEnv = process.env.SLOPLLM_TAIL_MERGE_MAX;
    delete process.env.SLOPLLM_TAIL_MERGE_MAX;
  });
  after(() => {
    if (savedTailEnv === undefined) delete process.env.SLOPLLM_TAIL_MERGE_MAX;
    else process.env.SLOPLLM_TAIL_MERGE_MAX = savedTailEnv;
  });

  it("cold 8,198: split [8192,6]/replays [0,1] becomes merged [8198]/replay [0]", () => {
    const split = traceRow(idsOf(8198), true, 8192, SPLIT_ARM);
    assert.deepEqual(chunkShape(split), [8192, 6]);
    assert.deepEqual(replayShape(split), [0, 1]);
    const merged = traceRow(idsOf(8198), true, 8192);
    assert.deepEqual(chunkShape(merged), [8198]);
    assert.deepEqual(replayShape(merged), [0]);
  });

  it("near-boundary regrouping (MTP on, chunkSize 8192) matches the corrected arithmetic", () => {
    const cases: { len: number; merged: number[]; mergedReplay: number[]; split: number[] }[] = [
      // exact multiple: nothing to merge; a fresh row's first chunk has replay 0.
      { len: 8192, merged: [8192], mergedReplay: [0], split: [8192] },
      { len: 8196, merged: [8196], mergedReplay: [0], split: [8192, 4] },
      // 8,199 merges -> phased halves floor/ceil 4,099/4,100 (size-math note).
      { len: 8199, merged: [8199], mergedReplay: [0], split: [8192, 7] },
      // cold 16,384 (exact multiple): the replay-induced one-token tail merges
      // into chunk 2 -> an 8,193-row chunk (halves 4,096/4,097: the ONLY half
      // that could ever hit the b12x 4,096 whitelist row).
      { len: 16384, merged: [8192, 8192], mergedReplay: [0, 1], split: [8192, 8191, 1] },
      { len: 16390, merged: [8192, 8198], mergedReplay: [0, 1], split: [8192, 8191, 7] },
      // inclusive endpoint: remainder 256 merges (executed rows 8,448 = C+T),
      // 257 does not.
      { len: 8448, merged: [8448], mergedReplay: [0], split: [8192, 256] },
      { len: 8449, merged: [8192, 257], mergedReplay: [0, 1], split: [8192, 257] },
    ];
    for (const c of cases) {
      const merged = traceRow(idsOf(c.len), true, 8192);
      const split = traceRow(idsOf(c.len), true, 8192, SPLIT_ARM);
      assert.deepEqual(chunkShape(merged), c.merged, `merged grouping ${c.len}`);
      assert.deepEqual(replayShape(merged), c.mergedReplay, `merged replay ${c.len}`);
      assert.deepEqual(chunkShape(split), c.split, `split grouping ${c.len}`);
      // Token conservation: consumed new tokens == prompt length, both arms.
      const sum = (shape: number[]) => shape.reduce((a, b) => a + b, 0);
      assert.equal(sum(chunkShape(merged)), c.len, `conservation merged ${c.len}`);
      assert.equal(sum(chunkShape(split)), c.len, `conservation split ${c.len}`);
    }
  });

  it("cached-prefix merges KEEP their entry overlap (the removed replay is only the eliminated boundary)", () => {
    // Cached row, 8,447 new tokens: greedy take is 8,191 (budget minus the
    // entry overlap), remainder 256 -> merges. The executed row is
    // 8,448 = C+T INCLUDING the prepended entry overlap, and the trace shows
    // replay 1 — the cached entry overlap survives the merge.
    const merged = traceRow(idsOf(8447), true, 8192, MTP_ARM, true);
    assert.deepEqual(chunkShape(merged), [8447]);
    assert.deepEqual(replayShape(merged), [1]);
    // Cached row, 8,448 new: remainder 257 -> no merge; BOTH chunks keep
    // their overlap (entry overlap + boundary replay).
    const noMerge = traceRow(idsOf(8448), true, 8192, MTP_ARM, true);
    assert.deepEqual(chunkShape(noMerge), [8191, 257]);
    assert.deepEqual(replayShape(noMerge), [1, 1]);
    // The split arm of the merged case keeps the boundary replay too.
    const split = traceRow(idsOf(8447), true, 8192, SPLIT_ARM, true);
    assert.deepEqual(chunkShape(split), [8191, 256]);
    assert.deepEqual(replayShape(split), [1, 1]);
  });

  it("64k/128k schedules are IDENTICAL (tails 7,175 / 6,023 far above the bound)", () => {
    // Corrected long-prompt accounting (verdict Q1 errata): cold 64,513 =
    // 8192 + 6*8191 + 7,175; cold 128,889 = 8192 + 14*8191 + 6,023.
    const expected64k = [8192, ...new Array(6).fill(8191), 7175];
    const expected128k = [8192, ...new Array(14).fill(8191), 6023];
    for (const [len, expected] of [[64513, expected64k], [128889, expected128k]] as const) {
      const merged = chunkShape(traceRow(idsOf(len), true, 8192));
      const split = chunkShape(traceRow(idsOf(len), true, 8192, SPLIT_ARM));
      assert.deepEqual(merged, expected, `merged ${len}`);
      assert.deepEqual(split, expected, `split ${len}`);
    }
  });

  it("preserves the no-progress throw and invalid-budget throws (caller inputs intact)", () => {
    // Regression contract (tests/test_glm51.ts chunked-MTP-budgets case):
    // chunkSize 1 with a required overlap cannot make progress and must throw
    // — a take==0 merge shortcut must NOT bypass this.
    assert.throws(() => chunker.selectPrefillChunk([[30]], 1, [-1], MTP_ARM), /required overlap/);
    assert.throws(() => chunker.selectPrefillChunk([[30]], 0, [], MTP_ARM), /Invalid prefill chunk size/);
    assert.throws(() => chunker.selectPrefillChunk([[30]], 1.5, [], MTP_ARM), /Invalid prefill chunk size/);
    assert.throws(() => chunker.selectPrefillChunk([[30]], 8192, [], { tailMergeMax: -1 }), /tail merge max/);
    // Caller inputs are never mutated.
    const ids = idsOf(8198);
    const snapshot = ids.slice();
    chunker.selectPrefillChunk([ids], 8192, [undefined], MTP_ARM);
    assert.deepEqual(ids, snapshot);
  });

  it("tailMergeMax = 0 reproduces pre-Fix-A slicing exactly (reproduction hook)", () => {
    assert.deepEqual(chunkShape(traceRow(idsOf(8198), true, 8192, SPLIT_ARM)), [8192, 6]);
    assert.deepEqual(chunkShape(traceRow(idsOf(8199), true, 8192, SPLIT_ARM)), [8192, 7]);
    assert.deepEqual(chunkShape(traceRow(idsOf(16384), true, 8192, SPLIT_ARM)), [8192, 8191, 1]);
  });

  it("overflow eligibility is exactly the qualified GLM single-request phased path", () => {
    const ctx: PrefillTailMergeContext = {
      chunkSize: 8192, tailMergeMax: PREFILL_TAIL_MERGE_MAX,
      maxSeqLen: 8192, batchLength: 1, mtp: true,
    };
    const rows = 8198;
    assert.equal(prefillTailMergeOverflowEligible(ctx, rows), true);
    // original inputIdsList.length must be ONE — empty rows count, so
    // [8198, []] is ineligible (verdict Q3a/Q3c).
    assert.equal(prefillTailMergeOverflowEligible({ ...ctx, batchLength: 2 }, rows), false);
    // caller approval: non-GLM-MTP callers stay greedy.
    assert.equal(prefillTailMergeOverflowEligible({ ...ctx, mtp: false }, rows), false);
    // phased mode off -> the merged chunk would NOT run the phased planner.
    const saved = process.env.SLOPLLM_PHASED_PREFILL;
    try {
      process.env.SLOPLLM_PHASED_PREFILL = "0";
      assert.equal(prefillTailMergeOverflowEligible(ctx, rows), false);
    } finally {
      if (saved === undefined) delete process.env.SLOPLLM_PHASED_PREFILL;
      else process.env.SLOPLLM_PHASED_PREFILL = saved;
    }
    // capacity guard: ceil(merged/2) must fit ws.maxSeqLen (planPrefill rejects
    // totals above it).
    assert.equal(prefillTailMergeOverflowEligible({ ...ctx, maxSeqLen: 4098 }, rows), false);
    // the merged row must be big enough for the ACTUAL planner to go phased.
    assert.equal(prefillTailMergeOverflowEligible(ctx, PHASED_PREFILL_MIN_ROWS - 1), false);
    // only the qualified 8,192-row budget carries the 256-token ceiling
    // (verdict Q3f: other chunk sizes keep greedy slicing).
    assert.equal(prefillTailMergeOverflowEligible({ ...ctx, chunkSize: 4096 }, rows), false);
    assert.equal(PREFILL_TAIL_MERGE_MAX, 256);
    assert.equal(PREFILL_TAIL_MERGE_QUALIFIED_CHUNK_SIZE, 8192);
    // Inclusive endpoint semantics (<= 256 merges, 257 does not) live in the
    // take rule's global cap (merged rows <= C + T = 8,448); the eligibility
    // predicate above is about the supported path/capacity only.
    assert.deepEqual(selectPrefillTakes([8448], [0], ctx), [8448]);
    assert.deepEqual(selectPrefillTakes([8449], [0], ctx), [8192]);
    // Serving-path arm override (SLOPLLM_TAIL_MERGE_MAX) resolves at call
    // time and reproduces both arm endpoints without a rebuild.
    const savedTail = process.env.SLOPLLM_TAIL_MERGE_MAX;
    try {
      process.env.SLOPLLM_TAIL_MERGE_MAX = "0";
      assert.equal(prefillTailMergeMaxSetting(), 0);
      assert.deepEqual(selectPrefillTakes([8198], [0], { chunkSize: 8192, maxSeqLen: 8192, batchLength: 1, mtp: true }), [8192]);
      process.env.SLOPLLM_TAIL_MERGE_MAX = "256";
      assert.equal(prefillTailMergeMaxSetting(), 256);
      assert.deepEqual(selectPrefillTakes([8198], [0], { chunkSize: 8192, maxSeqLen: 8192, batchLength: 1, mtp: true }), [8198]);
      process.env.SLOPLLM_TAIL_MERGE_MAX = "-1";
      assert.throws(() => prefillTailMergeMaxSetting(), /SLOPLLM_TAIL_MERGE_MAX/);
    } finally {
      if (savedTail === undefined) delete process.env.SLOPLLM_TAIL_MERGE_MAX;
      else process.env.SLOPLLM_TAIL_MERGE_MAX = savedTail;
    }
  });

  it("deferred rows are never merged early and never rewound (take>0, no take==0 shortcut)", () => {
    // Batch (greedy arm — batch length 2 is never eligible): row 2 is wholly
    // deferred when row 1 consumes the budget. Its small remainder must start
    // the NEXT chunk as-is; it must not be absorbed into row 1's chunk. Replay
    // semantics: a row's FIRST executed chunk has no replayed overlap even at
    // chunk 2 (matches tests/test_glm51.ts chunked-MTP-budgets contract).
    const trace = traceBatch([{ length: 8192 }, { length: 6 }], true, 8192, SPLIT_ARM);
    assert.deepEqual(trace.map(t => t.newTokens), [[8192, 0], [0, 6]]);
    assert.deepEqual(trace.map(t => t.replayTokens), [[0, 0], [0, 0]]);
    // A deferred row with CACHED history keeps its entry overlap when it
    // finally executes — deferral never rewinds or drops it.
    const cached = traceBatch([{ length: 8192 }, { length: 50, cached: true }], true, 8192, SPLIT_ARM);
    assert.deepEqual(cached.map(t => t.newTokens), [[8192, 0], [0, 50]]);
    assert.deepEqual(cached.map(t => t.replayTokens), [[0, 0], [0, 1]]);
    // White-box cap arithmetic (synthetic ctx): with the overflow allowance
    // already spent by an earlier row (Eused > 0), a later row's take is
    // clamped to 0 and the row DEFERS whole — the global cap is on the SUM of
    // executed rows incl. prepended overlaps (Eused + len + o <= C + T), not
    // per request (verdict Q3a). This can only be reached by a future policy
    // extension; today batchLength===1 makes it unreachable.
    const ctx: PrefillTailMergeContext = {
      chunkSize: 8192, tailMergeMax: 256, maxSeqLen: 8192, batchLength: 1, mtp: true,
    };
    const takes = selectPrefillTakes([8198, 6], [0, 0], { ...ctx, batchLength: 1 });
    assert.deepEqual(takes, [8198, 0], "second row defers after the overflow allowance is used");
  });

  it("selector-vs-mirror differential: identical per-chunk lengths and counts", () => {
    const cases: { name: string; rows: { length: number; cached?: boolean }[];
      mtp: boolean; chunkSize: number; maxSeqLen?: number }[] = [
      { name: "cold 8,198 eligible", rows: [{ length: 8198 }], mtp: true, chunkSize: 8192 },
      { name: "cold 8,199 eligible", rows: [{ length: 8199 }], mtp: true, chunkSize: 8192 },
      { name: "cold 8,448 cap", rows: [{ length: 8448 }], mtp: true, chunkSize: 8192 },
      { name: "cold 8,449 over cap", rows: [{ length: 8449 }], mtp: true, chunkSize: 8192 },
      { name: "cold 16,384 tail merge", rows: [{ length: 16384 }], mtp: true, chunkSize: 8192 },
      { name: "cached entry overlap 8,447", rows: [{ length: 8447, cached: true }], mtp: true, chunkSize: 8192 },
      { name: "cached entry overlap 8,448", rows: [{ length: 8448, cached: true }], mtp: true, chunkSize: 8192 },
      // empty rows: [8198, []] must look INELIGIBLE in prediction exactly as
      // in selection (original batch length 2).
      { name: "empty row batch", rows: [{ length: 8198 }, { length: 0 }], mtp: true, chunkSize: 8192 },
      { name: "two near-tail rows", rows: [{ length: 8194 }, { length: 8194 }], mtp: true, chunkSize: 8192 },
      { name: "overlap at budget limit", rows: [{ length: 8191, cached: true }, { length: 10 }], mtp: true, chunkSize: 8192 },
      { name: "mtp off", rows: [{ length: 8198 }], mtp: false, chunkSize: 8192 },
      { name: "64k unchanged", rows: [{ length: 64513 }], mtp: true, chunkSize: 8192 },
    ];
    for (const c of cases) {
      const actual = traceBatch(c.rows, c.mtp, c.chunkSize,
        c.mtp ? MTP_ARM : SPLIT_ARM, c.maxSeqLen ?? c.chunkSize);
      const mirror = mirrorChunkTakes(c.rows, c.mtp, c.chunkSize, c.maxSeqLen ?? c.chunkSize);
      assert.equal(actual.length, mirror.length, `${c.name}: chunk count`);
      for (let k = 0; k < mirror.length; k++) {
        assert.deepEqual(actual[k].newTokens, mirror[k], `${c.name}: per-chunk new tokens @${k}`);
      }
      // and the REAL private mirror agrees on the count.
      assert.equal(mirrorTotal(c.rows, c.mtp, c.chunkSize, c.maxSeqLen ?? c.chunkSize),
        actual.length, `${c.name}: real mirror total`);
    }
    // Invalid budgets: the mirror runs the SAME shared validation (same error),
    // and its no-progress case stops predicting instead of looping — actual
    // selection throws /required overlap/ (asserted above).
    assert.throws(() => mirrorTotal([{ length: 30 }], true, 0), /Invalid prefill chunk size/);
    assert.throws(() => mirrorTotal([{ length: 30 }], true, 1.5), /Invalid prefill chunk size/);
    assert.equal(mirrorTotal([{ length: 30, cached: true }], true, 1), 1,
      "no-progress prediction (C=1 + required overlap) stops at 1");
  });

  it("size-math oracle: phased halves fit the workspace (floor/ceil, no KV changes)", () => {
    // planPhasedPrefill splits total rows floor/ceil (A=floor(total/2)).
    const halves = (total: number) => [Math.floor(total / 2), total - Math.floor(total / 2)];
    assert.deepEqual(halves(8448), [4224, 4224]); // max merged: 4,224 = +3.125%
    assert.deepEqual(halves(8199), [4099, 4100]); // odd total
    assert.deepEqual(halves(8198), [4099, 4099]);
    assert.deepEqual(halves(8193), [4096, 4097]); // b12x: only the 4,096 half is whitelisted
    for (const total of [8193, 8198, 8199, 8448]) {
      assert.ok(Math.ceil(total / 2) <= 8192, `halves of ${total} fit ws.maxSeqLen=8192`);
    }
    // page unit note (documentation assertion): physical PAGE_SIZE=64 vs CP8
    // logical 512; the maxPages CLI default divides by the PHYSICAL 64
    // (openai-server.ts:107-108). No KV pool/config change is part of Fix A.
    assert.equal(8448 / 2, 4224);
    assert.ok(4224 / 512 <= 9, "a merged half needs <= 9 CP8 logical pages");
  });
});

// ---------------------------------------------------------------------------
// GPU section: the Q4 five-point oracle on the real GLM-5.x fixture model.
//
// Model host: tests/python/test_models/glm51_small/glm51_small_bf16 — the
// house tiny GLM-5.x model WITH a genuine MTP head (num_nextn_predict_layers
// 1: eh_proj/enorm/hnorm/shared_head.norm + a full MTP decoder layer) and
// sparse/MoE coverage (fixture-notes.md §4: "a model without MTP cannot
// prove its replay contract"). tests/csf_model_fixture.ts's tiny pair has
// num_nextn_predict_layers 0 and cannot exercise the replay contract; using
// it for merged-arm execution would silently skip the MTP rewind path.
//
// Arms run through the REAL Glm51Model.planChunkedPrefill caller
// (executePrefill loop) with SLOPLLM_TAIL_MERGE_MAX switching the shared
// policy: 0 = split/pre-Fix-A, 256 = merged. Cold 8,198 prompt = the
// observed bench case (split [8192,6]/replays [0,1] -> merged [8198]/[0]).
//
// Recording rule (fixture-notes.md §4): the block logs what EXECUTED. A
// clean skip under the disposition gate is NOT GPU equivalence evidence.
// TODO(consult #8 Q4) items below name the exact missing binding where a
// real API cannot yet express an oracle — no result is ever faked.
const GLM_NODE = path.resolve(__dirname, "../build/Release/glm.node");
const SMALL_MODEL_DIR = path.resolve(__dirname, "python/test_models/glm51_small/glm51_small_bf16");
const GPU_FIXTURES_ENV = process.env.SLOPLLM_CSF_GPU_TESTS;
const gpuFixturesSkip = (): string | false => {
  if (GPU_FIXTURES_ENV === "0") {
    return "SLOPLLM_CSF_GPU_TESTS=0 pins the GPU fixtures off (pure-CPU run)";
  }
  if (!fs.existsSync(GLM_NODE)) {
    return GPU_FIXTURES_ENV === "1"
      ? "SLOPLLM_CSF_GPU_TESTS=1 but build/Release/glm.node is missing (native build not finished)"
      : "build/Release/glm.node is missing (native build not finished); GPU equivalence is env-gated";
  }
  try {
    require(GLM_NODE);
  } catch (err) {
    return GPU_FIXTURES_ENV === "1"
      ? `SLOPLLM_CSF_GPU_TESTS=1 but build/Release/glm.node does not load: ${(err as Error).message}`
      : `build/Release/glm.node is present but not loadable in this process (${(err as Error).message}); GPU equivalence is env-gated`;
  }
  if (!fs.existsSync(path.join(SMALL_MODEL_DIR, "model.safetensors"))) {
    return `MTP fixture model missing at ${SMALL_MODEL_DIR}; generate it with `
      + "tests/python/test_models/glm51_small/generate_glm51_small.py (needs torch) — "
      + "a skip is NOT GPU equivalence evidence (record the block as deferred)";
  }
  return false;
};
const GPU_SKIP = gpuFixturesSkip();

const CHUNK = PREFILL_TAIL_MERGE_QUALIFIED_CHUNK_SIZE; // 8,192: the qualified budget
const MERGE_PROMPT_LEN = 8198; // cold bench case: tail 6 -> merges under T=256
// Teacher-forced probe positions: the moved A/B crossing (x4099/x4100 of the
// merged 4,099/4,099 halves), the eliminated outer boundary (x8192: where
// split chunk 2 prepended its replay), their neighborhoods, and the final
// position (equal-prompt-length logits). PROBE ORDER IS DESCENDING: each
// probe truncates the arm cache at its position, so later (smaller) probes
// only consume KV rows that earlier probes preserved.
const PROBE_POSITIONS = [8197, 8192, 8191, 4100, 4099, 4098];
// KV snapshot positions: same neighborhoods, gathered by LOGICAL position.
const KV_POSITIONS = [0, 4098, 4099, 4100, 8191, 8192, 8193, 8197];

interface ChunkTrace {
  /** Executed input ids of the row's chunk, replayed overlap prepend included. */
  input: number[];
  newTokens: number;
  /** Replay id (the eliminated boundary token) when an overlap was prepended. */
  replayToken: number | undefined;
  remainingAfter: number;
}

interface KvSnapshot {
  positions: number[];
  /** Packed base CKV(+KPE) row bytes per cache layer (flat U8 layout). */
  ckv: { layer: number; mtp: boolean; rows: Buffer[] }[];
  /** Indexer K rows (U8) + per-token scales (F32) per cache layer. */
  indexer: { layer: number; mtp: boolean; kRows: Buffer[]; scales: Buffer[] }[];
}

interface DecodeStepTrace {
  tokens: number[][];
  numAccepted: number[];
  numDraftTokens: number;
  warmup: boolean;
}

interface ArmRecord {
  arm: "split" | "merged";
  chunks: ChunkTrace[];
  committed: number[];
  targetToken: number | undefined;
  reported: number;
  allocLen: number;
  kv: KvSnapshot;
  decode: DecodeStepTrace[];
  probes: { position: number; logits: Buffer }[];
}

const readBytes = (ops: GlmOps, tensor: Tensor): Buffer => {
  const bytes = Buffer.alloc(tensor.bytes);
  tensor.d2h(bytes);
  ops.synchronize();
  return bytes;
};

/** GATHER BY LOGICAL TOKEN POSITION (verdict Q4 point 2): map the logical
 *  position through the sequence's page table to its packed row and copy
 *  exactly that row — never physical page IDs or unused allocation bytes are
 *  compared. d2h transport covers the whole layer tensor; only the sliced
 *  rows enter the oracle. */
const gatherLogicalRows = (ops: GlmOps, cache: PagedKVCache,
  sequence: { pages: { id: number }[] }, tensor: Tensor,
  positions: number[]): Buffer[] => {
  const bytes = readBytes(ops, tensor);
  const rowBytes = tensor.bytes / (cache.maxPages * cache.pageSize);
  return positions.map(position => {
    const page = sequence.pages[Math.floor(position / cache.pageSize)];
    assert.ok(page, `logical position ${position} must map to a content page`);
    const offset = (page.id * cache.pageSize + (position % cache.pageSize)) * rowBytes;
    return Buffer.from(bytes.subarray(offset, offset + rowBytes));
  });
};

const snapshotKv = (ops: GlmOps, cache: PagedKVCache,
  sequence: { pages: { id: number }[] }, positions: number[], nBaseLayers: number): KvSnapshot => {
  const ckv = cache.ckvData.map((tensor, layer) => ({
    layer,
    mtp: layer >= nBaseLayers,
    // Sparse flat layout (PagedKVCache sparseMode): the packed row holds the
    // base CKV/KPE payload; see the TODO(consult #8 Q4) in point (2) for the
    // field-unpack binding.
    rows: gatherLogicalRows(ops, cache, sequence, tensor, positions),
  }));
  const indexer = cache.kData.map((tensor, layer) => ({
    layer,
    mtp: layer >= nBaseLayers,
    kRows: tensor ? gatherLogicalRows(ops, cache, sequence, tensor, positions) : [],
    scales: cache.kScaleData[layer] ? gatherLogicalRows(ops, cache, sequence, cache.kScaleData[layer], positions) : [],
  }));
  return { positions, ckv, indexer };
};

/**
 * One full arm run: real chunked MTP prefill to completion, KV snapshot at
 * matching logical positions, real MTP decode steps, then teacher-forced
 * probes consuming the arm's committed KV (descending positions).
 */
const runPrefillArm = async (model: Glm51Model, ops: GlmOps,
  prompt: number[], tailMergeMax: number, arm: "split" | "merged"): Promise<ArmRecord> => {
  const savedTail = process.env.SLOPLLM_TAIL_MERGE_MAX;
  process.env.SLOPLLM_TAIL_MERGE_MAX = String(tailMergeMax);
  const ws = new ExecutionWorkspace(ops, 2, CHUNK);
  const cache: ChatCache = model.createChatCache(256, 2, CHUNK);
  try {
    cache.reset(1);
    const chunks: ChunkTrace[] = [];
    let remaining = [prompt.slice()];
    let committed = false;
    while (remaining.some(ids => ids.length)) {
      const res = await model.executePrefill(ws, cache, remaining, undefined, CHUNK);
      assert.equal(res.remainingInputIdsList.length, 1, "single-row arm");
      const input = res.prefillInputIdsList[0];
      assert.ok(input.length, "chunk made progress");
      // MTP semantics: the replayed overlap is prepended iff the row already
      // had committed history (cold chunk 1 has none).
      const replayToken: number | undefined = committed ? input[0] : undefined;
      if (replayToken !== undefined) {
        const expected = cache.getPagedKV().sequences[0].getTokenIds().at(-1);
        assert.equal(replayToken, expected, "replay id is the last committed token");
      }
      chunks.push({
        input: [...input],
        newTokens: input.length - (replayToken === undefined ? 0 : 1),
        replayToken,
        remainingAfter: res.remainingInputIdsList[0].length,
      });
      remaining = res.remainingInputIdsList;
      committed = true;
    }
    // Post-prefill state captured BEFORE decode/probes mutate anything.
    const sequence = cache.getPagedKV().sequences[0];
    const committedIds = sequence.getTokenIds();
    const targetToken = sequence.targetToken;
    const reported = sequence.reportedTokenCount();
    const allocLen = sequence.allocLen;
    const kv = snapshotKv(ops, cache.getPagedKV(), sequence, KV_POSITIONS,
      model.cfg.numHiddenLayers);

    // Output/decode evidence (point 4): multiple MTP decode steps after EACH
    // prefill arm (argmax greedy; default token-comparison verification).
    const decode: DecodeStepTrace[] = [];
    for await (const step of model.generateMtpDecode(ws, cache, 2)) {
      decode.push({
        tokens: step.tokens.map(row => [...row]),
        numAccepted: [...step.numAccepted],
        numDraftTokens: step.numDraftTokens,
        warmup: step.warmup,
      });
      if (decode.length >= 3) break;
    }
    assert.ok(decode.length >= 3, "multiple decode/MTP steps ran");

    // Teacher-forced probes (points 2/3): each probe truncates the arm cache
    // at position p, re-forwards the token at p against the arm's committed
    // KV[0..p), and captures next-token logits at p. Same probe machinery in
    // both arms; the comparison domain is same-prefix by construction.
    // DESCENDING order: each probe truncates, so later probes only consume KV
    // rows earlier probes preserved.
    const probes: { position: number; logits: Buffer }[] = [];
    for (const position of PROBE_POSITIONS) {
      sequence.truncate(position);
      const state = ws.planPrefill(model, 1, [1], cache);
      state.setInput([[prompt[position]]]);
      using hidden = model.forward(state);
      using logits = state.computeLogits(hidden, model);
      probes.push({ position, logits: readBytes(ops, logits) });
      ws.clearTracking();
    }
    return {
      arm,
      chunks,
      committed: committedIds,
      targetToken,
      reported,
      allocLen,
      kv,
      decode,
      probes,
    };
  } finally {
    cache.free();
    ws.free();
    if (savedTail === undefined) delete process.env.SLOPLLM_TAIL_MERGE_MAX;
    else process.env.SLOPLLM_TAIL_MERGE_MAX = savedTail;
  }
};

describe("fix-a merged-vs-split oracle (gpu)", {
  skip: GPU_SKIP,
  concurrency: false,
}, () => {
  let scratch: string;
  let ops: GlmOps;
  let modelSplit: Glm51Model;
  let modelMerged: Glm51Model;
  let runSplitA: ArmRecord;
  let runSplitB: ArmRecord;
  let runMerged: ArmRecord;

  before(async () => {
    scratch = fs.mkdtempSync(path.join(os.tmpdir(), "slopllm-fixa-equiv-"));
    const deviceId = parseInt(process.env.SLOPLLM_GPU ?? "0", 10);
    ops = new GlmOps(deviceId);
    // Same immutable checkpoint bytes per arm; separate fresh caches and
    // clean workspaces per run (two loads: no cross-arm model state at all).
    modelSplit = await Glm51Model.fromPretrained(ops, SMALL_MODEL_DIR, false, true);
    modelMerged = await Glm51Model.fromPretrained(ops, SMALL_MODEL_DIR, false, true);
    const savedPhased = process.env.SLOPLLM_PHASED_PREFILL;
    process.env.SLOPLLM_PHASED_PREFILL = "1"; // the qualified GLM MTP planner path
    try {
      const prompt = idsOf(MERGE_PROMPT_LEN);
      // Baseline repeatability recorded BEFORE the candidate runs (ladder §3).
      runSplitA = await runPrefillArm(modelSplit, ops, prompt, 0, "split");
      runSplitB = await runPrefillArm(modelSplit, ops, prompt, 0, "split");
      runMerged = await runPrefillArm(modelMerged, ops, prompt, PREFILL_TAIL_MERGE_MAX, "merged");
    } finally {
      if (savedPhased === undefined) delete process.env.SLOPLLM_PHASED_PREFILL;
      else process.env.SLOPLLM_PHASED_PREFILL = savedPhased;
    }
    console.log(`tail-merge GPU oracle EXECUTED: device ${deviceId}, ${path.join(SMALL_MODEL_DIR)} `
      + `mtp=true, prompts ${MERGE_PROMPT_LEN} tok; arms split x2 + merged x1; `
      + `probes ${PROBE_POSITIONS.join(",")}; kv positions ${KV_POSITIONS.join(",")}`);
  });

  after(() => {
    modelSplit.free();
    modelMerged.free();
    ops.free();
    fs.rmSync(scratch, { recursive: true, force: true });
  });

  // (5) Byte-identity-first ladder (fixture-notes.md §3) runs FIRST: baseline
  // repeatability, then byte identity as the first candidate oracle.
  it("byte-identity-first ladder: baseline repeatability frozen before candidate scoring", () => {
    // Baseline repeatability must hold BEFORE any candidate comparison — if
    // it does not, byte identity is not a supported oracle and the ladder
    // requires a frozen, owner-signed tolerance decision FIRST.
    assert.deepEqual(runSplitA.chunks, runSplitB.chunks, "baseline repeatability: chunk traces");
    assert.deepEqual(runSplitA.committed, runSplitB.committed, "baseline repeatability: committed history");
    assert.equal(runSplitA.targetToken, runSplitB.targetToken, "baseline repeatability: pending target");
    assert.deepEqual(runSplitA.kv, runSplitB.kv, "baseline repeatability: KV rows at logical positions");
    for (let i = 0; i < runSplitA.probes.length; i++) {
      assert.deepEqual(runSplitA.probes[i].logits, runSplitB.probes[i].logits,
        `baseline repeatability: teacher-forced logits @${runSplitA.probes[i].position}`);
    }
    assert.deepEqual(runSplitA.decode, runSplitB.decode, "baseline repeatability: decode/MTP steps");
    // Candidate byte identity is REQUESTED as the first oracle. If this
    // fails: shape-induced drift — it must be REPORTED, never silently
    // advertised as strict numerical identity. Freeze any tolerance +
    // quality criteria BEFORE inspecting candidate errors and obtain README
    // owner sign-off (fixture-notes.md §3 ladder); do NOT loosen here.
    assert.deepEqual(runMerged.kv, runSplitA.kv,
      "candidate KV byte identity at matching logical positions (if this fails: shape-induced drift — "
      + "report + freeze owner-signed tolerances before any re-run; never loosen after seeing errors)");
    for (let i = 0; i < runSplitA.probes.length; i++) {
      assert.deepEqual(runMerged.probes[i].logits, runSplitA.probes[i].logits,
        `candidate teacher-forced logits byte identity @${runSplitA.probes[i].position} (ladder as above)`);
    }
  });

  // (1) Structural oracle at state level.
  it("consumed new-token lists, replay ID/position, remainders, committed histories", () => {
    // Cold 8,198: split consumes [8192,6] new with replays [0,1]; merged
    // consumes [8198] with replay [0]. The token at the eliminated boundary
    // (x8192 = zero-based 8,191) remains in the prompt EXACTLY ONCE in the
    // merged arm (only its boundary replay disappears).
    assert.deepEqual(runSplitA.chunks.map(c => c.newTokens), [8192, 6], "split new tokens");
    assert.deepEqual(runSplitA.chunks.map(c => c.replayToken === undefined ? 0 : 1), [0, 1], "split replays");
    assert.equal(runSplitA.chunks[1].replayToken, idsOf(MERGE_PROMPT_LEN)[8191],
      "split chunk 2 replays x8192 (zero-based 8,191)");
    assert.deepEqual(runMerged.chunks.map(c => c.newTokens), [8198], "merged new tokens");
    assert.deepEqual(runMerged.chunks.map(c => c.replayToken === undefined ? 0 : 1), [0], "merged replays");
    // Remaining lists per chunk and conservation.
    assert.deepEqual(runSplitA.chunks.map(c => c.remainingAfter), [6, 0]);
    assert.deepEqual(runMerged.chunks.map(c => c.remainingAfter), [0]);
    const sum = (xs: number[]) => xs.reduce((a, b) => a + b, 0);
    assert.equal(sum(runSplitA.chunks.map(c => c.newTokens)), MERGE_PROMPT_LEN);
    assert.equal(sum(runMerged.chunks.map(c => c.newTokens)), MERGE_PROMPT_LEN);
    // Final committed histories are identical across arms; x8192 appears once.
    const prompt = idsOf(MERGE_PROMPT_LEN);
    assert.deepEqual(runSplitA.committed, prompt, "split committed history");
    assert.deepEqual(runMerged.committed, prompt, "merged committed history");
    // reportedTokenCount == allocLen and a pending target exists in both arms.
    assert.equal(runSplitA.reported, runSplitA.allocLen);
    assert.equal(runMerged.reported, runMerged.allocLen);
    assert.equal(runSplitA.allocLen, MERGE_PROMPT_LEN, "committed length is the prompt length");
    assert.equal(runMerged.allocLen, MERGE_PROMPT_LEN, "committed length is the prompt length");
    assert.notEqual(runSplitA.targetToken, undefined, "pending target valid (split)");
    assert.notEqual(runMerged.targetToken, undefined, "pending target valid (merged)");
    // Pending target equality across arms is an OUTPUT oracle (near-tie
    // selection) — grouped with the ladder, asserted here for localization.
    assert.equal(runMerged.targetToken, runSplitA.targetToken,
      "pending target equal across arms at equal prompt length (ladder §3 applies if this fails)");
  });

  it("page-sharing/COW isolation and untouched deferred rows", async () => {
    // Deferred rows (small scale, chunkSize 100; batch length 2 can never
    // merge — verdict Q3a): row 2 defers whole across chunk 1 and keeps its
    // cache/history untouched until its own chunk.
    const deferredRun = async () => {
      const ws = new ExecutionWorkspace(ops, 2, 100);
      const cache: ChatCache = modelSplit.createChatCache(64, 2, 100);
      try {
        cache.reset(2);
        const rows = [idsOf(130), idsOf(6)];
        const perChunk: number[][] = [];
        let remaining = rows.map(r => r.slice());
        while (remaining.some(ids => ids.length)) {
          const res = await modelSplit.executePrefill(ws, cache, remaining, undefined, 100);
          perChunk.push(res.prefillInputIdsList.map(ids => ids.length));
          // Deferred row: untouched while its input is still pending.
          if (remaining[1].length && !res.prefillInputIdsList[1].length) {
            const seq2 = cache.getPagedKV().sequences[1];
            assert.equal(seq2.reportedTokenCount(), 0, "deferred row untouched (reported)");
            assert.equal(seq2.targetToken, undefined, "deferred row untouched (no target)");
          }
          remaining = res.remainingInputIdsList;
        }
        assert.deepEqual(perChunk, [[100, 0], [31, 6]],
          "deferred row starts its first chunk as-is (chunk 2 row 0 = 30 new + 1 replay)");
        assert.deepEqual(cache.getPagedKV().sequences[0].getTokenIds(), rows[0]);
        assert.deepEqual(cache.getPagedKV().sequences[1].getTokenIds(), rows[1]);
      } finally {
        cache.free();
        ws.free();
      }
    };
    // COW/page-sharing isolation: copySequence shares FULL pages (refs > 1);
    // a non-page-aligned truncate detaches (COW) the shared partial page, and
    // writes to one sequence must never alter the sibling's KV rows.
    const cowRun = async () => {
      const ws = new ExecutionWorkspace(ops, 2, 100);
      const cache: ChatCache = modelSplit.createChatCache(64, 2, 100);
      try {
        cache.reset(2);
        const prompt = idsOf(100);
        await modelSplit.executePrefill(ws, cache, [prompt], undefined, 100);
        const kv = cache.getPagedKV();
        kv.copySequence(1, 0);
        const seq0 = kv.sequences[0];
        const seq1 = kv.sequences[1];
        // Sharing provenance: the full page is shared between the sequences.
        assert.equal(seq0.pages[0].id, seq1.pages[0].id, "copySequence shares the full page");
        assert.ok(seq0.pages[0].refs >= 2, `shared page carries refs>=2 (got ${seq0.pages[0].refs})`);
        const before = gatherLogicalRows(ops, kv, seq0, kv.ckvData[0], [0, 50, 63]);
        // Non-page-aligned truncate on the sibling triggers COW of the shared
        // partial page; then a real append writes only into the detached copy.
        seq1.truncate(50);
        await modelSplit.executePrefill(ws, cache, [[], idsOf(20)], undefined, 100);
        assert.notEqual(seq0.pages[0].id, seq1.pages[0].id, "truncate+append detached (COW) the shared page");
        assert.deepEqual(seq0.getTokenIds(), prompt, "sibling committed history untouched");
        const after = gatherLogicalRows(ops, kv, seq0, kv.ckvData[0], [0, 50, 63]);
        assert.deepEqual(after, before, "sibling KV rows untouched by the COW write");
      } finally {
        cache.free();
        ws.free();
      }
    };
    await deferredRun();
    await cowRun();
  });

  // (2) Numerical state oracle at matching LOGICAL positions.
  it("base CKV/KPE + indexer K/scales equal at matching logical positions", () => {
    // Gathered per layer BY LOGICAL TOKEN POSITION (the packed flat rows for
    // base CKV/KPE in sparseMode; indexer K U8 rows + F32 scales), including
    // the eliminated boundary (x8192 neighborhood) and the moved A/B crossing
    // (x4099/x4100) specifically. Unused allocation bytes are never compared.
    for (const [label, a, b] of [["ckv", runSplitA.kv.ckv, runMerged.kv.ckv],
      ["indexer", runSplitA.kv.indexer, runMerged.kv.indexer]] as const) {
      assert.equal(a.length, b.length, `${label} layer count`);
      for (let layer = 0; layer < a.length; layer++) {
        assert.deepEqual(a[layer], b[layer], `${label} layer ${layer} rows at logical positions`);
      }
    }
    // Scales are plain F32 rows: require finite values (the packed U8/U8-scale
    // payload cannot be field-unpacked yet — see the TODO below).
    for (const entry of runMerged.kv.indexer) {
      for (const row of entry.scales) {
        assert.ok(Number.isFinite(row.readFloatLE(0)), `indexer kScale finite (layer ${entry.layer})`);
      }
    }
    // TODO(consult #8 Q4) — missing binding: the packed flat CKV row layout
    // (concatAndCacheDsMla: base CKV bf16 + 4-byte scales per 128 + KPE bf16)
    // is not exported to TS, so per-field unpacking ("CKV/KPE and indexer
    // K/scales per layer/rank ... require finite values", verdict Q4 point 2)
    // cannot be asserted field-by-field yet. The oracle above compares the
    // PACKED rows byte-identically across arms (strictly stronger than
    // field equality modulo the finiteness check); a test-visible unpack
    // helper or a documented row-layout constant is needed to add the
    // per-field finite assertions.
  });

  it("shifted MTP prefix KV, final hidden/logits at equal prompt length", () => {
    // Stable shifted MTP prefix KV: the MTP cache layers (layer >=
    // numHiddenLayers) rows at the same logical positions — including the
    // neighborhoods of the eliminated boundary and the moved A/B crossing.
    const mtpLayers = (kv: KvSnapshot) => ({
      ckv: kv.ckv.filter(entry => entry.mtp),
      indexer: kv.indexer.filter(entry => entry.mtp),
    });
    assert.ok(mtpLayers(runSplitA.kv).ckv.length > 0, "MTP cache layers exist (genuine MTP head)");
    assert.deepEqual(mtpLayers(runMerged.kv), mtpLayers(runSplitA.kv),
      "shifted MTP prefix KV equal at matching logical positions");
    // Final hidden/logits at equal prompt length: the teacher-forced probe at
    // the final position (PROBE_POSITIONS[0] == 8197 = MERGE_PROMPT_LEN - 1),
    // byte-compared across arms in the ladder it above; the pending target
    // comparison lives in the structural it. The probe consumes each arm's
    // committed KV[0..8197) — a same-prefix forward in both arms.
    const finalProbe = (run: ArmRecord) => run.probes.find(p => p.position === MERGE_PROMPT_LEN - 1)!;
    assert.ok(finalProbe(runSplitA).logits.length > 0, "final-position logits captured");
    assert.equal(finalProbe(runSplitA).logits.length, modelSplit.cfg.vocabSize * 2,
      "logits row is one vocab-wide BF16 row");
    // TODO(consult #8 Q4) — missing binding: the in-plan final logits tensor
    // (planPrefillChunk's state.computeLogits result inside the chunk
    // generator) is not exposed across ChunkedPrefillPlan — only the selected
    // targets cross the boundary (readPrefillTarget reads selected ids). The
    // probe above recomputes the final-position logits from each arm's
    // committed KV through the same 1-token path in both arms; a direct
    // in-plan logits capture (a test-visible hook on the plan) would remove
    // the recompute. Also missing: the decoded MTP STARTUP SEED after decode's
    // mandatory rewind (verdict Q4 point 2) — generateMtpDecode keeps the
    // post-rewind seed in an UNNAMED workspace alloc (retained.seed copied
    // from forwardMtp's output; glm51_model.ts generateMtpDecode startup) and
    // exposes no readback; comparing the seed states requires a named alloc
    // or a snapshot hook. Partial evidence wired instead in the decode it:
    // the first post-rewind MTP step's tokens/acceptance condition on that
    // seed and are compared across arms. The prefill-terminal MTP row stays a
    // PLACEHOLDER by design and is only compared against the other arm's
    // placeholder state (the MTP rows above).
  });

  // (3) Comparison domain.
  it("teacher-forced interior logits at the same prefix; nothing at the eliminated boundary", () => {
    // Interior comparisons are teacher-forced at the same global prefix (the
    // probe machinery guarantees identical prefixes across arms). The moved
    // A/B crossing (4,099/4,099 split -> x4099/x4100) and the eliminated
    // outer boundary neighborhood are probed explicitly.
    for (const position of [4098, 4099, 4100, 8191, 8192]) {
      const a = runSplitA.probes.find(p => p.position === position)!;
      const b = runMerged.probes.find(p => p.position === position)!;
      assert.ok(a && b, `probe @${position} recorded in both arms`);
    }
    // There is NO requirement that intermediate next-token logits at the
    // eliminated outer boundary exist in the merged arm (never published to
    // the client): split's chunk-1 endpoint logits are NOT compared against
    // anything, and replay counts intentionally differ ([0,1] vs [0]). The
    // only boundary-position comparison is the teacher-forced probe @8192
    // (same global prefix in both arms) — byte-compared in the ladder it.
    assert.notDeepEqual(runSplitA.chunks.map(c => c.replayToken === undefined ? 0 : 1),
      runMerged.chunks.map(c => c.replayToken === undefined ? 0 : 1),
      "replay counts intentionally differ — they are outside the comparison domain");
  });

  // (4) Output/decode evidence.
  it("exact greedy token sequences: fixed fixtures + held-out production prompts", () => {
    // Multiple decode/MTP steps on the fixed 8,198-token fixture ran in every
    // arm (3 steps each, page-crossing from the 8,198-row committed history);
    // exact greedy sequences must match across arms and across reruns.
    for (let step = 0; step < runSplitA.decode.length; step++) {
      assert.deepEqual(runMerged.decode[step].tokens, runSplitA.decode[step].tokens,
        `greedy decode/MTP step ${step} identical across arms (ladder §3 applies if this fails)`);
    }
    assert.deepEqual(runMerged.decode.map(s => s.numAccepted), runSplitA.decode.map(s => s.numAccepted),
      "MTP accepted lengths identical across arms");
    // TODO(consult #8 Q4) — missing binding: held-out PRODUCTION prompts and
    // the real 8k/CP8/A16 cases (verdict Q4 point 5: "held-out production
    // prompts", fixture-notes §4 "real 8k/CP8/A16 cases for actual dispatch,
    // row geometry, and scratch") need the production NVFP4 checkpoint and a
    // CP8 ParallelOps fixture. CP8 requires 8 GPUs — the live service owns
    // 1-4,13-16 and 5-12 is another tenant — so those runs belong to the
    // maintenance window on an owner-sanctioned GPU set (RUNBOOK §D), not to
    // this fixture. The tiny model here covers plumbing, replay contracts and
    // scaled/real-8k row geometry only.
  });

  it("decode isolation (identical canonical input/KV) and end-to-end decode per arm", () => {
    // Decode isolation: runSplitB rebuilt an identical canonical input/KV
    // (byte-equal snapshot, ladder it) and produced the identical decode
    // stream (repeatability = isolation on identical canonical state).
    assert.deepEqual(runSplitB.decode, runSplitA.decode,
      "decode identical on identical canonical input/KV (isolation + repeatability)");
    // End-to-end decode after EACH prefill arm: both arms decoded (above) and
    // the merged arm's decode is compared to the split arm's in the previous
    // it. Token equality alone does not prove KV equivalence and similar KV
    // does not prove identical selection near ties — both comparisons are
    // asserted (KV rows in point (2), tokens here).
    assert.ok(runMerged.decode.length >= 3 && runSplitA.decode.length >= 3,
      "end-to-end decode ran after each prefill arm");
  });

  it("generator closure/reuse and graph replay survive the MTP decode restart", async () => {
    // Small fixed prompt (cheap): close the MTP decode generator mid-stream,
    // restart on the same cache, and require clean continued steps plus CUDA
    // graph capture/replay through the CaptureManager (house pattern:
    // tests/test_glm51.ts graph assertions).
    const ws = new ExecutionWorkspace(ops, 1, 256);
    const cache: ChatCache = modelSplit.createChatCache(64, 1, 256);
    const manager = new CaptureManager(ops);
    try {
      cache.reset(1);
      await modelSplit.executePrefill(ws, cache, [idsOf(300)], undefined, 256);
      let steps = 0;
      {
        const gen = modelSplit.generateMtpDecode(ws, cache, 2, manager);
        for await (const step of gen) {
          assert.equal(step.numDraftTokens, 2);
          if (++steps >= 3) break; // closure: break exits through the finally
        }
      }
      assert.equal(steps, 3, "first generator produced 3 steps before closure");
      // Reuse: a fresh generator on the same cache continues cleanly.
      {
        const gen = modelSplit.generateMtpDecode(ws, cache, 2, manager);
        for await (const step of gen) {
          assert.equal(step.numDraftTokens, 2, "restarted generator keeps the MTP contract");
          if (++steps >= 6) break;
        }
      }
      assert.equal(steps, 6, "restarted generator produced further steps");
      // Graph replay: at least one MTP decode capture instantiated a graph.
      assert.ok([...manager.captured].some(([key, entry]) =>
        key.includes("mtp-decode") && entry.graphExec !== null),
        "CUDA graph captured/replayed for the MTP decode path");
    } finally {
      manager[Symbol.dispose]();
      cache.free();
      ws.free();
    }
  });
});