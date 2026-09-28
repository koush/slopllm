import { it } from "node:test";
import assert from "node:assert/strict";
import { GlmOps } from "../src/glm_ops";
import { ParallelOps, ParallelTensor } from "../src/parallel_ops";
import { PrefixTierPolicy } from "../src/generation-scheduler";
import { PAGE_SIZE, PagedKVCache } from "../src/paged_kv";
import { Tensor } from "../src/tensor";

// End-to-end copyPrefixFrom under context parallelism: two GPUs as CP shards,
// a small sparse-MLA layout (2 layers, 8 pages, 656 bytes/token). The source
// and verification caches are pinned-host so page contents can be written and
// compared on the host; the middle cache is the real device-backed CP cache.
// This exercises the paths the shell tests in test_prefix_cache.ts stub out:
// PagedKVCache allocation with Row-sharded parallel tensors, and the
// ParallelOps/GlmOps memcpyBatchAsync implementations (HostToDevice and
// DeviceToHost shard copies).
const NLAYERS = 2;
const MAX_PAGES = 8;

const tokens = (length: number) => Array.from({ length }, (_, i) => i + 100);

function makeCpCache(ops: ParallelOps, pinned: boolean): PagedKVCache {
  return new PagedKVCache(ops, 1, 1, NLAYERS, MAX_PAGES, 2, PAGE_SIZE, 512, 64, true, 128, [], pinned);
}

// Same family order as copyPrefixFrom's layerCachePairs.
function cacheTensors(cache: PagedKVCache, layer: number): Tensor[] {
  return [cache.kData[layer], cache.kScaleData[layer], cache.ckvData[layer]].filter(tensor => tensor);
}

// Writes a distinct constant per (page, layer, family, shard) so a dropped,
// duplicated, or permuted page cannot pass the round-trip comparison.
function fillPageRows(cache: PagedKVCache, tag: number): void {
  for (let layer = 0; layer < cache.nLayers; layer++) {
    cacheTensors(cache, layer).forEach((tensor, family) => {
      (tensor as ParallelTensor).shards.forEach((shard, shardIdx) => {
        const rowBytes = Tensor.byteCount(shard.shape.slice(1), shard.type);
        shard.withPinnedBuffer(buf => {
          for (let page = 0; page < MAX_PAGES; page++) {
            buf.fill((tag + page * 31 + layer * 7 + family * 3 + shardIdx) & 0xff, page * rowBytes, (page + 1) * rowBytes);
          }
        });
      });
    });
  }
}

// Page rows of every layer/family/shard as host bytes, in a stable order.
function readPinnedPageRows(cache: PagedKVCache, pages: number[]): Buffer[] {
  const rows: Buffer[] = [];
  for (let layer = 0; layer < cache.nLayers; layer++) {
    for (const tensor of cacheTensors(cache, layer)) {
      for (const shard of (tensor as ParallelTensor).shards) {
        const rowBytes = Tensor.byteCount(shard.shape.slice(1), shard.type);
        shard.withPinnedBuffer(buf => {
          for (const page of pages) {
            rows.push(Buffer.from(buf.subarray(page * rowBytes, (page + 1) * rowBytes)));
          }
        });
      }
    }
  }
  return rows;
}

it("copyPrefixFrom round-trips full pages between pinned-host and CP device caches", async () => {
  const glm0 = new GlmOps(0);
  const glm1 = new GlmOps(1);
  const ops = new ParallelOps([glm0, glm1]);
  let pinnedFrom: PagedKVCache | undefined;
  let device: PagedKVCache | undefined;
  let pinnedBack: PagedKVCache | undefined;
  try {
    pinnedFrom = makeCpCache(ops, true);
    device = makeCpCache(ops, false);
    pinnedBack = makeCpCache(ops, true);
    assert.equal(device.pageSize, PAGE_SIZE * ops.worldSize, "a logical CP page spans all shards");
    assert.deepEqual(
      (device.ckvData[0] as ParallelTensor).shards.map(shard => shard.shape),
      [[MAX_PAGES, PAGE_SIZE, 656], [MAX_PAGES, PAGE_SIZE, 656]],
      "each shard holds every page's 64-token slice",
    );

    // Source: 260 committed tokens = 2 full 128-token CP pages + a 4-token tail, plus a pending decode token.
    const srcSeq = pinnedFrom.ensureSequence(0);
    pinnedFrom.allocAppendPages(0, 260);
    pinnedFrom.reportTokens(0, tokens(260), 777);
    fillPageRows(pinnedFrom, 5);
    const expectedRows = readPinnedPageRows(pinnedFrom, [0, 1]);

    // Pinned host -> device: full pages move via HostToDevice shard copies.
    const suffix = device.copyPrefixFrom(pinnedFrom, srcSeq, 0);
    assert.deepEqual(suffix, tokens(260).slice(256));
    const deviceSeq = device.sequences[0];
    assert.equal(deviceSeq.allocLen, 256);
    assert.deepEqual(deviceSeq.getTokenIds(), tokens(256));
    assert.equal(deviceSeq.targetToken, tokens(260)[256]);

    // Device -> a fresh pinned cache: the same pages back via DeviceToHost copies.
    // The device sequence is page-aligned (the 4-token tail never left the source),
    // so nothing remains to prefill and the pending token propagates.
    const backSuffix = pinnedBack.copyPrefixFrom(device, deviceSeq, 0);
    assert.deepEqual(backSuffix, []);
    assert.deepEqual(pinnedBack.sequences[0].getTokenIds(), tokens(256));
    assert.equal(pinnedBack.sequences[0].targetToken, tokens(260)[256]);

    await ops.synchronizeAsync();

    // Every copied page row must survive the round-trip byte-for-byte, per layer, cache family, and CP shard.
    assert.deepEqual(readPinnedPageRows(pinnedBack, [0, 1]), expectedRows);
  } finally {
    pinnedFrom?.free();
    device?.free();
    pinnedBack?.free();
    ops.free();
    glm0.free();
    glm1.free();
  }
});

// End-to-end PrefixTierPolicy under context parallelism: a pinned-host cache
// as the offload/restore tier behind the CP device cache. Exercises both
// tiers' real parallel-tensor bookkeeping and both copy directions through
// parallel_ops memcpyBatchAsync: host->device restore at admit time, and
// device->host offload under page pressure.
it("PrefixTierPolicy restores from and offloads into a pinned-host CP tier", async () => {
  const glm0 = new GlmOps(0);
  const glm1 = new GlmOps(1);
  const ops = new ParallelOps([glm0, glm1]);
  let host: PagedKVCache | undefined;
  let device: PagedKVCache | undefined;
  let victimSource: PagedKVCache | undefined;
  let pinnedBack: PagedKVCache | undefined;
  try {
    host = makeCpCache(ops, true);
    device = makeCpCache(ops, false);
    victimSource = makeCpCache(ops, true);
    pinnedBack = makeCpCache(ops, true);
    const policy = new PrefixTierPolicy(device, ops, host);

    // Host tier: 260 committed tokens = 2 full 128-token CP pages + a tail.
    const original = host.ensureSequence(0);
    host.allocAppendPages(0, 260);
    host.reportTokens(0, tokens(260));
    fillPageRows(host, 5);

    // Admit one request whose prompt extends the host content: the host prime
    // moves both full pages into an active slot, the device is empty, so the
    // restore copies the delta across tiers inside the policy's drain.
    const prompt = [...tokens(260), 501, 502];
    const suffixes = await policy.prefixMatch([prompt]);
    assert.deepEqual(suffixes, [prompt.slice(256)]);
    const restored = device.sequences[0];
    assert.equal(restored.allocLen, 256);
    assert.deepEqual(restored.getTokenIds(), tokens(256));
    assert.equal(restored.targetToken, tokens(260)[256]);
    assert.equal(host.staging.size, 1, "the staged original survives the restore");
    assert.equal(original.pages[0].refs, 2, "the active host prime slot shares its pages");
    assert.equal(host.sequences.length, 1);

    // Byte-verify the H2D restore with a device->pinned round trip.
    assert.deepEqual(pinnedBack.copyPrefixFrom(device, restored, 0), []);
    await ops.synchronizeAsync();
    assert.deepEqual(readPinnedPageRows(pinnedBack, [0, 1]), readPinnedPageRows(host, [0, 1]));

    // Offload direction: fill a device row with known bytes from another
    // pinned cache, retain it as a finished row, then allocate past the
    // remaining device pool. The victim is evicted and lands as a fresh
    // active host slot, byte-for-byte.
    const victimTokens = Array.from({ length: 2 * device.pageSize }, (_, i) => i + 5000);
    const victimSeq = victimSource.ensureSequence(0);
    victimSource.allocAppendPages(0, victimTokens.length);
    victimSource.reportTokens(0, victimTokens);
    fillPageRows(victimSource, 9);
    const expectedVictimRows = readPinnedPageRows(victimSource, [0, 1]);

    device.copyPrefixFrom(victimSource, victimSeq, 1);
    policy.retainSequence(1);
    assert.equal(device.staging.size, 1);

    // 8 device pages: 2 restored + 2 victim = 4 available; this allocation
    // needs 5, so the victim must be offloaded to free its pages.
    device.ensureSequence(2);
    device.allocAppendPages(2, 5 * device.pageSize);
    assert.equal(device.staging.size, 0, "victim fully evicted");

    const offloaded = host.sequences[1];
    assert.equal(host.sequences.length, 2, "offload destination appended as a second active slot");
    assert.deepEqual(offloaded.getTokenIds(), victimTokens);
    assert.deepEqual(offloaded.pages.map(page => page.id), [3, 4], "fresh host pages past the original's three");

    await ops.synchronizeAsync();
    assert.deepEqual(readPinnedPageRows(host, [3, 4]), expectedVictimRows, "D2H offload must preserve every page byte");
  } finally {
    host?.free();
    device?.free();
    victimSource?.free();
    pinnedBack?.free();
    ops.free();
    glm0.free();
    glm1.free();
  }
});