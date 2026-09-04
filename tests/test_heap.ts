import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { Heap } from "../src/heap";

describe("Heap", () => {
  it("aligns allocations and preserves prefix and suffix ranges", () => {
    const heap = new Heap();
    heap.manage(1, 1023);

    assert.deepEqual(heap.alloc(256), { ptr: 256, length: 256 });
    assert.ok(heap.contains(1, 255));
    assert.ok(heap.contains(512, 512));
  });

  it("returns aligned capacity while still accepting a shorter tail", () => {
    const rounded = new Heap();
    rounded.manage(256, 1024);
    assert.deepEqual(rounded.alloc(100), { ptr: 256, length: 256 });

    const tail = new Heap();
    tail.manage(256, 100);
    assert.deepEqual(tail.alloc(100), { ptr: 256, length: 100 });
  });

  it("coalesces adjacent returned ranges", () => {
    const heap = new Heap();
    heap.manage(256, 256);
    heap.manage(768, 256);
    heap.manage(512, 256);

    assert.deepEqual(heap.alloc(768), { ptr: 256, length: 768 });
    assert.throws(() => heap.alloc(1), /Heap OOM/);
  });

  it("rejects overlapping and invalid ranges", () => {
    const heap = new Heap();
    heap.manage(256, 256);

    assert.throws(() => heap.manage(384, 64), /overlaps/);
    assert.throws(() => heap.manage(-1, 64), /non-negative safe integer/);
    assert.throws(() => heap.manage(1024, 0), /positive safe integer/);
  });

  it("claims an exact captured range from a coalesced region", () => {
    const heap = new Heap();
    heap.manage(256, 1024);

    assert.deepEqual(heap.claim(512, 256), { ptr: 512, length: 256 });
    assert.ok(heap.contains(256, 256));
    assert.ok(heap.contains(768, 512));
    assert.equal(heap.claim(512, 256), undefined);
  });

  it("drains atomically and coalesces with the destination", () => {
    const source = new Heap();
    const destination = new Heap();
    source.manage(512, 256);
    destination.manage(256, 256);

    source.drainTo(destination);
    assert.deepEqual(destination.alloc(512), { ptr: 256, length: 512 });
    assert.throws(() => source.alloc(1), /Heap OOM/);
  });

  it("tracks the allocation layout independently of returned ranges", () => {
    const heap = new Heap();
    heap.manage(4096, 4096);
    const initial = heap.layoutSignature();
    const allocation = heap.alloc(100);
    const allocated = heap.layoutSignature();
    heap.manage(allocation.ptr, allocation.length);

    assert.notEqual(allocated, initial);
    assert.equal(heap.layoutSignature(), allocated);
  });
});
