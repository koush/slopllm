import { after, before, describe, it } from "node:test";
import assert from "node:assert/strict";
import { GlmOps, f32ToBf16Bytes } from "../src/glm_ops";
import { WorkspaceBase } from "../src/workspace";

describe("DeviceOps.appendSelectedMtpCaches", () => {
  let glm: GlmOps;
  let ws: WorkspaceBase;

  before(() => {
    glm = new GlmOps(parseInt(process.env.GLM_GPU ?? "0", 10));
    ws = new WorkspaceBase(glm);
  });

  after(() => {
    ws.free();
    glm.free();
  });

  it("matches per-layer sparse MLA and indexer appends", () => {
    const rows = 6;
    const kvRank = 128;
    const peDim = 64;
    const indexDim = 64;
    const pageSize = 4;
    const maxPages = 3;
    const bytesPerToken = kvRank + kvRank / 128 * 4 + peDim * 2;
    const selected = new Int32Array([0, 2, 5]);
    const pageIndices = new Int32Array([1, 2]);
    const pageIndptr = new Int32Array([0, 1, 2]);
    const batchIndices = new Int32Array([0, 0, 1]);
    const positions = new Int32Array([1, 3, 0]);

    const makeSource = (width: number, bias: number) => {
      const values = new Float32Array(rows * width);
      for (let row = 0; row < rows; row++) {
        for (let col = 0; col < width; col++) values[row * width + col] = bias + row + (col % 8) / 8;
      }
      const tensor = ws.alloc([rows, width], "BF16");
      tensor.h2d(f32ToBf16Bytes(values));
      return tensor;
    };
    const uploadI32 = (values: Int32Array) => {
      const tensor = ws.alloc([values.length], "I32");
      tensor.h2d(Buffer.from(values.buffer));
      return tensor;
    };

    const srcCkv = [makeSource(kvRank, 1), makeSource(kvRank, 9)];
    const srcKpe = [makeSource(peDim, 17), makeSource(peDim, 25)];
    const srcIndexer = [makeSource(indexDim, 33)];
    const dstCkv = srcCkv.map(() => ws.alloc([maxPages, pageSize, bytesPerToken], "U8"));
    const refCkv = srcCkv.map(() => ws.alloc([maxPages, pageSize, bytesPerToken], "U8"));
    const dstIndexer = [ws.alloc([maxPages, pageSize, indexDim], "U8")];
    const dstIndexerScale = [ws.alloc([maxPages, pageSize], "F32")];
    const refIndexer = [ws.alloc([maxPages, pageSize, indexDim], "U8")];
    const refIndexerScale = [ws.alloc([maxPages, pageSize], "F32")];
    for (const tensor of [...dstCkv, ...refCkv, ...dstIndexer, ...dstIndexerScale, ...refIndexer, ...refIndexerScale]) tensor.h2d(Buffer.alloc(tensor.bytes));

    using sourceRows = uploadI32(selected);
    using indices = uploadI32(pageIndices);
    using indptr = uploadI32(pageIndptr);
    using batches = uploadI32(batchIndices);
    using pos = uploadI32(positions);
    using lastPageLen = uploadI32(new Int32Array([4, 1]));

    using srcCkvPtrs = ws.alloc([srcCkv.length], "I64");
    using srcKpePtrs = ws.alloc([srcKpe.length], "I64");
    using dstCkvPtrs = ws.alloc([dstCkv.length], "I64");
    using srcIndexerPtrs = ws.alloc([srcIndexer.length], "I64");
    using dstIndexerPtrs = ws.alloc([dstIndexer.length], "I64");
    using dstIndexerScalePtrs = ws.alloc([dstIndexerScale.length], "I64");
    srcCkvPtrs.writePointers(srcCkv);
    srcKpePtrs.writePointers(srcKpe);
    dstCkvPtrs.writePointers(dstCkv);
    srcIndexerPtrs.writePointers(srcIndexer);
    dstIndexerPtrs.writePointers(dstIndexer);
    dstIndexerScalePtrs.writePointers(dstIndexerScale);

    glm.appendSelectedMtpCaches(
      srcCkvPtrs, srcKpePtrs, dstCkvPtrs, undefined,
      srcIndexerPtrs, dstIndexerPtrs, dstIndexerScalePtrs,
      sourceRows, indices, indptr, batches, pos,
      pageSize, kvRank, peDim, indexDim, true,
    );

    for (let layer = 0; layer < srcCkv.length; layer++) {
      using selectedCkv = srcCkv[layer].indexSelect(sourceRows);
      using selectedKpe = srcKpe[layer].indexSelect(sourceRows);
      using _cache = glm.concatAndCacheDsMla(undefined as never, layer, refCkv[layer], selectedCkv, selectedKpe,
        indices, indptr, batches, pos, selected.length, kvRank, peDim, kvRank, peDim, pageSize);
    }
    using selectedIndexer = srcIndexer[0].indexSelect(sourceRows);
    const indexerCache = glm.mlaKvCacheAppend(undefined as never, 0, refIndexer[0], refIndexerScale[0], indices, indptr, lastPageLen,
      selectedIndexer, null, batches, pos, selected.length, indexDim, 0, indexDim, 0, pageSize);
    indexerCache.ckv[Symbol.dispose]();
    indexerCache.kpe?.[Symbol.dispose]();
    glm.synchronize();

    for (let layer = 0; layer < dstCkv.length; layer++) {
      const actual = Buffer.alloc(dstCkv[layer].bytes);
      const expected = Buffer.alloc(refCkv[layer].bytes);
      dstCkv[layer].d2h(actual);
      refCkv[layer].d2h(expected);
      assert.deepEqual(actual, expected);
    }
    const actualIndexer = Buffer.alloc(dstIndexer[0].bytes);
    const expectedIndexer = Buffer.alloc(refIndexer[0].bytes);
    dstIndexer[0].d2h(actualIndexer);
    refIndexer[0].d2h(expectedIndexer);
    assert.deepEqual(actualIndexer, expectedIndexer);
    const actualIndexerScale = Buffer.alloc(dstIndexerScale[0].bytes);
    const expectedIndexerScale = Buffer.alloc(refIndexerScale[0].bytes);
    dstIndexerScale[0].d2h(actualIndexerScale);
    refIndexerScale[0].d2h(expectedIndexerScale);
    assert.deepEqual(actualIndexerScale, expectedIndexerScale);
  });
});
