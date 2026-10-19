import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  CsfContractError,
  PADDED_SCALE_CODE,
  concatFc1ScalePlanes,
  expandScalePlanes,
  expandScaleWindow,
  fixedStreamLength,
  nvfp4ScaleGeometry,
  sliceScalePlanes,
  tpExtent,
} from "../src/csf_codec";
import { makeScaleMatrix } from "./csf_fixture";
import { expandMxfp4ScalePlanes, expandMxfp4ScaleWindow } from "../src/mxfp4_csf_codec";

describe("mxfp4 csf codec", () => {
  // A 16x9 matrix exercises LSB order, the partial final selector byte,
  // byte wraparound, and exceptions straddling a sliced row boundary.
  const fixed = Buffer.concat([Buffer.alloc(16, 255), Buffer.from(Array(16).fill([0x55, 1]).flat())]);
  const exceptions = Buffer.from([8, 0, 0, 127, 9, 0, 0, 128]);
  it("recovers bytes and bounded windows across selector/exception boundaries", () => {
    const expected = Buffer.from(Array(16).fill([0, 255, 0, 255, 0, 255, 0, 255, 0]).flat());
    expected[8] = 127;
    expected[9] = 128;
    assert.deepEqual(expandMxfp4ScalePlanes(fixed, exceptions, 16, 9), expected);
    assert.deepEqual(expandMxfp4ScaleWindow(fixed, exceptions, 16, 9, 0, 2, 7, 2),
      Buffer.from([255, 127, 255, 0]));
    assert.deepEqual(expandMxfp4ScaleWindow(fixed, exceptions, 16, 9, 1, 1, 0, 2),
      Buffer.from([128, 255]));
  });
  it("rejects truncated, duplicate, out-of-range and unsorted exception planes", () => {
    for (const bad of [exceptions.subarray(0, 7), Buffer.concat([exceptions, exceptions]),
      Buffer.from([144, 0, 0, 127]), Buffer.from([9, 0, 0, 127, 8, 0, 0, 128])]) {
      assert.throws(() => expandMxfp4ScalePlanes(fixed, bad, 16, 9), CsfContractError);
    }
    assert.throws(() => expandMxfp4ScalePlanes(fixed.subarray(1), exceptions, 16, 9), CsfContractError);
    assert.throws(() => expandMxfp4ScaleWindow(fixed, exceptions, 16, 9, 16, 1, 0, 9), CsfContractError);
  });
});

// §9.7.1 CPU fixtures: exact NVFP4 codec recovery (byte-window4 fixed
// stream + u24 exceptions), invalid-plane rejection, TP slice/exception
// rebasing, exceptions at slice boundaries, asymmetric gate/up values.

const ROWS = 32;
const COLUMNS = 8;

function expectContractError(fn: () => void, match: RegExp): void {
  assert.throws(fn, (error: Error) => {
    assert.ok(error instanceof CsfContractError, `expected CsfContractError, got ${error.constructor.name}: ${error.message}`);
    assert.match(error.message, match);
    return true;
  });
}

describe("nvfp4 csf codec", () => {
  it("recovers exact scale bytes for asymmetric gate/up matrices", () => {
    const gate = makeScaleMatrix(ROWS, COLUMNS, {
      seed: 5,
      exceptionPositions: [0, 7, 8, 31, 32, 39, 40, 63, 127, 128, 255],
      exceptionValues: [0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x2a],
    });
    const up = makeScaleMatrix(ROWS, COLUMNS, {
      seed: 11,
      exceptionPositions: [15, 16, 24, 25, 240, 255],
      exceptionValues: [0x20, 0x21, 0x22, 0x23, 0x24, 0x25],
    });
    const gateOut = expandScalePlanes(gate.fixed, gate.exceptions, ROWS, COLUMNS);
    const upOut = expandScalePlanes(up.fixed, up.exceptions, ROWS, COLUMNS);
    assert.deepEqual([...gateOut], [...gate.logical]);
    assert.deepEqual([...upOut], [...up.logical]);
    // The two matrices are asymmetric: neither may leak into the other.
    assert.notDeepEqual([...gateOut], [...upOut]);
  });

  it("keeps exceptions at scale-matrix boundaries", () => {
    const matrix = makeScaleMatrix(ROWS, COLUMNS, {
      seed: 3,
      exceptionPositions: [0, COLUMNS - 1, (ROWS / 2) * COLUMNS - 1, (ROWS / 2) * COLUMNS, ROWS * COLUMNS - 1],
      exceptionValues: [0x10, 0x11, 0x12, 0x13, 0x14],
    });
    const out = expandScalePlanes(matrix.fixed, matrix.exceptions, ROWS, COLUMNS);
    assert.deepEqual([...out], [...matrix.logical]);
  });

  it("rejects malformed planes explicitly", () => {
    const matrix = makeScaleMatrix(ROWS, COLUMNS, { seed: 1, exceptionPositions: [1], exceptionValues: [0x21] });

    expectContractError(
      () => expandScalePlanes(matrix.fixed.subarray(0, matrix.fixed.length - 1), matrix.exceptions, ROWS, COLUMNS),
      /fixed stream must be \d+ bytes/,
    );

    const badBase = Buffer.from(matrix.fixed);
    badBase[0] = 241;
    expectContractError(() => expandScalePlanes(badBase, matrix.exceptions, ROWS, COLUMNS), /row base 241 exceeds/);

    const unsorted = Buffer.alloc(8);
    unsorted.writeUInt32LE((0x21 << 24) | 5, 0);
    unsorted.writeUInt32LE((0x22 << 24) | 4, 4);
    expectContractError(() => expandScalePlanes(matrix.fixed, unsorted, ROWS, COLUMNS), /sorted and unique/);

    const duplicate = Buffer.alloc(8);
    duplicate.writeUInt32LE((0x21 << 24) | 4, 0);
    duplicate.writeUInt32LE((0x22 << 24) | 4, 4);
    expectContractError(() => expandScalePlanes(matrix.fixed, duplicate, ROWS, COLUMNS), /sorted and unique/);

    const outOfRange = Buffer.alloc(4);
    outOfRange.writeUInt32LE((0x21 << 24) | (ROWS * COLUMNS), 0);
    expectContractError(() => expandScalePlanes(matrix.fixed, outOfRange, ROWS, COLUMNS), /outside the 32x8 scale matrix/);

    const oddWords = Buffer.alloc(6);
    expectContractError(() => expandScalePlanes(matrix.fixed, oddWords, ROWS, COLUMNS), /whole uint32 words/);

    // Valid scale values: positive finite e4m3 codes only.
    const nanException = Buffer.alloc(4);
    nanException.writeUInt32LE(0xff000000, 0);
    expectContractError(() => expandScalePlanes(matrix.fixed, nanException, ROWS, COLUMNS), /invalid E4M3 scale code 0xff/);

    const nanWindow = makeScaleMatrix(ROWS, COLUMNS, {
      seed: 1, bases: Array(ROWS).fill(0x7e), exceptionPositions: [], exceptionValues: [],
    });
    // 0x7e base with offset 1 yields the 0x7f e4m3fn NaN code in column 0.
    expectContractError(() => expandScalePlanes(nanWindow.fixed, nanWindow.exceptions, ROWS, COLUMNS), /invalid E4M3 scale code 0x7f/);

    const negativeWindow = makeScaleMatrix(ROWS, COLUMNS, {
      seed: 1, bases: Array(ROWS).fill(120), exceptionPositions: [], exceptionValues: [],
    });
    // base 120 + offset >= 8 sets the sign bit: negative scales are invalid.
    expectContractError(() => expandScalePlanes(negativeWindow.fixed, negativeWindow.exceptions, ROWS, COLUMNS), /invalid E4M3 scale code/);

    expectContractError(() => fixedStreamLength(30, 8), /16-row slabs/);
    expectContractError(() => nvfp4ScaleGeometry([32, 65]), /nibble pack/);
  });

  it("decodes bounded windows identically to full expansion", () => {
    const matrix = makeScaleMatrix(ROWS, COLUMNS, {
      seed: 9,
      exceptionPositions: [2, 9, 30, 64, 255],
      exceptionValues: [0x26, 0x27, 0x28, 0x29, 0x2a],
    });
    const full = expandScalePlanes(matrix.fixed, matrix.exceptions, ROWS, COLUMNS);
    const window = expandScaleWindow(matrix.fixed, matrix.exceptions, ROWS, COLUMNS, 8, 16, 2, 4);
    for (let r = 0; r < 16; r++) {
      for (let c = 0; c < 4; c++) {
        assert.equal(window[r * 4 + c], full[(8 + r) * COLUMNS + (2 + c)]);
      }
    }
    expectContractError(
      () => expandScaleWindow(matrix.fixed, matrix.exceptions, ROWS, COLUMNS, 8, 16, 2, 8),
      /outside the 32x8 scale matrix/,
    );
  });

  it("rebases TP slice exceptions including slice boundaries", () => {
    const matrix = makeScaleMatrix(ROWS, COLUMNS, {
      seed: 7,
      // Boundaries of the row slice [16, 32) and column slice [2, 6):
      // positions 0-based flat in the 32x8 scale matrix.
      exceptionPositions: [
        0,                     // outside (row 0)
        16 * COLUMNS + 2,      // first kept cell (16, 2)
        16 * COLUMNS + 5,      // last kept cell of first kept row (16, 5)
        16 * COLUMNS + 6,      // outside columns
        31 * COLUMNS + 2,      // first cell of last kept row
        31 * COLUMNS + 5,      // last kept cell (31, 5)
        31 * COLUMNS + 7,      // outside columns
      ],
      exceptionValues: [0x1f, 0x20, 0x21, 0x22, 0x23, 0x24, 0x25],
    });
    const slice = sliceScalePlanes(matrix.fixed, matrix.exceptions, ROWS, COLUMNS, [16, 32], [2, 6]);
    const expanded = expandScalePlanes(slice.fixed, slice.exceptions, 16, 4);
    for (let r = 0; r < 16; r++) {
      for (let c = 0; c < 4; c++) {
        assert.equal(expanded[r * 4 + c], matrix.logical[(16 + r) * COLUMNS + (2 + c)]);
      }
    }
    // Exceptions outside the slice were dropped; the kept ones are linear in
    // the 16x4 output matrix (row 0..15, column 0..3).
    assert.equal(slice.exceptions.length, 4 * 4);
    const positions = [];
    for (let i = 0; i < 4; i++) positions.push(slice.exceptions.readUInt32LE(i * 4) & 0xffffff);
    assert.deepEqual(positions, [0 * 4 + 0, 0 * 4 + 3, 15 * 4 + 0, 15 * 4 + 3]);
  });

  it("pads TP slices like zero-weight shards", () => {
    const matrix = makeScaleMatrix(ROWS, COLUMNS, {
      seed: 13,
      exceptionPositions: [16 * COLUMNS, 16 * COLUMNS + 1],
      exceptionValues: [0x2b, 0x2c],
    });
    const padded = sliceScalePlanes(matrix.fixed, matrix.exceptions, ROWS, COLUMNS, [16, 32], [0, 4], 32, 8);
    const expanded = expandScalePlanes(padded.fixed, padded.exceptions, 32, 8);
    for (let r = 0; r < 16; r++) {
      const sourceRow = 16 + r;
      for (let c = 0; c < 4; c++) {
        // Real rows/columns keep their source values (exceptions included).
        assert.equal(expanded[r * 8 + c], matrix.logical[sourceRow * COLUMNS + c]);
      }
      for (let c = 4; c < 8; c++) {
        // Appended columns carry zero offset codes over their row's base.
        assert.equal(expanded[r * 8 + c], matrix.bases[sourceRow]);
      }
    }
    for (let r = 16; r < 32; r++) {
      for (let c = 0; c < 8; c++) {
        // Appended rows get the E4M3 1.0 scale with zero offsets.
        assert.equal(expanded[r * 8 + c], PADDED_SCALE_CODE);
      }
    }
  });

  it("fuses FC1 projections with offset exceptions", () => {
    const up = makeScaleMatrix(16, 4, {
      seed: 2,
      exceptionPositions: [0, 63],
      exceptionValues: [0x21, 0x22],
    });
    const gate = makeScaleMatrix(16, 4, {
      seed: 4,
      exceptionPositions: [1, 63],
      exceptionValues: [0x23, 0x24],
    });
    const fused = concatFc1ScalePlanes(
      { fixed: up.fixed, exceptions: up.exceptions },
      { fixed: gate.fixed, exceptions: gate.exceptions },
      16, 4,
    );
    const expanded = expandScalePlanes(fused.fixed, fused.exceptions, 32, 4);
    for (let i = 0; i < 16 * 4; i++) {
      assert.equal(expanded[i], up.logical[i]);
      assert.equal(expanded[16 * 4 + i], gate.logical[i]);
    }
  });

  it("validates TP extents", () => {
    assert.deepEqual(tpExtent(2048, 0, 2, 64), [0, 1024]);
    assert.deepEqual(tpExtent(2048, 1, 2, 64), [1024, 2048]);
    expectContractError(() => tpExtent(2048, 2, 2, 64), /valid TP rank/);
    expectContractError(() => tpExtent(2000, 0, 2, 64), /aligned to 64/);
  });
});
