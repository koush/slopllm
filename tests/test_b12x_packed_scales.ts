import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { CsfContractError, expandScalePlanes } from "../src/csf_codec";
import {
  AUDITED_MAX_SCALE_CODE,
  AUDITED_MIN_SCALE_CODE,
  E4M3_MAX_FINITE_CODE,
  E8M0_K32_BF16_MAX_SCALE_BYTE,
  E8M0_LOGICAL_TAIL_SCALE_N_ALIGNMENT,
  NVFP4_DISCARD_BOUNDARY,
  computePackedScaleFactor,
  e4m3Decode,
  e4m3K16ScaleCapacity,
  e4m3K16ScaleColumns,
  fp16Bits,
  packedScaleByte,
  preparePackedScaleContract,
  processPackedGlobalScale,
  processPackedScales,
  restorePackedScale,
  swizzleScaleGroupLanes,
} from "./b12x_packed_scales_fixture";
import { makeScaleMatrix } from "./csf_fixture";

// M3 prep for the A4/A16 hybrid port: CPU-only fixture tests for the packed
// NVFP4 scale-boundary contract (design doc
// `docs/design_docs/b12x_nvfp4_packed_scale_contract.md`, executable
// reference `w4a16/prepare.py` and
// `tests/experimental/b12x/moe/test_w4a16_packed_format.py` in the flashinfer
// reference tree). No GPU, no native addon.
//
// Scope notes:
// - This checkpoint uses E4M3 K/16 block scales (one e4m3 scale byte per 16 K
//   channels), NOT E8M0 K/32. The reference's E8M0-specific tail guards and
//   serving clamps are asserted NOT applicable below instead of being
//   force-fitted.
// - A4 (4-bit activation prefill) is NOT lossless: it has a characterized
//   prefill regression. "Lossless" here means only the CSF
//   container-to-buffer scale round-trip.

function expectContractError(fn: () => void, match: RegExp): void {
  assert.throws(fn, (error: Error) => {
    assert.ok(error instanceof CsfContractError, `expected CsfContractError, got ${error.constructor.name}: ${error.message}`);
    assert.match(error.message, match);
    return true;
  });
}

describe("b12x packed nvfp4 scale boundary", () => {
  // Golden boundary table from the design doc: the boundary test "covers all
  // finite nonnegative E4M3 codes within the normalized range at factors 1,
  // 2, 4, and 8" and checks exact reconstruction of retained values plus the
  // respective 7, 3, 1, and 0 discarded positive source codes.
  for (const [factor, discardedCodes] of [
    [1, 7],
    [2, 3],
    [4, 1],
    [8, 0],
  ] as const) {
    it(`retains normalized normals exactly and flushes normalized subnormals at factor ${factor}`, () => {
      const source: number[] = [];
      for (let code = 0; code <= E4M3_MAX_FINITE_CODE; code++) {
        if (e4m3Decode(code) * factor <= 448) source.push(code);
      }
      // The reference harness repeats each source value across 4 lanes (the
      // packed grid stores scales in groups of 4).
      const scales = source.flatMap((code) => [code, code, code, code]);
      const packed = processPackedScales(scales, factor);
      let discarded = 0;
      for (let i = 0; i < source.length; i++) {
        const lanes = [...packed.subarray(i * 4, i * 4 + 4)];
        assert.deepEqual(lanes, [lanes[0], lanes[0], lanes[0], lanes[0]]);
        const sourceValue = e4m3Decode(source[i]);
        const isDiscarded = sourceValue > 0 && sourceValue * factor < NVFP4_DISCARD_BOUNDARY;
        const restored = restorePackedScale(lanes[0], factor);
        if (isDiscarded) {
          discarded += 1;
          // Flush is permanent: the packed byte is zero and reads back zero.
          assert.equal(lanes[0], 0);
          assert.equal(restored, 0);
        } else {
          // rtol=0, atol=0: retained scales are represented exactly.
          assert.equal(restored, sourceValue);
        }
        if (sourceValue === 0) {
          assert.equal(lanes[0], 0); // zero remains zero
        }
      }
      assert.equal(discarded, discardedCodes);
    });
  }

  it("sits the flush boundary exactly at s*f = 1/64 and encodes bit 7 for nonzero bytes", () => {
    // Code 7 is the largest raw subnormal: s = 7/512 < 1/64, flushed at f=1.
    assert.equal(packedScaleByte(7, 1), 0);
    // Code 8 is the smallest normal: s = 1/64 exactly, retained at f=1.
    assert.equal(e4m3Decode(8), 1 / 64);
    assert.equal(packedScaleByte(8, 1), 0x80);
    assert.equal(restorePackedScale(0x80, 1), 1 / 64);
    // At factor 8, code 1 normalizes to exactly 1/64 and is retained: the
    // boundary is on the NORMALIZED value (doc: 0 discarded codes at f=8).
    assert.equal(packedScaleByte(1, 8), 0x80);
    assert.equal(restorePackedScale(0x80, 8), e4m3Decode(1));
    // The discard is an encoding constraint, not an FP16 range limit:
    // code 4 normalizes to v = 1, exactly FP16-representable (0x3c00), yet it
    // is still flushed because nonzero lifted bytes require bit 7 for the
    // packed BF16 decoder.
    assert.equal(fp16Bits(e4m3Decode(4) * 128), 0x3c00);
    assert.equal(packedScaleByte(4, 1), 0);
    for (let code = 1; code <= E4M3_MAX_FINITE_CODE; code++) {
      const byte = packedScaleByte(code, 1);
      if (byte !== 0) {
        assert.equal(byte & 0x80, 0x80, `code 0x${code.toString(16)} lifted without bit 7`);
      }
    }
  });

  it("lifts retained scales as FP16 bits 14 through 7 of s*f*128", () => {
    // Golden lift bytes from the doc's audited-range codes and 1.0:
    // code -> packed byte at factor 1.
    const goldens: [number, number][] = [
      [0x31, 0xa9], // 0.5625 (audit minimum)
      [0x38, 0xb0], // 1.0
      [0x30, 0xa8], // 0.5
      [0x08, 0x80], // 1/64 (flush boundary)
      [0x7e, 0xf6], // 448 (audit maximum)
    ];
    for (const [code, expected] of goldens) {
      assert.equal(packedScaleByte(code, 1), expected);
      // Definition check: the byte is exactly FP16 bits 14..7 of s*f*128.
      assert.equal(packedScaleByte(code, 1), (fp16Bits(e4m3Decode(code) * 128) >>> 7) & 0xff);
      assert.equal(restorePackedScale(expected, 1), e4m3Decode(code));
    }
    // Factor-lifted golden: 0.5625 at f=512 normalizes to 288 and lifts to
    // FP16 bits of 288*128 = 36864 = 0x7880, byte 0xf1.
    assert.equal(packedScaleByte(0x31, 512), 0xf1);
    assert.equal(restorePackedScale(0xf1, 512), 0.5625);
  });

  it("retains the audited checkpoint range exactly at factor 1", () => {
    // Design doc "Checkpoint audit" (metadata revision dec48abd...): all
    // 19,025,362,944 routed-expert scales were retained, every
    // projection/rank normalization factor was 1, the smallest positive scale
    // was 0.5625 and the largest 448. The doc's "lossless" wording there is
    // scoped to scale bytes of that checkpoint revision; it is not a claim
    // about other checkpoints, other tensor families, or A4 prefill.
    assert.equal(e4m3Decode(AUDITED_MIN_SCALE_CODE), 0.5625);
    assert.equal(e4m3Decode(AUDITED_MAX_SCALE_CODE), 448);
    const audited: number[] = [];
    for (let code = 0; code <= AUDITED_MAX_SCALE_CODE; code++) {
      if (e4m3Decode(code) >= e4m3Decode(AUDITED_MIN_SCALE_CODE)) audited.push(code);
    }
    // Repeat each code across 4 lanes so scale rows cover whole groups of 4.
    const rows = (codes: number[]): number[] => codes.flatMap((code) => [code, code, code, code]);
    const experts = [rows(audited), rows(audited.slice().reverse())];
    assert.equal(computePackedScaleFactor(experts, "bf16"), 1);
    for (const expert of experts) {
      const packed = processPackedScales(expert, 1);
      for (let i = 0; i < expert.length; i += 4) {
        const byte = packedScaleByte(expert[i], 1);
        const sourceValue = e4m3Decode(expert[i]);
        assert.deepEqual([...packed.subarray(i, i + 4)], [byte, byte, byte, byte]);
        assert.notEqual(byte, 0);
        assert.equal(restorePackedScale(byte, 1), sourceValue);
      }
    }
  });

  it("chooses one power-of-two factor per projection across all experts; fp16 fixes f=1", () => {
    // FP16 preparation uses f=1 and the same discard boundary.
    assert.equal(computePackedScaleFactor([[0x7e], [0x08]], "fp16"), 1);
    // BF16 goldens: max 448 -> f=1; max 0.5625 -> f=512; max 8.0 -> f=32.
    assert.equal(computePackedScaleFactor([[0x7e, 0x31]], "bf16"), 1);
    assert.equal(computePackedScaleFactor([[0x31, 0x30]], "bf16"), 512);
    assert.equal(computePackedScaleFactor([[0x50, 0x30]], "bf16"), 32);
    // One factor per projection across ALL experts: expert 1's small scales
    // inherit the factor implied by expert 0's maximum.
    assert.equal(computePackedScaleFactor([[0x50], [0x30]], "bf16"), 32);
    // All-zero projections fall back to f=1.
    assert.equal(computePackedScaleFactor([[0, 0], [0]], "bf16"), 1);
    // The factor is a power of two and keeps every normalized scale <= 448.
    const experts = [[0x31, 0x08, 0x50], [0x30, 0x7e]];
    const factor = computePackedScaleFactor(experts, "bf16");
    assert.ok(factor >= 1 && Number.isInteger(Math.log2(factor)));
    for (const expert of experts) {
      for (const code of expert) {
        assert.ok(e4m3Decode(code) * factor <= 448);
      }
    }
  });

  it("compensates the global scale by 1/f and the activation decode convention", () => {
    // prepare.py: global_scale * 2^(bias-7) / f with
    // bias = 2^(target_exponent-1) - 2^(fp4_exponent-1): fp16 -> 2^7,
    // bf16 -> 2^119.
    assert.equal(processPackedGlobalScale(1, 1, "bf16"), 2 ** 119);
    assert.equal(processPackedGlobalScale(1, 1, "fp16"), 2 ** 7);
    assert.equal(processPackedGlobalScale(3, 512, "bf16"), (3 * 2 ** 119) / 512);
    // Compensation invariant: decoded normalized scale times compensated
    // global scale reproduces source scale times source global scale times
    // the decode convention constant (f cancels exactly).
    const sourceScale = 0.5625;
    const sourceGlobal = 2.5;
    const factor = 512;
    const compensated = processPackedGlobalScale(sourceGlobal, factor, "bf16");
    const restored = restorePackedScale(packedScaleByte(0x31, factor), factor);
    assert.equal(restored, sourceScale);
    assert.equal(restored * factor * compensated, sourceScale * sourceGlobal * 2 ** 119);
    // Without the convention constant, (s*f)*(g/f) = s*g exactly.
    assert.equal(sourceScale * factor * (sourceGlobal / factor), sourceScale * sourceGlobal);
    // Whole-projection preparation wires the pieces together (scale rows
    // cover whole groups of 4 lanes).
    const contract = preparePackedScaleContract(
      [[0x31, 0x08, 0x50, 0x38], [0x50, 0x30, 0x31, 0x30]],
      [2.5, 0.75],
      "bf16",
    );
    assert.equal(contract.scaleFactor, 32);
    assert.equal(contract.compensatedGlobalScales[0], (2.5 * 2 ** 119) / 32);
    assert.equal(contract.packedExperts[1].length, 4);
    expectContractError(() => processPackedGlobalScale(1, 1, "fp8" as never), /unsupported W4A16 activation dtype/);
  });

  it("keeps the group-of-4 lane swizzle a layout-only involution", () => {
    // `_process_nvfp4_packed_scales` swaps lanes 1 and 2 of every group of 4
    // (view(-1, 4)[:, [0, 2, 1, 3]]) before the boundary map; the permutation
    // is storage layout and never changes a boundary outcome.
    const pattern = [1, 2, 3, 4, 5, 6, 7, 8, 0x31, 0x08, 0x7e, 0x38];
    const swizzled = swizzleScaleGroupLanes(pattern);
    assert.deepEqual(swizzled, [1, 3, 2, 4, 5, 7, 6, 8, 0x31, 0x7e, 0x08, 0x38]);
    assert.deepEqual(swizzleScaleGroupLanes(swizzled), pattern);
    // Runs of 4 equal scales are fixed points (the reference boundary
    // harness's repeated layout).
    const repeated = [0x31, 0x31, 0x31, 0x31, 7, 7, 7, 7];
    assert.deepEqual(swizzleScaleGroupLanes(repeated), [...repeated]);
    // Boundary semantics are per-scale: undoing the lane swizzle recovers the
    // per-value normalize/flush/lift map for any lane order.
    const packed = processPackedScales(pattern, 1);
    assert.deepEqual(swizzleScaleGroupLanes([...packed]), pattern.map((code) => packedScaleByte(code, 1)));
  });

  it("makes the flush irreversible through CSF expansion and A4 reconstruction", () => {
    // CSF container-to-buffer scale round-trip is byte-exact ("lossless" in
    // this codebase refers only to this round-trip). The planted runs keep
    // lane-swizzle invariance: [code x4] per group of 4.
    const positions = Array.from({ length: 16 }, (_, i) => i);
    const values = [
      0x07, 0x07, 0x07, 0x07, // flushed subnormal at f=1
      0x00, 0x00, 0x00, 0x00, // source zero
      0x08, 0x08, 0x08, 0x08, // retained flush boundary
      0x7e, 0x7e, 0x7e, 0x7e, // retained maximum
    ];
    const matrix = makeScaleMatrix(32, 8, {
      seed: 5,
      exceptionPositions: positions,
      exceptionValues: values,
    });
    const expanded = expandScalePlanes(matrix.fixed, matrix.exceptions, 32, 8);
    assert.deepEqual([...expanded], [...matrix.logical]);
    const factor = computePackedScaleFactor([[...expanded]], "bf16");
    assert.equal(factor, 1);
    const packed = processPackedScales([...expanded], factor);
    assert.deepEqual([...packed.subarray(0, 4)], [0, 0, 0, 0]); // 0x07 flushed
    assert.deepEqual([...packed.subarray(4, 8)], [0, 0, 0, 0]); // zero stays zero
    assert.deepEqual([...packed.subarray(8, 12)], [0x80, 0x80, 0x80, 0x80]);
    assert.deepEqual([...packed.subarray(12, 16)], [0xf6, 0xf6, 0xf6, 0xf6]);
    // Compressed scale storage preserves the encoded packed bytes exactly,
    // but neither CSF expansion nor hybrid A4 reconstruction restores
    // discarded values: the flushed source subnormal is indistinguishable
    // from a genuine zero in the packed bytes. Hybrid A4 consumes these same
    // packed scales and is not lossless (characterized prefill regression);
    // no reconstruction path can resurrect the discarded subnormal.
    assert.deepEqual([...packed.subarray(0, 4)], [...packed.subarray(4, 8)]);
    assert.equal(restorePackedScale(0, factor), 0);
    // Re-expanding the source planes and re-preparing is deterministic.
    const reexpanded = expandScalePlanes(matrix.fixed, matrix.exceptions, 32, 8);
    assert.deepEqual([...processPackedScales([...reexpanded], factor)], [...packed]);
  });

  it("does not apply the E8M0 K/32 logical-tail guards to E4M3 K/16", () => {
    // NOT APPLICABLE TO E4M3 K/16. The reference E8M0 K/32 implementation
    // guards its microkernel logical tail with
    // `_E8M0_LOGICAL_TAIL_SCALE_N_ALIGNMENT = 64` (`_e8m0_logical_tail_scale_n`)
    // and offers a compact K-tail mode (`allow_k_tail`: K divisible by 8,
    // ceil(K/32) columns). Per the port plan: "Apply the E8M0 tail guard only
    // to an E8M0 K/32 implementation; this checkpoint uses E4M3 K/16.
    // Preserve NVFP4's own alignment/capacity guards." So this checkpoint
    // must NOT round scale-N up to 64 and must NOT accept K-tail geometry:
    // E4M3 K/16 carries one scale byte per 16 K channels with no tail
    // padding.
    assert.equal(E8M0_LOGICAL_TAIL_SCALE_N_ALIGNMENT, 64);
    // 768 K channels -> exactly 48 scale columns; the E8M0 tail guard would
    // round 48 up to a multiple of 64, E4M3 K/16 does not.
    assert.equal(e4m3K16ScaleColumns(768), 48);
    assert.equal(e4m3K16ScaleCapacity(1, 768), 48);
    // K=48 is divisible by 16 but not 32: legal for E4M3 K/16 even though an
    // E8M0 K/32 contract would need its K-tail exception.
    assert.equal(e4m3K16ScaleColumns(48), 3);
    const row = new Array(48).fill(0x38);
    assert.equal(processPackedScales(row, 1).length, 48);
    // The E8M0 K-tail acceptance (K divisible by 8) does not carry over:
    // K=24 passes an E8M0 K-tail check but violates NVFP4's own K%16 guard.
    expectContractError(() => e4m3K16ScaleColumns(24), /K divisible by 16/);
  });

  it("does not apply the E8M0 BF16 247-byte serving clamp to E4M3 K/16", () => {
    // NOT APPLICABLE TO E4M3 K/16. `_E8M0_K32_BF16_MAX_SCALE_BYTE = 247` is a
    // UE8M0-alphabet BF16 serving clamp in the E8M0 K/32 path (the x4t
    // exact-byte mode preserves the full UE8M0 alphabet; the clamp is a
    // distinct serving operation). E4M3 K/16 has no such clamp: its valid
    // alphabet is exactly 0x00..0x7e and valid codes pass through
    // preparation with no value ever clamped toward 247.
    assert.equal(E8M0_K32_BF16_MAX_SCALE_BYTE, 247);
    // 247 is not even a legal E4M3 scale code (bit 7 set: negative).
    expectContractError(() => e4m3Decode(247), /invalid E4M3 scale code/);
    // The maximum valid code survives preparation untouched at f=1.
    const packed = processPackedScales([0x7e, 0x7e, 0x7e, 0x7e], 1);
    assert.deepEqual([...packed], [0xf6, 0xf6, 0xf6, 0xf6]);
    assert.equal(restorePackedScale(packed[0], 1), 448);
    // Every code above the documented flush boundary is byte-exact and none
    // is biased toward the E8M0 clamp value.
    for (let code = 8; code <= E4M3_MAX_FINITE_CODE; code++) {
      const byte = packedScaleByte(code, 1);
      assert.equal(restorePackedScale(byte, 1), e4m3Decode(code));
    }
  });

  it("preserves NVFP4's own alignment and capacity guards", () => {
    // E4M3 K/16 geometry: K divisible by 16, exactly K/16 bytes per row.
    assert.equal(e4m3K16ScaleColumns(16), 1);
    assert.equal(e4m3K16ScaleCapacity(32, 768), 32 * 48);
    for (const badK of [0, -16, 15, 24, 16.5]) {
      expectContractError(() => e4m3K16ScaleColumns(badK), /K divisible by 16/);
    }
    expectContractError(() => e4m3K16ScaleCapacity(0, 16), /positive integer/);
  });

  it("rejects invalid scale codes and malformed preparation inputs", () => {
    // Scale-code validity: finite nonnegative E4M3 only (0x7f is the e4m3fn
    // NaN code; >= 0x80 is negative).
    expectContractError(() => e4m3Decode(0x7f), /invalid E4M3 scale code 0x7f/);
    expectContractError(() => e4m3Decode(0x80), /invalid E4M3 scale code/);
    expectContractError(() => e4m3Decode(0xff), /invalid E4M3 scale code/);
    // Normalization factor must be a power of two >= 1.
    expectContractError(() => processPackedScales([0x38, 0x38, 0x38, 0x38], 1.5), /power of two/);
    expectContractError(() => processPackedScales([0x38, 0x38, 0x38, 0x38], 0.5), /power of two/);
    // Packed scale rows cover whole groups of 4 lanes.
    expectContractError(() => processPackedScales([0x38, 0x38, 0x38], 1), /groups of 4/);
    // One global scale per expert.
    expectContractError(
      () => preparePackedScaleContract([[0x38, 0x38, 0x38, 0x38]], [1, 2], "bf16"),
      /one global scale per expert/,
    );
  });
});