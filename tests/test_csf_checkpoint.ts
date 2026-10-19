import assert from "node:assert/strict";
import crypto from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { after, before, describe, it } from "node:test";

import {
  CsfContractError,
  openCheckpointSource,
  resolveCheckpointConfigDir,
  validateCalibrationScalar,
} from "../src/checkpoint_source";
import { reconstructSourceShard, verifyCsfIntegrity, readCsfContract } from "../src/csf_checkpoint";
import { SafeTensorFile } from "../src/safetensors";
import {
  buildCsfFixture,
  readJsonFile,
  sha256,
  writeJsonFile,
  type CsfFixture,
} from "./csf_fixture";

// §9.7.1 CPU fixtures at the container level: exact NVFP4 codec recovery
// through the checkpoint source, invalid-container rejection (malformed CSF
// must fail explicitly, never fall back to safetensors), calibration
// preservation, and the separate integrity/provenance pass.

const PFX = "model.layers.3.mlp.experts.0";

it("reads a bounded row beyond 4 GiB without materializing an embedding table", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "slopllm-large-table-"));
  const size = 8 * 1024 ** 3, rows = size / 256;
  const header = Buffer.from(JSON.stringify({ table: { dtype: "F8_E4M3", shape: [rows, 256], data_offsets: [0, size] } }));
  const prefix = Buffer.alloc(8);
  prefix.writeBigUInt64LE(BigInt(header.length));
  const fd = fs.openSync(path.join(root, "model.safetensors"), "w");
  const expected = Buffer.from(Array.from({ length: 256 }, (_, i) => i));
  try {
    fs.writeSync(fd, prefix);
    fs.writeSync(fd, header);
    fs.writeSync(fd, expected, 0, expected.length, 8 + header.length + size - 256);
  } finally {
    fs.closeSync(fd);
  }
  const source = openCheckpointSource(root);
  const read = SafeTensorFile.prototype.readTensor;
  SafeTensorFile.prototype.readTensor = () => { throw new Error("Unbounded table read"); };
  try {
    assert.deepEqual(source.tensor("table").readSlice({ offsets: [rows - 1, 0], shape: [1, 256] }), expected);
    assert.throws(() => source.tensor("table").readSlice({ offsets: [rows - .5, 0], shape: [0, 256] }), /bounds/);
  } finally {
    SafeTensorFile.prototype.readTensor = read;
    source.close();
    fs.rmSync(root, { recursive: true, force: true });
  }
});

describe("csf checkpoint container", () => {
  let scratch: string;
  let fixture: CsfFixture;

  before(() => {
    scratch = fs.mkdtempSync(path.join(os.tmpdir(), "slopllm-csf-"));
    fixture = buildCsfFixture(path.join(scratch, "container"));
  });

  after(() => {
    fs.rmSync(scratch, { recursive: true, force: true });
  });

  it("opens the container and recovers exact logical tensors", () => {
    const source = openCheckpointSource(fixture.root);
    try {
      assert.equal(source.kind, "csf");
      assert.equal(source.configDir(), path.join(fixture.root, "metadata"));
      const names = source.names();
      // Logical source names only: physical planes never leak into the
      // inventory, replaced scales appear under their source names.
      assert.ok(names.includes(`${PFX}.gate_proj.weight_scale`));
      assert.ok(!names.some(n => n.includes("nvfp4_csf")));
      assert.deepEqual([...names].sort(), [...fixture.logical.keys()].sort());

      for (const [name, tensor] of fixture.logical) {
        const meta = source.meta(name);
        assert.equal(meta.dtype, tensor.dtype, name);
        assert.deepEqual(meta.shape, tensor.shape, name);
        assert.deepEqual([...source.tensor(name).read()], [...tensor.data], name);
      }
    } finally {
      source.close();
    }
  });

  it("serves bounded logical slice reads", () => {
    const source = openCheckpointSource(fixture.root);
    try {
      const weight = source.tensor(`${PFX}.gate_proj.weight`);
      const slice = weight.readSlice({ shape: [8, 16], offsets: [4, 8] });
      const full = weight.read();
      for (let r = 0; r < 8; r++) {
        for (let c = 0; c < 16; c++) {
          assert.equal(slice[r * 16 + c], full[(4 + r) * 64 + (8 + c)]);
        }
      }
      // Compressed scale windows decode from the planes directly.
      const scale = source.tensor(`${PFX}.gate_proj.weight_scale`);
      const window = scale.readSlice({ shape: [16, 4], offsets: [16, 2] });
      const fullScale = scale.read();
      for (let r = 0; r < 16; r++) {
        for (let c = 0; c < 4; c++) {
          assert.equal(window[r * 4 + c], fullScale[(16 + r) * 8 + (2 + c)]);
        }
      }
      assert.throws(() => scale.readSlice({ shape: [16, 4], offsets: [16, 6] }), /out of bounds/);
    } finally {
      source.close();
    }
  });

  it("retains calibration scalars unreciprocated", () => {
    const source = openCheckpointSource(fixture.root);
    try {
      // Asymmetric gate/up calibration values survive byte-exact as F32.
      for (const [proj, expected] of [["gate_proj", 0.03125], ["up_proj", 0.0625], ["down_proj", 0.125]] as const) {
        const name = `${PFX}.${proj}.input_scale`;
        const tensor = source.tensor(name);
        const meta = source.meta(name);
        assert.equal(meta.dtype, "F32");
        assert.deepEqual(meta.shape, []);
        const bytes = tensor.read();
        assert.equal(bytes.length, 4);
        // Retention keeps source values: no F32-to-BF16 conversion and no
        // reciprocal at load time (the reference preparation reciprocates
        // gate/up max and down input scales at A4-preparation time).
        const value = validateCalibrationScalar(name, meta, bytes);
        assert.equal(value, expected);
        assert.equal(bytes.readFloatLE(0), expected);
      }
    } finally {
      source.close();
    }
  });

  it("rejects invalid calibration scalars", () => {
    const scalar = (value: number) => {
      const buf = Buffer.alloc(4);
      buf.writeFloatLE(value);
      return buf;
    };
    const meta = { dtype: "F32", shape: [] as number[], dataOffsets: [0, 4] as [number, number] };
    for (const bad of [0, -1, Number.NaN, Number.POSITIVE_INFINITY]) {
      assert.throws(
        () => validateCalibrationScalar("x.input_scale", meta, scalar(bad)),
        /positive and finite/,
      );
    }
    assert.throws(
      () => validateCalibrationScalar("x.input_scale", { ...meta, shape: [1] }, Buffer.alloc(8)),
      /one FP32 value/,
    );
    assert.throws(
      () => validateCalibrationScalar("x.input_scale", { ...meta, dtype: "BF16" }, Buffer.alloc(2)),
      /must be F32/,
    );
    assert.equal(validateCalibrationScalar("x.input_scale", meta, scalar(0.03125)), 0.03125);
  });

  it("rejects recognized-but-malformed containers without falling back", () => {
    const clone = (name: string): string => {
      const dir = path.join(scratch, name);
      fs.cpSync(fixture.root, dir, { recursive: true });
      return dir;
    };
    const cases: [string, (dir: string) => void, RegExp][] = [
      ["manifest schema", dir => {
        const manifest = readJsonFile(dir, "manifest.json");
        manifest.schema = "lil-mxfp4-csf-checkpoint/1";
        writeJsonFile(dir, "manifest.json", manifest);
      }, /requires schema=lil-nvfp4-csf-checkpoint\/1/],
      ["unsupported codec variant", dir => {
        for (const file of ["manifest.json", "build-contract.json"]) {
          const json = readJsonFile(dir, file);
          json.codec = "row-base-offset1-u24-exceptions/1";
          writeJsonFile(dir, file, json);
        }
      }, /requires codec=byte-window4-fixed-stream-u24-exceptions\/1/],
      ["unsupported family", dir => {
        for (const file of ["manifest.json", "build-contract.json"]) {
          const json = readJsonFile(dir, file);
          json.family = "deepseek_v41";
          writeJsonFile(dir, file, json);
        }
      }, /family deepseek_v41 is not supported/],
      ["inventory disagreement", dir => {
        const contract = readJsonFile(dir, "build-contract.json");
        delete contract.source_names["model.norm.weight"];
        writeJsonFile(dir, "build-contract.json", contract);
      }, /disagree/],
      ["source index disagreement", dir => {
        const index = readJsonFile(dir, "metadata/model.safetensors.index.json");
        index.weight_map["extra.tensor"] = "model-00001-of-00001.safetensors";
        writeJsonFile(dir, "metadata/model.safetensors.index.json", index);
      }, /source index and contract source_names disagree/],
      ["fixed-stream size", dir => {
        const receipt = readJsonFile(dir, "receipts/model-00001-of-00001.safetensors.json");
        receipt.scales[`${PFX}.gate_proj.weight_scale`].fixed_bytes += 1;
        writeJsonFile(dir, "receipts/model-00001-of-00001.safetensors.json", receipt);
      }, /fixed stream size disagrees/],
      ["scale/weight geometry", dir => {
        const receipt = readJsonFile(dir, "receipts/model-00001-of-00001.safetensors.json");
        receipt.scales[`${PFX}.gate_proj.weight_scale`].shape = [32, 16];
        receipt.scales[`${PFX}.gate_proj.weight_scale`].source_bytes = 32 * 16;
        writeJsonFile(dir, "receipts/model-00001-of-00001.safetensors.json", receipt);
      }, /does not match .* geometry|fixed stream size disagrees/],
      ["shard extent", dir => {
        const manifest = readJsonFile(dir, "manifest.json");
        manifest.shards[0].target_file_bytes += 1;
        writeJsonFile(dir, "manifest.json", manifest);
      }, /extent is \d+ bytes/],
      ["missing exception plane", dir => {
        // Physical tensor coverage check: rename the plane away.
        const file = path.join(dir, "tensors/model-00001-of-00001.safetensors");
        const bytes = fs.readFileSync(file);
        const headerLen = Number(bytes.readBigUInt64LE(0));
        const header = Buffer.from(
          bytes.subarray(8, 8 + headerLen).toString("utf-8").replace(".nvfp4_csf_exceptions", ".nvfp4_csf_exceptiozz"),
          "utf-8",
        );
        const prefix = Buffer.alloc(8);
        prefix.writeBigUInt64LE(BigInt(header.length));
        fs.writeFileSync(file, Buffer.concat([prefix, header, bytes.subarray(8 + headerLen)]));
      }, /missing one of its fixed\/exception planes/],
      ["source header disagreement", dir => {
        const receipt = readJsonFile(dir, "receipts/model-00001-of-00001.safetensors.json");
        const raw = Buffer.from(receipt.source_header_base64, "base64");
        const headerLen = Number(raw.readBigUInt64LE(0));
        const header = JSON.parse(raw.subarray(8).toString("utf-8"));
        header["model.norm.weight"].shape = [4];
        const json = Buffer.from(JSON.stringify(header), "utf-8");
        const prefix = Buffer.alloc(8);
        prefix.writeBigUInt64LE(BigInt(json.length));
        receipt.source_header_base64 = Buffer.concat([prefix, json]).toString("base64");
        writeJsonFile(dir, "receipts/model-00001-of-00001.safetensors.json", receipt);
      }, /disagrees with its source header/],
    ];
    for (const [label, mutate, match] of cases) {
      const dir = clone(`bad-${label.replace(/\W+/g, "-")}`);
      mutate(dir);
      // A valid plain-safetensors root must not be reached: the container is
      // recognized and its malformation is an explicit error.
      fs.writeFileSync(path.join(dir, "model.safetensors"), Buffer.from("not a checkpoint"));
      assert.throws(() => openCheckpointSource(dir), (error: Error) => {
        assert.ok(error instanceof CsfContractError, `${label}: expected CsfContractError, got ${error.constructor.name}: ${error.message}`);
        assert.match(error.message, match);
        return true;
      }, label);
    }
  });

  it("rejects malformed planes at decode time", () => {
    const dir = path.join(scratch, "bad-planes");
    fs.cpSync(fixture.root, dir, { recursive: true });
    // Corrupt one exception word in place (unsorted positions).
    const file = path.join(dir, "tensors/model-00001-of-00001.safetensors");
    const bytes = fs.readFileSync(file);
    const headerLen = Number(bytes.readBigUInt64LE(0));
    const header = JSON.parse(bytes.subarray(8, 8 + headerLen).toString("utf-8")) as Record<string, any>;
    const plane = header[`${PFX}.gate_proj.weight_scale.nvfp4_csf_exceptions`];
    const planeOffset = 8 + headerLen + plane.data_offsets[0];
    // Swap two words so the positions become unsorted.
    const first = bytes.subarray(planeOffset, planeOffset + 4);
    const second = bytes.subarray(planeOffset + 4, planeOffset + 8);
    Buffer.from(second).copy(bytes, planeOffset);
    Buffer.from(first).copy(bytes, planeOffset + 4);
    fs.writeFileSync(file, bytes);
    const source = openCheckpointSource(dir);
    try {
      assert.throws(() => source.tensor(`${PFX}.gate_proj.weight_scale`).read(), /sorted and unique/);
    } finally {
      source.close();
    }
  });

  it("resolves config from the container metadata directory", () => {
    assert.equal(resolveCheckpointConfigDir(fixture.root), path.join(fixture.root, "metadata"));
    const plain = path.join(scratch, "plain");
    fs.mkdirSync(plain, { recursive: true });
    fs.writeFileSync(path.join(plain, "config.json"), "{}");
    fs.mkdirSync(path.join(plain, "tensors"));
    fs.writeFileSync(path.join(plain, "lil-manifest.json"), '{"schema":"lil-manifest/1"}');
    assert.equal(resolveCheckpointConfigDir(plain), plain);
    const malformed = path.join(scratch, "malformed-csf");
    fs.mkdirSync(path.join(malformed, "receipts"), { recursive: true });
    assert.throws(() => resolveCheckpointConfigDir(malformed), /metadata\/config.json/);
  });
});

describe("csf integrity and provenance", () => {
  let scratch: string;
  let fixture: CsfFixture;

  before(() => {
    scratch = fs.mkdtempSync(path.join(os.tmpdir(), "slopllm-csf-integrity-"));
    fixture = buildCsfFixture(path.join(scratch, "container"));
  });

  after(() => {
    fs.rmSync(scratch, { recursive: true, force: true });
  });

  it("verifies target and metadata hashes while ignoring recorded claims", async () => {
    const report = await verifyCsfIntegrity(fixture.root);
    assert.equal(report.ok, true, report.problems.join("; "));
    assert.deepEqual(report.checkedShards, ["model-00001-of-00001.safetensors"]);
    // recordedVerification is reported for information only: a passed:true
    // is never treated as verification of the files being opened now.
    assert.equal(report.recordedVerification?.passed, true);
  });

  it("detects corrupted target files even when verification.json passed", async () => {
    const dir = path.join(scratch, "corrupt");
    fs.cpSync(fixture.root, dir, { recursive: true });
    const file = path.join(dir, "tensors/model-00001-of-00001.safetensors");
    const bytes = fs.readFileSync(file);
    bytes[bytes.length - 1] ^= 0xff;
    fs.writeFileSync(file, bytes);
    const report = await verifyCsfIntegrity(dir);
    assert.equal(report.ok, false);
    assert.ok(report.problems.some(p => /target hash/.test(p)), report.problems.join("; "));
    assert.equal(report.recordedVerification?.passed, true);
  });

  it("detects receipt/manifest drift and metadata hash drift", async () => {
    const dir = path.join(scratch, "drift");
    fs.cpSync(fixture.root, dir, { recursive: true });
    const receipt = readJsonFile(dir, "receipts/model-00001-of-00001.safetensors.json");
    receipt.source_sha256 = "0".repeat(64);
    writeJsonFile(dir, "receipts/model-00001-of-00001.safetensors.json", receipt);
    fs.writeFileSync(path.join(dir, "metadata/config.json"), "{}");
    const report = await verifyCsfIntegrity(dir);
    assert.equal(report.ok, false);
    assert.ok(report.problems.some(p => /disagrees with the manifest on source_sha256/.test(p)), report.problems.join("; "));
    assert.ok(report.problems.some(p => /metadata file config.json/.test(p)), report.problems.join("; "));
  });

  it("reconstructs source shards losslessly from the compressed planes", async () => {
    const { contract } = readCsfContract(fixture.root);
    const result = await reconstructSourceShard(fixture.root, contract, "model-00001-of-00001.safetensors");
    const sourceBytes = fixture.sourceShards.get("model-00001-of-00001.safetensors")!;
    assert.equal(result.bytes, sourceBytes.length);
    assert.equal(result.sha256, sha256(sourceBytes));
    assert.equal(result.matchesRecorded, true);
    assert.equal(result.sha256, crypto.createHash("sha256").update(sourceBytes).digest("hex"));
  });
});
