import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { describe, it } from "node:test";
import { GLM51_REPO, parseModelArgs, resolveModelSelection } from "../src/model_cli";
import { modelRegistry } from "../src/model_registry";

function fixture() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "model-cli-"));
  return { dir, [Symbol.dispose]() { fs.rmSync(dir, { recursive: true, force: true }); } };
}

describe("checkpoint-based model selection", () => {
  it("selects implementations by top-level config, independent of directory name or precision", () => {
    using f = fixture();
    for (const modelType of Object.keys(modelRegistry)) {
      fs.writeFileSync(path.join(f.dir, "config.json"), JSON.stringify({
        model_type: modelType, text_config: { model_type: "unregistered_text_type" },
        quantization_config: { quant_method: "fp8" },
      }));
      const args = parseModelArgs(["--model-dir", f.dir]);
      const selection = resolveModelSelection(args);
      assert.equal(selection.modelType, modelType);
      assert.equal(selection.modelDir, f.dir);
      assert.equal(selection.repoId, f.dir);
      assert.equal(args.arena, 0);
    }
  });

  it("resolves --model through the HF cache and --model-dir directly", () => {
    using f = fixture();
    const oldHome = process.env.HF_HOME;
    process.env.HF_HOME = f.dir;
    try {
      const cache = path.join(f.dir, "hub", "models--test--checkpoint");
      const snapshot = path.join(cache, "snapshots", "revision");
      fs.mkdirSync(snapshot, { recursive: true });
      fs.mkdirSync(path.join(cache, "refs"));
      fs.writeFileSync(path.join(cache, "refs", "main"), "revision\n");
      fs.writeFileSync(path.join(snapshot, "config.json"), JSON.stringify({ model_type: "qwen3" }));
      const selection = resolveModelSelection(parseModelArgs(["--model", "test/checkpoint"]));
      assert.equal(selection.modelDir, snapshot);
      assert.equal(selection.repoId, "test/checkpoint");
      assert.equal(selection.modelType, "qwen3");
      assert.equal(resolveModelSelection(parseModelArgs(["--model-dir", snapshot])).modelDir, snapshot);
      assert.throws(() => resolveModelSelection(parseModelArgs(["--model-dir", "test/checkpoint"])), /ENOENT/);
    } finally {
      if (oldHome === undefined) delete process.env.HF_HOME;
      else process.env.HF_HOME = oldHome;
    }
  });

  it("rejects missing/unknown types without falling back to text_config", () => {
    using f = fixture();
    for (const modelType of [undefined, "unknown", "toString"]) {
      fs.writeFileSync(path.join(f.dir, "config.json"), JSON.stringify({ model_type: modelType, text_config: { model_type: "qwen3" } }));
      assert.throws(() => resolveModelSelection(parseModelArgs(["--model-dir", f.dir])), /Unsupported model_type.*config.json.*Supported types/);
    }
  });

  it("validates model options after detection", () => {
    using f = fixture();
    fs.writeFileSync(path.join(f.dir, "config.json"), JSON.stringify({ model_type: "qwen3" }));
    for (const option of ["--cp", "--mtp"]) {
      assert.throws(() => resolveModelSelection(parseModelArgs(["--model-dir", f.dir, option])), /does not support/);
    }
    fs.writeFileSync(path.join(f.dir, "config.json"), JSON.stringify({ model_type: "glm_moe_dsa" }));
    assert.equal(resolveModelSelection(parseModelArgs(["--model-dir", f.dir, "--cp", "--mtp", "2"])).modelType, "glm_moe_dsa");
  });

  it("validates selectors and preserves the production default", () => {
    assert.equal(parseModelArgs([]).arena, 92);
    assert.equal(parseModelArgs(["--model", GLM51_REPO]).arena, 92);
    assert.equal(parseModelArgs(["--model", GLM51_REPO, "--arena", "0"]).arena, 0);
    assert.throws(() => parseModelArgs(["--model", "test/checkpoint", "--model-dir", "/tmp/model"]), /mutually exclusive/);
    for (const option of ["--model", "--model-dir"]) {
      assert.throws(() => parseModelArgs([option]), /requires a value/);
      assert.throws(() => parseModelArgs([option, "--cp"]), /requires a value/);
    }
    assert.throws(() => parseModelArgs(["--model", "/tmp/model"]), /repo ID/);
    for (const option of ["--glm51", "--glm51-small", "--glm53-flash", "--qwen3", "--qwen35", "--fp8", "--nvfp4"]) {
      assert.throws(() => parseModelArgs([option]), /removed.*--model/);
    }
  });
});
