import assert from "node:assert/strict";
import path from "node:path";
import { describe, it } from "node:test";
import { parseLoaderArgs, parseWorkerCommand } from "../src/run_model_loader";

describe("model loader arguments", () => {
  it("allows startup without an executor command", () => {
    const args = parseLoaderArgs([
      "--control-host", "0.0.0.0",
      "--control-port", "9000",
      "--gpus", "0,1",
      "--arena", "48",
      "--glm51",
    ]);

    assert.equal(args.controlHost, "0.0.0.0");
    assert.equal(args.controlPort, 9000);
    assert.deepEqual(args.sharedArgs, ["--gpus", "0,1", "--arena", "48", "--glm51"]);
    assert.equal(args.initialCommand, null);
  });

  it("preserves startup executor behavior", () => {
    const args = parseLoaderArgs([
      "--arena", "48",
      "src/run_qwen3_unified.ts",
      "--max-tokens", "32",
    ]);

    assert.deepEqual(args.initialCommand, {
      entry: path.resolve("src/run_qwen3_unified.ts"),
      args: ["--arena", "48", "--max-tokens", "32"],
    });
  });

  it("creates an executor command from a JSON array", () => {
    const command = parseWorkerCommand(
      ["src/run_qwen3_unified.ts", "--max-tokens", "32"],
      ["--arena", "48", "--glm51"],
    );

    assert.deepEqual(command, {
      entry: path.resolve("src/run_qwen3_unified.ts"),
      args: ["--arena", "48", "--glm51", "--max-tokens", "32"],
    });
  });

  it("rejects malformed executor commands", () => {
    assert.throws(() => parseWorkerCommand({}, []), /JSON string array/);
    assert.throws(() => parseWorkerCommand([], []), /JSON string array/);
    assert.throws(() => parseWorkerCommand(["executor", 1], []), /JSON string array/);
    assert.throws(() => parseWorkerCommand(["executor"], []), /Invalid executor entry point/);
  });
});
