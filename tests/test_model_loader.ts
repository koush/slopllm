import assert from "node:assert/strict";
import path from "node:path";
import { describe, it } from "node:test";
import { parseLoaderArgs, parseWorkerCommand, validateWorkerModelArgs, workerEnvironment } from "../src/run_model_loader";
import { parseModelArgs } from "../src/model_cli";

describe("model loader arguments", () => {
  it("loads a standalone DFlash2 workspace and rejects target-model flags", () => {
    const args = parseModelArgs(["--dflash2", "--arena", "8", "--gpus", "0,1"]);
    assert.equal(args.useDflash2, true);
    assert.equal(args.useGlm51, false);
    assert.equal(parseModelArgs(["--dflash2"]).arena, 0);
    for (const flag of ["--glm51", "--qwen3", "--qwen35", "--cp", "--mtp", "--fp8", "--nvfp4"]) {
      assert.throws(() => parseModelArgs(["--dflash2", flag]));
      assert.throws(() => parseModelArgs([flag, "--dflash2"]));
    }
    validateWorkerModelArgs(["--dflash2", "--arena", "8", "--gpus", "0,1", "--layerwise"], args);
    assert.throws(() => validateWorkerModelArgs(["--qwen3", "--arena", "8", "--gpus", "0,1"], args), /cannot override/);
  });

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
      mode: "fork",
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
      mode: "fork",
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

  it("preserves arbitrary spawn commands without injecting model arguments", () => {
    const args = ["profile", "--output=/tmp/a b", "node", "--require", "tsx/cjs", "src/run_glm51_multiple_mtp.ts", "--arena", "92"];
    const command = parseWorkerCommand(["nsys", ...args], ["--arena", "48"], null, "spawn");
    assert.deepEqual(command, { mode: "spawn", entry: "nsys", args });
    const updated = parseWorkerCommand({ env: { FLAG: "1" } }, [], command);
    assert.equal(updated.mode, "spawn");
    assert.deepEqual(updated.args, args);
    const replaced = parseWorkerCommand(["/bin/true"], ["--arena", "48"], updated);
    assert.deepEqual(replaced, { mode: "spawn", entry: "/bin/true", args: [] });
    assert.throws(() => parseWorkerCommand({ env: {} }, [], command, "fork"), /No executor command/);
    for (const value of [[""], ["nsys\0"], ["nsys", "bad\0arg"]]) {
      assert.throws(() => parseWorkerCommand(value, [], null, "spawn"));
    }
    assert.throws(() => parseWorkerCommand({ command: ["nsys"], env: { SLOPLLM_SKIP_MMAP_LOAD: null } }, [], null, "spawn"), /loader-managed/);
  });

  it("supports executor-only environment overrides and unsetting inherited variables", () => {
    const command = parseWorkerCommand({ command: ["src/openai-server.ts"], env: { SLOPLLM_CHAT_DEBUG: "0", REMOVE_ME: null } }, ["--arena", "48"]);
    const inherited = { SLOPLLM_CHAT_DEBUG: "1", REMOVE_ME: "yes", KEEP_ME: "yes" };
    assert.deepEqual(workerEnvironment(command, inherited), { SLOPLLM_CHAT_DEBUG: "0", KEEP_ME: "yes" });
    assert.equal(inherited.SLOPLLM_CHAT_DEBUG, "1");
    const updated = parseWorkerCommand({ env: { NEW_FLAG: "1" } }, [], command);
    assert.equal(updated.entry, command.entry);
    assert.deepEqual(updated.args, command.args);
    assert.deepEqual(updated.env, { NEW_FLAG: "1" });
    assert.deepEqual(parseWorkerCommand({ env: {} }, [], updated).env, {});
  });

  it("rejects invalid or loader-managed environment overrides", () => {
    for (const env of [[], null, { FLAG: true }, { FLAG: 1 }, { "BAD=NAME": "x" }, { FLAG: "x\0y" },
      { SLOPLLM_SKIP_MMAP_LOAD: "0" }, { SLOPLLM_ARENA_IPC_HANDLE_0: null }, { SLOPLLM_MODEL_LAYOUT_0: "x" }]) {
      assert.throws(() => parseWorkerCommand({ command: ["src/openai-server.ts"], env }, []));
    }
    assert.throws(() => parseWorkerCommand({ env: {} }, []), /No executor command/);
  });

  it("rejects executor overrides of the resident model layout", () => {
    const expected = parseModelArgs(["--arena", "48", "--gpus", "0,1", "--glm51", "--cp"]);
    validateWorkerModelArgs(["--arena", "48", "--gpus", "0,1", "--glm51", "--cp", "--port", "8000"], expected);
    validateWorkerModelArgs(["--arena", "48", "--gpus", "0,1", "--glm51", "--port", "8000"], expected);
    assert.throws(
      () => validateWorkerModelArgs(["--arena", "48", "--gpus", "2,3", "--glm51", "--cp"], expected),
      /cannot override/,
    );
    assert.throws(
      () => validateWorkerModelArgs(["--arena", "64", "--gpus", "0,1", "--glm51", "--cp"], expected),
      /cannot override/,
    );
  });
});
