import http from "node:http";
import path from "node:path";
import util from "node:util";
import { fork, spawn, type ChildProcess } from "node:child_process";
import { freeModelRuntime, loadModelRuntime, modelArenaLayoutSignatures, modelLabel, parseModelArgs, type ModelCliArgs } from "./model_cli";

export interface LoaderArgs {
  controlHost: string;
  controlPort: number;
  sharedArgs: string[];
  initialCommand: WorkerCommand | null;
}

export interface WorkerCommand {
  mode: "fork" | "spawn";
  entry: string;
  args: string[];
  env?: Record<string, string | null>;
}

export function parseLoaderArgs(argv: string[]): LoaderArgs {
  const entryIndex = argv.findIndex(arg => /\.[cm]?[jt]s$/.test(arg));

  let controlHost = "127.0.0.1";
  let controlPort = 8099;
  const sharedArgs: string[] = [];
  const loaderArgsEnd = entryIndex < 0 ? argv.length : entryIndex;
  for (let i = 0; i < loaderArgsEnd; i++) {
    const arg = argv[i];
    if (arg === "--control-host" && i + 1 < loaderArgsEnd) controlHost = argv[++i];
    else if (arg === "--control-port" && i + 1 < loaderArgsEnd) controlPort = parseInt(argv[++i], 10);
    else sharedArgs.push(arg);
  }
  if (!Number.isInteger(controlPort) || controlPort <= 0 || controlPort > 65535) {
    throw new Error(`Invalid control port: ${controlPort}`);
  }

  return {
    controlHost,
    controlPort,
    sharedArgs,
    initialCommand: entryIndex < 0 ? null : {
      mode: "fork",
      entry: path.resolve(argv[entryIndex]),
      args: [...sharedArgs, ...argv.slice(entryIndex + 1)],
    },
  };
}

function sendJson(res: http.ServerResponse, status: number, body: object): void {
  res.writeHead(status, { "content-type": "application/json" });
  res.end(JSON.stringify(body));
}

const CONSOLE_LOG_LIMIT = 128 * 1024;
const consoleLog: Buffer[] = [];
let consoleLogSize = 0;
const consoleFollowers = new Set<http.ServerResponse>();

function recordConsoleOutput(chunk: Buffer): void {
  consoleLog.push(chunk);
  consoleLogSize += chunk.length;
  while (consoleLogSize > CONSOLE_LOG_LIMIT && consoleLog.length > 1) {
    consoleLogSize -= consoleLog[0].length;
    consoleLog.shift();
  }
  for (const follower of consoleFollowers) {
    if (!follower.writableEnded && !follower.destroyed) follower.write(chunk);
  }
}

function mirrorConsole(): void {
  const patch = (orig: (...args: any[]) => void) => (...args: any[]) => {
    recordConsoleOutput(Buffer.from(`${util.format(...args)}\n`));
    orig(...args);
  };
  console.log = patch(console.log);
  console.warn = patch(console.warn);
  console.error = patch(console.error);
}

export function parseWorkerCommand(value: unknown, sharedArgs: string[], previous?: WorkerCommand | null, mode: WorkerCommand["mode"] = previous?.mode ?? "fork"): WorkerCommand {
  if (previous && previous.mode !== mode) previous = null;
  if (value && typeof value === "object" && !Array.isArray(value)) {
    const options = value as { command?: unknown; env?: unknown };
    if (options.command === undefined && options.env === undefined) {
      throw new Error("Expected a JSON string array or an object with command and/or env");
    }
    const command = options.command === undefined ? previous : parseWorkerCommand(options.command, sharedArgs, null, mode);
    if (!command) throw new Error("No executor command has been configured");
    if (options.env === undefined) return command;
    if (!options.env || typeof options.env !== "object" || Array.isArray(options.env)) {
      throw new Error("env must be an object of strings or null (to unset a variable)");
    }
    for (const [name, value] of Object.entries(options.env)) {
      if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(name) || (value !== null && (typeof value !== "string" || value.includes("\0")))) {
        throw new Error(`Invalid executor environment variable: ${name}`);
      }
      if (/^GLM_(ARENA_IPC_HANDLE_|ARENA_LAYOUT_|MODEL_LAYOUT_)/.test(name) || ["GLM_SKIP_MMAP_LOAD", "GLM_MODEL_LOAD_REPLAY"].includes(name)) {
        throw new Error(`Cannot override loader-managed environment variable: ${name}`);
      }
    }
    return { ...command, env: { ...options.env } as Record<string, string | null> };
  }
  if (!Array.isArray(value) || value.length === 0 || value.some(arg => typeof arg !== "string" || arg.includes("\0"))) {
    throw new Error('Expected a JSON string array such as ["src/run_qwen3_unified.ts", "--batch"]');
  }
  const [entry, ...args] = value as string[];
  if (!entry) throw new Error("Executor entry point cannot be empty");
  if (mode === "spawn") return { mode, entry, args };
  if (!/\.[cm]?[jt]s$/.test(entry)) {
    throw new Error(`Invalid executor entry point: ${entry}`);
  }
  return { mode, entry: path.resolve(entry), args: [...sharedArgs, ...args] };
}

export function validateWorkerModelArgs(commandArgs: string[], expected: ModelCliArgs): void {
  const actual = parseModelArgs(commandArgs);
  const { cp: _actualCp, ...actualLoadArgs } = actual;
  const { cp: _expectedCp, ...expectedLoadArgs } = expected;
  if (JSON.stringify(actualLoadArgs) !== JSON.stringify(expectedLoadArgs)) {
    throw new Error("Executor arguments cannot override the loader's model, GPU, parallelism, or arena configuration");
  }
}

async function readWorkerCommand(req: http.IncomingMessage): Promise<unknown> {
  const chunks: Buffer[] = [];
  let length = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
    length += buffer.length;
    if (length > 64 * 1024) throw new Error("Request body is too large");
    chunks.push(buffer);
  }
  if (length === 0) return null;

  let value: unknown;
  try {
    value = JSON.parse(Buffer.concat(chunks).toString("utf8"));
  } catch {
    throw new Error("Request body must be valid JSON");
  }
  return value;
}

export function workerEnvironment(command: WorkerCommand, inherited: NodeJS.ProcessEnv): NodeJS.ProcessEnv {
  const env = { ...inherited };
  for (const [name, value] of Object.entries(command.env ?? {})) {
    if (value === null) delete env[name];
    else env[name] = value;
  }
  return env;
}

async function main(): Promise<void> {
  mirrorConsole();
  const loaderArgs = parseLoaderArgs(process.argv.slice(2));
  const modelArgs = parseModelArgs(loaderArgs.sharedArgs);
  if (loaderArgs.initialCommand) validateWorkerModelArgs(loaderArgs.initialCommand.args, modelArgs);
  if (!modelArgs.arena) throw new Error("run_model_loader requires --arena <GiB>");
  if (modelArgs.useQwen35 || modelArgs.useFp8) {
    throw new Error("The model loader currently supports Qwen3 and GLM-5.1");
  }

  console.log(`Loading ${modelLabel(modelArgs)} for persistent worker execution...`);
  const runtime = await loadModelRuntime(modelArgs);
  const modelLayouts = modelArenaLayoutSignatures(runtime.model, runtime.gpuDevices);
  for (const device of runtime.gpuDevices) {
    if (device.arenaBase === undefined) throw new Error(`GPU ${device.device} did not create an arena`);
    process.env[`GLM_ARENA_IPC_HANDLE_${device.device}`] = device.exportArenaIpcHandle().toString("base64");
    process.env[`GLM_ARENA_LAYOUT_${device.device}`] = device.arenaLayoutSignature();
    process.env[`GLM_MODEL_LAYOUT_${device.device}`] = modelLayouts.get(device.device);
  }
  process.env.GLM_SKIP_MMAP_LOAD = "1";
  process.env.GLM_MODEL_LOAD_REPLAY = "1";
  console.log(`Model loaded from ${runtime.modelDir}`);

  let worker: ChildProcess | null = null;
  let lastExitCode: number | null = null;
  let lastSignal: NodeJS.Signals | null = null;
  let lastError: string | null = null;
  let stopping: Promise<void> | null = null;
  let workerCommand = loaderArgs.initialCommand;
  let shuttingDown = false;
  let lifecycle: Promise<void> = Promise.resolve();
  const spawnedProcesses = new WeakSet<ChildProcess>();

  const serializeLifecycle = (operation: () => void | Promise<void>): Promise<void> => {
    const next = lifecycle.then(operation, operation);
    lifecycle = next.catch(() => {});
    return next;
  };

  const startWorker = (follow?: http.ServerResponse): void => {
    if (shuttingDown) throw new Error("Model loader is shutting down");
    if (worker) throw new Error("Executor worker is already running");
    if (!workerCommand) throw new Error("No executor command has been configured");
    if (workerCommand.mode === "fork") validateWorkerModelArgs(workerCommand.args, modelArgs);
    lastExitCode = null;
    lastSignal = null;
    lastError = null;
    const next = workerCommand.mode === "spawn" ? spawn(workerCommand.entry, workerCommand.args, {
      env: workerEnvironment(workerCommand, process.env),
      detached: true,
      stdio: ["inherit", "pipe", "pipe"],
    }) : fork(workerCommand.entry, workerCommand.args, {
      env: workerEnvironment(workerCommand, process.env),
      execArgv: ["--require", require.resolve("tsx/cjs")],
      stdio: ["inherit", "pipe", "pipe", "ipc"],
    });
    worker = next;
    next.once("spawn", () => spawnedProcesses.add(next));
    next.stdout!.pipe(process.stdout, { end: false });
    next.stderr!.pipe(process.stderr, { end: false });
    next.stdout!.on("data", recordConsoleOutput);
    next.stderr!.on("data", recordConsoleOutput);
    if (follow) {
      consoleFollowers.add(follow);
      follow.once("close", () => consoleFollowers.delete(follow));
    }
    let exited = false;
    let stdoutEnded = false;
    let stderrEnded = false;
    const finishOutput = () => {
      if (!exited || !stdoutEnded || !stderrEnded) return;
      const followers = [...consoleFollowers];
      consoleFollowers.clear();
      for (const follower of followers) {
        if (!follower.writableEnded && !follower.destroyed) follower.end();
      }
    };
    next.stdout!.once("end", () => { stdoutEnded = true; finishOutput(); });
    next.stderr!.once("end", () => { stderrEnded = true; finishOutput(); });
    next.once("exit", () => { exited = true; setImmediate(finishOutput); });
    next.once("error", () => { exited = true; setImmediate(finishOutput); });
    next.once("error", error => {
      lastError = error instanceof Error ? (error.stack ?? error.message) : String(error);
      if (!spawnedProcesses.has(next) && worker === next) worker = null;
      console.error("Executor process failed:", error);
    });
    next.once("close", (code, signal) => {
      lastExitCode = code;
      lastSignal = signal;
      if (worker === next) worker = null;
      console.log(`Executor process exited with ${signal ? `signal ${signal}` : `code ${code}`}`);
    });
    console.log(`Started executor process ${next.pid}: ${workerCommand.entry} ${workerCommand.args.join(" ")}`);
  };

  const stopWorker = (): Promise<void> => {
    if (!worker) return Promise.resolve();
    if (stopping) return stopping;
    const current = worker;
    const isSpawn = workerCommand?.mode === "spawn";
    const signal = (name: NodeJS.Signals) => {
      if (isSpawn && current.pid) {
        try { process.kill(-current.pid, name); }
        catch (error) { if ((error as NodeJS.ErrnoException).code !== "ESRCH") throw error; }
      } else current.kill(name);
    };
    stopping = new Promise<void>(resolve => {
      let settled = false;
      const finish = () => {
        if (settled) return;
        settled = true;
        clearTimeout(timeout);
        if (worker === current) worker = null;
        stopping = null;
        resolve();
      };
      const timeout = setTimeout(() => {
        console.warn("Executor did not stop gracefully; killing process");
        signal("SIGKILL");
      }, isSpawn ? 30000 : 5000);
      current.once("close", finish);
      current.once("error", () => {
        if (!spawnedProcesses.has(current)) finish();
      });
      if (isSpawn) signal("SIGTERM");
      else if (current.connected) {
        current.send({ type: "shutdown" }, error => {
          if (error && current.exitCode === null && !current.killed) {
            console.warn("Failed to request graceful executor shutdown:", error);
          }
        });
      }
    });
    return stopping;
  };

  if (workerCommand) startWorker();
  else console.log("No executor command configured; waiting for POST /fork or /spawn");

  const server = http.createServer((req, res) => {
    const url = new URL(req.url ?? "/", `http://${req.headers.host ?? loaderArgs.controlHost}`);
    if (req.method === "GET" && url.pathname === "/status") {
      sendJson(res, 200, {
        state: worker ? "running" : "idle",
        pid: worker?.pid ?? null,
        mode: workerCommand?.mode ?? null,
        entry: workerCommand?.entry ?? null,
        args: workerCommand?.args ?? [],
        env: workerCommand?.env ?? {},
        lastExitCode,
        lastSignal,
        lastError,
      });
      return;
    }
    if (req.method === "GET" && url.pathname === "/model-args") {
      sendJson(res, 200, { args: loaderArgs.sharedArgs });
      return;
    }
    if (req.method === "GET" && url.pathname === "/follow") {
      res.writeHead(200, {
        "content-type": "text/plain; charset=utf-8",
        "cache-control": "no-cache",
        "x-accel-buffering": "no",
      });
      res.socket?.setNoDelay(true);
      res.flushHeaders();
      if (consoleLog.length > 0) res.write(Buffer.concat(consoleLog));
      consoleFollowers.add(res);
      res.once("close", () => consoleFollowers.delete(res));
      return;
    }
    if (req.method === "POST" && (url.pathname === "/fork" || url.pathname === "/spawn")) {
      void readWorkerCommand(req).then(value => serializeLifecycle(() => {
        if (worker) throw new Error("Executor worker is already running");
        const mode = url.pathname === "/fork" ? "fork" : "spawn";
        const command = value === null ? workerCommand : parseWorkerCommand(value, loaderArgs.sharedArgs, workerCommand, mode);
        if (command && command.mode !== mode) throw new Error(`No ${mode} command has been configured; provide a command`);
        if (command?.mode === "fork") validateWorkerModelArgs(command.args, modelArgs);
        workerCommand = command;
        if (!workerCommand) throw new Error("No executor command was provided");
        if (url.searchParams.has("follow")) {
          res.writeHead(200, {
            "content-type": "text/plain; charset=utf-8",
            "cache-control": "no-cache",
            "x-accel-buffering": "no",
          });
          res.socket?.setNoDelay(true);
          res.flushHeaders();
          startWorker(res);
        } else {
          startWorker();
          sendJson(res, 202, { state: "running", pid: worker!.pid });
        }
      })).catch(error => {
        if (!res.headersSent) sendJson(res, 400, { error: error instanceof Error ? error.message : String(error) });
        else if (!res.writableEnded) res.end(`Executor failed: ${error instanceof Error ? error.message : String(error)}\n`);
      });
      return;
    }
    if (req.method === "POST" && url.pathname === "/stop") {
      void serializeLifecycle(() => stopWorker())
        .then(() => sendJson(res, 200, { state: "idle" }))
        .catch(error => sendJson(res, 500, { error: error instanceof Error ? error.message : String(error) }));
      return;
    }
    if (req.method === "POST" && url.pathname === "/restart") {
      if (!workerCommand) {
        sendJson(res, 400, { error: "No executor command has been configured" });
        return;
      }
      void readWorkerCommand(req).then(value => serializeLifecycle(async () => {
        const command = value === null ? workerCommand : parseWorkerCommand(value, loaderArgs.sharedArgs, workerCommand);
        if (!command) throw new Error("No executor command has been configured");
        if (command.mode === "fork") validateWorkerModelArgs(command.args, modelArgs);
        await stopWorker();
        workerCommand = command;
        if (url.searchParams.has("follow")) {
          res.writeHead(200, {
            "content-type": "text/plain; charset=utf-8",
            "cache-control": "no-cache",
            "x-accel-buffering": "no",
          });
          res.socket?.setNoDelay(true);
          res.flushHeaders();
          startWorker(res);
        } else {
          startWorker();
          sendJson(res, 202, { state: "running", pid: worker!.pid });
        }
      })).catch(error => {
        if (!res.headersSent) sendJson(res, 500, { error: error instanceof Error ? error.message : String(error) });
        else if (!res.writableEnded) res.end(`Executor failed: ${error instanceof Error ? error.message : String(error)}\n`);
      });
      return;
    }
    sendJson(res, 404, { error: "Not found" });
  });

  server.listen(loaderArgs.controlPort, loaderArgs.controlHost, () => {
    console.log(`Model loader control server: http://${loaderArgs.controlHost}:${loaderArgs.controlPort}`);
  });

  const shutdown = async () => {
    if (shuttingDown) return;
    shuttingDown = true;
    await serializeLifecycle(() => stopWorker());
    await new Promise<void>(resolve => server.close(() => resolve()));
    freeModelRuntime(runtime);
  };
  process.once("SIGINT", () => void shutdown().then(() => process.exit(0)));
  process.once("SIGTERM", () => void shutdown().then(() => process.exit(0)));
}

if (require.main === module) {
  main().catch(error => {
    console.error("Model loader failed:", error);
    process.exit(1);
  });
}
