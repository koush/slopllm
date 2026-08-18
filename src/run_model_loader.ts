import http from "node:http";
import path from "node:path";
import { Worker } from "node:worker_threads";
import { freeModelRuntime, loadModelRuntime, modelLabel, parseModelArgs } from "./model_cli";

export interface LoaderArgs {
  controlHost: string;
  controlPort: number;
  sharedArgs: string[];
  initialCommand: WorkerCommand | null;
}

export interface WorkerCommand {
  entry: string;
  args: string[];
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
      entry: path.resolve(argv[entryIndex]),
      args: [...sharedArgs, ...argv.slice(entryIndex + 1)],
    },
  };
}

function sendJson(res: http.ServerResponse, status: number, body: object): void {
  res.writeHead(status, { "content-type": "application/json" });
  res.end(JSON.stringify(body));
}

export function parseWorkerCommand(value: unknown, sharedArgs: string[]): WorkerCommand {
  if (!Array.isArray(value) || value.length === 0 || value.some(arg => typeof arg !== "string")) {
    throw new Error('Expected a JSON string array such as ["src/run_qwen3_unified.ts", "--batch"]');
  }
  const [entry, ...args] = value as string[];
  if (!/\.[cm]?[jt]s$/.test(entry)) {
    throw new Error(`Invalid executor entry point: ${entry}`);
  }
  return { entry: path.resolve(entry), args: [...sharedArgs, ...args] };
}

async function readWorkerCommand(req: http.IncomingMessage, sharedArgs: string[]): Promise<WorkerCommand | null> {
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
  return parseWorkerCommand(value, sharedArgs);
}

async function main(): Promise<void> {
  const loaderArgs = parseLoaderArgs(process.argv.slice(2));
  const modelArgs = parseModelArgs(loaderArgs.initialCommand?.args ?? loaderArgs.sharedArgs);
  if (!modelArgs.arena) throw new Error("run_model_loader requires --arena <GiB>");
  if (modelArgs.useQwen35 || modelArgs.useFp8) {
    throw new Error("The model loader currently supports Qwen3 and GLM-5.1");
  }

  console.log(`Loading ${modelLabel(modelArgs)} for persistent worker execution...`);
  const runtime = await loadModelRuntime(modelArgs);
  for (const device of runtime.gpuDevices) {
    if (device.arenaBase === undefined) throw new Error(`GPU ${device.device} did not create an arena`);
    process.env[`GLM_ARENA_BASE_${device.device}`] = String(device.arenaBase);
  }
  process.env.GLM_SKIP_MMAP_LOAD = "1";
  process.env.GLM_MODEL_LOAD_REPLAY = "1";
  console.log(`Model loaded from ${runtime.modelDir}`);

  let worker: Worker | null = null;
  let lastExitCode: number | null = null;
  let lastError: string | null = null;
  let stopping: Promise<void> | null = null;
  let workerCommand = loaderArgs.initialCommand;

  const startWorker = (follow?: http.ServerResponse): void => {
    if (worker) throw new Error("Executor worker is already running");
    if (!workerCommand) throw new Error("No executor command has been configured");
    lastExitCode = null;
    lastError = null;
    const next = new Worker(workerCommand.entry, {
      argv: workerCommand.args,
      execArgv: ["--require", require.resolve("tsx/cjs")],
      stdout: true,
      stderr: true,
    });
    worker = next;
    next.stdout.pipe(process.stdout, { end: false });
    next.stderr.pipe(process.stderr, { end: false });
    if (follow) {
      let exited = false;
      let stdoutEnded = false;
      let stderrEnded = false;
      const finish = () => {
        if (exited && stdoutEnded && stderrEnded && !follow.writableEnded) follow.end();
      };
      const write = (chunk: Buffer | string) => {
        if (!follow.writableEnded && !follow.destroyed) follow.write(chunk);
      };
      next.stdout.on("data", write);
      next.stderr.on("data", write);
      next.stdout.once("end", () => { stdoutEnded = true; finish(); });
      next.stderr.once("end", () => { stderrEnded = true; finish(); });
      next.once("exit", () => {
        exited = true;
        setImmediate(finish);
      });
    }
    next.once("error", error => {
      lastError = error instanceof Error ? (error.stack ?? error.message) : String(error);
      console.error("Executor worker failed:", error);
    });
    next.once("exit", code => {
      lastExitCode = code;
      if (worker === next) worker = null;
      console.log(`Executor worker exited with code ${code}`);
    });
    console.log(`Started executor worker ${next.threadId}: ${workerCommand.entry} ${workerCommand.args.join(" ")}`);
  };

  const stopWorker = (): Promise<void> => {
    if (!worker) return Promise.resolve();
    if (stopping) return stopping;
    const current = worker;
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
      current.once("exit", finish);
      current.postMessage({ type: "shutdown" });
      const timeout = setTimeout(() => {
        console.warn("Executor did not stop gracefully; terminating worker");
        void current.terminate().then(finish);
      }, 5000);
    });
    return stopping;
  };

  if (workerCommand) startWorker();
  else console.log("No executor command configured; waiting for POST /run");

  const server = http.createServer((req, res) => {
    const url = new URL(req.url ?? "/", `http://${req.headers.host ?? loaderArgs.controlHost}`);
    if (req.method === "GET" && url.pathname === "/status") {
      sendJson(res, 200, {
        state: worker ? "running" : "idle",
        threadId: worker?.threadId ?? null,
        entry: workerCommand?.entry ?? null,
        args: workerCommand?.args ?? [],
        lastExitCode,
        lastError,
      });
      return;
    }
    if (req.method === "POST" && url.pathname === "/run") {
      void readWorkerCommand(req, loaderArgs.sharedArgs).then(command => {
        if (worker) {
          sendJson(res, 409, { error: "Executor worker is already running" });
          return;
        }
        if (command) workerCommand = command;
        if (!workerCommand) {
          sendJson(res, 400, { error: "No executor command was provided" });
          return;
        }
        startWorker();
        sendJson(res, 202, { state: "running", threadId: worker!.threadId });
      }).catch(error => {
        sendJson(res, 400, { error: error instanceof Error ? error.message : String(error) });
      });
      return;
    }
    if (req.method === "POST" && url.pathname === "/stop") {
      void stopWorker().then(() => sendJson(res, 200, { state: "idle" }));
      return;
    }
    if (req.method === "POST" && url.pathname === "/restart") {
      if (!workerCommand) {
        sendJson(res, 400, { error: "No executor command has been configured" });
        return;
      }
      void stopWorker().then(() => {
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
          sendJson(res, 202, { state: "running", threadId: worker!.threadId });
        }
      });
      return;
    }
    sendJson(res, 404, { error: "Not found" });
  });

  server.listen(loaderArgs.controlPort, loaderArgs.controlHost, () => {
    console.log(`Model loader control server: http://${loaderArgs.controlHost}:${loaderArgs.controlPort}`);
  });

  let shuttingDown = false;
  const shutdown = async () => {
    if (shuttingDown) return;
    shuttingDown = true;
    await stopWorker();
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
