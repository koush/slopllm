import http from "node:http";
import path from "node:path";
import { Worker } from "node:worker_threads";
import { freeModelRuntime, loadModelRuntime, modelLabel, parseModelArgs } from "./model_cli";

interface LoaderArgs {
  controlHost: string;
  controlPort: number;
  entry: string;
  workerArgs: string[];
}

function parseLoaderArgs(argv: string[]): LoaderArgs {
  const entryIndex = argv.findIndex(arg => /\.[cm]?[jt]s$/.test(arg));
  if (entryIndex < 0) {
    throw new Error("Expected an executor entry point, for example src/run_qwen3_unified.ts");
  }

  let controlHost = "127.0.0.1";
  let controlPort = 8099;
  const sharedArgs: string[] = [];
  for (let i = 0; i < entryIndex; i++) {
    const arg = argv[i];
    if (arg === "--control-host" && i + 1 < entryIndex) controlHost = argv[++i];
    else if (arg === "--control-port" && i + 1 < entryIndex) controlPort = parseInt(argv[++i], 10);
    else sharedArgs.push(arg);
  }
  if (!Number.isInteger(controlPort) || controlPort <= 0 || controlPort > 65535) {
    throw new Error(`Invalid control port: ${controlPort}`);
  }

  return {
    controlHost,
    controlPort,
    entry: path.resolve(argv[entryIndex]),
    workerArgs: [...sharedArgs, ...argv.slice(entryIndex + 1)],
  };
}

function sendJson(res: http.ServerResponse, status: number, body: object): void {
  res.writeHead(status, { "content-type": "application/json" });
  res.end(JSON.stringify(body));
}

async function main(): Promise<void> {
  const loaderArgs = parseLoaderArgs(process.argv.slice(2));
  const modelArgs = parseModelArgs(loaderArgs.workerArgs);
  if (!modelArgs.arena) throw new Error("run_model_loader requires --arena <GiB>");
  if (modelArgs.useGlm51 || modelArgs.useQwen35 || modelArgs.useFp8) {
    throw new Error("The initial model-loader implementation supports default Qwen3 only");
  }

  console.log(`Loading ${modelLabel(modelArgs)} for persistent worker execution...`);
  const runtime = await loadModelRuntime(modelArgs);
  for (const device of runtime.gpuDevices) {
    if (device.arenaBase === undefined) throw new Error(`GPU ${device.device} did not create an arena`);
    process.env[`GLM_ARENA_BASE_${device.device}`] = String(device.arenaBase);
  }
  process.env.GLM_SKIP_MMAP_LOAD = "1";
  console.log(`Model loaded from ${runtime.modelDir}`);

  let worker: Worker | null = null;
  let lastExitCode: number | null = null;
  let lastError: string | null = null;
  let stopping: Promise<void> | null = null;

  const startWorker = (): void => {
    if (worker) throw new Error("Executor worker is already running");
    lastExitCode = null;
    lastError = null;
    const next = new Worker(loaderArgs.entry, {
      argv: loaderArgs.workerArgs,
      execArgv: ["--require", require.resolve("tsx/cjs")],
    });
    worker = next;
    next.once("error", error => {
      lastError = error instanceof Error ? (error.stack ?? error.message) : String(error);
      console.error("Executor worker failed:", error);
    });
    next.once("exit", code => {
      lastExitCode = code;
      if (worker === next) worker = null;
      console.log(`Executor worker exited with code ${code}`);
    });
    console.log(`Started executor worker ${next.threadId}: ${loaderArgs.entry} ${loaderArgs.workerArgs.join(" ")}`);
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

  startWorker();

  const server = http.createServer((req, res) => {
    const url = new URL(req.url ?? "/", `http://${req.headers.host ?? loaderArgs.controlHost}`);
    if (req.method === "GET" && url.pathname === "/status") {
      sendJson(res, 200, {
        state: worker ? "running" : "idle",
        threadId: worker?.threadId ?? null,
        entry: loaderArgs.entry,
        args: loaderArgs.workerArgs,
        lastExitCode,
        lastError,
      });
      return;
    }
    if (req.method === "POST" && url.pathname === "/run") {
      if (worker) {
        sendJson(res, 409, { error: "Executor worker is already running" });
      } else {
        startWorker();
        sendJson(res, 202, { state: "running", threadId: worker!.threadId });
      }
      return;
    }
    if (req.method === "POST" && url.pathname === "/stop") {
      void stopWorker().then(() => sendJson(res, 200, { state: "idle" }));
      return;
    }
    if (req.method === "POST" && url.pathname === "/restart") {
      void stopWorker().then(() => {
        startWorker();
        sendJson(res, 202, { state: "running", threadId: worker!.threadId });
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
