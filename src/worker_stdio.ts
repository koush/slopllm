import { parentPort } from "node:worker_threads";

let installed = false;

export function installWorkerStdioForwarding(): void {
  if (!parentPort || installed) return;
  installed = true;
  const port = parentPort;

  const forward = (stream: "stdout" | "stderr") => {
    return (chunk: string | Uint8Array, encodingOrCallback?: BufferEncoding | (() => void), callback?: () => void): boolean => {
      const encoding = typeof encodingOrCallback === "string" ? encodingOrCallback : undefined;
      const text = typeof chunk === "string" ? chunk : Buffer.from(chunk).toString(encoding);
      port.postMessage({ type: "stdio", stream, text });
      const done = typeof encodingOrCallback === "function" ? encodingOrCallback : callback;
      done?.();
      return true;
    };
  };

  process.stdout.write = forward("stdout") as typeof process.stdout.write;
  process.stderr.write = forward("stderr") as typeof process.stderr.write;
}
