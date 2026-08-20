# Persistent Model Loader

`src/run_model_loader.ts` loads the model and GPU arena once, then starts executor workers against that resident model. Stopping or restarting a worker does not reload the weights. Stopping the loader process releases the model runtime.

The loader currently supports Qwen3 and GLM-5.1. It requires `--arena <GiB>` and does not support Qwen3.5 or FP8.

## Start the Loader

Start the loader and its initial executor in one command:

```bash
npx tsx src/run_model_loader.ts \
  --arena 92 --gpus 0,1,2,3,4,5,6,7 --cp --glm51 --mtp \
  src/openai-server.ts --host 0.0.0.0 --port 8010
```

Arguments before the executor path are shared model arguments and are passed to every worker. Arguments after the path apply only to that executor. Loader options default to `--control-host 127.0.0.1 --control-port 8099` and must appear before the executor path.

The model is ready when the control endpoint responds and the worker reports its own service as ready:

```bash
curl http://127.0.0.1:8099/status
curl http://127.0.0.1:8010/health
```

## Control the Worker

```bash
# Inspect the current command and worker state.
curl http://127.0.0.1:8099/status

# Restart the configured worker without reloading the model.
curl -X POST http://127.0.0.1:8099/restart

# Stop only the worker. The model remains resident on the GPUs.
curl -X POST http://127.0.0.1:8099/stop

# Start the last configured worker again.
curl -X POST http://127.0.0.1:8099/run
```

To replace the executor or its worker-specific arguments, stop the current worker and provide a JSON command array:

```bash
curl -X POST http://127.0.0.1:8099/stop
curl -X POST http://127.0.0.1:8099/run \
  -H 'content-type: application/json' \
  -d '["src/openai-server.ts", "--host", "0.0.0.0", "--port", "8010"]'
```

Add `?follow` to `/run` or `/restart` to stream worker output until that worker exits:

```bash
curl -N -X POST 'http://127.0.0.1:8099/restart?follow'
```

Changing model/shared arguments requires restarting the loader itself. The control server has no authentication, so keep it bound to `127.0.0.1` unless it is protected by other means.
