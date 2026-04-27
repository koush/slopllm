# Capture Greedy Argmax in CUDA Graph (Revised)

## Key Insight
With `ws.freeze()` after graph capture, workspace allocations are frozen. The argmax result Tensor from the capture step retains its GPU address permanently — no pre-allocation or `argmaxInto()` needed.

## Why This Works
- CUDA graph capture **executes** kernels (not just records them), so the argmax result is available immediately after capture
- `ws.freeze()` prevents any new allocations, so the argmax buffer address is never reused
- `logits.argmax()` is called AFTER `model.forward()` returns (after the tracker disposes), so the argmax result goes into `tracked` with no active tracker — it stays alive
- During replay, `model.forward()` is skipped, so `startTracking()` is never called — the argmax result is never disposed
- `planDecode()` uses only named tensors, so it doesn't allocate from the workspace

## Changes (single file: `src/run_qwen3_unified.ts`)

### Add local variable before the decode loop:
```ts
let greedyArgmaxResult: Tensor | null = null;
```

### Restructure the decode loop:

**Replay path** (graphExec !== null):
```ts
glm.graphLaunch(graphState!.graphExec);
graphSteps++;
// No synchronize for greedy — readInt32LE provides the sync point
if (!sampling && greedyArgmaxResult) {
    currentToken = greedyArgmaxResult.readInt32LE()[0];
} else if (sampling) {
    glm.synchronize();  // need sync before sampling
    using sampleResult = logits!.sampleTokenGPU(sampling, tokenHistory);
    currentToken = sampleResult.readInt32LE()[0];
}
```

**Warmup/Capture path** (else branch):
```ts
if (useGraph && graphState!.warmupRemaining === 0 && !capturing) {
    capturing = true;
    glm.graphBeginCapture();
}

logits = model.forward(state);

// Include greedy argmax in graph capture
const includeArgmaxInGraph = useGraph && !sampling && capturing;
if (includeArgmaxInGraph) {
    greedyArgmaxResult = logits.argmax();
    // Don't dispose — saved for replay, address frozen by ws.freeze()
}

if (capturing) {
    const graph = glm.graphEndCapture();
    ws.freeze();  // already added by user
    graphState!.graphExec = glm.graphInstantiate(graph);
    glm.graphDestroy(graph);
    capturing = false;
}

// Read token
if (includeArgmaxInGraph) {
    // Capture step: read from saved argmax result (kernels executed during capture)
    currentToken = greedyArgmaxResult!.readInt32LE()[0];
} else if (!sampling) {
    // Warmup or no-graph: normal argmax
    using argmaxResult = logits!.argmax();
    currentToken = argmaxResult!.readInt32LE()[0];
} else {
    // Sampling
    using sampleResult = logits!.sampleTokenGPU(sampling, tokenHistory);
    currentToken = sampleResult.readInt32LE()[0];
}
```

## Multi-GPU
- `logits.argmax()` on ParallelTensor handles allGather/allReduce internally
- The returned ParallelTensor (Replicated, shape [1]) is saved as `greedyArgmaxResult`
- `readInt32LE()` reads from shard 0 (device 0) — correct since all shards have the same argmax index
- Stream ordering ensures other devices complete before next `planDecode()` H2D copies

## No Changes Needed To
- `src/tensor.ts` — no `argmaxInto()` needed
- `src/glm_ops.ts` — no changes
- `src/parallel_ops.ts` — no changes

## Expected Impact
- Eliminates separate `argmax()` kernel launch (~1-5μs) and `synchronize()` call per decode step
- For greedy mode: single sync point via `readInt32LE()` instead of `synchronize()` + `argmax()` + `readInt32LE()`
- sampleMs should drop from ~34ms to ~1-5ms (just the d2h copy + sync)
