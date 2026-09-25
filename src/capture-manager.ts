import { Tensor } from "./tensor";
import { type DeviceOps } from "./device_ops";
import type { WorkspaceBase } from "./workspace";
import { mapTensors, type TensorTree } from "./tensor-tree";
import type { ExecutionManager, ExecutionOptions, ExecutionResult } from "./execution-manager";
import type { ExecutionState } from "./execution-workspace";


interface Captured {
    diagnosticBindings?: Record<string, string>;
    warmupSteps: number;
    graphExec: number | null;
    result: TensorTree;
    inputs: { [name: string]: Tensor, };
    capturedWorkspaces: Set<WorkspaceBase>;
}

/**
 * Which power-of-2 "padded" dims a given base graph actually sizes its buffers
 * by. Learned lazily: a graph is length-invariant by default and only becomes
 * variant once it calls getGraphVariantPadded*() during a run (see
 * ExecutionState). Keying is then stable across KV-length buckets for invariant
 * graphs (capture once, replay always) and per-bucket for variant graphs.
 */
interface LengthVariant {
    kvLen: boolean;
}

export class CaptureManager implements Disposable, ExecutionManager {
    static capturing?: Captured;
    disabled = false;
    captured = new Map<string, Captured>();

    // baseKey (caller key params + batchSize, WITHOUT padded dims) -> learned variance.
    private lengthVariant = new Map<string, LengthVariant>();

    constructor(public ops: DeviceOps) {
    }

    get captureEnabled(): boolean {
        return !this.disabled;
    }

    execute<T, I extends Record<string, Tensor>>(
        options: ExecutionOptions<I>, fn: (inputs: I) => T,
    ): ExecutionResult<T> {
        const warmup = this.captureEnabled && !!options.key?.length && !this.isStateCaptured(options);
        const result = this.runStates(options, (_capturing, retained) => fn(retained));
        return { warmup, result };
    }

    // Learn length dependence lazily; invariant graphs share one key across KV
    // lengths. Variant multi-state graphs include each state's own length bucket.
    private stateKeys(states: readonly ExecutionState[], key: readonly (string | number)[]) {
        const params = [...key];
        for (const state of states) params.push(`batchSize:${state.batchSize}`, `totalTokens:${state.totalTokens}`,
            ...this.ops.getCaptureKeys(state));
        const base = params.join(",");
        if (this.getLengthVariant(base).kvLen) {
            for (const state of states) params.push(`paddedKvLen:${state.paddedKvLen}`);
        }
        return { base, params };
    }

    isStateCaptured<I extends Record<string, Tensor>>(options: ExecutionOptions<I>): boolean {
        return !!options.key?.length && this.isCaptured(this.stateKeys(options.states, options.key).params, options.inputs);
    }

    /** Legacy capture callbacks also receive whether this invocation records a graph. */
    runStates<T, I extends Record<string, Tensor>>(
        { states, inputs, key }: ExecutionOptions<I>, fn: (capturing: boolean, inputs: I) => T,
    ): T {
        if (!key?.length) return fn(false, inputs);
        const { base, params } = this.stateKeys(states, key);
        const bindings: Record<string, string> | undefined = process.env.GLM_GRAPH_DIAGNOSTICS === "1" ? {} : undefined;
        if (bindings) {
            const record = (name: string, tensor: Tensor) => {
                bindings[name] = `${tensor.name ?? "unnamed"}:${tensor.type}[${tensor.shape}]:${JSON.stringify(tensor.memoryRanges())}`;
            };
            for (const [index, state] of states.entries()) {
                for (const [name, value] of Object.entries(state)) {
                    if (value instanceof Tensor && name !== "input") record(`state${index}.${name}`, value);
                }
                for (const [name, value] of Object.entries(state.customMask ?? {})) {
                    if (value instanceof Tensor) record(`state${index}.mask.${name}`, value);
                }
                for (const [name, value] of state.cache.getPagedKV().tensors) record(`state${index}.cache.${name}`, value);
            }
        }
        return this.run(inputs, (capturing, retained) => {
            const result = fn(capturing, retained);
            this.recordLengthVariant(base, states.some(state => !state.paddedKvLenInvariant));
            return result as TensorTree;
        }, params, bindings) as T;
    }

    static trackWorkspaceAlloc(workspace: WorkspaceBase) {
        const { capturing } = CaptureManager;
        if (!capturing)
            return;
        // if (workspace instanceof PagedKVCache) {
        //     throw new Error("Cannot capture a PagedKVCache workspace");
        // }
        const { capturedWorkspaces } = capturing;
        if (capturedWorkspaces.has(workspace)) {
            return;
        }
        if (workspace.tracked.size) {
            throw new Error("Cannot capture a workspace that has tracked tensors: " + workspace.tracked.size);
        }
        capturedWorkspaces.add(workspace);
    }

    [Symbol.dispose]() {
        for (const captured of this.captured.values()) {
            if (captured.graphExec !== null) {
                this.ops.graphExecDestroy(captured.graphExec);
            }
        }
        this.captured.clear();
    }

    /** Learned variance for a base graph; defaults to length-invariant. */
    getLengthVariant(baseKey: string): LengthVariant {
        return this.lengthVariant.get(baseKey) ?? { kvLen: false };
    }

    /** Monotonically record that a base graph sizes buffers by a padded dim. */
    recordLengthVariant(baseKey: string, kvLen: boolean): void {
        if (!kvLen) return;
        const cur = this.lengthVariant.get(baseKey);
        if (!cur) {
            this.lengthVariant.set(baseKey, { kvLen });
        } else {
            cur.kvLen ||= kvLen;
        }
    }

    private graphKey(keyParams: any[], inputs: { [name: string]: Tensor }): string {
        const signature = Object.keys(inputs).sort().filter(name => !!inputs[name]).map(name => {
            const tensor = inputs[name];
            return [name, tensor.shape, tensor.type, tensor.pinned, tensor.parallelism, tensor.memoryRanges()];
        });
        return keyParams.join(",") + (signature.length ? `,inputs:${JSON.stringify(signature)}` : "");
    }

    run<I extends { [name: string]: Tensor }>(inputs: I, fn: (capturing: boolean, capturedInputs: I) => TensorTree, keyParams?: any[], diagnosticBindings?: Record<string, string>): TensorTree {
        if (CaptureManager.capturing) {
            throw new Error("Cannot run a capture while another capture is in progress");
        }

        let captured: Captured | undefined;
        let capturing = false;

        if (!this.disabled && keyParams?.length) {
            const key = this.graphKey(keyParams, inputs);
            captured = this.captured.get(key);

            if (captured) {
                if (captured.graphExec !== null) {
                    if (diagnosticBindings && captured.diagnosticBindings) {
                        for (const name of new Set([...Object.keys(diagnosticBindings), ...Object.keys(captured.diagnosticBindings)])) {
                            if (diagnosticBindings[name] !== captured.diagnosticBindings[name]) {
                                console.error(`[cuda-graph] BINDING MISMATCH key=${key} graphExec=${captured.graphExec} name=${name} captured=${captured.diagnosticBindings[name]} current=${diagnosticBindings[name]}`);
                            }
                        }
                    }
                    if (process.env.GLM_GRAPH_DIAGNOSTICS === "1") {
                        console.warn(`[cuda-graph] checkpoint before-replay key=${key} graphExec=${captured.graphExec}`);
                        this.ops.synchronize();
                    }
                    for (const [name, input] of Object.entries(inputs)) {
                        if (!input)
                            continue;
                        input.stage();
                        const capturedInput = captured.inputs[name];
                        if (!capturedInput.same(input)) {
                            throw new Error(`[cuda-graph] pointer-keyed replay input mismatch key=${key} name=${name}`);
                        }
                    }
                    for (const ws of captured.capturedWorkspaces) {
                        if (ws.tracked.size) {
                            throw new Error("Cannot replay a capture with exported or tracked tensors in a captured workspace");
                        }
                    }
                    for (const [name, input] of Object.entries(inputs)) {
                        if (!input)
                            continue;
                        input.unstage();
                    }
                    try {
                        this.ops.graphLaunch(captured.graphExec);
                    } catch (error) {
                        console.error(`[cuda-graph] graphLaunch failed synchronously key=${key} graphExec=${captured.graphExec}`, error);
                        throw error;
                    }
                    // Replay skips the host heap migrations performed while
                    // recording barriers. Reconcile before claiming outputs.
                    // The caller synchronizes before subsequent GPU phases;
                    // this bookkeeping hook does not establish GPU completion.
                    this.ops.hostSynchronizeWorld();
                    return mapTensors(captured.result, tensor => tensor.uncapture());
                }

                // Run three eager executions per key before capturing on the
                // fourth invocation. This is a conservative performance mitigation,
                // not a CUDA requirement or a substitute for correct initialization.
                //
                // In the eight-GPU GLM CP/MTP investigation (September 2026),
                // capturing after one eager execution left steady-state replay
                // roughly 0.5-0.7 ms/step slower (~3-4%). Two eager executions
                // recovered most of the loss; use three to leave some margin.
                // The gap persisted with KV length fixed and matching acceptance,
                // and thousands of graph replays did not recover it. Forcing
                // KV-length graph variants also helped, but changing capture
                // history was sufficient: increasing KV length was not required.
                //
                // Slow/fast graph dumps matched kernel names, displayed launch
                // attributes, node types, and dependency edges on all eight GPUs
                // after address normalization. No extra initialization nodes were
                // found. Crucially, ONE eager main-decode execution made the SAME
                // retained graph faster, without prefill, boundary replay, or
                // recapture; replay-only controls remained slower. This points to
                // execution-history-dependent state outside graph structure, not
                // necessarily a bad graph captured on the earlier invocation.
                // Kernel arguments, buffer contents/aliasing, and library/runtime
                // state were not fully isolated. Explicit cuBLAS workspaces did
                // not remove the gap; sampled SM/memory clocks showed no matching
                // frequency increase across recovery, but that was not a definitive
                // exclusion of all power-management effects.
                //
                // Root cause remains unresolved. Do not reduce this count based
                // only on successful capture or matching outputs: compare sustained
                // replay timing, including fixed-length and retained-graph/eager
                // conditioning controls. Extra warmup is not a correctness proof.
                if (captured.warmupSteps === 3) {
                    const capturedInputs: typeof inputs = {} as any;
                    for (const [name, tensor] of Object.entries(inputs)) {
                        if (!tensor)
                            continue;
                        (capturedInputs as any)[name] = tensor.capture();
                    }
                    captured.inputs = capturedInputs;
                    captured.diagnosticBindings = diagnosticBindings;
                    if (process.env.GLM_GRAPH_DIAGNOSTICS === "1") {
                        console.warn(`[cuda-graph] capture bindings key=${key} bindings=${JSON.stringify(diagnosticBindings)}`);
                    }

                    console.warn("\n====capturing====", key)
                    this.ops.graphBeginCapture();
                    capturing = true;
                }
                else {
                    console.warn("\n====warmup+1====", key)
                }
                captured.warmupSteps++;
            }
            else {
                console.warn("\n====warmup====", key)
                captured = { warmupSteps: 1, graphExec: null, result: undefined, inputs: undefined!, capturedWorkspaces: new Set() };
                this.captured.set(key, captured);
            }
        }

        let result: TensorTree;
        try {
            const capturedInputs: any = {};
            for (const [name, input] of Object.entries(inputs)) {
                if (!input)
                    continue;
                input.stage();
                capturedInputs[name] = input.capture();
            }

            captured?.capturedWorkspaces.clear();
            CaptureManager.capturing = captured;

            result = fn(capturing, capturedInputs);
        }
        catch (e) {
            console.warn("Error during capture run:", e);

            if (capturing) {
                const graph = this.ops.graphEndCapture();
                this.ops.graphDestroy(graph);
            }
            throw e;
        }
        finally {
            CaptureManager.capturing = undefined;

            for (const [name, input] of Object.entries(inputs)) {
                if (!input)
                    continue;
                input.unstage();
            }
        }

        if (capturing) {
            captured!.result = mapTensors(result, tensor => tensor.capture());
            const graph = this.ops.graphEndCapture();
            try {
                captured!.graphExec = this.ops.graphInstantiate(graph);
            }
            finally {
                this.ops.graphDestroy(graph);
            }
            this.ops.graphLaunch(captured!.graphExec);
            this.ops.hostSynchronizeWorld();
        }
        return result;
    }

    isCaptured(keyParams: any[], inputs: { [name: string]: Tensor } = {}): boolean {
        if (!keyParams?.length) {
            return false;
        }
        const key = this.graphKey(keyParams, inputs);
        const captured = this.captured.get(key);
        // graphExec may be 0
        return captured?.graphExec != null;
    }
}
