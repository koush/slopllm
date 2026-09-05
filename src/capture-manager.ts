import { Tensor } from "./tensor";
import { type DeviceOps } from "./device_ops";
import { MemcpyKind } from "./enums";
import type { WorkspaceBase } from "./workspace";
import { mapTensors, type TensorTree } from "./tensor-tree";


interface Captured {
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

export class CaptureManager implements Disposable {
    static capturing?: Captured;
    disabled = false;
    captured = new Map<string, Captured>();

    // baseKey (caller key params + batchSize, WITHOUT padded dims) -> learned variance.
    private lengthVariant = new Map<string, LengthVariant>();

    constructor(public ops: DeviceOps) {
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

    run<I extends { [name: string]: Tensor }>(inputs: I, fn: (capturing: boolean, capturedInputs: I) => TensorTree, keyParams?: any[]): TensorTree {
        if (CaptureManager.capturing) {
            throw new Error("Cannot run a capture while another capture is in progress");
        }

        let captured: Captured | undefined;
        let capturing = false;

        if (!this.disabled && keyParams?.length) {
            const key = keyParams.join(",");
            captured = this.captured.get(key);

            if (captured) {
                if (captured.graphExec !== null) {
                    const replayInputs = Object.entries(inputs)
                        .filter((entry): entry is [string, Tensor] => !!entry[1])
                        .map(([name, input]) => ({
                            name,
                            input,
                            capturedInput: captured!.inputs[name],
                            needsCopy: !captured!.inputs[name].same(input),
                        }));
                    for (let writerIndex = 0; writerIndex < replayInputs.length; writerIndex++) {
                        const writer = replayInputs[writerIndex];
                        if (!writer.needsCopy || !writer.capturedInput.matches(writer.input)) continue;
                        const destinations = writer.capturedInput.memoryRanges();
                        for (let victimIndex = 0; victimIndex < replayInputs.length; victimIndex++) {
                            const victim = replayInputs[victimIndex];
                            const sourceIsStillNeeded = victim.needsCopy
                                ? victimIndex >= writerIndex
                                : true;
                            if (!sourceIsStillNeeded) continue;
                            const sources = victim.input.memoryRanges();
                            const numRanges = Math.min(destinations.length, sources.length);
                            for (let shard = 0; shard < numRanges; shard++) {
                                const destination = destinations[shard];
                                const source = sources[shard];
                                const overlapStart = Math.max(destination.data, source.data);
                                const overlapEnd = Math.min(destination.data + destination.bytes, source.data + source.bytes);
                                if (overlapStart >= overlapEnd) continue;
                                console.warn(
                                    `[cuda-graph] HAZARDOUS input memcpy overlap key=${key} graphExec=${captured.graphExec}`
                                    + ` writer=${writer.name} victim=${victim.name} shard=${shard}`
                                    + ` destination=[0x${destination.data.toString(16)},0x${(destination.data + destination.bytes).toString(16)})`
                                    + ` source=[0x${source.data.toString(16)},0x${(source.data + source.bytes).toString(16)})`
                                    + ` overlap=[0x${overlapStart.toString(16)},0x${overlapEnd.toString(16)})`,
                                );
                            }
                        }
                    }
                    for (const [name, input] of Object.entries(inputs)) {
                        if (!input)
                            continue;
                        input.stage();
                        const capturedInput = captured.inputs[name];
                        if (!capturedInput.same(input)) {
                            console.warn(`[cuda-graph] input mismatch key=${key} graphExec=${captured.graphExec} name=${name}`);
                            console.warn(`[cuda-graph] current input: ${input.debugDescription()}`);
                            console.warn(`[cuda-graph] captured input: ${capturedInput.debugDescription()}`);
                            if (!capturedInput.matches(input)) {
                                throw new Error(`[cuda-graph] incompatible replay input key=${key} name=${name}`);
                            }
                            capturedInput.memcpy(input, capturedInput.bytes, MemcpyKind.DeviceToDevice);
                            console.warn(`[cuda-graph] enqueued input memcpy key=${key} name=${name} bytes=${capturedInput.bytes}`);
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
                    return mapTensors(captured.result, tensor => tensor.uncapture());
                }

                if (captured.warmupSteps === 3) {
                    const capturedInputs: typeof inputs = {} as any;
                    for (const [name, tensor] of Object.entries(inputs)) {
                        if (!tensor)
                            continue;
                        (capturedInputs as any)[name] = tensor.capture();
                    }
                    captured.inputs = capturedInputs;

                    // console.warn("\n====capturing====", key)
                    this.ops.graphBeginCapture();
                    capturing = true;
                }
                else {
                    // console.warn("\n====warmingup====", key)
                }
                captured.warmupSteps++;
            }
            else {
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
        }
        return result;
    }

    isCaptured(keyParams: any[]): boolean {
        if (!keyParams?.length) {
            return false;
        }
        const key = keyParams.join(",");
        const captured = this.captured.get(key);
        // graphExec may be 0
        return captured?.graphExec != null;
    }
}
