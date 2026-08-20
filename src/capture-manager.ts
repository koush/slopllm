import { Tensor } from "./tensor";
import { type DeviceOps } from "./device_ops";
import { MemcpyKind } from "./enums";
import type { WorkspaceBase } from "./workspace";


interface Captured {
    warmupSteps: number;
    graphExec: number | null;
    result: CaptureReturn;
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

export type CaptureReturn = Tensor | CaptureReturn[] | string | number | boolean | null | undefined | { [name: string]: CaptureReturn };

function mapCaptureTensors(object: CaptureReturn, mapTensor: (tensor: Tensor) => Tensor): CaptureReturn {
    if (object instanceof Tensor) {
        return mapTensor(object);
    }
    else if (Array.isArray(object)) {
        return object.map(item => mapCaptureTensors(item, mapTensor));
    }
    else if (object && typeof object === "object") {
        const result: Record<string, CaptureReturn> = {};
        for (const [key, item] of Object.entries(object)) {
            result[key] = mapCaptureTensors(item, mapTensor);
        }
        return result;
    }
    else {
        return object;
    }
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
            throw new Error("Cannot capture a workspace that has tracked tensors");
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

    run<I extends { [name: string]: Tensor }>(inputs: I, fn: (capturing: boolean, capturedInputs: I) => CaptureReturn, keyParams?: any[]): CaptureReturn {
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
                    for (const [name, input] of Object.entries(inputs)) {
                        if (!input)
                            continue;
                        input.stage();
                        const capturedInput = captured.inputs[name];
                        if (!capturedInput.same(input)) {
                            console.warn('need memcpy')
                            capturedInput.memcpy(input, capturedInput.bytes, MemcpyKind.DeviceToDevice);
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
                    this.ops.graphLaunch(captured.graphExec);
                    return mapCaptureTensors(captured.result, tensor => tensor.uncapture().removeTracking());
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

        let result: CaptureReturn;
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
            captured!.result = mapCaptureTensors(result, tensor => tensor.capture());
            const graph = this.ops.graphEndCapture();
            try {
                captured!.graphExec = this.ops.graphInstantiate(graph);
            }
            finally {
                this.ops.graphDestroy(graph);
            }
            this.ops.graphLaunch(captured!.graphExec);
        }
        return mapCaptureTensors(result, tensor => tensor.removeTracking());
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
