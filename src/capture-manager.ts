import { type DeviceOps } from "./device_ops";
import { MemcpyKind } from "./enums";
import { type Tensor } from "./tensor";

interface Captured {
    warmupSteps: number;
    graphExec: number | null;
    result: any;
    inputs: { [name: string]: Tensor, };
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
    disabled = false;
    captured = new Map<string, Captured>();
    // baseKey (caller key params + batchSize, WITHOUT padded dims) -> learned variance.
    private lengthVariant = new Map<string, LengthVariant>();
    static capturing?: CaptureManager;

    constructor(public ops: DeviceOps) {
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

    run<T, I extends { [name: string]: Tensor }>(inputs: I, fn: (capturing: boolean, capturedInputs: I) => T, keyParams?: any[]): T {
        if (CaptureManager.capturing) {
            throw new Error("Cannot run a capture while another capture is in progress");
        }

        let capturing: string | undefined;
        if (!this.disabled && keyParams?.length) {
            const key = keyParams.join(",");
            const captured = this.captured.get(key);

            if (captured) {
                if (captured.graphExec !== null) {
                    for (const [name, input] of Object.entries(inputs)) {
                        const capturedInput = captured.inputs[name];
                        if (!capturedInput.same(input)) {
                            capturedInput.memcpy(input, capturedInput.bytes, MemcpyKind.DeviceToDevice);
                        }
                    }
                    this.ops.graphLaunch(captured.graphExec);
                    return captured.result;
                }

                if (captured.warmupSteps === 3) {
                    const capturedInputs: typeof inputs = {} as any;
                    for (const [name, tensor] of Object.entries(inputs)) {
                        (capturedInputs as any)[name] = tensor.capture();
                    }
                    captured.inputs = capturedInputs;

                    // console.warn("\n====capturing====", key)
                    this.ops.graphBeginCapture();
                    capturing = key;
                }
                else {
                    // console.warn("\n====warmingup====", key)
                }
                captured.warmupSteps++;
            }
            else {
                this.captured.set(key, { warmupSteps: 1, graphExec: null, result: undefined, inputs: undefined! });
            }
        }

        let result: T;
        try {
            const capturedInputs: any = {};
            for (const [name, input] of Object.entries(inputs)) {
                capturedInputs[name] = input.capture();
            }

            CaptureManager.capturing = this;
            result = fn(!!capturing, capturedInputs);
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
        }

        if (capturing) {
            const captured = this.captured.get(capturing)!;
            captured.result = result;
            const graph = this.ops.graphEndCapture();
            captured.graphExec = this.ops.graphInstantiate(graph);
            this.ops.graphDestroy(graph);
            this.ops.graphLaunch(captured.graphExec);
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
