import { DeviceOps } from "./device_ops";

interface Captured {
    warmupSteps: number;
    graphExec: number | null;
    result: any;
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
    qLen: boolean;
}

export class CaptureManager implements Disposable {
    disabled = false;
    captured = new Map<string, Captured>();
    // baseKey (caller key params + batchSize, WITHOUT padded dims) -> learned variance.
    private lengthVariant = new Map<string, LengthVariant>();

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
        return this.lengthVariant.get(baseKey) ?? { kvLen: false, qLen: false };
    }

    /** Monotonically record that a base graph sizes buffers by a padded dim. */
    recordLengthVariant(baseKey: string, kvLen: boolean, qLen: boolean): void {
        if (!kvLen && !qLen) return;
        const cur = this.lengthVariant.get(baseKey);
        if (!cur) {
            this.lengthVariant.set(baseKey, { kvLen, qLen });
        } else {
            cur.kvLen ||= kvLen;
            cur.qLen ||= qLen;
        }
    }

    run<T>(fn: (capturing: boolean) => T, keyParams?: any[]): T {
        let capturing: string | undefined;
        if (!this.disabled && keyParams?.length) {
            const key = keyParams.join(",");
            const captured = this.captured.get(key);
            if (captured) {
                if (captured.graphExec !== null) {
                    this.ops.graphLaunch(captured.graphExec);
                    return captured.result;
                }

                if (captured.warmupSteps === 3) {
                    this.ops.graphBeginCapture();
                    capturing = key;
                }
                captured.warmupSteps++;
            }
            else {
                this.captured.set(key, { warmupSteps: 1, graphExec: null, result: undefined });
            }
        }

        const result = fn(!!capturing);
        if (capturing)
            this.captured.get(capturing!)!.result = result;

        if (capturing) {
            const graph = this.ops.graphEndCapture();
            const captured = this.captured.get(capturing)!;
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
