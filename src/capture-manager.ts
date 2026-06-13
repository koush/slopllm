import { DeviceOps } from "./device_ops";

interface Captured {
    warmupSteps: number;
    graphExec: number | null;
    result: any;
}

export class CaptureManager implements Disposable {
    disabled = false;
    captured = new Map<string, Captured>();

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