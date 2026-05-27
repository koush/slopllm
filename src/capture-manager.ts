import { DeviceOps } from "./device_ops";

interface Captured {
    warmupSteps: number;
    graphExec: number | null;
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

    run(fn: () => void, keyParams?: any[]) {
        let capturing: string | undefined;
        if (!this.disabled && keyParams?.length) {
            const key = keyParams.join(",");
            const captured = this.captured.get(key);
            if (captured) {
                if (captured.graphExec !== null) {
                    this.ops.graphLaunch(captured.graphExec);
                    return;
                }

                if (captured.warmupSteps === 3) {
                    this.ops.graphBeginCapture();
                    capturing = key;
                }
                captured.warmupSteps++;
            }
            else {
                this.captured.set(key, { warmupSteps: 1, graphExec: null });
            }
        }

        fn();

        if (capturing) {
            const graph = this.ops.graphEndCapture();
            const captured = this.captured.get(capturing)!;
            captured.graphExec = this.ops.graphInstantiate(graph);
            this.ops.graphDestroy(graph);
            this.ops.graphLaunch(captured.graphExec);
        }
    }

    isCaptured(keyParams: any[]): boolean {
        if (!keyParams?.length) {
            return false;
        }
        const key = keyParams.join(",");
        const captured = this.captured.get(key);
        return captured?.graphExec !== null;
    }
}