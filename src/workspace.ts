import { DeviceOps, TensorParallelism } from "./device_ops";
import { Tensor } from "./tensor";

export class WorkspaceBase implements Disposable {
  readonly glm: DeviceOps;
  tensors = new Map<string, Tensor>();
  tracked = new Set<Tensor>();
  disposed = new Set<Tensor>();
  exported = new Set<Tensor>();
  private tracking: Disposable & { [Symbol.dispose](): void } | null = null;
  frozen = false;

  constructor(glm: DeviceOps) {
    this.glm = glm;
  }

  freeze() {
    this.frozen = true;
  }

  stats(): { namedCount: number; namedBytes: number; namedDetails: { name: string; shape: number[]; type: string; allocSize: number; parallelism: string }[]; disposedCount: number; disposedBytes: number; disposedDetails: { shape: number[]; type: string; allocSize: number }[]; trackedCount: number; trackedBytes: number; exportedCount: number; exportedBytes: number } {
    const namedDetails: { name: string; shape: number[]; type: string; allocSize: number; parallelism: string }[] = [];
    let namedBytes = 0;
    for (const [name, t] of this.tensors) {
      namedBytes += t.allocSize;
      namedDetails.push({ name, shape: t.shape, type: t.type, allocSize: t.allocSize, parallelism: t.parallelism ?? "none" });
    }
    const disposedDetails: { shape: number[]; type: string; allocSize: number }[] = [];
    let disposedBytes = 0;
    for (const t of this.disposed) {
      disposedBytes += t.allocSize;
      disposedDetails.push({ shape: t.shape, type: t.type, allocSize: t.allocSize });
    }
    let trackedBytes = 0;
    for (const t of this.tracked) {
      trackedBytes += t.allocSize;
    }
    let exportedBytes = 0;
    for (const t of this.exported) {
      exportedBytes += t.allocSize;
    }
    return {
      namedCount: this.tensors.size, namedBytes, namedDetails,
      disposedCount: this.disposed.size, disposedBytes, disposedDetails,
      trackedCount: this.tracked.size, trackedBytes,
      exportedCount: this.exported.size, exportedBytes,
    };
  }

  alloc(shape: number[], type: string, name?: string, parallelism?: TensorParallelism): Tensor {
    return this._alloc(shape, type, false, name, parallelism);
  }

  allocPinned(shape: number[], type: string, name?: string, parallelism?: TensorParallelism): Tensor {
    return this._alloc(shape, type, true, name, parallelism);
  }

  allocRaw(bytes: number, name: string): Tensor {
    return this.alloc([bytes], "U8", name);
  }

  private _alloc(shape: number[], type: string, pinned: boolean, name?: string, parallelism?: TensorParallelism): Tensor {
    if (this.frozen) {
      throw new Error("Workspace is frozen");
    }

    const bytes = Tensor.byteCount(shape, type);

    if (name !== undefined) {
      const existing = this.tensors.get(name);
      if (existing !== undefined) {
        existing[Symbol.dispose]();
      }

    }

    let best: Tensor | undefined;
    for (const t of this.disposed) {
      if (t.view)
        throw new Error("disposed tensor should not have a view");
      if (!t.data)
        throw new Error("disposed tensor should have data");
      if (t.pinned === pinned && t.allocSize >= bytes && (best === undefined || t.allocSize < best.allocSize)) {
        if (name === undefined || t.allocSize === bytes) {
          best = t;
        }
      }
    }
    let tensor: Tensor;
    if (best !== undefined) {
      this.disposed.delete(best);
      const data = best.data;
      best.detachData();
      tensor = this.glm.wrapTensor(this, data, best.allocSize, shape, type, pinned, undefined);
    } else {
      tensor = this.glm.newTensor(this, shape, type, pinned, undefined, parallelism);
    }

    if (name !== undefined) {
      tensor.setName(name);
      this.tensors.set(name, tensor);
    } else {
      this.tracked.add(tensor);
    }
    return tensor;
  }

  free(): void {
    for (const tensor of this.tensors.values()) {
      tensor.free();
    }
    for (const tensor of this.tracked) {
      tensor.free();
    }
    for (const tensor of this.disposed) {
      tensor.free();
    }
    for (const tensor of this.exported) {
      tensor.free();
    }
    this.tensors.clear();
    this.tracked.clear();
    this.disposed.clear();
    this.exported.clear();
  }

  [Symbol.dispose](): void {
    this.free();
  }

  startTracking(): Disposable & { [Symbol.dispose](): void } {
    if (this.tracking !== null) throw new Error("startTracking already active");
    for (const tensor of this.exported) {
      tensor[Symbol.dispose]();
    }
    this.exported.clear();
    const ws = this;
    const tracker: Disposable & { [Symbol.dispose](): void } = {
      [Symbol.dispose]() {
        for (const tensor of ws.tracked) {
          tensor[Symbol.dispose]();
        }
        ws.tracked.clear();
        ws.tracking = null;
      },
    };
    this.tracking = tracker;
    return tracker;
  }
}
