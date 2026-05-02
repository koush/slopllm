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
      const tensor = this.glm.newTensor(this, shape, type, pinned, name, parallelism);
      const existing = this.tensors.get(name);
      if (existing !== undefined) {
        existing[Symbol.dispose]();
      }
      this.tensors.set(name, tensor);
      return tensor;
    }

    let best: Tensor | undefined;
    for (const t of this.disposed) {
      // shouldn't be possible but defensive check.
      if (t.view)
        throw new Error("disposed tensor should not have a view");
      if (!t.data)
        throw new Error("disposed tensor should have data");
      if (t.pinned === pinned && t.allocSize >= bytes && (best === undefined || t.allocSize < best.allocSize)) {
        best = t;
      }
    }
    if (best !== undefined) {
      this.disposed.delete(best);
      const data = best.data;
      best.detachData();
      const tensor = this.glm.wrapTensor(this, data, best.allocSize, shape, type, pinned, undefined);
      this.tracked.add(tensor);
      return tensor;
    }

    const tensor = this.glm.newTensor(this, shape, type, pinned, undefined, parallelism);

    this.tracked.add(tensor);
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
