import { DeviceOps } from "./device_ops";
import { Tensor } from "./tensor";

export class WorkspaceBase {
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

  alloc(shape: number[], type: string, name?: string): Tensor {
    return this._alloc(shape, type, false, name);
  }

  allocPinned(shape: number[], type: string, name?: string): Tensor {
    return this._alloc(shape, type, true, name);
  }

  private _alloc(shape: number[], type: string, pinned: boolean, name?: string): Tensor {
    if (this.frozen) {
      throw new Error("Workspace is frozen");
    }

    const bytes = Tensor.byteCount(shape, type);

    if (name !== undefined) {
      const tensor = this.glm.newTensor(shape, type, pinned, name);
      tensor.workspace = this;
      const existing = this.tensors.get(name);
      if (existing !== undefined) {
        this.disposed.add(existing);
      }
      this.tensors.set(name, tensor);
      return tensor;
    }

    let best: Tensor | undefined;
    for (const t of this.disposed) {
      if (t.pinned === pinned && t.allocSize >= bytes && (best === undefined || t.allocSize < best.allocSize)) {
        best = t;
      }
    }
    if (best !== undefined) {
      this.disposed.delete(best);
      const data = best.data;
      (best as { data: number }).data = 0;
      const tensor = new Tensor(data, best.allocSize, shape, type, undefined, pinned);
      tensor.workspace = this;
      this.tracked.add(tensor);
      return tensor;
    }

    const tensor = this.glm.newTensor(shape, type, pinned, undefined);
    tensor.workspace = this;
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

  startTracking(): Disposable & { [Symbol.dispose](): void } {
    if (this.tracking !== null) throw new Error("startTracking already active");
    for (const tensor of this.exported) {
      this.disposed.add(tensor);
    }
    this.exported.clear();
    const ws = this;
    const tracker: Disposable & { [Symbol.dispose](): void } = {
      [Symbol.dispose]() {
        for (const tensor of ws.tracked) {
          ws.disposed.add(tensor);
        }
        ws.tracked.clear();
        ws.tracking = null;
      },
    };
    this.tracking = tracker;
    return tracker;
  }
}
