import { CaptureManager } from "./capture-manager";
import { DeviceOps, TensorParallelism } from "./device_ops";
import { Tensor } from "./tensor";

export class WorkspaceBase implements Disposable {
  readonly glm: DeviceOps;
  tensors = new Map<string, Tensor>();
  tracked = new Set<Tensor>();
  staged = new Set<Tensor>();
  disposedDevice = new Set<Tensor>();
  disposedHost = new Set<Tensor>();
  synchronizingHost = new Set<Tensor>();
  frozen = false;
  allocLogger = false;
  tracking: Disposable & { [Symbol.dispose](): void } | null = null;
  private readonly synchronizeRef: WeakRef<WorkspaceBase>;

  constructor(glm: DeviceOps) {
    this.glm = glm;
    this.synchronizeRef = new WeakRef(this);
    this.glm.synchronizeListeners.push(this.synchronizeRef);
  }

  synchronizeComplete(): void {
    for (const tensor of this.synchronizingHost) {
      this.disposedHost.add(tensor);
    }
    this.synchronizingHost.clear();
  }

  _runClear(keepExports = new Set<Tensor>(), callback: (tracked: Tensor) => boolean) {
    const keep = new Set<Tensor>();
    for (const tensor of keepExports) {
      keep.add(tensor);
      let root = tensor;
      while (root.view) {
        root = root.view;
        keep.add(root);
      }
    }
    for (const tracked of this.tracked) {
      if (!keep.has(tracked)) {
        if (!callback(tracked))
          break;
      }
    }
  }

  assertClear(keepExports = new Set<Tensor>()) {
    this._runClear(keepExports, (t) => {
      // console.warn(t.stack);
      throw new Error("assertClear was called with tensors already allocated that were not in keepExports, this may result in non-deterministic allocations.");
    });
  }

  _clearTracking(keepExports = new Set<Tensor>()) {
    if (this.tracking !== null) {
      throw new Error("tracking already active");
    }
    if (this.staged.size) {
      throw new Error("dangling staged tensors were found from a previous incomplete operation");
    }
    this._runClear(keepExports, tracked => {
      console.warn(new Error("clearTracking was called with tensors already allocated that were not in keepExports, this may result in non-deterministic allocations."));
      // console.warn(tracked.shape, this.tracked.size, tracked.stack);
      tracked[Symbol.dispose]();
      return true;
    });
  }

  clearTracking(keepExports = new Set<Tensor>()) {
    this._clearTracking(keepExports);
  }

  startTracking(keepExports = new Set<Tensor>()): Disposable & { [Symbol.dispose](): void } {
    this._clearTracking(keepExports);

    const ws = this;
    const tracker: Disposable & { [Symbol.dispose](): void } = {
      [Symbol.dispose]() {
        for (const tensor of ws.tracked) {
          tensor.views.clear();
          tensor[Symbol.dispose]();
        }
        ws.tracked.clear();
        for (const tensor of ws.staged) {
          ws.tracked.add(tensor);
        }
        ws.staged.clear();
        ws.tracking = null;
      },
    };
    this.tracking = tracker;
    return tracker;
  }

  freeze() {
    this.frozen = true;
  }

  unfreeze() {
    this.frozen = false;
  }

  alloc(shape: number[], type: string, name?: string, parallelism?: TensorParallelism): Tensor {
    return this._alloc(shape, type, false, name, parallelism);
  }

  ensureAlloc(shape: number[], type: string, name: string, parallelism?: TensorParallelism, fill?: number): Tensor {
    const existing = this.tensors.get(name);
    if (existing !== undefined) {
      if (existing.shape.length !== shape.length || existing.shape.some((v, i) => v !== shape[i]) || existing.type !== type) {
        throw new Error(`Tensor with name ${name} already exists with different shape or type`);
      }
      return existing;
    }
    const ret = this._alloc(shape, type, false, name, parallelism);
    if (fill !== undefined) {
      ret.fill(fill, ret.numElements);
    }
    return ret;
  }

  ensureAllocPinned(shape: number[], type: string, name: string, parallelism?: TensorParallelism): Tensor {
    const existing = this.tensors.get(name);
    if (existing !== undefined) {
      if (existing.shape.length !== shape.length || existing.shape.some((v, i) => v !== shape[i]) || existing.type !== type) {
        throw new Error(`Tensor with name ${name} already exists with different shape or type`);
      }
      return existing;
    }
    return this._alloc(shape, type, true, name, parallelism);
  }

  allocPinned(shape: number[], type: string, name?: string, parallelism?: TensorParallelism): Tensor {
    return this._alloc(shape, type, true, name, parallelism);
  }

  allocRaw(bytes: number, name?: string): Tensor {
    return this.alloc([bytes], "U8", name);
  }

  addTracked(tensor: Tensor) {
    try {
      CaptureManager.trackWorkspaceAlloc(this);
    }
    finally {
      this.addTrackedInternal(tensor);
    }
  }

  private addTrackedInternal(tensor: Tensor) {
    this.tracked.add(tensor);
  }

  protected _alloc(shape: number[], type: string, pinned: boolean, name?: string, parallelism?: TensorParallelism): Tensor {
    if (this.frozen) {
      throw new Error("Workspace is frozen");
    }

    CaptureManager.trackWorkspaceAlloc(this);

    const bytes = Tensor.byteCount(shape, type);

    if (name !== undefined) {
      const existing = this.tensors.get(name);
      if (existing !== undefined) {
        throw new Error(`Tensor with name ${name} already exists`);
      }
    }

    let best: Tensor | undefined;
    const disposed = pinned ? this.disposedHost : this.disposedDevice;
    for (const t of disposed) {
      if (t.view)
        throw new Error("disposed tensor should not have a view");
      if (!t.data)
        throw new Error("disposed tensor should have data");
      if (t.allocSize >= bytes && (best === undefined || t.allocSize < best.allocSize)) {
        if (name === undefined) {
          best = t;
        }
      }
    }
    let tensor: Tensor;
    if (best !== undefined) {
      if (best.allocSize !== bytes && this.allocLogger) {
        console.warn(`Reusing disposed tensor of size ${best.allocSize} bytes for allocation of ${bytes} bytes (${shape.join("x")} ${type}${pinned ? " pinned" : ""}${parallelism ? ` ${parallelism}` : ""})`);
      }
      disposed.delete(best);
      const data = best.data;
      best.detachData();
      tensor = this.glm.wrapTensor(this, data, best.allocSize, shape, type, pinned, undefined);
    } else {
      if (this.allocLogger) {
        console.warn(`Allocating new tensor ${name ?? "<unnamed>"} of size ${bytes} bytes (${shape.join("x")} ${type}${pinned ? " pinned" : ""}${parallelism ? ` ${parallelism}` : ""})`);
      }
      tensor = this.glm.newTensor(this, shape, type, pinned, name, parallelism);
    }

    if (name !== undefined) {
      this.tensors.set(name, tensor);
    } else {
      this.addTrackedInternal(tensor);
    }
    return tensor;
  }

  free(): void {
    const synchronizeIndex = this.glm.synchronizeListeners.indexOf(this.synchronizeRef);
    if (synchronizeIndex !== -1) {
      this.glm.synchronizeListeners.splice(synchronizeIndex, 1);
    }
    for (const tensor of this.tensors.values()) {
      tensor.free();
    }
    for (const tensor of this.tracked) {
      tensor.free();
    }
    for (const tensor of this.disposedHost) {
      tensor.free();
    }
    for (const tensor of this.disposedDevice) {
      tensor.free();
    }
    for (const tensor of this.synchronizingHost) {
      tensor.free();
    }
    for (const tensor of this.staged) {
      tensor.free();
    }
    this.tensors.clear();
    this.tracked.clear();
    this.disposedHost.clear();
    this.disposedDevice.clear();
    this.synchronizingHost.clear();
    this.staged.clear();
  }

  [Symbol.dispose](): void {
    this.free();
  }
}
