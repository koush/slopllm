import { CaptureManager } from "./capture-manager";
import { DeviceOps, TensorParallelism } from "./device_ops";
import { Tensor } from "./tensor";
import { collectTensors, type TensorTree } from "./tensor-tree";
import { Heap, type HeapKey } from "./heap";

export class WorkspaceBase implements Disposable {
  readonly ops: DeviceOps;
  tensors = new Map<string, Tensor>();
  tracked = new Set<Tensor>();
  staged = new Set<Tensor>();
  heapByKey = new Map<HeapKey, Heap>();
  disposedHost = new Set<Tensor>();
  synchronizingHost = new Set<Tensor>();
  frozen = false;
  allocLogger = false;
  tracking: Disposable & { [Symbol.dispose](): void } | null = null;
  private readonly synchronizeRef: WeakRef<WorkspaceBase>;

  constructor(ops: DeviceOps) {
    this.ops = ops;
    this.synchronizeRef = new WeakRef(this);
    this.ops.synchronizeListeners.push(this.synchronizeRef);
  }

  synchronizeComplete(): void {
    for (const tensor of this.synchronizingHost) {
      this.disposedHost.add(tensor);
    }
    this.synchronizingHost.clear();
    for (const key of [...this.heapByKey.keys()]) {
      if (key !== undefined) this.drainHeap(key, undefined);
    }
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

  assertClear(keep: TensorTree = undefined) {
    this._runClear(collectTensors(keep), (t) => {
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

  clearTracking(keep: TensorTree = undefined) {
    this._clearTracking(collectTensors(keep));
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
        // Restore exports through the tensor API so composite tensors also
        // unstage their backing shards and views in their owning workspaces.
        for (const tensor of [...ws.staged]) {
          tensor.unstage();
        }
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

  getHeap(key: HeapKey): Heap {
    let heap = this.heapByKey.get(key);
    if (!heap) {
      heap = new Heap();
      this.heapByKey.set(key, heap);
    }
    return heap;
  }

  getDisposedPools(pinned: boolean): Set<Tensor>[] {
    if (!pinned) throw new Error("Device tensors are recycled through heaps");
    return [this.disposedHost];
  }

  private allocationLineage(lineage?: readonly HeapKey[]): HeapKey[] {
    if (lineage !== undefined) {
      if (lineage.length === 0) throw new Error("Allocation lineage must not be empty");
      return [...new Set(lineage)];
    }
    const streams = [...this.ops.activeStreams].reverse();
    return [...new Set<HeapKey>([...streams, undefined])];
  }

  recycleDevice(ptr: number, length: number, key: HeapKey): void {
    this.getHeap(key).manage(ptr, length);
  }

  claimDevice(ptr: number, length: number, recycleKey: HeapKey | null): boolean {
    const lineage = recycleKey === null ? undefined : [recycleKey, undefined];
    for (const key of this.allocationLineage(lineage)) {
      if (this.heapByKey.get(key)?.claim(ptr, length)) return true;
    }
    return false;
  }

  describeDeviceRange(ptr: number, length: number): string {
    const heaps = [...this.heapByKey.entries()].map(([key, heap]) => {
      const keyName = key === undefined
        ? "synchronized"
        : typeof key === "number"
          ? `stream:${key}`
          : typeof key === "symbol"
            ? key.toString()
            : `object:${key.constructor?.name ?? "unknown"}`;
      return `${keyName}{${heap.describeRange(ptr, length)}}`;
    });
    return heaps.length ? heaps.join(" ") : "no workspace heaps";
  }

  drainHeap(sourceKey: HeapKey, destinationKey: HeapKey): void {
    if (sourceKey === destinationKey) return;
    const source = this.heapByKey.get(sourceKey);
    if (!source) return;
    source.drainTo(this.getHeap(destinationKey));
    this.heapByKey.delete(sourceKey);
  }

  disposeStream(stream: number, destinationStream: number): void {
    if (stream === destinationStream) return;
    this.drainHeap(stream, destinationStream);
  }

  alloc(shape: number[], type: string, name?: string, parallelism?: TensorParallelism, lineage?: readonly HeapKey[]): Tensor {
    return this._alloc(shape, type, false, name, parallelism, lineage);
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

  protected _alloc(shape: number[], type: string, pinned: boolean, name?: string, parallelism?: TensorParallelism, lineage?: readonly HeapKey[]): Tensor {
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

    let tensor: Tensor;
    const recycleKey = lineage === undefined ? null : this.allocationLineage(lineage)[0];
    if (pinned) {
      let best: Tensor | undefined;
      for (const disposed of this.disposedHost) {
        if (disposed.view)
          throw new Error("disposed tensor should not have a view");
        if (!disposed.data)
          throw new Error("disposed tensor should have data");
        if (disposed.allocSize >= bytes && (best === undefined || disposed.allocSize < best.allocSize)) {
          if (name === undefined) {
            best = disposed;
          }
        }
      }
      if (best !== undefined) {
        if (best.allocSize !== bytes && this.allocLogger) {
          console.warn(`Reusing disposed tensor of size ${best.allocSize} bytes for allocation of ${bytes} bytes (${shape.join("x")} ${type} pinned${parallelism ? ` ${parallelism}` : ""})`);
        }
        this.disposedHost.delete(best);
        const data = best.data;
        best.detachData();
        tensor = this.ops.wrapTensor(this, data, best.allocSize, shape, type, true, undefined, recycleKey);
      } else {
        tensor = this.ops.newTensor(this, shape, type, true, name, parallelism, recycleKey);
      }
    } else {
      let allocation;
      if (name === undefined) {
        for (const key of this.allocationLineage(lineage)) {
          allocation = this.heapByKey.get(key)?.tryAlloc(bytes);
          if (allocation) break;
        }
      }
      if (allocation) {
        tensor = this.ops.wrapTensor(this, allocation.ptr, allocation.length, shape, type, false, undefined, recycleKey);
      } else {
        if (this.allocLogger) {
          console.warn(`Allocating new tensor ${name ?? "<unnamed>"} of size ${bytes} bytes (${shape.join("x")} ${type}${parallelism ? ` ${parallelism}` : ""})`);
        }
        tensor = this.ops.newTensor(this, shape, type, false, name, parallelism, recycleKey);
      }
    }

    if (name !== undefined) {
      this.tensors.set(name, tensor);
    } else {
      this.addTrackedInternal(tensor);
    }
    return tensor;
  }

  free(): void {
    const synchronizeIndex = this.ops.synchronizeListeners.indexOf(this.synchronizeRef);
    if (synchronizeIndex !== -1) {
      this.ops.synchronizeListeners.splice(synchronizeIndex, 1);
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
    for (const tensor of this.synchronizingHost) {
      tensor.free();
    }
    for (const tensor of this.staged) {
      tensor.free();
    }
    this.tensors.clear();
    this.tracked.clear();
    this.disposedHost.clear();
    this.heapByKey.clear();
    this.synchronizingHost.clear();
    this.staged.clear();
  }

  [Symbol.dispose](): void {
    this.free();
  }
}
