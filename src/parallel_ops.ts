import { DeviceOps, TensorParallelism } from "./device_ops";
import { GlmOps } from "./glm_ops";
import { Tensor } from "./tensor";
import { WorkspaceBase } from "./workspace";

export class ParallelTensor extends Tensor {
  readonly parallelism: TensorParallelism;
  readonly shards: readonly Tensor[];
  readonly fullShape: number[];
  private readonly devices: readonly GlmOps[];
  private _disposed = false;

  constructor(
    workspace: WorkspaceBase,
    devices: readonly GlmOps[],
    parallelism: TensorParallelism,
    shards: readonly Tensor[],
    fullShape: number[],
    type: string,
    name: string | undefined,
    pinned: boolean,
  ) {
    super(workspace, 0, 0, fullShape, type, name, pinned);
    this.devices = devices;
    this.parallelism = parallelism;
    this.shards = shards;
    this.fullShape = fullShape;
  }

  free(): void {
    this._disposed = true;
    for (let i = 0; i < this.shards.length; i++) {
      const shard = this.shards[i];
      if (shard.data !== 0) {
        if (this.pinned) {
          this.devices[i].freePinned(shard);
        } else {
          this.devices[i].freeBuf(shard);
        }
        (shard as { data: number }).data = 0;
      }
    }
  }

  [Symbol.dispose](): void {
    if (this.name !== undefined) {
      throw new Error("Cannot dispose named tensor");
    }
    if (this._disposed) return;
    this._disposed = true;
    this.workspace.tracked.delete(this);
    for (const shard of this.shards) {
      if (shard.data !== 0) {
        shard[Symbol.dispose]();
      }
    }
  }

  shard(rank: number): Tensor {
    return this.shards[rank];
  }
}

export class ParallelOps implements DeviceOps {
  readonly devices: readonly GlmOps[];
  readonly worldSize: number;
  private readonly shardWorkspaces = new WeakMap<WorkspaceBase, WorkspaceBase[]>();

  constructor(devices: GlmOps[]) {
    if (devices.length === 0) {
      throw new Error("ParallelOps requires at least one device");
    }
    if ((devices.length & (devices.length - 1)) !== 0) {
      throw new Error("ParallelOps requires power-of-2 device count");
    }
    this.devices = devices;
    this.worldSize = devices.length;
  }

  private getShardWorkspaces(workspace: WorkspaceBase): WorkspaceBase[] {
    let wss = this.shardWorkspaces.get(workspace);
    if (wss === undefined) {
      wss = this.devices.map(glm => new WorkspaceBase(glm));
      this.shardWorkspaces.set(workspace, wss);
    }
    return wss;
  }

  shardWorkspacesFor(workspace: WorkspaceBase): readonly WorkspaceBase[] {
    return this.shardWorkspaces.get(workspace) ?? [];
  }

  shardShape(fullShape: number[], parallelism: TensorParallelism): number[] {
    switch (parallelism) {
      case TensorParallelism.Column:
        if (fullShape[0] % this.worldSize !== 0) {
          throw new Error(`Column parallel: shape[0]=${fullShape[0]} not divisible by worldSize=${this.worldSize}`);
        }
        return [fullShape[0] / this.worldSize, ...fullShape.slice(1)];
      case TensorParallelism.Row:
        if (fullShape.length < 2) {
          throw new Error("Row parallel: requires at least 2D shape");
        }
        if (fullShape[1] % this.worldSize !== 0) {
          throw new Error(`Row parallel: shape[1]=${fullShape[1]} not divisible by worldSize=${this.worldSize}`);
        }
        return [fullShape[0], fullShape[1] / this.worldSize, ...fullShape.slice(2)];
      case TensorParallelism.Replicated:
      default:
        return [...fullShape];
    }
  }

  newTensor(workspace: WorkspaceBase, shape: number[], type: string, pinned: boolean, name?: string, parallelism?: TensorParallelism): ParallelTensor {
    const par = parallelism ?? TensorParallelism.Replicated;
    const ss = this.shardShape(shape, par);
    const shardWss = this.getShardWorkspaces(workspace);
    const shards: Tensor[] = shardWss.map(ws =>
      pinned ? ws.allocPinned(ss, type) : ws.alloc(ss, type),
    );
    return new ParallelTensor(workspace, this.devices, par, shards, shape, type, name, pinned);
  }

  freeBuf(_ptr: Tensor): void { throw new Error("ParallelOps.freeBuf not implemented"); }
  freePinned(_ptr: Tensor): void { throw new Error("ParallelOps.freePinned not implemented"); }
  h2d(): void { throw new Error("ParallelOps.h2d not implemented"); }
  d2h(): void { throw new Error("ParallelOps.d2h not implemented"); }
  synchronize(): void { throw new Error("ParallelOps.synchronize not implemented"); }
  rmsnorm(): void { throw new Error("ParallelOps.rmsnorm not implemented"); }
  fusedAddRmsnorm(): void { throw new Error("ParallelOps.fusedAddRmsnorm not implemented"); }
  fusedNormRope(): void { throw new Error("ParallelOps.fusedNormRope not implemented"); }
  siluAndMul(): void { throw new Error("ParallelOps.siluAndMul not implemented"); }
  linear(): void { throw new Error("ParallelOps.linear not implemented"); }
  embedding(): void { throw new Error("ParallelOps.embedding not implemented"); }
  fill(): void { throw new Error("ParallelOps.fill not implemented"); }
  arange(): void { throw new Error("ParallelOps.arange not implemented"); }
  argmax(): void { throw new Error("ParallelOps.argmax not implemented"); }
  indexSelect(): void { throw new Error("ParallelOps.indexSelect not implemented"); }
  kvCacheWrite(): void { throw new Error("ParallelOps.kvCacheWrite not implemented"); }
  rotaryEmbedding(): void { throw new Error("ParallelOps.rotaryEmbedding not implemented"); }
  fp8LinearDecode(): void { throw new Error("ParallelOps.fp8LinearDecode not implemented"); }
  gdnRecurrentStep(): void { throw new Error("ParallelOps.gdnRecurrentStep not implemented"); }
  gdnPrefill(): void { throw new Error("ParallelOps.gdnPrefill not implemented"); }
  causalConv1d(): void { throw new Error("ParallelOps.causalConv1d not implemented"); }
  causalConv1dUpdate(): void { throw new Error("ParallelOps.causalConv1dUpdate not implemented"); }
  rmsnormGated(): void { throw new Error("ParallelOps.rmsnormGated not implemented"); }
  gateSigmoidMul(): void { throw new Error("ParallelOps.gateSigmoidMul not implemented"); }
  writePinned(): void { throw new Error("ParallelOps.writePinned not implemented"); }
  batchDecodePlan(): void { throw new Error("ParallelOps.batchDecodePlan not implemented"); }
  batchDecodeRun(): void { throw new Error("ParallelOps.batchDecodeRun not implemented"); }
  batchPrefillPagedPlan(): void { throw new Error("ParallelOps.batchPrefillPagedPlan not implemented"); }
  batchPrefillPagedRun(): void { throw new Error("ParallelOps.batchPrefillPagedRun not implemented"); }
  sampleBatch(): void { throw new Error("ParallelOps.sampleBatch not implemented"); }
  graphBeginCapture(): void { throw new Error("ParallelOps.graphBeginCapture not implemented"); }
  graphEndCapture(): number { throw new Error("ParallelOps.graphEndCapture not implemented"); }
  graphInstantiate(): number { throw new Error("ParallelOps.graphInstantiate not implemented"); }
  graphLaunch(): void { throw new Error("ParallelOps.graphLaunch not implemented"); }
  graphDestroy(): void { throw new Error("ParallelOps.graphDestroy not implemented"); }
  graphExecDestroy(): void { throw new Error("ParallelOps.graphExecDestroy not implemented"); }
  mmapOpen(): number { throw new Error("ParallelOps.mmapOpen not implemented"); }
  mmapLoad(): void { throw new Error("ParallelOps.mmapLoad not implemented"); }
  mmapClose(): void { throw new Error("ParallelOps.mmapClose not implemented"); }
}
