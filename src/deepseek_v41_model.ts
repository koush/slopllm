import fs from "node:fs";
import path from "node:path";
import { ChatModel, type ChatCache, type CommonModelConfig, type TokenSelector, type ChunkedPrefillPlan, type ChatTemplateKwargs, type MtpDecodeStepResult } from "./chat_model";
import { resolveCheckpointConfigDir, type CheckpointTensor } from "./checkpoint_source";
import { type DeviceOps, TensorParallelism } from "./device_ops";
import type { ExecutionState, ExecutionWorkspace } from "./execution-workspace";
import { CompressorCarry, DeepseekV41Cache, DSV41_KV_SOURCES as KV_SOURCES, DSV41_INDEX_SOURCES as INDEX_SOURCES } from "./deepseek_v41_cache";
import { DeepseekV41EngramHasher } from "./deepseek_v41_engram";
import { UsingHolder, UsingHolderBase, DisposableSet } from "./using-holder";
import { EagerExecution, type ExecutionManager } from "./execution-manager";
import { PagedKVCache } from "./paged_kv";
import { MemcpyKind } from "./enums";
import { DEEPSEEK_V41_CHAT_TEMPLATE, DeepseekV41ChatParser } from "./deepseek_v41_chat";
import { ParallelTensor } from "./parallel_ops";
import { SafeTensorFile, type TensorMeta } from "./safetensors";
import type { Tensor } from "./tensor";

export const DEEPSEEK_V41_REPO = "local-inference-lab/DeepSeek-V4.1-Flash-lossless-CSF";

export interface DeepseekV41Config extends CommonModelConfig {
  text: Record<string, any>;
  eosTokenId: number;
}

export class LayerHolders extends DisposableSet {
  readonly localSlots = new UsingHolder<Tensor>(undefined!);
  readonly sharedSlots = new UsingHolder<Tensor>(undefined!);
  readonly candidates = new UsingHolder<Tensor>(undefined!);
  readonly compressorInputs = new Map<number, CompressorCarry>();
  readonly compressorOutputs = new Map<number, CompressorCarry>();
  readonly compressorRows = new Map<number, UsingHolder<Tensor>>();

  constructor(private readonly state: ExecutionState) {
    super();
    this.add(this.localSlots);
    this.add(this.sharedSlots);
    this.add(this.candidates);
    const cache = state.cache as DeepseekV41Cache;
    try {
      // Retain each plan's inputs before publishing fresh carry, including
      // plan+plan+run+run. The caller owns these references through execution.
      for (const [source, holder] of cache.compressorCarry) {
        if (holder.value) {
          const input = holder.value.viewClone();
          this.add(input);
          this.compressorInputs.set(source, input);
        }
        const pending = cache.compressorWorkspace.alloc([cache.maxBatch, 1024], "F32");
        this.add(pending);
        const position = cache.compressorWorkspace.alloc([cache.maxBatch], "I32");
        this.add(position);
        if (state.batchSize < cache.maxBatch) {
          using unused = position.narrow(state.batchSize, cache.maxBatch - state.batchSize);
          unused.fill(-1, unused.numElements);
        }
        const output = new CompressorCarry(pending, position);
        this.compressorOutputs.set(source, output);
        holder.replace(output.viewClone());
        if (state.extras.get("dsv41.retainCompressorRows")) {
          const rows = new UsingHolder<Tensor>(undefined!);
          this.add(rows);
          this.compressorRows.set(source, rows);
        }
      }
    } catch (error) {
      this[Symbol.dispose]();
      throw error;
    }
  }

  detachCompressorRows(): Record<string, Tensor> {
    return Object.fromEntries([...this.compressorRows].map(([layer, holder]) => [layer, holder.detach()]));
  }

  replaceCompressorRows(rows: Record<string, Tensor>): void {
    for (const [layer, tensor] of Object.entries(rows)) this.compressorRows.get(Number(layer))!.replace(tensor);
  }

  override [Symbol.dispose](): void {
    super[Symbol.dispose]();
    this.compressorInputs.clear();
    this.compressorOutputs.clear();
    this.compressorRows.clear();
    if (this.state.extras.get("dsv41.layerHolders") === this) this.state.extras.delete("dsv41.layerHolders");
  }
}

interface DsparkPlan {
  cache: PagedKVCache;
  positions: Tensor;
  pages: Tensor;
  slots?: Tensor;
  rows?: Tensor[];
}

// Own a precomputed next-sublayer mix across phased yields. Its residual has
// already transferred to the hidden holder; only these four outputs remain.
type MhcMix = Omit<ReturnType<Tensor["dsv41MhcPre"]>, "residual">;
class MhcMixHolder extends UsingHolderBase<MhcMix> {
  dispose(value: MhcMix) {
    for (const tensor of Object.values(value)) {
      tensor[Symbol.dispose]();
    }
  }
}

export function loadDeepseekV41Config(dir: string): DeepseekV41Config {
  const raw = JSON.parse(fs.readFileSync(path.join(resolveCheckpointConfigDir(dir), "config.json"), "utf8"));
  const c = raw.text_config;
  if (raw.model_type !== "deepseek_v41" || !c || c.hidden_size !== 5120 || c.num_hidden_layers !== 40
    || c.head_dim !== 512 || c.hc_mult !== 4 || c.n_routed_experts !== 384
    || c.num_attention_heads !== 64 || c.num_key_value_heads !== 1 || c.q_lora_rank !== 1280
    || c.o_lora_rank !== 1024 || c.o_groups !== 8 || c.moe_intermediate_size !== 2304 || c.n_shared_experts !== 1
    || c.qk_rope_head_dim !== 64 || raw.quantization_config?.expert_dtype !== "fp4"
    || raw.quantization_config?.scale_fmt !== "ue8m0"
    || JSON.stringify(raw.quantization_config?.weight_block_size) !== "[32,32]"
    || JSON.stringify(c.kv_source_layer_ids) !== "[2,8,14,20]"
    || JSON.stringify(c.index_source_layer_ids) !== "[2,8,14,20,24,28,32,36]"
    || JSON.stringify(c.engram_layer_ids) !== "[1,14]") {
    throw new Error("Expected DeepSeek V4.1 Flash with native MXFP4 experts");
  }
  const rope = c.rope_scaling;
  const ratios = Array.from({ length: 40 }, (_, i) => i < 2 ? 0 : i < 20 ? 2 : 1);
  if (c.rms_norm_eps !== 1e-20 || c.hc_eps !== 1e-6 || c.hc_sinkhorn_iters !== 20
    || c.rope_theta !== 10000 || c.compress_rope_theta !== 160000
    || rope?.rope_type !== "yarn" || rope.factor !== 16 || rope.beta_fast !== 32
    || rope.beta_slow !== 1 || rope.original_max_position_embeddings !== 65536
    || JSON.stringify(c.compress_ratios?.slice(0, 40)) !== JSON.stringify(ratios)
    || c.sliding_window !== 128 || c.index_n_heads !== 32 || c.index_head_dim !== 128 || c.index_topk !== 512
    || c.candidate_source_layer_id !== 20 || c.candidate_topk_blocks !== 2048 || c.candidate_block_size !== 8
    || c.num_experts_per_tok !== 6 || c.scoring_func !== "sqrtsoftplus" || c.topk_method !== "noaux_tc"
    || c.norm_topk_prob !== true || c.routed_scaling_factor !== 1.5 || c.swiglu_limit !== 10) {
    throw new Error("Unsupported DeepSeek V4.1 text numerical configuration");
  }
  return {
    text: c, eosTokenId: raw.eos_token_id,
    hiddenSize: c.hidden_size, intermediateSize: c.moe_intermediate_size,
    numHiddenLayers: c.num_hidden_layers, numNextNPredictLayers: c.num_nextn_predict_layers,
    rmsNormEps: c.rms_norm_eps, vocabSize: c.vocab_size, tieWordEmbeddings: false,
    numAttentionHeads: c.num_attention_heads, numKeyValueHeads: c.num_key_value_heads,
    headDim: c.head_dim, ropeTheta: c.rope_theta, numKeyValueGroups: c.num_attention_heads,
    scaling: c.head_dim ** -0.5, qkRopeHeadDim: c.qk_rope_head_dim, kvLoraRank: 512,
    indexTopk: c.index_topk, maxPositionEmbeddings: c.max_position_embeddings,
  };
}

/** Checkpoint-native text weights with explicit MXFP4 and E8M0 storage types. */
export class DeepseekV41Model extends ChatModel {
  protected readonly loadSyncInterval = 1024;
  readonly eosIds: Set<number>;
  readonly sourceMetadata = new Map<string, TensorMeta>();
  readonly engramRows = new Map<string, number>();
  private hashConstants!: { map: Tensor; primes: Tensor; offsets: Tensor };
  private expertWeights = new Map<string, Tensor[]>();

  constructor(ops: DeviceOps, public readonly cfg: DeepseekV41Config, readonly mtp = false) {
    super(ops);
    this.eosIds = new Set([cfg.eosTokenId]);
    if (![1, 2, 4, 8].includes(ops.worldSize)) {
      throw new Error("DeepSeek V4.1 requires TP 1, 2, 4, or 8");
    }
    if (mtp && (cfg.text.num_nextn_predict_layers !== 3 || cfg.text.dspark_block_size !== 5
      || cfg.text.dspark_noise_token_id !== 128799 || JSON.stringify(cfg.text.dspark_target_layer_ids) !== "[37,38,39]"
      || cfg.text.dspark_markov_rank !== 256 || cfg.text.dspark_n_routed_experts !== 128 || cfg.text.dspark_num_experts_per_tok !== 3)) {
      throw new Error("Unsupported DeepSeek DSpark configuration");
    }
  }

  static async fromPretrained(ops: DeviceOps, dir: string, mtp = false): Promise<DeepseekV41Model> {
    const model = new DeepseekV41Model(ops, loadDeepseekV41Config(dir), mtp);
    try {
      await model.fromPretrained(dir, DEEPSEEK_V41_REPO);
      return model;
    } catch (error) {
      model.free();
      throw error;
    }
  }

  private validateWeights(): void {
    const require = (name: string, shape: number[], dtype?: string) => {
      const meta = this.sourceMetadata.get(name);
      if (!meta || JSON.stringify(meta.shape) !== JSON.stringify(shape) || (dtype && meta.dtype !== dtype)) {
        throw new Error(`DeepSeek V4.1 requires ${name} ${dtype ?? ""} [${shape}]`);
      }
    };
    require("embed.weight", [this.cfg.vocabSize, 5120], "BF16");
    require("head.weight", [this.cfg.vocabSize, 5120], "BF16");
    require("norm.weight", [5120], "BF16");
    for (let layer = 0; layer < (this.mtp ? 43 : 40); layer++) {
      const p = this.layerPrefix(layer) + ".", experts = layer < 40 ? 384 : 128;
      for (const block of ["attn", "ffn"]) {
        require(p + `hc_${block}_fn`, [24, 20480], "F32");
        require(p + `hc_${block}_base`, [24], "F32");
        require(p + `hc_${block}_scale`, [3], "F32");
        require(p + `${block}_norm.weight`, [5120], "BF16");
      }
      require(p + "attn.attn_sink", [64], "F32");
      require(p + "attn.q_norm.weight", [1280], "BF16");
      require(p + "attn.kv_norm.weight", [512], "BF16");
      require(p + "ffn.gate.weight", [experts, 5120], "BF16");
      require(p + "ffn.gate.bias", [experts], "F32");
      for (const [name, n, k] of [["wq_a", 1280, 5120], ["wq_b", 32768, 1280], ["wkv", 512, 5120],
      ["wo_a", 8192, 4096], ["wo_b", 5120, 8192]] as const) {
        require(p + `attn.${name}.weight`, [n, k], "F8_E4M3");
      }
      for (const proj of [1, 2, 3]) {
        require(p + `ffn.shared_experts.w${proj}.weight`, proj === 2 ? [5120, 2304] : [2304, 5120], "F8_E4M3");
      }
      if ([2, 8, 14, 20].includes(layer)) {
        require(p + "attn.compressor.wkv.weight", [512, 5120], "BF16");
        require(p + "attn.compressor.norm.weight", [512], "BF16");
        if (layer !== 20) {
          require(p + "attn.compressor.wgate.weight", [512, 5120], "BF16");
        }
        require(p + "attn.indexer.wk.weight", [128, 512], "BF16");
        require(p + "attn.indexer.k_norm.weight", [128], "BF16");
      }
      if ([2, 8, 14, 20, 24, 28, 32, 36].includes(layer)) {
        require(p + "attn.indexer.wq_b.weight", [4096, 1280], "F8_E4M3");
        require(p + "attn.indexer.weights_proj.weight", [32, 5120], "BF16");
      }
      for (let expert = 0; expert < experts; expert++) {
        for (const proj of [1, 2, 3]) {
          const [n, k] = proj === 2 ? [5120, 2304] : [2304, 5120];
          const base = p + `ffn.experts.${expert}.w${proj}`;
          require(base + ".weight", [n, k / 2], "I8");
          require(base + ".scale", [n, k / 32], "F8_E8M0");
        }
      }
    }
    if (this.mtp) {
      require("mtp.0.main_proj.weight", [5120, 15360], "F8_E4M3");
      require("mtp.0.main_norm.weight", [5120], "BF16");
      require("mtp.2.norm.weight", [5120], "BF16");
      require("mtp.2.markov_head.embed.weight", [this.cfg.vocabSize, 256], "BF16");
      require("mtp.2.markov_head.head.weight", [this.cfg.vocabSize, 256], "BF16");
    }
    for (const [layer, rows] of [[1, 384006168], [14, 384016682]]) {
      const p = `layers.${layer}.engram.`;
      require(p + "embed.weight", [rows, 256], "F8_E4M3");
      require(p + "wkv.weight", [25600, 6144], "F8_E4M3");
      require(p + "q_weight", [4, 5120], "BF16");
      require(p + "k_weight", [4, 5120], "BF16");
    }
    // Block-scaled FP8 linears use 32x32 scales. Engram's embedding table
    // instead has a separate scale for each row's 32 consecutive features.
    for (const [name, meta] of this.sourceMetadata) {
      if (meta.dtype !== "F8_E4M3") {
        continue;
      }
      const [n, k] = meta.shape;
      if (meta.shape.length !== 2) {
        throw new Error(`Unsupported FP8 tensor geometry: ${name}`);
      }
      require(name.replace(/\.weight$/, ".scale"),
        [name.includes(".engram.embed.") ? n : Math.ceil(n / 32), Math.ceil(k / 32)], "F8_E8M0");
    }
  }

  private parallelism(name: string): TensorParallelism {
    if (name.endsWith(".attn.attn_sink")) {
      return TensorParallelism.Column;
    }
    if (name === "embed.weight" || name === "head.weight" || /\.markov_head\.(embed|head)\.weight$/.test(name) || /\.engram\.embed\.(weight|scale)$/.test(name)) {
      return TensorParallelism.Column;
    }
    // q_a/KV outputs are gathered by RMSNorm; indexer Q is gathered before
    // scoring. Keep their quantized payloads and scales column-sharded too.
    if (/\.ffn\.(experts\.\d+|shared_experts)\.w[13]\.(weight|scale)$/.test(name)
      || /\.attn\.(wq_a|wq_b|wkv|wo_a)\.(weight|scale)$/.test(name)
      || /\.attn\.indexer\.wq_b\.(weight|scale)$/.test(name)
      || /\.engram\.wkv\.(weight|scale)$/.test(name)) {
      return TensorParallelism.Column;
    }
    if (/\.ffn\.(experts\.\d+|shared_experts)\.w2\.(weight|scale)$/.test(name)
      || /\.attn\.wo_b\.(weight|scale)$/.test(name)) {
      return TensorParallelism.Row;
    }
    return TensorParallelism.Replicated;
  }

  protected async loadTensor(name: string, meta: TensorMeta, source: CheckpointTensor): Promise<void> {
    if (/^(vision\.|aligner\.|image_)/.test(name) || (!this.mtp && name.startsWith("mtp."))) {
      return;
    }
    if (!/^(layers\.\d+\.|mtp\.[012]\.|embed\.weight$|head\.weight$|norm\.weight$)/.test(name)) {
      throw new Error(`Unrecognized DeepSeek V4.1 tensor ${name}`);
    }
    if (!["BF16", "F32", "F8_E4M3", "F8_E8M0", "I8"].includes(meta.dtype)) {
      throw new Error(`Unsupported DeepSeek V4.1 dtype ${meta.dtype}: ${name}`);
    }
    this.sourceMetadata.set(name, meta);
    const dtype = this.resolveWeightType(name, meta);
    const shape = [...meta.shape];
    const engram = /\.engram\.embed\.(weight|scale)$/.test(name);
    if (engram) {
      this.engramRows.set(name, shape[0]);
      shape[0] = Math.ceil(shape[0] / this.ops.worldSize) * this.ops.worldSize;
    }
    const tensor = this.alloc(shape, dtype, name, this.parallelism(name));
    if (!engram) {
      await this.loadTensorBytes(tensor, source);
      return;
    }
    if (this.loadReplay) {
      return;
    }
    const mapping = source.mapping();
    if (!mapping) {
      throw new Error(`Engram tables require mapped loading: ${name}`);
    }
    const shards = tensor instanceof ParallelTensor ? tensor.shards : [tensor];
    const shardRows = shape[0] / shards.length;
    const rowBytes = shape[1] * SafeTensorFile.dtypeBytes(meta.dtype);
    // Vocabulary-range shards are contiguous on disk. Only the final shard
    // has padding; lookups must compare against the original vocabulary size.
    await Promise.all(shards.map(async (shard, rank) => {
      const rows = Math.min(shardRows, meta.shape[0] - rank * shardRows);
      if (rows < shardRows) {
        using padding = shard.narrow(rows, shardRows - rows);
        padding.fill(0, (shardRows - rows) * shape[1]);
      }
      // Bound pageable-host CUDA staging for tables exceeding 90 GiB each.
      const chunkRows = Math.max(1, Math.floor(64 * 1024 ** 2 / rowBytes));
      for (let row = 0; row < rows; row += chunkRows) {
        const count = Math.min(chunkRows, rows - row);
        using chunk = shard.narrow(row, count);
        await chunk.mmapLoad(mapping.mmapPtr, mapping.offset + (rank * shardRows + row) * rowBytes, count * rowBytes);
      }
    }));
  }

  override get outputWeight(): Tensor {
    return this.tensors.get("head.weight")!;
  }

  override createParser(kwargs: ChatTemplateKwargs = {}) {
    return new DeepseekV41ChatParser(this.tokenizer, kwargs);
  }

  createChatCache(maxPages = 256, maxBatch = 1, maxSeqLen = 4096, _pageSize = 64, pinned = false): ChatCache {
    if (pinned) {
      throw new Error("V4.1 caches cannot be host-offloaded");
    }
    if (!Number.isInteger(maxSeqLen) || maxSeqLen < 1 || maxSeqLen > this.cfg.maxPositionEmbeddings!) {
      throw new Error("Invalid V4.1 cache context length");
    }
    // maxSeqLen is the execution workspace's prefill chunk capacity, not a
    // sequence-length limit. Decode must be able to continue past that chunk.
    return new DeepseekV41Cache(this, maxPages, maxBatch, Math.min(this.cfg.maxPositionEmbeddings!, maxPages * 64));
  }

  forwardModel(state: ExecutionState, layerHolders?: LayerHolders): Tensor {
    return this.runPhased(this.forwardPhased(state, layerHolders));
  }

  override prepareForward(state: ExecutionState): LayerHolders {
    const cache = state.cache;
    if (!(cache instanceof DeepseekV41Cache)) {
      throw new Error("V4.1 requires its own cache");
    }
    const lookback = state.ws.ensureAlloc([state.ws.maxBatch, 3], "I32", `dsv41.lookback:${state.slot}`);
    const host = state.ws.ensureAllocPinned([state.ws.maxBatch, 3], "I32", `dsv41.lookback.host:${state.slot}`);
    const values = new Int32Array(state.ws.maxBatch * 3).fill(2);
    const pageName = `dsv41.pages:${state.slot}:${cache.maxPages}`;
    const pageTable = state.ws.ensureAlloc([state.ws.maxBatch, cache.maxPages], "I32", pageName);
    const pageHost = state.ws.ensureAllocPinned([state.ws.maxBatch, cache.maxPages], "I32", pageName + ".host");
    const pageIds = new Int32Array(state.ws.maxBatch * cache.maxPages);
    for (let batch = 0; batch < state.batchSize; batch++) {
      const start = state.seqKvLens[batch] - state.seqLens[batch];
      if (state.seqKvLens[batch] > cache.maxContext) {
        throw new Error("V4.1 context budget exceeded");
      }
      const history = (state.extras.get("dsv41.history") as number[][] | undefined)?.[batch] ?? cache.sequences[batch].getTokenIds();
      for (let j = 0; j < 3; j++) {
        if (start - 3 + j >= 0) {
          values[batch * 3 + j] = history[start - 3 + j];
        }
      }
      cache.sequences[batch].pages.slice(0, Math.ceil(state.seqKvLens[batch] / 64))
        .forEach((page, j) => pageIds[batch * cache.maxPages + j] = page.id);
    }
    host.withPinnedBuffer(buffer => Buffer.from(values.buffer).copy(buffer));
    lookback.memcpy(host, host.bytes, MemcpyKind.HostToDevice);
    state.extras.set("dsv41.lookback", lookback);
    pageHost.withPinnedBuffer(buffer => Buffer.from(pageIds.buffer).copy(buffer));
    pageTable.memcpy(pageHost, pageHost.bytes, MemcpyKind.HostToDevice);
    state.extras.set("dsv41.pages", pageTable);
    const holders = state.extras.get("dsv41.layerHolders") as LayerHolders | undefined ?? new LayerHolders(state);
    state.extras.set("dsv41.layerHolders", holders);
    return holders;
  }

  override forwardCaptureKey(state: ExecutionState): (string | number)[] {
    const cache = state.cache as DeepseekV41Cache;
    const holders = state.extras.get("dsv41.layerHolders") as LayerHolders;
    return ["dsv41-paged", cache.maxPages, cache.maxBatch, JSON.stringify([
      cache.ckvData[0], ...[...holders.compressorInputs.values(), ...holders.compressorOutputs.values()].flatMap(carry => [carry.pending, carry.pendingPosition]),
      state.positionIds, state.qoIndptrD, state.mlaBatchIndices,
      state.extras.get("dsv41.lookback") as Tensor, state.extras.get("dsv41.pages") as Tensor,
    ].map(tensor => tensor.memoryRanges()))];
  }

  private *forwardWithLayerHolders<T>(state: ExecutionState, layerHolders: LayerHolders | undefined,
    cb: (layerHolders: Required<LayerHolders>) => Generator<void, T, void>): Generator<void, T, void> {
    if (!(state.cache instanceof DeepseekV41Cache) || state.cache.model !== this || state.customMask) {
      throw new Error("V4.1 forward requires its own cache and ordinary causal positions");
    }
    layerHolders ??= state.extras.get("dsv41.layerHolders") as LayerHolders | undefined;
    if (!layerHolders) throw new Error("V4.1 holders must be prepared before forward/capture");
    // Attention temporaries are local to the forward. Compressor references
    // remain owned by the caller until replay/verification has completed.
    using localSlots = layerHolders.localSlots;
    using sharedSlots = layerHolders.sharedSlots;
    using candidates = layerHolders.candidates;
    return yield* cb(layerHolders);
  }

  *forwardPhased(state: ExecutionState, layerHolders?: LayerHolders): Generator<void, Tensor, void> {
    const self = this;
    return yield* this.forwardWithLayerHolders(state, layerHolders, function* (layerHolders) {
      const { totalTokens: tokens } = state, ops = self.ops;
      using rotaryStream = ops.withStream(() => state.positionIds.dsv41RotaryEmbedding());
      using rotary = rotaryStream.result;
      using hidden = new UsingHolder(self.weight("embed.weight").embedding(state.input!, { streams: 4 }));
      const constants = self.hashConstants;
      using hashes = ops.dsv41Hash(state.input!, state.extras.get("dsv41.lookback") as Tensor,
        constants.map, constants.primes, constants.offsets, state.qoIndptrD);
      using previous = new UsingHolder<Tensor>(undefined!);
      using preparedMix = new MhcMixHolder(undefined!);
      using features = new DisposableSet();
      const taps: Tensor[] = [];
      const draftContext = state.extras.get("dsv41.dsparkContext") as DsparkPlan | undefined;
      using mean = !draftContext
        ? undefined
        : state.ws.alloc([tokens, 4], "F32");
      mean?.fill(0.25, mean.numElements);
      rotaryStream.streamWaitEvent();

      for (let layer = 0; layer < 40; layer++) {
        const p = `layers.${layer}`;
        if (layer === 1 || layer === 14) {
          using hashLayer = hashes.narrow(layer === 1 ? 0 : 1, 1);
          using ids = hashLayer.reshape([tokens, 24]);
          using lookup = self.weight(p + ".engram.embed.weight").embedding(ids,
            { originalRows: self.engramRows.get(p + ".engram.embed.weight")! });
          using kv = self.linear(lookup, p + ".engram.wkv");
          hidden.replace(hidden.value.dsv41EngramGate(kv, self.weight(p + ".engram.q_weight"), self.weight(p + ".engram.k_weight")));
        }
        if (draftContext && layer >= 37) {
          const feature = hidden.value.dsv41Collapse(mean!);
          features.add(feature);
          taps.push(feature);
        }
        yield* self.mlaLayerPhased(hidden, previous, preparedMix, layer, state, layerHolders, rotary);
      }
      if (draftContext) {
        self.prepareDsparkContext(state, taps, draftContext, rotary);
      }
      using collapsed = hidden.value.dsv41Collapse(previous.value);
      return collapsed.rmsnorm(self.weight("norm.weight"), 1e-20);
    });
  }

  protected override planPrefillChunk(ws: ExecutionWorkspace, cache: ChatCache,
    inputs: readonly number[][], sampling: TokenSelector = { selectTarget: logits => logits.argmax() }): ChunkedPrefillPlan {
    // DSV4 uses layer-major batching, without alternating-chunk phased prefill.
    const lengths = inputs.map(ids => ids.length);
    if (!lengths.some(Boolean)) {
      return this.createChunkedPrefillPlan([], cache, inputs, (function* () {
        return [];
      })());
    }
    const state = ws.planPrefill(this, inputs.length, lengths, cache);
    state.setInput(inputs);
    const self = this;
    return this.createChunkedPrefillPlan([state], cache, inputs, (function* () {
      using holders = self.prepareForward(state);
      using hidden = yield* self.forwardPhased(state);
      using logits = state.computeLogits(hidden, self);
      using selected = sampling.selectTarget(logits);
      return selected.readInt32LEArray().map((token, i) => lengths[i] ? token : undefined);
    })());
  }

  private weight(name: string): Tensor {
    const weight = this.tensors.get(name);
    if (!weight) {
      throw new Error(`Missing V4.1 weight ${name}`);
    }
    return weight;
  }

  private layerPrefix(layer: number): string {
    return layer < 40 ? `layers.${layer}` : `mtp.${layer - 40}`;
  }

  private linear(input: Tensor, prefix: string): Tensor {
    return input.linear(this.weight(prefix + ".weight"));
  }

  protected override async initializeConstants(): Promise<void> {
    if (!this.tokenizer.chat_template) {
      this.tokenizer.chat_template = DEEPSEEK_V41_CHAT_TEMPLATE;
    }
    this.validateWeights();
    const hasher = new DeepseekV41EngramHasher(this.tokenizer);
    const ints = (name: string, values: ArrayLike<number>) => {
      const tensor = this.alloc([values.length], "I32", `engram.${name}`);
      // Executors replay the same allocations; the loader owns the immutable data.
      if (!this.loadReplay) {
        tensor.h2d(Buffer.from(Int32Array.from(values).buffer));
      }
      return tensor;
    };
    this.hashConstants = {
      map: ints("token_map", hasher.tokenMap), primes: ints("primes", hasher.primes.flat()),
      offsets: ints("offsets", hasher.offsets.flat())
    };
    for (let layer = 0; layer < (this.mtp ? 43 : 40); layer++) {
      for (const projection of [1, 2, 3]) {
        const prefix = `${this.layerPrefix(layer)}.ffn`;
        const weights = Array.from({ length: layer < 40 ? 384 : 128 }, (_, expert) => this.weight(`${prefix}.experts.${expert}.w${projection}.weight`));
        const name = `${prefix}.w${projection}`;
        this.expertWeights.set(name, weights);
      }
    }
    // These uploads must finish before a resident loader exports the arena.
    if (!this.loadReplay) {
      await this.ops.synchronizeAsync();
    }
  }

  private prepareMhc(hidden: UsingHolder<Tensor>, previous: UsingHolder<Tensor>, layer: number,
    sublayer: "attn" | "ffn", fuseNorm: boolean, postMix?: { block: Tensor; post: Tensor; comb: Tensor }): MhcMix {
    const pfx = this.layerPrefix(layer);
    const { residual, ...mix } = hidden.value.dsv41MhcPre(this.weight(`${pfx}.hc_${sublayer}_fn`),
      this.weight(`${pfx}.hc_${sublayer}_base`), this.weight(`${pfx}.hc_${sublayer}_scale`), previous.value,
      1e-20, 1e-6, 20, fuseNorm ? this.weight(`${pfx}.${sublayer}_norm.weight`) : undefined, postMix);
    if (residual) {
      hidden.replace(residual);
    }
    return mix;
  }

  private *mlaLayerPhased(hidden: UsingHolder<Tensor>, previous: UsingHolder<Tensor>, preparedMix: MhcMixHolder,
    layer: number, state: ExecutionState, layerHolders: Required<LayerHolders>, rotary: Tensor, draft?: DsparkPlan): Generator<void, void, void> {
    const pfx = this.layerPrefix(layer);
    const normMode = process.env.SLOPLLM_DSV41_MHC_DECODE_NORM;
    if (normMode !== undefined && normMode !== "0" && normMode !== "1") {
      throw new Error("SLOPLLM_DSV41_MHC_DECODE_NORM must be 0 or 1");
    }
    const fuseNorm = state.totalTokens < 128 && normMode !== "0";
    const postMode = process.env.SLOPLLM_DSV41_MHC_POST_PRE;
    if (postMode !== undefined && postMode !== "0" && postMode !== "1") {
      throw new Error("SLOPLLM_DSV41_MHC_POST_PRE must be 0 or 1");
    }
    const fusePost = state.totalTokens <= 32 && postMode !== "0";
    using oProjBuf = new UsingHolder<Tensor>(undefined!);
    using preparedMlp = new MhcMixHolder(undefined!);
    const attnMix = preparedMix.detach() ?? this.prepareMhc(hidden, previous, layer, "attn", fuseNorm);
    using attnInput = attnMix.input;
    using attnPost = attnMix.post;
    using attnComb = attnMix.comb;
    previous.replace(attnMix.pre);
    using normed = fuseNorm ? attnInput.viewClone() : attnInput.rmsnorm(this.weight(pfx + ".attn_norm.weight"), 1e-20);
    oProjBuf.replace(draft ? this.dsparkAttention(normed, layer, state, draft, rotary) : this.attention(normed, layer, state, layerHolders, rotary));

    yield;
    if (fusePost) {
      preparedMlp.replace(this.prepareMhc(hidden, previous, layer, "ffn", fuseNorm,
        { block: oProjBuf.value, post: attnPost, comb: attnComb }));
    } else {
      hidden.replace(this.ops.mhcPost(hidden.value, oProjBuf.value, attnPost, attnComb));
    }
    oProjBuf.release();
    yield;

    const mlpMix = preparedMlp.detach() ?? this.prepareMhc(hidden, previous, layer, "ffn", fuseNorm);
    using mlpInput = mlpMix.input;
    using mlpPost = mlpMix.post;
    using mlpComb = mlpMix.comb;
    previous.replace(mlpMix.pre);
    using mlpNormed = fuseNorm ? mlpInput.viewClone() : mlpInput.rmsnorm(this.weight(pfx + ".ffn_norm.weight"), 1e-20);
    using downBuf = this.mlpSparse(mlpNormed, layer);
    using rounded = downBuf.toType("BF16");

    yield;
    // Engram changes the next layer's residual before attention pre-mixing.
    // Feature taps only read it, so they can use the materialized fused result.
    const nextLayer = layer + 1;
    if (fusePost && nextLayer < (draft ? 43 : 40) && nextLayer !== 1 && nextLayer !== 14) {
      preparedMix.replace(this.prepareMhc(hidden, previous, nextLayer, "attn", fuseNorm,
        { block: rounded, post: mlpPost, comb: mlpComb }));
    } else {
      hidden.replace(this.ops.mhcPost(hidden.value, rounded, mlpPost, mlpComb));
    }
    yield;
  }

  private attention(normed: Tensor, layer: number, state: ExecutionState, layerHolders: Required<LayerHolders>, rotary: Tensor): Tensor {
    const { localSlots, sharedSlots, candidates } = layerHolders;
    const cache = state.cache as DeepseekV41Cache, ws = state.ws, ops = this.ops;
    const tokens = state.totalTokens, position = state.positionIds;
    const p = `layers.${layer}.attn`, ratio = this.cfg.text.compress_ratios[layer] as 0 | 1 | 2;
    const source = ratio ? KV_SOURCES.filter(i => i <= layer).at(-1)! : -1;
    const indexSource = ratio ? INDEX_SOURCES.filter(i => i <= layer).at(-1)! : -1;
    // Stable per-plan page rows and per-token sequence IDs let every sequence
    // share one launch while preserving independent positions and causal masks.
    const pages = state.extras.get("dsv41.pages") as Tensor, batchIndices = state.mlaBatchIndices;

    using kvcacheIndex = source !== layer
      ? undefined
      : ops.withStream(() => {
        using gateStream = ratio !== 2
          ? undefined
          : ops.withStream(() => normed.linear(this.weight(p + ".compressor.wgate.weight"), { outputType: "F32" }));
        using projected = normed.linear(this.weight(p + ".compressor.wkv.weight"), { outputType: "F32" });
        gateStream?.streamWaitEvent();
        using gate = gateStream?.result;
        using raw = gate ? projected.cat([gate], 1) : projected.viewClone();
        layerHolders.compressorRows.get(layer)?.replace(raw.viewClone());
        const previous = layerHolders.compressorInputs.get(layer);
        const result = raw.dsv41Compress(position, state.qoIndptrD, this.weight(p + ".compressor.norm.weight"), ratio as 1 | 2,
          previous?.pending, previous?.pendingPosition, 1e-20, layerHolders.compressorOutputs.get(layer));
        using latent = result.latent;
        using valid = result.valid;
        using cacheAppend = this.ops.withStream(() => {
          using compressed = latent.dsv41Rope(rotary, true, false, 512, ratio);
          ops.dsv4CacheAppend(cache.compressed(layer), compressed, position, pages, ratio as 1 | 2, batchIndices);
        });
        using keyRaw = latent.linear(this.weight(p + ".indexer.wk.weight"));
        using keyNorm = keyRaw.rmsnorm(this.weight(p + ".indexer.k_norm.weight"), 1e-20);
        using key = keyNorm.dsv41Rope(rotary, true, false, 128, ratio);
        ops.dsv4IndexAppend(cache.indexKeys(layer), key, position, pages, ratio as 1 | 2, batchIndices);
        cacheAppend.streamWaitEvent();
      });

    // Cache preparation only depends on input, so start it before the query path.
    using kvcache = ops.withStream(() => {
      using projectedKv = this.linear(normed, p + ".wkv");
      using kv = projectedKv.rmsnorm(this.weight(p + ".kv_norm.weight"), 1e-20);
      using rotated = kv.dsv41Rope(rotary, ratio > 0);
      ops.dsv4CacheAppend(cache.ckvData[layer], rotated, position, pages, 1, batchIndices);
    });

    using idxWeightsStream = indexSource !== layer
      ? undefined
      : ops.withStream(() => {
        return normed.linear(this.weight(p + ".indexer.weights_proj.weight"));
      });
    using qr = this.linear(normed, p + ".wq_a");
    using qNorm = qr.rmsnorm(this.weight(p + ".q_norm.weight"), 1e-20);
    const projectQuery = () => {
      using projectedQ = this.linear(qNorm, p + ".wq_b");
      using qFlat = projectedQ.dsv41Rope(rotary, ratio > 0);
      return qFlat.reshape([tokens, 64, 512]);
    };
    // Full indexers overlap query expansion with selection. Shared indexers
    // project on the calling stream while cache preparation is still in flight.
    using qStream = indexSource !== layer
      ? undefined
      : ops.withStream(projectQuery);
    using directQuery = qStream
      ? undefined
      : projectQuery();
    using streamedQuery = qStream?.result;
    using idxWeights = idxWeightsStream?.result;
    using localSlotsStream = localSlots.value
      ? undefined
      : ops.withStream(() => ops.dsv4LocalSlots(position, pages, batchIndices));
    if (indexSource === layer) {
      using indexQ = this.linear(qNorm, p + ".indexer.wq_b");
      using rotatedQ = indexQ.dsv41Rope(rotary, true, false, 128);
      using quantizedQ = ops.dsv41IndexQuantize(rotatedQ, position);
      const capacity = Math.min(cache.maxContext, state.isDecode || state.extras.get("dsv41.retainCompressorRows")
        ? state.getGraphVariantPaddedKvLen()
        : Math.max(...state.seqKvLens));
      kvcacheIndex?.streamWaitEvent();
      idxWeightsStream!.streamWaitEvent();
      using scores = ops.dsv4IndexScores(quantizedQ, cache.indexKeys(source), idxWeights!, position, pages, ratio as 1 | 2, capacity, batchIndices);
      if (layer === 20) {
        candidates.replace(ops.dsv41Candidates(scores, position));
      }
      using selected = ops.dsv41Select(scores, position, ratio, layer > 20 ? candidates.value : undefined);
      sharedSlots.release();
      sharedSlots.replace(ops.dsv4TopkToSlots(selected, position, pages, ratio as 1 | 2, batchIndices));
    }
    const global = !ratio
      ? undefined
      : sharedSlots.value;
    if (ratio && !global) {
      throw new Error(`Missing shared MLA slots at layer ${layer}`);
    }
    if (indexSource !== layer) {
      kvcacheIndex?.streamWaitEvent();
    }
    qStream?.streamWaitEvent();
    kvcache.streamWaitEvent();
    if (localSlotsStream) {
      localSlotsStream.streamWaitEvent();
      localSlots.replace(localSlotsStream.result);
    }
    const result = ops.dsv4SparseAttention((streamedQuery ?? directQuery)!, { cache: cache.ckvData[layer], indices: localSlots.value },
      global ? { cache: cache.compressed(source), indices: global } : undefined,
      this.weight(p + ".attn_sink"));
    using output = result.output;
    using lse = result.lse;
    using inverse = output.dsv41Rope(rotary, ratio > 0, true);
    using grouped = inverse.linear(this.weight(p + ".wo_a.weight"), { groups: 8 });
    return this.linear(grouped, p + ".wo_b");
  }

  private mlpSparse(normed: Tensor, layer: number): Tensor {
    const p = `${this.layerPrefix(layer)}.ffn`, topK = layer < 40 ? 6 : 3;
    using sharedStream = this.ops.withStream(() => normed.swiGluMlp({
      gate: this.weight(`${p}.shared_experts.w1.weight`),
      up: this.weight(`${p}.shared_experts.w3.weight`),
      down: this.weight(`${p}.shared_experts.w2.weight`),
    }, 10));
    using shared = sharedStream.result;
    using logits = normed.linear(this.weight(p + ".gate.weight"), { outputType: "F32" });
    const routing = logits.moeRoute({
      scoring: "sqrt_softplus", correctionBias: this.weight(p + ".gate.bias"),
      numExpertsPerToken: topK, scalingFactor: 1.5, normalize: true
    });
    using ids = routing.indices;
    using weightsStream = routing.normalizedWeightsStream;
    using weights = weightsStream.result;
    using flatIds = ids.reshape([ids.numElements]);
    const gate = this.expertWeights.get(`${p}.w1`)!;
    sharedStream.streamWaitEvent();
    return normed.swiGluMlpMoeReduce({
      gate, up: this.expertWeights.get(`${p}.w3`)!, down: this.expertWeights.get(`${p}.w2`)!,
      normalizedWeightsStream: weightsStream, shared, activationLimit: 10, outputType: "F32",
    }, flatIds, topK, flatIds.numElements, gate[0].shape[0], normed.shape[1], p);
  }

  /** Metadata only: cache coordinates are bounded/local, RoPE uses state.positionIds. */
  private planDspark(state: ExecutionState, cache: PagedKVCache, tag: string, block = false): DsparkPlan {
    const { ws, batchSize, totalTokens } = state;
    const upload = (name: string, values: Int32Array, shape: number[]) => {
      const key = `dspark.${tag}.${name}:${batchSize}:${totalTokens}`;
      const tensor = ws.ensureAlloc(shape, "I32", key);
      const host = ws.ensureAllocPinned(shape, "I32", key + ".host");
      host.withPinnedBuffer(buffer => Buffer.from(values.buffer).copy(buffer));
      tensor.memcpy(host, host.bytes, MemcpyKind.HostToDevice);
      return tensor;
    };
    const starts = cache.sequences.map(sequence => sequence.allocLen);
    state.seqLens.forEach((length, b) => cache.allocAppendPages(b, length));
    const positions = new Int32Array(totalTokens), pages = new Int32Array(batchSize * cache.maxPages);
    const slots = !block
      ? undefined
      : new Int32Array(totalTokens * 192).fill(-1);
    let row = 0;
    for (let b = 0; b < batchSize; b++) {
      const sequence = cache.sequences[b], start = starts[b];
      sequence.pages.forEach((page, i) => {
        pages[b * cache.maxPages + i] = page.id;
      });
      const slot = (position: number) => sequence.pages[Math.floor(position / 64)].id * 64 + position % 64;
      for (let t = 0; t < state.seqLens[b]; t++, row++) {
        positions[row] = start + t;
        if (slots) {
          let column = 0;
          for (let p = Math.max(0, start - 128); p < start + 5; p++) {
            slots[row * 192 + column++] = slot(p);
          }
        }
      }
    }
    return {
      cache, positions: upload("positions", positions, [totalTokens]),
      pages: upload("pages", pages, [batchSize, cache.maxPages]),
      slots: !slots
        ? undefined
        : upload("slots", slots, [totalTokens, 192]),
      rows: !block
        ? undefined
        : Array.from({ length: 5 }, (_, d) => upload(`rows.${d}`,
          Int32Array.from({ length: batchSize }, (_, b) => b * 5 + d), [batchSize]))
    };
  }

  private prepareDsparkContext(state: ExecutionState, features: readonly Tensor[], plan: DsparkPlan, rotary: Tensor): void {
    using joined = features[0].cat(features.slice(1), 1);
    using projected = this.linear(joined, "mtp.0.main_proj");
    using context = projected.rmsnorm(this.weight("mtp.0.main_norm.weight"), 1e-20);
    for (let layer = 0; layer < 3; layer++) {
      const prefix = `mtp.${layer}.attn`;
      using projectedKv = this.linear(context, prefix + ".wkv");
      using kv = projectedKv.rmsnorm(this.weight(prefix + ".kv_norm.weight"), 1e-20);
      using rotated = kv.dsv41Rope(rotary, false);
      this.ops.dsv4CacheAppend(plan.cache.ckvData[layer], rotated, plan.positions, plan.pages, 1, state.mlaBatchIndices);
    }
  }

  private dsparkAttention(normed: Tensor, layer: number, state: ExecutionState, plan: DsparkPlan, rotary: Tensor): Tensor {
    const prefix = this.layerPrefix(layer) + ".attn";
    using kvStream = this.ops.withStream(() => {
      using projected = this.linear(normed, prefix + ".wkv");
      using kv = projected.rmsnorm(this.weight(prefix + ".kv_norm.weight"), 1e-20);
      using rotated = kv.dsv41Rope(rotary, false);
      this.ops.dsv4CacheAppend(plan.cache.ckvData[layer - 40], rotated, plan.positions, plan.pages, 1, state.mlaBatchIndices);
    });
    using qr = this.linear(normed, prefix + ".wq_a");
    using qNorm = qr.rmsnorm(this.weight(prefix + ".q_norm.weight"), 1e-20);
    using projectedQ = this.linear(qNorm, prefix + ".wq_b");
    using rotatedQ = projectedQ.dsv41Rope(rotary, false);
    using q = rotatedQ.reshape([state.totalTokens, 64, 512]);
    kvStream.streamWaitEvent();
    const attention = this.ops.dsv4SparseAttention(q, { cache: plan.cache.ckvData[layer - 40], indices: plan.slots! },
      undefined, this.weight(prefix + ".attn_sink"));
    using output = attention.output;
    using lse = attention.lse;
    using inverse = output.dsv41Rope(rotary, false, true);
    using grouped = inverse.linear(this.weight(prefix + ".wo_a.weight"), { groups: 8 });
    return this.linear(grouped, prefix + ".wo_b");
  }

  /** Five parallel transformer positions, followed by five cheap conditional
   * Markov selections. Every vocabulary bias is applied before argmax. */
  private forwardDspark(state: ExecutionState, plan: DsparkPlan): Tensor {
    using rotaryStream = this.ops.withStream(() => state.positionIds.dsv41RotaryEmbedding());
    using rotary = rotaryStream.result;
    using hidden = new UsingHolder(this.weight("embed.weight").embedding(state.input!, { streams: 4 }));
    using previous = new UsingHolder<Tensor>(undefined!);
    using preparedMix = new MhcMixHolder(undefined!);
    rotaryStream.streamWaitEvent();
    for (let layer = 40; layer < 43; layer++) {
      this.runPhased(this.mlaLayerPhased(hidden, previous, preparedMix, layer, state, {} as Required<LayerHolders>, rotary, plan));
    }
    using collapsed = hidden.value.dsv41Collapse(previous.value);
    using normed = collapsed.rmsnorm(this.weight("mtp.2.norm.weight"), 1e-20);
    using logits = normed.linear(this.outputWeight, { outputType: "F32" });
    using predecessor = new UsingHolder(state.ws.alloc([state.batchSize], "I32"));
    predecessor.value.memcpy2d(0, 4, state.input!, 0, 20, 4, state.batchSize, MemcpyKind.DeviceToDevice);
    const proposals = state.ws.alloc([state.batchSize, 5], "I32");
    for (let d = 0; d < 5; d++) {
      using embedding = this.weight("mtp.2.markov_head.embed.weight").embedding(predecessor.value);
      using bias = embedding.linear(this.weight("mtp.2.markov_head.head.weight"), { outputType: "F32" });
      using base = logits.indexSelect(plan.rows![d]);
      using adjusted = base.add(bias);
      predecessor.replace(adjusted.argmax());
      proposals.memcpy2d(d * 4, 20, predecessor.value, 0, 4, 4, state.batchSize, MemcpyKind.DeviceToDevice);
    }
    return proposals;
  }

  override async *generateMtpDecode(ws: ExecutionWorkspace, chatCache: ChatCache, depth: number,
    executionManager: ExecutionManager = new EagerExecution(),
    sampling: TokenSelector = { selectTarget: logits => logits.argmax() }): AsyncGenerator<MtpDecodeStepResult, void, void> {
    const cache = chatCache as DeepseekV41Cache, draft = cache.draft;
    if (!this.mtp || !draft || !Number.isInteger(depth) || depth < 1 || depth > 5) {
      throw new Error("DSpark requires --mtp depth 1..5");
    }
    const sequences = cache.sequences.slice(), batch = sequences.length, width = depth + 1;
    if (!batch || ws.maxSeqLen < batch * Math.max(5, width)) {
      throw new Error("DSpark workspace must fit draft and verification blocks");
    }
    const sampled = sampling.mtpEnabled === true;
    if (sampled && (!sampling.prepareDraft || !sampling.prepareDeterministicVerification || !sampling.verify)) {
      throw new Error("DSpark requires deterministic proposal verification");
    }
    // The aligned context replay below restores compressor state as well.
    const nextTokens = await super.prepareDecodeInput(ws, cache, sampling);
    const histories = sequences.map(sequence => sequence.getTokenIds());
    let committed = sequences.map(sequence => sequence.allocLen), speculative = false;
    const hostKey = `dspark.verify:${batch}:${depth}`;
    const inputHost = ws.ensureAllocPinned([batch, width], "I32", hostKey + ".input");
    const selectedHost = ws.ensureAllocPinned([batch, width], "I32", hostKey + ".selected");
    const acceptedHost = !sampled
      ? undefined
      : ws.ensureAllocPinned([batch], "I32", hostKey + ".accepted");
    draft.reset(batch);
    try {
      // Rebuild a bounded suffix on generator entry. Align its start to a full
      // compression pair; the earlier target KV supplies the causal prefix.
      const starts = histories.map(history => Math.max(0, history.length - 128) & ~1);
      sequences.forEach((sequence, b) => sequence.truncate(starts[b]));
      committed = starts.slice();
      const offsets = starts.slice();
      while (offsets.some((offset, b) => offset < histories[b].length)) {
        ws.assertClear();
        ws.clearTracking();
        const budget = Math.max(1, Math.floor(ws.maxSeqLen / batch));
        const chunks = histories.map((history, b) => history.slice(offsets[b], offsets[b] + budget));
        speculative = true;
        const target = ws.planPrefill(this, batch, chunks.map(chunk => chunk.length), cache);
        target.setInput(chunks);
        using targetHolders = this.prepareForward(target);
        target.extras.set("dsv41.dsparkContext", this.planDspark(target, draft, "recondition"));
        {
          using hidden = this.forwardModel(target);
        }
        await this.ops.synchronizeAsync();
        chunks.forEach((chunk, b) => {
          offsets[b] += chunk.length;
          cache.reportTokens(b, chunk, offsets[b] < histories[b].length ? histories[b][offsets[b]] : nextTokens[b]);
        });
        committed = sequences.map(sequence => sequence.allocLen);
        speculative = false;
      }
      for (; ;) {
        ws.assertClear();
        ws.clearTracking();
        if (cache.sequences.length !== batch || sequences.some((sequence, b) => cache.sequences[b] !== sequence || sequence.allocLen !== committed[b])) {
          throw new Error("DSpark batch changed; close and restart the generator");
        }
        for (const sequence of draft.sequences) {
          sequence.discardPrefixPages(Math.max(0, Math.floor((sequence.allocLen - 128) / 64)));
        }
        const draftLengths = draft.sequences.map(sequence => sequence.allocLen);
        speculative = true;
        const block = ws.planPrefill(this, batch, Array(batch).fill(5), cache);
        block.setInput(nextTokens.map(token => [token, 128799, 128799, 128799, 128799]));
        const draftPlan = this.planDspark(block, draft, "block", true);
        sequences.forEach((sequence, b) => {
          sequence.allocLen = committed[b];
        });
        draft.sequences.forEach((sequence, b) => {
          sequence.allocLen = draftLengths[b];
        });
        const verification = ws.planPrefill(this, batch, Array(batch).fill(width), cache);
        verification.setInput(nextTokens.map(token => [token, ...Array(depth).fill(0)]));
        verification.extras.set("dsv41.retainCompressorRows", true);
        using verificationHolders = this.prepareForward(verification);
        const context = this.planDspark(verification, draft, "context");
        verification.extras.set("dsv41.dsparkContext", context);
        if (sampled) {
          sampling.prepareDraft!(batch, depth);
        }
        const execution = executionManager.execute({
          states: [block, verification], inputs: {},
          key: ["dspark", depth, sampling.captureKey ?? "greedy", sampling.mtpCaptureKey ?? "greedy",
            ...this.forwardCaptureKey(verification), JSON.stringify(draft.ckvData.map(t => t.memoryRanges()))]
        }, () => {
          using proposed = this.forwardDspark(block, draftPlan);
          using prefix = ws.alloc([batch, depth], "I32");
          prefix.memcpy2d(0, depth * 4, proposed, 0, 20, depth * 4, batch, MemcpyKind.DeviceToDevice);
          verification.input!.memcpy2d(4, width * 4, prefix, 0, depth * 4, depth * 4, batch, MemcpyKind.DeviceToDevice);
          using inputCopy = this.ops.withStream(() => {
            using input = verification.input!.viewClone();
            inputHost.memcpy(input, inputHost.bytes, MemcpyKind.DeviceToHost);
          });
          if (sampled) {
            sampling.prepareDeterministicVerification!(prefix, batch);
          }
          using hidden = this.forwardModel(verification);
          using logits = verification.computeLogits(hidden, this, true);
          const result = sampled ? sampling.verify!(logits) : { tokens: sampling.selectTarget(logits), numAccepted: undefined };
          using tokens = result.tokens;
          using counts = result.numAccepted;
          using outputCopy = this.ops.withStream(() => {
            using selected = tokens.viewClone();
            selectedHost.memcpy(selected, selectedHost.bytes, MemcpyKind.DeviceToHost);
            using accepted = counts?.viewClone();
            if (accepted) {
              acceptedHost!.memcpy(accepted, acceptedHost!.bytes, MemcpyKind.DeviceToHost);
            }
          });
          inputCopy.streamWaitEvent();
          outputCopy.streamWaitEvent();
          return { compressorRows: verificationHolders.detachCompressorRows() };
        });
        verificationHolders.replaceCompressorRows(execution.result.compressorRows);
        await this.ops.synchronizeAsync();
        const input = inputHost.readPinnedBuffer(), selected = selectedHost.readPinnedBuffer(), counts = acceptedHost?.readPinnedBuffer();
        const accepted: number[] = [], tokens: number[][] = [];
        for (let b = 0; b < batch; b++) {
          const base = b * width;
          let count = counts ? counts.readInt32LE(b * 4) : 0;
          if (!counts) {
            while (count < depth && input.readInt32LE((base + count + 1) * 4) === selected.readInt32LE((base + count) * 4)) {
              count++;
            }
          }
          if (count < 0 || count > depth) {
            throw new Error("Invalid DSpark acceptance count");
          }
          const prefix = Array.from({ length: count + 1 }, (_, t) => input.readInt32LE((base + t) * 4));
          const replacement = selected.readInt32LE((base + count) * 4);
          sequences[b].truncate(committed[b] + count + 1);
          draft.sequences[b].truncate(draftLengths[b] + count + 1);
          cache.reportTokens(b, prefix, replacement);
          nextTokens[b] = replacement;
          accepted.push(count);
          tokens.push([...prefix.slice(1), replacement]);
        }
        cache.commitCompressor(verification, verificationHolders, accepted);
        verificationHolders[Symbol.dispose]();
        committed = sequences.map(sequence => sequence.allocLen);
        speculative = false;
        ws.assertClear();
        ws.clearTracking();
        yield { tokens, numAccepted: accepted, numDraftTokens: depth, warmup: execution.warmup };
      }
    } finally {
      await this.ops.synchronizeAsync();
      if (speculative) {
        sequences.forEach((sequence, b) => sequence.truncate(committed[b]));
      }
      draft.reset(0);
    }
  }

  override planChunkedPrefill(ws: ExecutionWorkspace, cache: ChatCache, inputs: readonly number[][],
    chunkSize = 8192, sampling: TokenSelector = { selectTarget: logits => logits.argmax() }): ChunkedPrefillPlan {
    const sequences = cache.getPagedKV().sequences;
    if (sequences.length !== inputs.length || sequences.some(sequence => sequence.reportedTokenCount() !== sequence.allocLen)) {
      throw new Error("DSV4 prefill requires one input row per committed sequence");
    }
    const overlaps = sequences.map((sequence, i) => inputs[i].length && sequence.allocLen % 2 ? sequence.getTokenIds().at(-1) : undefined);
    if (chunkSize === 1) {
      const chunk = this.selectPrefillChunk(inputs, 1);
      const overlap = sequences.map((_, i) => chunk.prefillInputIdsList[i].length && overlaps[i] !== undefined ? [overlaps[i]!] : []);
      if (overlap.some(row => row.length)) {
        const history = sequences.map(sequence => sequence.getTokenIds());
        for (let i = 0; i < sequences.length; i++) {
          if (overlap[i].length) {
            sequences[i].truncate(sequences[i].allocLen - 1);
          }
        }
        const first = ws.planPrefill(this, sequences.length, overlap.map(row => row.length), cache);
        first.setInput(overlap);
        const second = ws.planPrefill(this, sequences.length, chunk.prefillInputIdsList.map(row => row.length), cache);
        second.setInput(chunk.prefillInputIdsList);
        second.extras.set("dsv41.history", history);
        const self = this;
        const committed = chunk.prefillInputIdsList.map((row, i) => [...overlap[i], ...row]);
        const plan = this.createChunkedPrefillPlan([first, second], cache, committed, (function* () {
          using firstHolders = self.prepareForward(first);
          using secondHolders = self.prepareForward(second);
          {
            using hidden = yield* self.forwardPhased(first);
          }
          using hidden = yield* self.forwardPhased(second);
          using logits = second.computeLogits(hidden, self);
          using selected = sampling.selectTarget(logits);
          return selected.readInt32LEArray().map((token, i) => chunk.prefillInputIdsList[i].length ? token : undefined);
        })());
        plan.remainingInputIdsList = chunk.remainingInputIdsList;
        return plan;
      }
      const plan = this.planPrefillChunk(ws, cache, chunk.prefillInputIdsList, sampling);
      plan.remainingInputIdsList = chunk.remainingInputIdsList;
      return plan;
    }
    const chunk = this.selectPrefillChunk(inputs, chunkSize, overlaps);
    for (let i = 0; i < sequences.length; i++) {
      if (chunk.prefillInputIdsList[i].length && overlaps[i] !== undefined) {
        sequences[i].truncate(sequences[i].allocLen - 1);
      }
    }
    const plan = this.planPrefillChunk(ws, cache, chunk.prefillInputIdsList, sampling);
    plan.remainingInputIdsList = chunk.remainingInputIdsList;
    return plan;
  }

  protected override async prepareDecodeInput(ws: ExecutionWorkspace, cache: ChatCache, sampling: TokenSelector): Promise<number[]> {
    const targets = await super.prepareDecodeInput(ws, cache, sampling);
    const sequences = cache.getPagedKV().sequences;
    const overlap = sequences.map(sequence => sequence.allocLen % 2 ? [sequence.getTokenIds().at(-1)!] : []);
    if (!overlap.some(row => row.length)) {
      return targets;
    }
    for (let i = 0; i < sequences.length; i++) {
      if (overlap[i].length) {
        sequences[i].truncate(sequences[i].allocLen - 1);
      }
    }
    ws.assertClear();
    ws.clearTracking();
    {
      using tracking = ws.startTracking();
      const state = ws.planPrefill(this, sequences.length, overlap.map(row => row.length), cache);
      state.setInput(overlap);
      using holders = this.prepareForward(state);
      using hidden = this.forwardModel(state);
      await this.ops.synchronizeAsync();
      for (let i = 0; i < sequences.length; i++) {
        if (overlap[i].length) {
          cache.reportTokens(i, overlap[i], targets[i]);
        }
      }
    }
    ws.assertClear();
    ws.clearTracking();
    return targets;
  }

}
