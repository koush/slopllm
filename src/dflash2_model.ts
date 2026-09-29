import fs from "node:fs";
import path from "node:path";
import { ChatModel, type ChatCache, type CommonModelConfig } from "./chat_model";
import { type DeviceOps, MaskMode, TensorParallelism } from "./device_ops";
import { MemcpyKind } from "./enums";
import { type ExecutionState, type ExecutionWorkspace } from "./execution-workspace";
import { resolveModelPath } from "./model_path";
import { PagedKVCache } from "./paged_kv";
import { SafeTensorFile, type TensorMeta } from "./safetensors";
import { Tensor } from "./tensor";
import { UsingHolder } from "./using-holder";

export const DFLASH2_REPO = "incoai/GLM-5.3-DFlash2";

export interface Dflash2Config extends CommonModelConfig {
  blockSize: number;
  convGroupSize: number;
  slidingWindow: number;
  targetLayerIds: number[];
  maskTokenId: number;
  selectorRank: number;
  selectorTopK: number;
  ropeInterleaved: boolean;
}

export type Dflash2Trace = (name: string, value: Tensor) => void;

/** Standalone draft weight workspace. Inputs are target features and block embeddings,
 * not target-model objects. The checkpoint does not contain an embedding or LM head. */
export class Dflash2Model extends ChatModel {
  readonly cfg: Dflash2Config;
  readonly invFreq: Tensor;
  private expectedWeights!: Map<string, number[]>;

  get eosIds(): Set<number> {
    throw new Error("DFlash2 has no standalone EOS policy");
  }

  createChatCache(): ChatCache {
    throw new Error("DFlash2 does not support chat generation; use createCache()");
  }

  forwardPhased(_state: ExecutionState): Generator<void, Tensor, void> {
    throw new Error("DFlash2 requires target features and block embeddings; use forwardDflash2Phased()");
  }

  private constructor(ops: DeviceOps, raw: any) {
    super(ops);
    const d = raw.dflash_config;
    if (!raw.architectures?.includes("DFlash2DraftModel") || raw.is_causal !== false
      || d?.conv_kernel_size !== 2 || !Array.isArray(d.target_layer_ids) || d.target_layer_ids.length !== 6
      || raw.attention_bias || raw.hidden_act !== "silu" || raw.dtype !== "bfloat16"
      || raw.layer_types?.some((t: string) => t !== "sliding_attention")) {
      throw new Error("Unsupported DFlash2 configuration (expected BF16, six target features, two-tap conv, non-causal sliding GQA)");
    }
    this.cfg = {
      hiddenSize: raw.hidden_size, intermediateSize: raw.intermediate_size,
      numHiddenLayers: raw.num_hidden_layers, rmsNormEps: raw.rms_norm_eps,
      vocabSize: raw.vocab_size, tieWordEmbeddings: false,
      numAttentionHeads: raw.num_attention_heads, numKeyValueHeads: raw.num_key_value_heads,
      headDim: raw.head_dim, ropeTheta: raw.rope_parameters.rope_theta,
      numKeyValueGroups: raw.num_attention_heads / raw.num_key_value_heads,
      scaling: raw.head_dim ** -0.5, maxPositionEmbeddings: raw.max_position_embeddings,
      blockSize: d.block_size, convGroupSize: d.conv_group_size,
      slidingWindow: raw.sliding_window, targetLayerIds: d.target_layer_ids,
      maskTokenId: d.mask_token_id, selectorRank: d.selector_rank, selectorTopK: d.selector_top_k,
      ropeInterleaved: !(raw.rope_is_neox_style ?? raw.is_neox_style ?? true),
    };
    const c = this.cfg;
    for (const value of [c.hiddenSize, c.intermediateSize, c.numHiddenLayers, c.numAttentionHeads,
      c.numKeyValueHeads, c.headDim, c.blockSize, c.convGroupSize, c.slidingWindow, c.selectorRank, c.selectorTopK]) {
      if (!Number.isSafeInteger(value) || value < 1) throw new Error("Invalid DFlash2 dimensions");
    }
    if (c.blockSize < 2 || c.blockSize > c.slidingWindow || c.hiddenSize % c.convGroupSize
      || c.numAttentionHeads % c.numKeyValueHeads || c.numKeyValueHeads % ops.worldSize
      || c.numAttentionHeads % ops.worldSize || c.intermediateSize % ops.worldSize
      || c.headDim !== 128) throw new Error("Unsupported DFlash2 dimensions or TP size");
    this.invFreq = this.alloc([c.headDim / 2], "F32", "invFreq");
    const values = Float32Array.from({ length: c.headDim / 2 }, (_, i) => c.ropeTheta ** (-2 * i / c.headDim));
    this.invFreq.h2d(Buffer.from(values.buffer));
  }

  static async fromPretrained(ops: DeviceOps, modelDir = DFLASH2_REPO): Promise<Dflash2Model> {
    if (!fs.existsSync(modelDir)) modelDir = resolveModelPath(modelDir);
    const model = new Dflash2Model(ops, JSON.parse(fs.readFileSync(path.join(modelDir, "config.json"), "utf8")));
    try {
      model.expectedWeights = model.weightShapes();
      // Reuse the shared weight loader without initializing a target tokenizer.
      await model.loadWeights(modelDir);
      if (model.expectedWeights.size) throw new Error(`Missing DFlash2 weights: ${[...model.expectedWeights.keys()].join(", ")}`);
      model.freeze();
      return model;
    } catch (e) {
      await ops.synchronizeAsync();
      model.free();
      throw e;
    }
  }

  protected async loadTensor(name: string, meta: TensorMeta, st: SafeTensorFile, mmapPtr: number): Promise<void> {
    const shape = this.expectedWeights.get(name);
    if (!shape || meta.dtype !== "BF16" || JSON.stringify(shape) !== JSON.stringify(meta.shape)) {
      throw new Error(`Unexpected DFlash2 weight ${name}: ${meta.dtype} [${meta.shape}]`);
    }
    const par = /\.(q_proj|k_proj|v_proj|gate_proj|up_proj)\.weight$/.test(name)
      ? TensorParallelism.Column : /\.(o_proj|down_proj)\.weight$/.test(name)
        ? TensorParallelism.Row : TensorParallelism.Replicated;
    const tensor = this.alloc(shape, "BF16", name, par);
    this.expectedWeights.delete(name);
    await tensor.mmapLoad(mmapPtr, st.dataStart + meta.dataOffsets[0], tensor.bytes);
  }

  private weightShapes(): Map<string, number[]> {
    const c = this.cfg, h = c.hiddenSize, hd = c.headDim;
    const shapes = new Map<string, number[]>([
      ["fc.weight", [h, 6 * h]], ["hidden_norm.weight", [h]], ["norm.weight", [h]],
      ["candidate_selector.hidden_projection.weight", [c.selectorRank, h]],
      ["candidate_selector.predecessor_codebook", [c.vocabSize, c.selectorRank]],
      ["candidate_selector.successor_codebook", [c.vocabSize, c.selectorRank]],
    ]);
    for (let i = 0; i < c.numHiddenLayers; i++) {
      const p = `layers.${i}`;
      for (const name of ["input_layernorm", "post_attention_layernorm"]) shapes.set(`${p}.${name}.weight`, [h]);
      for (const name of ["attention_conv", "mlp_conv"]) {
        shapes.set(`${p}.${name}.base_kernel`, [2, 2, h]);
        shapes.set(`${p}.${name}.kernel_projection.weight`, [4 * h / c.convGroupSize, h]);
      }
      shapes.set(`${p}.self_attn.q_proj.weight`, [c.numAttentionHeads * hd, h]);
      for (const name of ["k", "v"]) shapes.set(`${p}.self_attn.${name}_proj.weight`, [c.numKeyValueHeads * hd, h]);
      shapes.set(`${p}.self_attn.o_proj.weight`, [h, c.numAttentionHeads * hd]);
      for (const name of ["q", "k"]) shapes.set(`${p}.self_attn.${name}_norm.weight`, [hd]);
      for (const name of ["gate", "up"]) shapes.set(`${p}.mlp.${name}_proj.weight`, [c.intermediateSize, h]);
      shapes.set(`${p}.mlp.down_proj.weight`, [h, c.intermediateSize]);
    }
    return shapes;
  }

  private weight(name: string): Tensor { return this.tensors.get(name)!; }

  /** Candidate logits/IDs can come from any externally supplied head or fixture.
   * Returns the complete pairwise lattice and the greedy path, all on device. */
  selectCandidates(hidden: Tensor, ids: Tensor, logits: Tensor, anchors: Tensor): { scores: Tensor, tokens: Tensor } {
    if (ids.shape[1] !== this.cfg.selectorTopK) throw new Error("DFlash2 selector candidate count mismatch");
    using gates = hidden.linear(this.weight("candidate_selector.hidden_projection.weight"));
    return gates.dflash2Select(ids, logits, this.weight("candidate_selector.predecessor_codebook"),
      this.weight("candidate_selector.successor_codebook"), anchors, this.cfg.blockSize - 1);
  }

  createCache(maxPages: number, maxBatch: number): PagedKVCache {
    return new PagedKVCache(this.ops, this.cfg.numKeyValueHeads, this.cfg.headDim,
      this.cfg.numHiddenLayers, maxPages, maxBatch);
  }

  planBlock(ws: ExecutionWorkspace, cache: PagedKVCache): ExecutionState {
    return ws.planPrefill(this, cache.sequences.length, cache.sequences.map(() => this.cfg.blockSize), cache,
      { mode: MaskMode.None, windowLeft: this.cfg.slidingWindow - 1 });
  }

  /** Writes target-derived context to the ordinary GQA cache. Its state is a
   * normal prefill plan for the supplied feature rows, possibly just an append. */
  *prepareContextPhased(state: ExecutionState, hidden: readonly Tensor[], trace?: Dflash2Trace): Generator<void, void, void> {
    const c = this.cfg;
    if (state.isDecode || state.totalTokens < 1 || hidden.length !== 6 || hidden.some(t => t.type !== "BF16"
      || t.shape.length !== 2 || t.shape[0] !== state.totalTokens || t.shape[1] !== c.hiddenSize)) {
      throw new Error("DFlash2 context requires six BF16 [contextTokens, hiddenSize] tensors in target-layer order");
    }
    using joined = hidden[0].cat(hidden.slice(1), 1);
    using projected = joined.linear(this.weight("fc.weight"));
    using context = projected.rmsnorm(this.weight("hidden_norm.weight"), c.rmsNormEps);
    trace?.("context", context);
    const rope = state.rotaryEmbedding(this.invFreq);
    using cos = rope.cos; using sin = rope.sin;
    for (let i = 0; i < c.numHiddenLayers; i++) {
      const p = `layers.${i}.self_attn`;
      using k = context.linear(this.weight(`${p}.k_proj.weight`));
      using v = context.linear(this.weight(`${p}.v_proj.weight`));
      using kr = k.fusedNormRope(this.weight(`${p}.k_norm.weight`), cos, sin, c.rmsNormEps,
        c.headDim, state.totalTokens, 1, undefined, c.ropeInterleaved);
      state.kvCacheWrite(kr, v, i, c.numKeyValueHeads, c.headDim);
      yield;
    }
  }

  *forwardDflash2Phased(contextState: ExecutionState, blockState: ExecutionState,
    targetHidden: readonly Tensor[], blockEmbeddings: Tensor, trace?: Dflash2Trace): Generator<void, Tensor, void> {
    if (contextState.cache !== blockState.cache || contextState.ws !== blockState.ws
      || contextState.batchSize !== blockState.batchSize) throw new Error("DFlash2 plans must share a workspace, cache, and batch");
    yield* this.prepareContextPhased(contextState, targetHidden, trace);
    return yield* this.forwardBlockPhased(blockState, blockEmbeddings, trace);
  }

  /** Returns [batch * (blockSize - 1), hiddenSize] normalized prediction rows.
   * Caller supplies eight embeddings per request; no target model is needed. */
  *forwardBlockPhased(state: ExecutionState, blockEmbeddings: Tensor, trace?: Dflash2Trace): Generator<void, Tensor, void> {
    const c = this.cfg, rows = state.totalTokens;
    if (state.isDecode || state.seqLens.some(n => n !== c.blockSize)
      || state.customMask?.mode !== MaskMode.None || state.customMask?.windowLeft !== c.slidingWindow - 1
      || blockEmbeddings.type !== "BF16" || blockEmbeddings.shape.length !== 2
      || blockEmbeddings.shape[0] !== rows || blockEmbeddings.shape[1] !== c.hiddenSize) {
      throw new Error("DFlash2 forward requires a non-causal sliding block plan and BF16 block embeddings");
    }
    using residual = new UsingHolder(blockEmbeddings.viewClone());
    using normed = new UsingHolder(residual.value.rmsnorm(this.weight("layers.0.input_layernorm.weight"), c.rmsNormEps));
    const rope = state.rotaryEmbedding(this.invFreq);
    using cos = rope.cos; using sin = rope.sin;
    for (let i = 0; i < c.numHiddenLayers; i++) {
      const result = yield* this.forwardLayerPhased(state, i, normed.value, residual.value, cos, sin, trace);
      residual.replace(result.residual); normed.replace(result.normed);
      yield;
    }
    const predictions = state.ws.alloc([state.batchSize * (c.blockSize - 1), c.hiddenSize], "BF16");
    const rowBytes = c.hiddenSize * 2;
    predictions.memcpy2d(0, (c.blockSize - 1) * rowBytes, normed.value, rowBytes,
      c.blockSize * rowBytes, (c.blockSize - 1) * rowBytes, state.batchSize, MemcpyKind.DeviceToDevice);
    trace?.("predictions", predictions);
    return predictions;
  }

  *forwardLayerPhased(state: ExecutionState, layer: number, normed: Tensor, residual: Tensor,
    cos: Tensor, sin: Tensor, trace?: Dflash2Trace): Generator<void, { normed: Tensor, residual: Tensor }, void> {
    const c = this.cfg, rows = state.totalTokens, p = `layers.${layer}`;
    if (!Number.isInteger(layer) || layer < 0 || layer >= c.numHiddenLayers) throw new Error("Invalid draft layer");
    using ac = normed.linear(this.weight(`${p}.attention_conv.kernel_projection.weight`));
    using ax = normed.dflash2Conv(ac, this.weight(`${p}.attention_conv.base_kernel`), c.blockSize, c.convGroupSize, 0);
    trace?.(`${p}.attention_input`, ax);
    using q = ax.linear(this.weight(`${p}.self_attn.q_proj.weight`));
    using k = ax.linear(this.weight(`${p}.self_attn.k_proj.weight`));
    using v = ax.linear(this.weight(`${p}.self_attn.v_proj.weight`));
    using qr = q.fusedNormRope(this.weight(`${p}.self_attn.q_norm.weight`), cos, sin, c.rmsNormEps, c.headDim, rows, 1, undefined, c.ropeInterleaved);
    using kr = k.fusedNormRope(this.weight(`${p}.self_attn.k_norm.weight`), cos, sin, c.rmsNormEps, c.headDim, rows, 1, undefined, c.ropeInterleaved);
    state.kvCacheWrite(kr, v, layer, c.numKeyValueHeads, c.headDim);
    using attention = state.ws.flashPrefillPaged(state, qr, layer, c.numAttentionHeads, c.numKeyValueHeads,
      c.headDim, c.headDim, rows * c.headDim, MaskMode.None, c.scaling);
    using flattened = attention.reshape([rows, c.numAttentionHeads * c.headDim]);
    using projected = flattened.outputProj(this.weight(`${p}.self_attn.o_proj.weight`));
    using projectedReplicated = projected.replicate();
    using ao = projectedReplicated.dflash2Conv(ac, this.weight(`${p}.attention_conv.base_kernel`), c.blockSize, c.convGroupSize, 1);
    const attn = residual.fusedAddRmsnorm(ao, this.weight(`${p}.post_attention_layernorm.weight`), c.rmsNormEps);
    using ar = attn.residual;
    using an = attn.normed;
    yield;
    using mc = an.linear(this.weight(`${p}.mlp_conv.kernel_projection.weight`));
    using mx = an.dflash2Conv(mc, this.weight(`${p}.mlp_conv.base_kernel`), c.blockSize, c.convGroupSize, 0);
    using mlp = mx.swiGluMlp({ gate: this.weight(`${p}.mlp.gate_proj.weight`), up: this.weight(`${p}.mlp.up_proj.weight`), down: this.weight(`${p}.mlp.down_proj.weight`) });
    using mlpReplicated = mlp.replicate();
    using mo = mlpReplicated.dflash2Conv(mc, this.weight(`${p}.mlp_conv.base_kernel`), c.blockSize, c.convGroupSize, 1);
    const nextNorm = layer + 1 < c.numHiddenLayers ? `layers.${layer + 1}.input_layernorm.weight` : "norm.weight";
    const result = ar.fusedAddRmsnorm(mo, this.weight(nextNorm), c.rmsNormEps);
    using resultNormed = new UsingHolder(result.normed);
    using resultResidual = new UsingHolder(result.residual);
    trace?.(`${p}.output`, result.residual);
    trace?.(`${p}.normalized`, result.normed);
    return { normed: resultNormed.detach(), residual: resultResidual.detach() };
  }
}
