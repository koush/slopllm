import fs from "node:fs";
import path from "node:path";
import { ChatModel, type ChatCache, type CommonModelConfig } from "./chat_model";
import type { CheckpointTensor } from "./checkpoint_source";
import { type DeviceOps, TensorParallelism } from "./device_ops";
import type { ExecutionState } from "./execution-workspace";
import { resolveModelPath } from "./model_path";
import type { TensorMeta } from "./safetensors";
import type { Tensor } from "./tensor";
import type { KdaWeights } from "./kda";

export const GLM53_FLASH_REPO = "local-inference-lab/GLM-5.3-Flash-NVFP4";

export interface Glm53FlashConfig extends CommonModelConfig {
  layerTypes: string[];
  qLoraRank: number;
  qkNopeHeadDim: number;
  vHeadDim: number;
  linearNumHeads: number;
  linearHeadDim: number;
  linearConvKernelSize: number;
  linearGateLowerBound: number;
  hcMult: number;
  hcEps: number;
  hcSinkhornIters: number;
  nRoutedExperts: number;
  nSharedExperts: number;
  numExpertsPerTok: number;
  moeIntermediateSize: number;
  firstKDenseReplace: number;
  routedScalingFactor: number;
  normTopkProb: boolean;
  nGroup: number;
  topkGroup: number;
  swigluLimit: number;
  indexHeadDim: number;
  indexNHeads: number;
  indexKpool: number;
  indexKpoolAlwaysSelectTail: boolean;
  eosTokenIds: number[];
}

export function loadGlm53FlashConfig(modelDir: string): Glm53FlashConfig {
  const raw = JSON.parse(fs.readFileSync(path.join(modelDir, "config.json"), "utf8"));
  const c = raw.text_config;
  if (raw.model_type !== "glm5_next" || !c) throw new Error("Expected a glm5_next checkpoint with text_config");
  const linear = c.linear_attn_config ?? {};
  const cfg: Glm53FlashConfig = {
    hiddenSize: c.hidden_size, intermediateSize: c.intermediate_size,
    numHiddenLayers: c.num_hidden_layers, numNextNPredictLayers: 0,
    rmsNormEps: c.rms_norm_eps, vocabSize: c.vocab_size,
    tieWordEmbeddings: c.tie_word_embeddings ?? false,
    numAttentionHeads: c.num_attention_heads, numKeyValueHeads: c.num_key_value_heads,
    headDim: c.qk_nope_head_dim, ropeTheta: 1, numKeyValueGroups: c.num_attention_heads / c.num_key_value_heads,
    scaling: c.qk_nope_head_dim ** -0.5, kvLoraRank: c.kv_lora_rank,
    qkRopeHeadDim: c.qk_rope_head_dim, indexTopk: c.index_topk,
    maxPositionEmbeddings: c.max_position_embeddings,
    layerTypes: c.layer_types, qLoraRank: c.q_lora_rank,
    qkNopeHeadDim: c.qk_nope_head_dim, vHeadDim: c.v_head_dim,
    linearNumHeads: c.linear_num_heads ?? linear.num_heads,
    linearHeadDim: c.linear_head_dim ?? linear.head_dim,
    linearConvKernelSize: c.linear_conv_kernel_dim ?? linear.short_conv_kernel_size,
    linearGateLowerBound: c.linear_lower_bound ?? linear.gate_lower_bound,
    hcMult: c.hc_mult, hcEps: c.hc_eps, hcSinkhornIters: c.hc_sinkhorn_iters,
    nRoutedExperts: c.n_routed_experts, nSharedExperts: c.n_shared_experts,
    numExpertsPerTok: c.num_experts_per_tok, moeIntermediateSize: c.moe_intermediate_size,
    firstKDenseReplace: c.first_k_dense_replace, routedScalingFactor: c.routed_scaling_factor,
    normTopkProb: c.norm_topk_prob, nGroup: c.n_group, topkGroup: c.topk_group,
    swigluLimit: c.swiglu_limit, indexHeadDim: c.index_head_dim,
    indexNHeads: c.index_n_heads, indexKpool: c.index_kpool,
    indexKpoolAlwaysSelectTail: c.index_kpool_always_select_tail,
    eosTokenIds: Array.isArray(c.eos_token_id) ? c.eos_token_id : [c.eos_token_id],
  };
  for (const key of ["hiddenSize", "intermediateSize", "numHiddenLayers", "vocabSize", "numAttentionHeads",
    "numKeyValueHeads", "qLoraRank", "kvLoraRank", "linearNumHeads", "hcSinkhornIters", "nRoutedExperts",
    "nSharedExperts", "numExpertsPerTok", "moeIntermediateSize", "indexHeadDim", "indexNHeads", "indexKpool"] as const) {
    const value = cfg[key];
    if (!Number.isSafeInteger(value) || !value || value < 1) throw new Error(`Invalid Flash config ${key}: ${value}`);
  }
  if (cfg.hcMult !== 4 || cfg.qkRopeHeadDim !== 0 || cfg.linearHeadDim !== 128 ||
      cfg.linearConvKernelSize !== 4 || cfg.linearGateLowerBound !== -5) {
    throw new Error("Unsupported GLM-5.3-Flash mHC/KDA/MLA geometry");
  }
  if (cfg.layerTypes?.length !== cfg.numHiddenLayers || cfg.layerTypes.some(t =>
    !["linear_attention", "deepseek_sparse_attention", "indexed_attention"].includes(t))) {
    throw new Error("Invalid GLM-5.3-Flash layer_types");
  }
  return cfg;
}

/** Text-only weight workspace. Decoder/cache integration is intentionally not implemented yet. */
export class Glm53FlashModel extends ChatModel {
  static readonly WEIGHT_PREFIX = "model.language_model.";
  readonly eosIds: Set<number>;

  constructor(ops: DeviceOps, public readonly cfg: Glm53FlashConfig) {
    super(ops);
    if (cfg.linearNumHeads % ops.worldSize || cfg.numAttentionHeads % ops.worldSize) {
      throw new Error("Flash attention heads must be divisible by TP size");
    }
    this.eosIds = new Set(cfg.eosTokenIds);
  }

  static async fromPretrained(ops: DeviceOps, repoIdOrDir = GLM53_FLASH_REPO): Promise<Glm53FlashModel> {
    const dir = fs.existsSync(repoIdOrDir) ? repoIdOrDir : resolveModelPath(repoIdOrDir);
    const model = new Glm53FlashModel(ops, loadGlm53FlashConfig(dir));
    try {
      await model.fromPretrained(dir, GLM53_FLASH_REPO);
      return model;
    } catch (e) {
      model.free();
      throw e;
    }
  }

  private weightParallelism(name: string): TensorParallelism {
    if (name === "lm_head.weight") return TensorParallelism.Column;
    if (name === `${Glm53FlashModel.WEIGHT_PREFIX}embed_tokens.weight`) return TensorParallelism.Row;
    // Preserve head-contiguous checkpoint layouts; MLA absorption belongs to the future forward path.
    const base = name.replace(/\.(weight_scale|weight)$/, "");
    if (/\.self_attn\.(q_proj|k_proj|v_proj|q_b_proj|kv_b_proj|b_proj|f_b_proj|g_b_proj|[qkv]_conv1d)$/.test(base) ||
        /\.self_attn\.(A_log|dt_bias)$/.test(base) ||
        /\.mlp\.(?:experts\.\d+\.|shared_experts\.)?(gate_proj|up_proj)$/.test(base)) return TensorParallelism.Column;
    if (/\.self_attn\.o_proj$/.test(base) ||
        /\.mlp\.(?:experts\.\d+\.|shared_experts\.)?down_proj$/.test(base)) return TensorParallelism.Row;
    return TensorParallelism.Replicated;
  }

  protected async loadTensor(name: string, meta: TensorMeta, source: CheckpointTensor): Promise<void> {
    if (name.startsWith("model.visual.")) return;
    const layer = /^model\.language_model\.layers\.(\d+)\./.exec(name);
    if (layer && Number(layer[1]) >= this.cfg.numHiddenLayers) return; // MTP, including MXFP8 experts.
    if (name !== "lm_head.weight" && !name.startsWith(Glm53FlashModel.WEIGHT_PREFIX)) {
      throw new Error(`Unsupported Flash tensor: ${name}`);
    }
    if (name.includes(".nvfp4_csf_") || (meta.dtype === "F8_E4M3" && !name.endsWith(".weight_scale"))) {
      throw new Error(`Flash loader expects BF16 linears and uncompressed NVFP4 experts: ${name}`);
    }
    // Retain F32 router/mHC/decay parameters and activation scales verbatim.
    const storeName = name.replace(/\.weight_scale(_2)?$/, ".weight_weight_scale$1");
    const dtype = this.resolveWeightType(name, meta);
    const tensor = this.alloc(meta.shape, dtype, storeName, this.weightParallelism(name));
    await this.loadTensorBytes(tensor, source);
  }

  createChatCache(): ChatCache {
    throw new Error("GLM-5.3-Flash hybrid chat cache is not implemented");
  }

  kdaWeights(layerIdx: number): KdaWeights {
    if (this.cfg.layerTypes[layerIdx] !== "linear_attention") throw new Error(`Layer ${layerIdx} is not KDA`);
    const prefix = `${Glm53FlashModel.WEIGHT_PREFIX}layers.${layerIdx}.self_attn.`;
    const get = (suffix: string): Tensor => {
      const weight = this.tensors.get(prefix + suffix);
      if (!weight) throw new Error(`Missing KDA weight: ${prefix}${suffix}`);
      return weight;
    };
    return {
      q: get("q_proj.weight"), k: get("k_proj.weight"), v: get("v_proj.weight"),
      qConv: get("q_conv1d.weight"), kConv: get("k_conv1d.weight"), vConv: get("v_conv1d.weight"),
      fA: get("f_a_proj.weight"), fB: get("f_b_proj.weight"), beta: get("b_proj.weight"),
      gA: get("g_a_proj.weight"), gB: get("g_b_proj.weight"), norm: get("o_norm.weight"), out: get("o_proj.weight"),
      aLog: get("A_log"), dtBias: get("dt_bias"),
    };
  }

  forwardModel(_state: ExecutionState): Tensor {
    throw new Error("GLM-5.3-Flash forward is not implemented");
  }

  *forwardPhased(_state: ExecutionState): Generator<void, Tensor, void> {
    throw new Error("GLM-5.3-Flash forward is not implemented");
  }
}
