import fs from "node:fs";
import path from "node:path";
import { GlmOps, f32ToBf16Bytes, I32 } from "./glm_ops";
import { SafeTensorFile } from "./safetensors";
import { resolveModelPath } from "./model_path";
import { PagedKVCache, WorkspaceBuffers } from "./paged_kv";
import { Tensor } from "./tensor";
import { Qwen35GdnState } from "./qwen35_gdn_state";
import type { ChatCache } from "./chat_model";
import { ChatModelBase, type BatchState, SamplingParams, SamplingWorkspaceBase } from "./chat_model";

class Qwen35ChatCache implements ChatCache {
  constructor(
    public readonly pagedKV: PagedKVCache,
    public readonly gdnState: Qwen35GdnState,
  ) {}

  reset(batchSize: number): void {
    this.pagedKV.reset(batchSize);
    this.gdnState.reset();
  }

  free(): void {
    this.gdnState.free();
    this.pagedKV.free();
  }

  prefixMatch(seqIdx: number, inputIds: number[]): number[] {
    const batchSize = Math.max(this.pagedKV.seqPages.length, 1);
    this.pagedKV.reset(batchSize);
    this.gdnState.reset();
    return inputIds.slice();
  }

  appendTokens(seqIdx: number, tokens: number[]): void {
    this.pagedKV.appendTokens(seqIdx, tokens);
  }
}

export type { SamplingParams };

const QWEN35_REPO = "Qwen/Qwen3.5-0.8B";
const EOS_TOKEN_IDS = new Set([248044]);

export interface Qwen35Config {
  hiddenSize: number;
  intermediateSize: number;
  numHiddenLayers: number;
  rmsNormEps: number;
  vocabSize: number;
  tieWordEmbeddings: boolean;
  numAttentionHeads: number;
  numKeyValueHeads: number;
  headDim: number;
  partialRotaryFactor: number;
  mropeSection: number[];
  mropeInterleaved: boolean;
  ropeTheta: number;
  attnOutputGate: boolean;
  linearNumKeyHeads: number;
  linearKeyHeadDim: number;
  linearNumValueHeads: number;
  linearValueHeadDim: number;
  linearConvKernelDim: number;
  layerTypes: string[];
  numKeyValueGroups: number;
  scaling: number;
  numFullAttnLayers: number;
  numGdnLayers: number;
  fullAttnLayerIndices: number[];
}

function loadConfig(modelDir: string): Qwen35Config {
  const raw = JSON.parse(fs.readFileSync(path.join(modelDir, "config.json"), "utf-8"));
  const tc = raw.text_config ?? raw;
  const layerTypes: string[] = tc.layer_types ?? [];
  const numFullAttnLayers = layerTypes.filter((t: string) => t === "full_attention").length;
  const numGdnLayers = layerTypes.filter((t: string) => t === "linear_attention").length;
  const fullAttnLayerIndices = layerTypes.map((t: string, i: number) => t === "full_attention" ? i : -1).filter((i: number) => i >= 0);
  const ropeParams = tc.rope_parameters ?? {};
  return {
    hiddenSize: tc.hidden_size,
    intermediateSize: tc.intermediate_size,
    numHiddenLayers: tc.num_hidden_layers,
    rmsNormEps: tc.rms_norm_eps ?? 1e-6,
    vocabSize: tc.vocab_size,
    tieWordEmbeddings: tc.tie_word_embeddings ?? false,
    numAttentionHeads: tc.num_attention_heads,
    numKeyValueHeads: tc.num_key_value_heads,
    headDim: tc.head_dim,
    partialRotaryFactor: ropeParams.partial_rotary_factor ?? 1.0,
    mropeSection: ropeParams.mrope_section ?? [],
    mropeInterleaved: ropeParams.mrope_interleaved ?? false,
    ropeTheta: ropeParams.rope_theta ?? 1000000,
    attnOutputGate: tc.attn_output_gate ?? false,
    linearNumKeyHeads: tc.linear_num_key_heads ?? 16,
    linearKeyHeadDim: tc.linear_key_head_dim ?? 128,
    linearNumValueHeads: tc.linear_num_value_heads ?? 16,
    linearValueHeadDim: tc.linear_value_head_dim ?? 128,
    linearConvKernelDim: tc.linear_conv_kernel_dim ?? 4,
    layerTypes,
    numKeyValueGroups: tc.num_attention_heads / tc.num_key_value_heads,
    scaling: Math.pow(tc.head_dim, -0.5),
    numFullAttnLayers,
    numGdnLayers,
    fullAttnLayerIndices,
  };
}

class Qwen35Workspace extends SamplingWorkspaceBase {
  hiddenA: Tensor;
  hiddenB: Tensor;
  normed: Tensor;
  gateBuf: Tensor;
  upBuf: Tensor;
  siluBuf: Tensor;
  downBuf: Tensor;
  logitsBuf: Tensor;
  hiddenLast: Tensor;
  lastIdx: Tensor;
  argmaxIdx: Tensor;
  positionIds: Tensor;
  cos: Tensor;
  sin: Tensor;
  inputIdsBuf: Tensor;
  flashOut: Tensor;
  oProjBuf: Tensor;
  gdnQkvBuf: Tensor;
  gdnABuf: Tensor;
  gdnBBuf: Tensor;
  gdnZBuf: Tensor;
  gdnOut: Tensor;
  gdnGatedOut: Tensor;
  gdnPrefillQkvLinear: Tensor;
  gdnPrefillQkvBuf: Tensor;
  gdnPrefillABuf: Tensor;
  gdnPrefillBBuf: Tensor;
  gdnPrefillZBuf: Tensor;
  gdnPrefillConvOut: Tensor;
  gdnPrefillOut: Tensor;
  gdnPrefillGatedOut: Tensor;
  attnQBuf: Tensor;
  attnKBuf: Tensor;
  attnVBuf: Tensor;
  attnVT: Tensor;
  attnQRope: Tensor;
  attnKRope: Tensor;
  qoIndptrD: Tensor;
  prefillSlotMapping: Tensor;

  constructor(glm: GlmOps, B: number, S: number, cfg: Qwen35Config) {
    super(glm, B, cfg.vocabSize);
    const hs = cfg.hiddenSize;
    const inter = cfg.intermediateSize;
    const nHeads = cfg.numAttentionHeads;
    const nKv = cfg.numKeyValueHeads;
    const hd = cfg.headDim;
    const linHeads = cfg.linearNumKeyHeads;
    const linKDim = cfg.linearKeyHeadDim;
    const linVDim = cfg.linearValueHeadDim;
    const convDim = linHeads * (linKDim * 2 + linVDim);
    const zDim = linHeads * linVDim;
    const BS = B * S;

    this.hiddenA = Tensor.alloc(glm, [B, S, hs], "BF16");
    this.hiddenB = Tensor.alloc(glm, [B, S, hs], "BF16");
    this.normed = Tensor.alloc(glm, [B, S, hs], "BF16");
    this.gateBuf = Tensor.alloc(glm, [B, S, inter], "BF16");
    this.upBuf = Tensor.alloc(glm, [B, S, inter], "BF16");
    this.siluBuf = Tensor.alloc(glm, [B, S, inter], "BF16");
    this.downBuf = Tensor.alloc(glm, [B, S, hs], "BF16");
    this.logitsBuf = Tensor.alloc(glm, [B, cfg.vocabSize], "BF16");
    this.hiddenLast = Tensor.alloc(glm, [B, hs], "BF16");
    this.lastIdx = Tensor.alloc(glm, [B], "I32");
    this.argmaxIdx = Tensor.alloc(glm, [B], "I32");
    this.positionIds = Tensor.alloc(glm, [B * S], "I32");
    this.cos = Tensor.alloc(glm, [B, S, hd], "BF16");
    this.sin = Tensor.alloc(glm, [B, S, hd], "BF16");
    this.inputIdsBuf = Tensor.alloc(glm, [B * S], "I32");
    this.flashOut = Tensor.alloc(glm, [B, nHeads, S, hd], "BF16");
    this.oProjBuf = Tensor.alloc(glm, [B, S, hs], "BF16");

    this.gdnQkvBuf = Tensor.alloc(glm, [B * convDim], "BF16");
    this.gdnABuf = Tensor.alloc(glm, [B * linHeads], "BF16");
    this.gdnBBuf = Tensor.alloc(glm, [B * linHeads], "BF16");
    this.gdnZBuf = Tensor.alloc(glm, [B * zDim], "BF16");
    this.gdnOut = Tensor.alloc(glm, [B * linHeads * linVDim], "BF16");
    this.gdnGatedOut = Tensor.alloc(glm, [B * linHeads * linVDim], "BF16");

    this.gdnPrefillQkvLinear = Tensor.alloc(glm, [BS, convDim], "BF16");
    this.gdnPrefillQkvBuf = Tensor.alloc(glm, [convDim, S], "BF16");
    this.gdnPrefillABuf = Tensor.alloc(glm, [BS * linHeads], "BF16");
    this.gdnPrefillBBuf = Tensor.alloc(glm, [BS * linHeads], "BF16");
    this.gdnPrefillZBuf = Tensor.alloc(glm, [BS, zDim], "BF16");
    this.gdnPrefillConvOut = Tensor.alloc(glm, [convDim, S], "BF16");
    this.gdnPrefillOut = Tensor.alloc(glm, [S * linHeads, linVDim], "BF16");
    this.gdnPrefillGatedOut = Tensor.alloc(glm, [S * linHeads, linVDim], "BF16");

    const qTotalDim = nHeads * hd;
    this.attnQBuf = Tensor.alloc(glm, [BS, qTotalDim * 2], "BF16");
    this.attnKBuf = Tensor.alloc(glm, [BS, nKv * hd], "BF16");
    this.attnVBuf = Tensor.alloc(glm, [BS, nKv * hd], "BF16");
    this.attnVT = Tensor.alloc(glm, [B, nKv, S, hd], "BF16");
    this.attnQRope = Tensor.alloc(glm, [B, nHeads, S, hd], "BF16");
    this.attnKRope = Tensor.alloc(glm, [B, nKv, S, hd], "BF16");

    this.qoIndptrD = Tensor.alloc(glm, [B + 1], "I32");
    this.prefillSlotMapping = Tensor.alloc(glm, [B * S], "I32");

    this.buildTensorMap();
  }
}


export class Qwen35Model extends ChatModelBase {
  readonly eosIds = new Set([248044]);
  declare glm: GlmOps;
  cfg: Qwen35Config;
  maxBatch: number;
  maxSeqLen: number;
  invFreq: Tensor;
  declare ws: Qwen35Workspace;

  private constructor(glm: GlmOps, config: Qwen35Config, maxBatch: number, maxSeqLen: number) {
    super();
    this.glm = glm;
    this.cfg = config;
    this.maxBatch = maxBatch;
    this.maxSeqLen = maxSeqLen;

    const ropeDim = Math.floor(config.headDim * config.partialRotaryFactor);
    const halfRopeDim = ropeDim / 2;
    const invFreqF32 = new Float32Array(halfRopeDim);
    for (let i = 0; i < halfRopeDim; i++) {
      invFreqF32[i] = 1.0 / Math.pow(config.ropeTheta, (2 * i) / ropeDim);
    }
    this.invFreq = this.alloc(glm, [halfRopeDim], "BF16", "invFreq");
    this.invFreq.h2d(f32ToBf16Bytes(invFreqF32));

    this.ws = new Qwen35Workspace(glm, maxBatch, maxSeqLen, config);
  }

  static fromPretrained(glm: GlmOps, repoId: string = QWEN35_REPO, maxBatch = 1, maxSeqLen = 4096): Qwen35Model {
    const modelDir = resolveModelPath(repoId);
    const config = loadConfig(modelDir);

    const stFiles = fs.readdirSync(modelDir).filter(f => f.endsWith('.safetensors') || f.endsWith('.safetensors.json'));
    const shards: string[] = [];
    if (stFiles.some(f => f === 'model.safetensors')) {
      shards.push(path.join(modelDir, 'model.safetensors'));
    } else {
      const indexFile = stFiles.find(f => f.endsWith('.json'));
      if (indexFile) {
        const idx = JSON.parse(fs.readFileSync(path.join(modelDir, indexFile), 'utf-8'));
        for (const f of Object.keys(idx.weight_map ?? idx)) {
          if (f.endsWith('.safetensors') && !shards.includes(path.join(modelDir, f))) {
            shards.push(path.join(modelDir, f));
          }
        }
      } else {
        shards.push(...stFiles.filter(f => f.endsWith('.safetensors')).map(f => path.join(modelDir, f)));
      }
    }

    const model = new Qwen35Model(glm, config, maxBatch, maxSeqLen);

    const prefix = "model.language_model.";
    const gemmaNormSuffixes = [
      "input_layernorm.weight",
      "post_attention_layernorm.weight",
      "q_norm.weight",
      "k_norm.weight",
    ];

    for (const stPath of shards) {
      const st = SafeTensorFile.open(stPath);
      const mmapPtr = glm.mmapOpen(stPath);
      const fileSize = fs.statSync(stPath).size;

      for (const name of st.tensorNames()) {
        let weightName = name.startsWith(prefix) ? name.slice(prefix.length) : name;
        const meta = st.meta(name);
        const isGemmaNorm = weightName === "norm.weight" ||
          gemmaNormSuffixes.some(s => weightName.endsWith(s));

        if (weightName.includes("A_log") || weightName.includes("dt_bias")) {
          const numElements = meta.shape.reduce((a, b) => a + b, 1);
          const tensor = model.alloc(glm, meta.shape, "F32", weightName);
          if (meta.dtype === "F32") {
            const offset = st.dataStart + meta.dataOffsets[0];
            glm.mmapLoad(tensor.data, mmapPtr, offset, tensor.bytes);
          } else {
            const rawBytes = st.readTensor(name);
            const f32Arr = new Float32Array(numElements);
            for (let i = 0; i < numElements; i++) {
              const u16 = rawBytes.readUInt16LE(i * 2);
              const u32 = u16 << 16;
              f32Arr[i] = new Float32Array(new Uint32Array([u32]).buffer)[0];
            }
            tensor.h2d(Buffer.from(f32Arr.buffer));
          }
        } else if (meta.dtype === "F32") {
          const numElements = meta.shape.reduce((a, b) => a + b, 1);
          const tensor = model.alloc(glm, meta.shape, "BF16", weightName);
          const f32Bytes = st.readTensor(name);
          const f32Arr = new Float32Array(f32Bytes.buffer, f32Bytes.byteOffset, numElements);
          if (isGemmaNorm) {
            for (let i = 0; i < numElements; i++) f32Arr[i] += 1.0;
          }
          tensor.h2d(f32ToBf16Bytes(f32Arr));
        } else if (isGemmaNorm) {
          const numElements = meta.shape.reduce((a, b) => a + b, 1);
          const tensor = model.alloc(glm, meta.shape, "BF16", weightName);
          const rawBytes = st.readTensor(name);
          const f32Arr = new Float32Array(numElements);
          for (let i = 0; i < numElements; i++) {
            const u16 = rawBytes.readUInt16LE(i * 2);
            const u32 = u16 << 16;
            f32Arr[i] = (new Float32Array(new Uint32Array([u32]).buffer)[0]) + 1.0;
          }
          tensor.h2d(f32ToBf16Bytes(f32Arr));
        } else {
          const dtype = meta.dtype === "F32" ? "F32" : meta.dtype;
          const tensor = model.alloc(glm, meta.shape, dtype, weightName);
          const offset = st.dataStart + meta.dataOffsets[0];
          glm.mmapLoad(tensor.data, mmapPtr, offset, tensor.bytes);
        }
      }

      glm.synchronize();
      st.close();
      glm.mmapClose(mmapPtr, fileSize);
    }

    if (config.tieWordEmbeddings && !model.tensors.has("lm_head.weight")) {
      const embedTensor = model.tensors.get("embed_tokens.weight")!;
      model.tensors.set("lm_head.weight", embedTensor);
    }

    return model;
  }

  free(): void {
    this.ws.free();
    super.free();
  }

  createGdnState(batchSize = 1): Qwen35GdnState {
    return new Qwen35GdnState(this.glm, this.cfg, batchSize);
  }

  createChatCache(maxPages = 256): ChatCache {
    const pagedKV = new PagedKVCache(this.glm, this.cfg.numKeyValueHeads, this.cfg.headDim, this.cfg.numFullAttnLayers, maxPages, this.maxBatch);
    const gdnState = this.createGdnState(this.maxBatch);
    return new Qwen35ChatCache(pagedKV, gdnState);
  }

  private fullAttnCacheIdx(layerIdx: number): number {
    let idx = 0;
    for (let i = 0; i < layerIdx; i++) {
      if (this.cfg.layerTypes[i] === "full_attention") idx++;
    }
    return idx;
  }

  protected getPagedKV(cache: ChatCache): PagedKVCache {
    if (!(cache instanceof Qwen35ChatCache)) throw new Error("Expected Qwen35ChatCache");
    return cache.pagedKV;
  }

  private mlp(pfx: string, BS: number): void {
    const cfg = this.cfg;
    const hs = cfg.hiddenSize;
    const inter = cfg.intermediateSize;
    this.ws.gateBuf.linear(this.ws.normed, this.tensors.get(`${pfx}.mlp.gate_proj.weight`)!, BS, inter, hs, this);
    this.ws.upBuf.linear(this.ws.normed, this.tensors.get(`${pfx}.mlp.up_proj.weight`)!, BS, inter, hs, this);
    this.ws.siluBuf.siluAndMul(this.ws.gateBuf, this.ws.upBuf, inter, BS);
    this.ws.downBuf.linear(this.ws.siluBuf, this.tensors.get(`${pfx}.mlp.down_proj.weight`)!, BS, hs, inter, this);
  }

  private gdnLayerPrefill(layerIdx: number, S: number, gdnState: Qwen35GdnState): void {
    const cfg = this.cfg;
    const glm = this.glm;
    const hs = cfg.hiddenSize;
    const linHeads = cfg.linearNumKeyHeads;
    const linKDim = cfg.linearKeyHeadDim;
    const linVDim = cfg.linearValueHeadDim;
    const convDim = linHeads * (linKDim * 2 + linVDim);
    const zDim = linHeads * linVDim;
    const pfx = `layers.${layerIdx}.linear_attn`;
    const BS = S;
    const batchSize = gdnState.batchSize;

    const qkvLinear = this.ws.gdnPrefillQkvLinear;
    const qkvBuf = this.ws.gdnPrefillQkvBuf;
    const aBuf = this.ws.gdnPrefillABuf;
    const bBuf = this.ws.gdnPrefillBBuf;
    const zBuf = this.ws.gdnPrefillZBuf;

    qkvLinear.linear(this.ws.normed, this.tensors.get(`${pfx}.in_proj_qkv.weight`)!, BS, convDim, hs, this);
    aBuf.linear(this.ws.normed, this.tensors.get(`${pfx}.in_proj_a.weight`)!, BS, linHeads, hs, this);
    bBuf.linear(this.ws.normed, this.tensors.get(`${pfx}.in_proj_b.weight`)!, BS, linHeads, hs, this);
    zBuf.linear(this.ws.normed, this.tensors.get(`${pfx}.in_proj_z.weight`)!, BS, zDim, hs, this);

    qkvBuf.transpose4d(qkvLinear, 1, S, convDim, 1, 0, 2, 1, 3);

    const convState = gdnState.convState[layerIdx];
    const recurrentState = gdnState.recurrentState[layerIdx];
    const kernelSize = cfg.linearConvKernelDim;

    const convOut = this.ws.gdnPrefillConvOut;
    glm.causalConv1d(convOut.data, convState.data, qkvBuf.data, this.tensors.get(`${pfx}.conv1d.weight`)!.data, gdnState.cuSeqlens.data, convDim, S, kernelSize, batchSize, gdnState.convStateStride);

    const gdnOut = this.ws.gdnPrefillOut;

    glm.gdnPrefill(
      gdnOut.data, recurrentState.data,
      convOut.data,
      aBuf.data, bBuf.data,
      this.tensors.get(`${pfx}.A_log`)!.data,
      this.tensors.get(`${pfx}.dt_bias`)!.data,
      gdnState.cuSeqlens.data, S, linHeads, linKDim, linVDim,
      batchSize, gdnState.recurrentStateStride, S,
    );

    const gatedOut = this.ws.gdnPrefillGatedOut;
    gatedOut.rmsnormGated(gdnOut, zBuf, this.tensors.get(`${pfx}.norm.weight`)!, cfg.rmsNormEps, linVDim, S * linHeads);

    this.ws.oProjBuf.linear(gatedOut, this.tensors.get(`${pfx}.out_proj.weight`)!, BS, hs, zDim, this);

    this.ws.normed.fusedAddRmsnorm(this.ws.hiddenB, this.ws.hiddenA, this.ws.oProjBuf, this.tensors.get(`layers.${layerIdx}.post_attention_layernorm.weight`)!, cfg.rmsNormEps, hs, BS);
    this.mlp(`layers.${layerIdx}`, BS);
  }

  private gdnLayerDecode(layerIdx: number, gdnState: Qwen35GdnState): void {
    const cfg = this.cfg;
    const glm = this.glm;
    const hs = cfg.hiddenSize;
    const linHeads = cfg.linearNumKeyHeads;
    const linKDim = cfg.linearKeyHeadDim;
    const linVDim = cfg.linearValueHeadDim;
    const convDim = linHeads * (linKDim * 2 + linVDim);
    const zDim = linHeads * linVDim;
    const pfx = `layers.${layerIdx}.linear_attn`;
    const BS = gdnState.batchSize;

    const qkvBuf = this.ws.gdnQkvBuf;
    const aBuf = this.ws.gdnABuf;
    const bBuf = this.ws.gdnBBuf;
    const zBuf = this.ws.gdnZBuf;

    qkvBuf.linear(this.ws.normed, this.tensors.get(`${pfx}.in_proj_qkv.weight`)!, BS, convDim, hs, this);
    aBuf.linear(this.ws.normed, this.tensors.get(`${pfx}.in_proj_a.weight`)!, BS, linHeads, hs, this);
    bBuf.linear(this.ws.normed, this.tensors.get(`${pfx}.in_proj_b.weight`)!, BS, linHeads, hs, this);
    zBuf.linear(this.ws.normed, this.tensors.get(`${pfx}.in_proj_z.weight`)!, BS, zDim, hs, this);

    const convState = gdnState.convState[layerIdx];
    const recurrentState = gdnState.recurrentState[layerIdx];
    const kernelSize = cfg.linearConvKernelDim;

    glm.causalConv1dUpdate(qkvBuf.data, convState.data, qkvBuf.data, this.tensors.get(`${pfx}.conv1d.weight`)!.data, convDim, kernelSize, BS, gdnState.convStateStride);

    const qkvT = this.ws.gdnPrefillQkvBuf;
    if (BS > 1) {
      qkvT.transpose4d(qkvBuf, 1, BS, convDim, 1, 0, 2, 1, 3);
    }
    const qkvSrc = BS === 1 ? qkvBuf.data : qkvT.data;

    const gdnOut = this.ws.gdnOut;

    glm.gdnRecurrentStep(
      gdnOut.data, recurrentState.data,
      qkvSrc,
      aBuf.data, bBuf.data,
      this.tensors.get(`${pfx}.A_log`)!.data,
      this.tensors.get(`${pfx}.dt_bias`)!.data,
      linHeads, linKDim, linVDim,
      BS, gdnState.recurrentStateStride, BS,
    );

    const gatedOut = this.ws.gdnGatedOut;
    gatedOut.rmsnormGated(gdnOut, zBuf, this.tensors.get(`${pfx}.norm.weight`)!, cfg.rmsNormEps, linVDim, BS * linHeads);

    this.ws.oProjBuf.linear(gatedOut, this.tensors.get(`${pfx}.out_proj.weight`)!, BS, hs, zDim, this);

    this.ws.normed.fusedAddRmsnorm(this.ws.hiddenB, this.ws.hiddenA, this.ws.oProjBuf, this.tensors.get(`layers.${layerIdx}.post_attention_layernorm.weight`)!, cfg.rmsNormEps, hs, BS);
    this.mlp(`layers.${layerIdx}`, BS);
  }

  private fullAttnLayer(layerIdx: number, state: BatchState, pagedKV: PagedKVCache, ws: WorkspaceBuffers): void {
    const cfg = this.cfg;
    const glm = this.glm;
    const hs = cfg.hiddenSize;
    const nHeads = cfg.numAttentionHeads;
    const nKv = cfg.numKeyValueHeads;
    const hd = cfg.headDim;
    const pageSize = pagedKV.pageSize;
    const cacheIdx = this.fullAttnCacheIdx(layerIdx);
    const pfx = `layers.${layerIdx}.self_attn`;
    const qTotalDim = nHeads * hd;
    const batchSize = state.batchSize;
    const totalTokens = state.totalTokens;
    const BS = totalTokens;
    const B = state.isDecode ? batchSize : 1;
    const S = state.isDecode ? 1 : totalTokens;

    const qBuf = this.ws.attnQBuf;
    const kBuf = this.ws.attnKBuf;
    const vBuf = this.ws.attnVBuf;

    qBuf.linear(this.ws.normed, this.tensors.get(`${pfx}.q_proj.weight`)!, BS, qTotalDim * 2, hs, this);
    kBuf.linear(this.ws.normed, this.tensors.get(`${pfx}.k_proj.weight`)!, BS, nKv * hd, hs, this);
    vBuf.linear(this.ws.normed, this.tensors.get(`${pfx}.v_proj.weight`)!, BS, nKv * hd, hs, this);

    const ropeDim = Math.floor(hd * cfg.partialRotaryFactor);
    const qRope = this.ws.attnQRope;
    const kRope = this.ws.attnKRope;

    qRope.fusedNormRope(qBuf, this.tensors.get(`${pfx}.q_norm.weight`)!, this.ws.cos, this.ws.sin, cfg.rmsNormEps, ropeDim, hd, nHeads, S, B, hd * 2);
    kRope.fusedNormRope(kBuf, this.tensors.get(`${pfx}.k_norm.weight`)!, this.ws.cos, this.ws.sin, cfg.rmsNormEps, ropeDim, hd, nKv, S, B);

    const vData = S === 1 ? vBuf.data : this.ws.attnVT.data;
    if (S > 1) {
      this.ws.attnVT.transpose4d(vBuf, B, S, nKv, hd, 0, 2, 1, 3);
    }

    const slotMapping = state.isDecode ? pagedKV.slotMapping : this.ws.prefillSlotMapping.data;
    const kStride = state.isDecode ? nKv * hd : hd;
    const vStride = state.isDecode ? hd : BS * hd;

    glm.kvCacheWrite(
      kRope.data, vData,
      pagedKV.kData[cacheIdx], pagedKV.vData[cacheIdx],
      slotMapping, BS, nKv, hd, pageSize,
      kStride, vStride
    );

    if (state.isDecode) {
      glm.batchDecodeRun(
        qRope.data, this.ws.flashOut.data,
        pagedKV.kData[cacheIdx], pagedKV.vData[cacheIdx],
        pagedKV.indices, pagedKV.indptrD, pagedKV.lastPageLen,
        ws.floatWs, ws.intWs,
        ws.decodePlanInfo,
        batchSize,
        nHeads, nKv, hd, pageSize,
        cfg.scaling
      );
    } else {
      const qStrideN = hd;
      const qStrideH = BS * hd;
      glm.batchPrefillPagedRun(
        qRope.data, this.ws.flashOut.data,
        pagedKV.kData[cacheIdx], pagedKV.vData[cacheIdx],
        pagedKV.indices, pagedKV.indptrD, pagedKV.lastPageLen,
        ws.floatWs, ws.intWs,
        this.ws.qoIndptrD.data,
        ws.prefillPlanInfo,
        BS, batchSize,
        nHeads, nKv, hd, pageSize,
        qStrideN, qStrideH, 1, cfg.scaling
      );
    }

    if (cfg.attnOutputGate) {
      glm.gateSigmoidMul(this.ws.flashOut.data, qBuf.data, BS, nHeads, hd);
    }

    this.ws.oProjBuf.linear(this.ws.flashOut, this.tensors.get(`${pfx}.o_proj.weight`)!, BS, hs, nHeads * hd, this);

    this.ws.normed.fusedAddRmsnorm(this.ws.hiddenB, this.ws.hiddenA, this.ws.oProjBuf, this.tensors.get(`layers.${layerIdx}.post_attention_layernorm.weight`)!, cfg.rmsNormEps, hs, BS);
    this.mlp(`layers.${layerIdx}`, BS);
  }

  protected prefillBatchPlanHook(
    _inputIdsList: number[][], seqLens: number[], totalTokens: number,
    _startPos: number[], cache: ChatCache,
  ): void {
    const { gdnState } = cache as Qwen35ChatCache;
    gdnState.uploadCuSeqlens(seqLens);
    if (totalTokens > this.maxBatch * this.maxSeqLen) {
      throw new Error(`Total tokens ${totalTokens} exceeds max (B=${this.maxBatch}, S=${this.maxSeqLen})`);
    }
  }

  forward(state: BatchState, ws: WorkspaceBuffers, cache: ChatCache): void {
    if (!(cache instanceof Qwen35ChatCache)) throw new Error("Expected Qwen35ChatCache");
    const { pagedKV, gdnState } = cache;
    const cfg = this.cfg;
    const glm = this.glm;
    const hs = cfg.hiddenSize;
    const hd = cfg.headDim;
    const batchSize = state.batchSize;
    const totalTokens = state.totalTokens;
    const BS = totalTokens;
    const B = state.isDecode ? batchSize : 1;
    const S = state.isDecode ? 1 : totalTokens;

    this.ws.hiddenA.embedding(this.tensors.get("embed_tokens.weight")!, this.ws.inputIdsBuf, hs, BS);

    const ropeDim = Math.floor(hd * cfg.partialRotaryFactor);
    glm.rotaryEmbedding(this.ws.cos.data, this.ws.sin.data, this.invFreq.data, this.ws.positionIds.data, ropeDim / 2, B, S);

    this.ws.normed.rmsnorm(this.ws.hiddenA, this.tensors.get(`layers.0.input_layernorm.weight`)!, cfg.rmsNormEps, hs, BS);

    for (let i = 0; i < cfg.numHiddenLayers; i++) {
      if (cfg.layerTypes[i] === "linear_attention") {
        if (state.isDecode) {
          this.gdnLayerDecode(i, gdnState);
        } else {
          this.gdnLayerPrefill(i, totalTokens, gdnState);
        }
      } else {
        this.fullAttnLayer(i, state, pagedKV, ws);
      }

      if (i < cfg.numHiddenLayers - 1) {
        const nextWeightKey = `layers.${i + 1}.input_layernorm.weight`;
        this.ws.normed.fusedAddRmsnorm(this.ws.hiddenA, this.ws.hiddenB, this.ws.downBuf, this.tensors.get(nextWeightKey)!, cfg.rmsNormEps, hs, BS);
      } else {
        this.ws.normed.fusedAddRmsnorm(this.ws.hiddenA, this.ws.hiddenB, this.ws.downBuf, this.tensors.get("norm.weight")!, cfg.rmsNormEps, hs, BS);
        if (state.isDecode) {
          this.ws.logitsBuf.linear(this.ws.normed, this.tensors.get("lm_head.weight")!, batchSize, cfg.vocabSize, hs, this);
        } else {
          this.ws.hiddenLast.indexSelect(this.ws.normed, this.ws.lastIdx, hs, batchSize);
          this.ws.logitsBuf.linear(this.ws.hiddenLast, this.tensors.get("lm_head.weight")!, batchSize, cfg.vocabSize, hs, this);
        }
      }
    }

    this.ws.argmaxIdx.argmax(this.ws.logitsBuf, cfg.vocabSize, batchSize);
  }
}
