import type { TensorMeta } from "./safetensors";

interface QuantizationGroup {
  targets?: string[];
  weights?: { num_bits?: number; type?: string; group_size?: number };
  quant_algo?: string;
}

export interface WeightQuantizationConfig {
  quant_method?: string;
  quant_algo?: string;
  expert_dtype?: string;
  scale_fmt?: string;
  ignore?: string[];
  modules_to_not_convert?: string[];
  config_groups?: Record<string, QuantizationGroup>;
}

function matches(pattern: string, module: string): boolean {
  // Packed matrices passed here are linear weights; other class targets are unsupported.
  if (pattern === "Linear") return true;
  if (pattern.startsWith("re:")) return new RegExp(pattern.slice(3)).test(module);
  const escaped = pattern.replace(/[.+^${}()|[\]\\]/g, "\\$&").replace(/\*/g, ".*").replace(/\?/g, ".");
  return new RegExp(`^${escaped}$`).test(module);
}

/** Checkpoint declarations select the encoding; scale metadata verifies that declaration. */
export function resolveWeightType(
  name: string, meta: TensorMeta, config: WeightQuantizationConfig | undefined,
  metadata: (name: string) => TensorMeta,
): string {
  if (!name.endsWith(".weight") || !["U8", "I8"].includes(meta.dtype)) return meta.dtype;
  const module = name.slice(0, -".weight".length);
  const fail = (reason: string): never => { throw new Error(`Cannot resolve packed weight ${name}: ${reason}`); };
  if (!config) return fail("missing quantization_config");
  if ([...(config.ignore ?? []), ...(config.modules_to_not_convert ?? [])].some(p => matches(p, module))) {
    return fail("packed weight is excluded by quantization_config");
  }

  const groups = Object.values(config.config_groups ?? {}).filter(g =>
    g.targets?.some(target => matches(target, module)));
  if (config.config_groups && groups.length === 0) return fail("no matching quantization group");

  const declared = (algo: string | undefined): string | undefined => {
    if (algo === undefined) return undefined;
    if (algo === "NVFP4" || algo === "MXFP4") return algo;
    return fail(`unsupported quant_algo ${algo}`);
  };
  const globalFormat = declared(config.quant_algo);
  const formats = groups.map(group => {
    const weights = group.weights;
    if (weights?.num_bits !== 4 || weights.type !== "float" || ![16, 32].includes(weights.group_size!)) {
      return fail("matching group does not describe supported float4 weights");
    }
    const format = declared(group.quant_algo) ?? globalFormat ?? (weights.group_size === 16 ? "NVFP4" : "MXFP4");
    if (weights.group_size !== (format === "NVFP4" ? 16 : 32)) return fail("quant_algo conflicts with group size");
    return format;
  });
  if (new Set(formats).size > 1) return fail("conflicting quantization groups");
  let format = formats[0] ?? globalFormat;
  // DeepSeek mixes FP8 linears with MXFP4 expert matrices in the same checkpoint.
  if (!format && config.quant_method === "fp8" && config.expert_dtype === "fp4" &&
    config.scale_fmt === "ue8m0" && /(?:^|\.)experts\.\d+\./.test(name)) format = "MXFP4";
  if (!format) return fail("unsupported or ambiguous quantization_config");

  const blockSize = format === "NVFP4" ? 16 : 32;
  const scale = metadata(format === "NVFP4" ? `${name}_scale` : name.replace(/\.weight$/, ".scale"));
  if (meta.shape.length !== 2 || meta.shape[1] % (blockSize / 2) !== 0 ||
    scale.dtype !== (format === "NVFP4" ? "F8_E4M3" : "F8_E8M0") || scale.shape.length !== 2 ||
    scale.shape[0] !== meta.shape[0] || scale.shape[1] !== meta.shape[1] * 2 / blockSize) {
    return fail(`invalid ${format} weight/scale metadata`);
  }
  if (format === "NVFP4") {
    const globalScale = metadata(`${name}_scale_2`);
    if (globalScale.dtype !== "F32" || globalScale.shape.reduce((a, b) => a * b, 1) !== 1) {
      return fail("invalid NVFP4 global scale metadata");
    }
  }
  return `U8_${format}`;
}
