# Diagnostic tools

Two standalone tools for measuring generation quality. They exist because this
repo previously had **no way to measure generation quality** — the only
available check was eyeballing a greedy sample, which is unreliable when the
underlying numerics are chaotic and sent an entire debugging session down three
dead ends (P2P allReduce, FP8 KV cache, sparse-MLA kernel) before the actual
cause turned up.

## `src/run_sweep.ts` — greedy termination sweep

Runs 20 fixed prompts through **one model load** and reports how many fail to
emit EOS or fall into a repetition loop. Answers "is generation healthy?" as a
**rate**, not an anecdote — single-prompt comparisons are uninformative when a
1-ulp numerical difference can flip a marginal decision.

```bash
npx tsx src/run_sweep.ts --gpus 0,1,2,3,4,5,6,7 --arena 92 \
  --max-seq-len 6144 --max-pages 2048 --start 0 --count 10
```

| flag | default | meaning |
|---|---|---|
| `--gpus` | `0` | comma-separated device ids |
| `--arena` | `0` | arena size in GB (`0` = per-alloc) |
| `--model-dir` | GLM-5.2-NVFP4 snapshot | checkpoint to load |
| `--start` / `--count` | `0` / all | slice of the prompt list (lets you batch under a wall-clock limit) |
| `--max-new-tokens` | `4000` | generation cap; hitting it counts as non-terminating |
| `--max-seq-len` / `--max-pages` | `8192` / `8192` | lower these for dense attention, whose bf16 KV cache is far larger |

Env: `GLM_DENSE_ATTN=1` for the dense A/B (reported in the summary line);
`GLM_REASONING_EFFORT=high` to match vLLM's default (the template otherwise
emits `Reasoning Effort: Max`).

Output — one line per prompt plus a summary:

```
[template] .../GLM-5.2-NVFP4/.../chat_template.jinja
    stop  2167tok  loop=false tell me about india
  LENGTH  4000tok  loop=true  explain CRISPR gene editing NON-TERMINATING LOOPED

glm.js greedy SPARSE: 0/20 prompts non-terminating or looping
```

Two independent failure signals: `LENGTH` (never emitted EOS) and `loop=true`
(a 12-word window where one word occupies >= 8 slots). It also prints the
resolved chat template path, which is a direct regression guard for the bug it
was written to catch.

**Watch the token counts, not just the failure count.** The strongest signal
that something was wrong was the *length distribution* collapsing (162-token
stubs, sd 1031) long before any prompt outright hung.

## `src/run_nll.ts` — teacher-forced NLL / perplexity

Objective quality metric independent of sampling: every configuration is scored
on the **identical token sequence**, so numbers are directly comparable across
kernel, collective, or attention-backend changes.

```bash
# prefill mode: one forward pass, logits for all positions
npx tsx src/run_nll.ts --gpus 0,1,2,3,4,5,6,7 --arena 92 --tokens 1024 --skip 60000

# decode mode: prefill a short prefix, then step token-by-token
npx tsx src/run_nll.ts --gpus 0,1,2,3,4,5,6,7 --arena 92 --tokens 600 \
  --decode --prefix 16 --dump nll.json
```

| flag | default | meaning |
|---|---|---|
| `--gpus` | `0` | comma-separated device ids |
| `--arena` | `0` | arena size in GB |
| `--model-dir` | GLM-5.2-NVFP4 snapshot | checkpoint to load |
| `--tokens` | `512` | tokens to score |
| `--skip` | `2000` | character offset into the text file |
| `--text` | `big.txt` | source text |
| `--decode` | off | use the decode path instead of a single prefill |
| `--prefix` | `16` | decode mode: prefill length before teacher forcing begins |
| `--dump` | — | decode mode: write per-token NLL as JSON, for paired statistics |

```
tokens=1023 vocab=154880 meanNLL=0.16080 ppl=1.1744 top1acc=96.29%
[decode] tokens=584 meanNLL=0.16098 ppl=1.1747 top1acc=95.38%
```

**`--decode` is not optional for decode-path work.** Decode-only kernels are
invisible to a prefill run — the P2P allReduce is gated to <= 262144 elements,
so at hidden 6144 it never fires above ~42 tokens, and a 1024-token prefill
returns byte-identical numbers whether it is enabled or not.

**Use `--dump` and compare paired per-token, not by mean.** Mean NLL is
dominated by tokens both configs get right; a real difference can hide
entirely. In practice a sparse-vs-dense comparison came back *not significant*
on mean NLL (t=+1.76) while the same data showed t=+19 on the probability mass
assigned to specific tail tokens. If you care about EOS or any rare token,
measure it directly.

## Env knobs

| var | effect |
|---|---|
| `GLM_DENSE_ATTN=1` | forces `indexHeadDim: 0`, disabling sparse indexing in favour of dense MLA — the cheapest sparse-vs-dense A/B |
| `GLM_P2P_ALLREDUCE=0` | escape hatch back to NCCL for the P2P allReduce (now on by default) |
| `GLM_REASONING_EFFORT` | passed as `reasoning_effort` to the chat template (`run_sweep` only) |
