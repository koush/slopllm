#!/usr/bin/env python3
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from helpers import GlmOps
from qwen3_model import Qwen3Model, _normalize_token_ids
from paged_kv import PagedKVCache, WorkspaceBuffers

from transformers import AutoTokenizer

QWEN3_REPO = "Qwen/Qwen3-0.6B"
EOS_TOKEN_IDS = {151645, 151643}


def main():
    parser = argparse.ArgumentParser(description="Batched interactive Qwen3-0.6B chat")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max tokens per response")
    parser.add_argument("--no-think", action="store_true", help="Disable thinking mode")
    parser.add_argument("--gpu", type=int, default=None, help="GPU device ID")
    parser.add_argument("--max-seq-len", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--max-pages", type=int, default=128, help="Max pages for paged KV cache")
    parser.add_argument("--max-batch", type=int, default=4, help="Max batch size")
    args = parser.parse_args()

    gpu_id = args.gpu if args.gpu is not None else int(os.environ.get("GLM_GPU", "0"))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"Loading model on GPU {gpu_id}...")
    glm = GlmOps(device_id=0)
    model = Qwen3Model.from_pretrained(glm, QWEN3_REPO, max_batch=args.max_batch, max_seq_len=args.max_seq_len)
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_REPO)

    cfg = model.cfg
    n_kv = cfg.num_key_value_heads
    hd = cfg.head_dim
    n_layers = cfg.num_hidden_layers

    ws = WorkspaceBuffers(glm)
    paged_kv = PagedKVCache(glm, n_kv, hd, n_layers, args.max_pages, max_batch=args.max_batch)

    enable_thinking = not args.no_think

    print(f"Qwen3-0.6B batched ready (max_tokens={args.max_tokens}, thinking={'off' if args.no_think else 'on'})")
    print("Enter prompts separated by blank lines. Double blank line to submit batch.")
    print("/clear to reset, /q to quit.")

    try:
        while True:
            prompts = []
            try:
                while True:
                    line = input(f"\nPrompt {len(prompts) + 1} (blank=done, double blank=submit): ")
                    if line.strip() == "":
                        if prompts:
                            break
                        continue
                    if line.strip().lower() in {"quit", "exit", "/q"}:
                        raise SystemExit(0)
                    if line.strip().lower() == "/clear":
                        prompts = []
                        print("Batch cleared.")
                        continue
                    prompts.append(line.strip())
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not prompts:
                continue

            input_ids_list = []
            for prompt in prompts:
                messages = [{"role": "user", "content": prompt}]
                try:
                    ids = tokenizer.apply_chat_template(
                        messages, tokenize=True, add_generation_prompt=True,
                        enable_thinking=enable_thinking,
                    )
                except TypeError:
                    ids = tokenizer.apply_chat_template(
                        messages, tokenize=True, add_generation_prompt=True,
                    )
                ids = _normalize_token_ids(ids, tokenizer)
                input_ids_list.append(ids)

            max_prompt_len = max(len(ids) for ids in input_ids_list)
            if max_prompt_len > args.max_seq_len:
                print(f"Longest prompt ({max_prompt_len} tokens) exceeds max_seq_len ({args.max_seq_len}). Skipping.")
                continue

            start = time.time()
            generated_ids = model.generate_batch(
                input_ids_list, ws, paged_kv,
                max_new_tokens=args.max_tokens,
                eos_token_ids=EOS_TOKEN_IDS,
            )
            elapsed = time.time() - start
            total_tokens = sum(len(ids) for ids in generated_ids)

            for i, (prompt, ids) in enumerate(zip(prompts, generated_ids)):
                text = tokenizer.decode(ids, skip_special_tokens=True)
                print(f"\n--- Response {i + 1} ---")
                print(text)

            print(f"\n  [{len(prompts)} prompts, {total_tokens} tokens, {elapsed:.1f}s, {total_tokens/elapsed:.1f} tok/s]")
    finally:
        paged_kv.free()
        ws.free()
        model.free()


if __name__ == "__main__":
    main()
