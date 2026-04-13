#!/usr/bin/env python3
import argparse
import os
import sys
import time
import torch

sys.path.insert(0, os.path.dirname(__file__))
from helpers import GlmOps, get_model_path
from qwen3_model import Qwen3Model, _normalize_token_ids

from transformers import AutoTokenizer

QWEN3_REPO = "Qwen/Qwen3-0.6B"
EOS_TOKEN_IDS = {151645, 151643}


def chat_loop(model, tokenizer, max_new_tokens, no_think):
    messages = []
    enable_thinking = not no_think

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "/q"}:
            break
        if user_input.lower() == "/clear":
            messages = []
            print("Conversation cleared.")
            continue

        messages.append({"role": "user", "content": user_input})

        try:
            input_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            input_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
            )

        input_ids = _normalize_token_ids(input_ids, tokenizer)
        input_tensor = torch.tensor([input_ids], dtype=torch.int64)

        if len(input_ids) > model.max_seq_len:
            print(f"Prompt ({len(input_ids)} tokens) exceeds max_seq_len ({model.max_seq_len}). Truncating conversation.")
            messages.pop()
            continue

        print("\nAssistant: ", end="", flush=True)
        generated_ids = []
        start = time.time()
        token_count = 0

        for token_id in model.generate_tokens(
            input_tensor, max_new_tokens=max_new_tokens,
            eos_token_ids=EOS_TOKEN_IDS,
        ):
            generated_ids.append(token_id)
            token_count += 1
            chunk = tokenizer.decode([token_id], skip_special_tokens=False)
            print(chunk, end="", flush=True)

            if token_id in EOS_TOKEN_IDS:
                break

        elapsed = time.time() - start
        print(f"\n  [{token_count} tokens, {elapsed:.1f}s, {token_count/elapsed:.1f} tok/s]")

        assistant_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        messages.append({"role": "assistant", "content": assistant_text})

    model.free()


def main():
    parser = argparse.ArgumentParser(description="Interactive Qwen3-0.6B chat")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens per response")
    parser.add_argument("--no-think", action="store_true", help="Disable thinking mode")
    parser.add_argument("--gpu", type=int, default=None, help="GPU device ID (default: GLM_GPU env var or 0)")
    parser.add_argument("--max-seq-len", type=int, default=2048, help="Max sequence length")
    args = parser.parse_args()

    gpu_id = args.gpu if args.gpu is not None else int(os.environ.get("GLM_GPU", "0"))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"Loading model on GPU {gpu_id}...")
    glm = GlmOps(device_id=0)
    model = Qwen3Model.from_pretrained(glm, QWEN3_REPO, max_batch=1, max_seq_len=args.max_seq_len)
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_REPO)

    print(f"Qwen3-0.6B ready (max_tokens={args.max_tokens}, thinking={'off' if args.no_think else 'on'})")
    print("Type a message to chat. /clear to reset, /q to quit.")

    chat_loop(model, tokenizer, args.max_tokens, args.no_think)


if __name__ == "__main__":
    main()
