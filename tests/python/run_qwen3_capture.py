#!/usr/bin/env python3
"""Qwen3 interactive chat with CUDA graph capture for decode acceleration.

Graph capture flow:
  1. Prefill prompt (normal, not captured)
  2. Warmup decode steps (normal execution, enable_cuda_graph=True)
  3. Capture CUDA graph of decode_batch_forward (once)
  4. Replay captured step immediately (kernels don't execute during capture)
  5. Decode loop: plan -> graph_launch -> sync -> read (graph replay every step)

Multi-turn: re-prefill full conversation each turn, reuse captured graph.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from helpers import GlmOps
from paged_kv import PagedKVCache, WorkspaceBuffers
from qwen3_model import Qwen3Model

QWEN3_REPO = "Qwen/Qwen3-0.6B"
EOS_TOKEN_IDS = {151645, 151643}


def load_tokenizer():
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(QWEN3_REPO)
    except ImportError:
        return None


def tokenize_messages(tokenizer, messages, enable_thinking=False):
    if tokenizer is None:
        raise RuntimeError("transformers required for tokenization")
    try:
        result = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        result = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
        )
    if hasattr(result, 'input_ids'):
        result = result['input_ids']
    if isinstance(result, list) and len(result) > 0 and isinstance(result[0], list):
        result = result[0]
    return result


def decode_tokens(tokenizer, tokens):
    if tokenizer is None:
        return f"[{len(tokens)} tokens]"
    return tokenizer.decode(tokens, skip_special_tokens=True)


def generate_response(model, glm, ws, paged_kv, input_ids, graph_exec,
                      max_new_tokens, warmup_steps):
    """Generate response using CUDA graph replay for decode.

    Returns (generated_tokens, graph_exec, timing_info).
    graph_exec is None on first call (needs warmup+capture), or the captured graph exec.
    """
    generated_tokens = []
    timing = {"prefill_ms": 0, "warmup_ms": [], "capture_ms": 0,
              "replay_ms": [], "first_token": True}

    # Prefill
    t0 = time.perf_counter()
    tokens = model.prefill_batch([input_ids], ws, paged_kv)
    timing["prefill_ms"] = (time.perf_counter() - t0) * 1000
    current_token = tokens[0]
    generated_tokens.append(current_token)

    if current_token in EOS_TOKEN_IDS:
        return generated_tokens, graph_exec, timing

    # Warmup + capture (first call only)
    if graph_exec is None:
        for i in range(warmup_steps):
            if current_token in EOS_TOKEN_IDS:
                return generated_tokens, None, timing
            t0 = time.perf_counter()
            state = model.decode_batch_plan([current_token], ws, paged_kv, enable_cuda_graph=True)
            model.decode_batch_forward(state, ws, paged_kv)
            current_token = model.decode_batch_read(state)[0]
            timing["warmup_ms"].append((time.perf_counter() - t0) * 1000)
            generated_tokens.append(current_token)

        if current_token in EOS_TOKEN_IDS:
            return generated_tokens, None, timing

        # Capture graph
        t0 = time.perf_counter()
        state = model.decode_batch_plan([current_token], ws, paged_kv, enable_cuda_graph=True)
        glm.graph_begin_capture()
        model.decode_batch_forward(state, ws, paged_kv)
        graph = glm.graph_end_capture()
        assert graph is not None and graph != 0, "Graph capture failed"
        graph_exec = glm.graph_instantiate(graph)
        assert graph_exec is not None and graph_exec != 0, "Graph instantiation failed"
        glm.graph_destroy(graph)
        timing["capture_ms"] = (time.perf_counter() - t0) * 1000

        # Replay the captured step (forward wasn't executed during capture)
        glm.graph_launch(graph_exec)
        glm.synchronize()
        current_token = model.decode_batch_read(state)[0]
        generated_tokens.append(current_token)

    # Decode loop with graph replay
    remaining = max_new_tokens - len(generated_tokens)
    for _ in range(remaining):
        if current_token in EOS_TOKEN_IDS:
            break
        t0 = time.perf_counter()
        state = model.decode_batch_plan([current_token], ws, paged_kv, enable_cuda_graph=True)
        glm.graph_launch(graph_exec)
        glm.synchronize()
        current_token = model.decode_batch_read(state)[0]
        timing["replay_ms"].append((time.perf_counter() - t0) * 1000)
        generated_tokens.append(current_token)

    return generated_tokens, graph_exec, timing


def interactive_chat(model, glm, ws, paged_kv, tokenizer, args):
    """Interactive chat loop with CUDA graph capture."""
    messages = []
    graph_exec = None

    print(f"Qwen3-0.6B  |  GPU {args.gpu}  |  max_seq_len={args.max_seq_len}")
    print(f"Graph capture: {args.warmup_steps} warmup steps, max {args.max_new_tokens} tokens/turn")
    print("Type /quit to exit, /clear to reset conversation\n")

    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input == "/quit":
            break
        if user_input == "/clear":
            messages = []
            graph_exec = None
            print("Conversation cleared.\n")
            continue

        messages.append({"role": "user", "content": user_input})

        input_ids = tokenize_messages(tokenizer, messages, enable_thinking=args.thinking)

        if len(input_ids) > args.max_seq_len - args.max_new_tokens:
            print(f"Warning: prompt ({len(input_ids)} tokens) too long, truncating conversation")
            while len(input_ids) > args.max_seq_len - args.max_new_tokens and len(messages) > 1:
                messages = messages[2:]  # remove oldest user+assistant pair
                input_ids = tokenize_messages(tokenizer, messages, enable_thinking=args.thinking)
            if len(input_ids) > args.max_seq_len - args.max_new_tokens:
                print("Conversation too long even after truncation. Use /clear to reset.")
                messages.pop()
                continue

        # Reset KV cache and prefill
        paged_kv.reset(1)
        tokens, graph_exec, timing = generate_response(
            model, glm, ws, paged_kv, input_ids, graph_exec,
            args.max_new_tokens, args.warmup_steps
        )

        # Strip EOS tokens from output
        response_tokens = [t for t in tokens if t not in EOS_TOKEN_IDS]
        response_text = decode_tokens(tokenizer, response_tokens)

        # Print response
        sys.stdout.write(response_text + "\n\n")
        sys.stdout.flush()

        # Add assistant response to conversation history
        messages.append({"role": "assistant", "content": response_text})

        # Print timing for first turn
        if timing.get("first_token"):
            timing["first_token"] = False
            if timing["capture_ms"] > 0:
                print(f"  [prefill {timing['prefill_ms']:.0f}ms + "
                      f"capture {timing['capture_ms']:.0f}ms + "
                      f"{len(timing['warmup_ms'])} warmup]")


def single_prompt(model, glm, ws, paged_kv, tokenizer, args):
    """Single prompt mode: generate and print timing."""
    messages = [{"role": "user", "content": args.prompt}]
    input_ids = tokenize_messages(tokenizer, messages, enable_thinking=args.thinking)

    print(f"Prompt: {args.prompt}")
    print(f"Tokens: {len(input_ids)}")

    paged_kv.reset(1)
    tokens, graph_exec, timing = generate_response(
        model, glm, ws, paged_kv, input_ids, None,
        args.max_new_tokens, args.warmup_steps
    )

    response_tokens = [t for t in tokens if t not in EOS_TOKEN_IDS]
    response_text = decode_tokens(tokenizer, response_tokens)

    # Timing
    print(f"\nPrefill: {timing['prefill_ms']:.1f}ms")
    if timing["capture_ms"] > 0:
        print(f"Graph capture: {timing['capture_ms']:.1f}ms")
    if timing["warmup_ms"]:
        avg = sum(timing["warmup_ms"]) / len(timing["warmup_ms"])
        print(f"Warmup decode: {avg:.2f}ms avg ({len(timing['warmup_ms'])} steps)")
    if timing["replay_ms"]:
        avg = sum(timing["replay_ms"]) / len(timing["replay_ms"])
        p50 = sorted(timing["replay_ms"])[len(timing["replay_ms"]) // 2]
        print(f"Graph replay: avg={avg:.2f}ms  p50={p50:.2f}ms  "
              f"min={min(timing['replay_ms']):.2f}ms  max={max(timing['replay_ms']):.2f}ms  "
              f"({len(timing['replay_ms'])} steps, {1000/avg:.0f} tok/s)")

    print(f"\nResponse: {response_text}")

    # Cleanup
    if graph_exec is not None:
        glm.graph_exec_destroy(graph_exec)


def main():
    parser = argparse.ArgumentParser(description="Qwen3 chat with CUDA graph capture")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Single prompt mode (default: interactive chat)")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--gpu", type=int, default=int(os.environ.get("GLM_GPU", "0")))
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--max-pages", type=int, default=512)
    parser.add_argument("--thinking", action="store_true", default=True,
                        help="Enable thinking mode")
    args = parser.parse_args()

    glm = GlmOps(device_id=args.gpu)
    model = Qwen3Model.from_pretrained(glm, QWEN3_REPO, max_batch=1, max_seq_len=args.max_seq_len)
    cfg = model.cfg
    tokenizer = load_tokenizer()
    ws = WorkspaceBuffers(glm)
    paged_kv = PagedKVCache(
        glm, cfg.num_key_value_heads, cfg.head_dim,
        cfg.num_hidden_layers, args.max_pages, max_batch=1
    )

    try:
        if args.prompt:
            single_prompt(model, glm, ws, paged_kv, tokenizer, args)
        else:
            interactive_chat(model, glm, ws, paged_kv, tokenizer, args)
    finally:
        paged_kv.free()
        ws.free()
        model.free()


if __name__ == "__main__":
    main()
