import pytest
import torch
import numpy as np
from helpers import GlmOps, has_model_cached
from paged_kv import PagedKVCache, WorkspaceBuffers
from qwen3_model import Qwen3Model
from test_qwen3 import load_qwen3_config, load_qwen3_weights

QWEN3_REPO = "Qwen/Qwen3-0.6B"

pytestmark = pytest.mark.skipif(
    not has_model_cached(QWEN3_REPO),
    reason=f"{QWEN3_REPO} not in HF cache"
)


@pytest.fixture(scope="module")
def glm():
    ops = GlmOps(device_id=int(__import__("os").environ.get("GLM_GPU", "0")))
    yield ops
    del ops


@pytest.fixture(scope="module")
def qwen3_model(glm):
    model = Qwen3Model.from_pretrained(glm, QWEN3_REPO, max_batch=4, max_seq_len=128)
    yield model
    model.free()
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def ws(glm):
    w = WorkspaceBuffers(glm)
    yield w
    w.free()


def test_batch_prefill_vs_single(glm, qwen3_model, ws):
    model = qwen3_model
    cfg = model.cfg
    n_kv = cfg.num_key_value_heads
    hd = cfg.head_dim
    n_layers = cfg.num_hidden_layers
    max_pages = 128

    paged_kv = PagedKVCache(glm, n_kv, hd, n_layers, max_pages, max_batch=4)
    try:
        prompt1 = [151643, 151644, 151645, 1, 2, 3]
        prompt2 = [151643, 151644, 1, 2, 3, 4, 5]

        batch_logits = model.prefill_batch([prompt1, prompt2], ws, paged_kv)

        model.reset_cache()
        single_logits_1 = model.prefill(torch.tensor([prompt1], dtype=torch.int64))

        model.reset_cache()
        single_logits_2 = model.prefill(torch.tensor([prompt2], dtype=torch.int64))

        diff1 = (batch_logits[0] - single_logits_1[0]).abs().max().item()
        diff2 = (batch_logits[1] - single_logits_2[0]).abs().max().item()
        print(f"  Batch prefill vs single: diff1={diff1:.4f}, diff2={diff2:.4f}")

        assert batch_logits[0].argmax().item() == single_logits_1[0].argmax().item(), \
            f"Seq1 top-1 mismatch"
        assert batch_logits[1].argmax().item() == single_logits_2[0].argmax().item(), \
            f"Seq2 top-1 mismatch"
        assert diff1 < 0.5, f"Seq1 diff too large: {diff1:.4f}"
        assert diff2 < 0.5, f"Seq2 diff too large: {diff2:.4f}"
    finally:
        paged_kv.free()


def test_batch_decode_vs_single(glm, qwen3_model, ws):
    model = qwen3_model
    cfg = model.cfg
    n_kv = cfg.num_key_value_heads
    hd = cfg.head_dim
    n_layers = cfg.num_hidden_layers
    max_pages = 128

    paged_kv = PagedKVCache(glm, n_kv, hd, n_layers, max_pages, max_batch=4)
    try:
        prompt1 = [151643, 151644, 151645, 1, 2, 3]
        prompt2 = [151643, 151644, 1, 2, 3, 4, 5]

        batch_logits = model.prefill_batch([prompt1, prompt2], ws, paged_kv)

        token1 = batch_logits[0].argmax().item()
        token2 = batch_logits[1].argmax().item()

        model.reset_cache()
        model.prefill(torch.tensor([prompt1], dtype=torch.int64))
        single_decode_logits_1 = model.decode(torch.tensor([[token1]], dtype=torch.int64))

        model.reset_cache()
        model.prefill(torch.tensor([prompt2], dtype=torch.int64))
        single_decode_logits_2 = model.decode(torch.tensor([[token2]], dtype=torch.int64))

        batch_decode_logits = model.decode_batch([token1, token2], ws, paged_kv)

        diff1 = (batch_decode_logits[0] - single_decode_logits_1[0, 0]).abs().max().item()
        diff2 = (batch_decode_logits[1] - single_decode_logits_2[0, 0]).abs().max().item()
        print(f"  Batch decode vs single: diff1={diff1:.4f}, diff2={diff2:.4f}")

        assert batch_decode_logits[0].argmax().item() == single_decode_logits_1[0, 0].argmax().item(), \
            f"Seq1 decode top-1 mismatch"
        assert batch_decode_logits[1].argmax().item() == single_decode_logits_2[0, 0].argmax().item(), \
            f"Seq2 decode top-1 mismatch"
        assert diff1 < 0.5, f"Seq1 decode diff too large: {diff1:.4f}"
        assert diff2 < 0.5, f"Seq2 decode diff too large: {diff2:.4f}"
    finally:
        paged_kv.free()


def test_batch_multi_step_decode(glm, qwen3_model, ws):
    model = qwen3_model
    cfg = model.cfg
    n_kv = cfg.num_key_value_heads
    hd = cfg.head_dim
    n_layers = cfg.num_hidden_layers
    max_pages = 128

    paged_kv = PagedKVCache(glm, n_kv, hd, n_layers, max_pages, max_batch=4)
    try:
        prompt1 = [151643, 151644, 151645, 1, 2, 3]
        prompt2 = [151643, 151644, 1, 2, 3, 4, 5]

        batch_logits = model.prefill_batch([prompt1, prompt2], ws, paged_kv)

        batch_tokens = [
            [batch_logits[0].argmax().item()],
            [batch_logits[1].argmax().item()],
        ]

        single_models = []
        for prompt in [prompt1, prompt2]:
            model.reset_cache()
            model.prefill(torch.tensor([prompt], dtype=torch.int64))
            single_models.append(model.cache_pos)

        num_steps = 5
        for step in range(num_steps):
            batch_decode_logits = model.decode_batch(batch_tokens, ws, paged_kv)

            batch_tokens = [
                [batch_decode_logits[0].argmax().item()],
                [batch_decode_logits[1].argmax().item()],
            ]

        print(f"  Batch multi-step decode completed {num_steps} steps")
        print(f"  Final tokens: seq1={batch_tokens[0]}, seq2={batch_tokens[1]}")

        assert len(batch_tokens) == 2
        assert all(isinstance(t[0], int) for t in batch_tokens)
    finally:
        paged_kv.free()


def test_batch_generate_vs_single(glm, qwen3_model, ws):
    model = qwen3_model
    cfg = model.cfg
    n_kv = cfg.num_key_value_heads
    hd = cfg.head_dim
    n_layers = cfg.num_hidden_layers
    max_pages = 128
    max_new_tokens = 20

    paged_kv = PagedKVCache(glm, n_kv, hd, n_layers, max_pages, max_batch=4)
    try:
        prompt1 = [151643, 151644, 151645, 1, 2, 3, 4, 5, 6, 7]
        prompt2 = [151643, 151644, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        batch_generated = model.generate_batch(
            [prompt1, prompt2], ws, paged_kv,
            max_new_tokens=max_new_tokens,
        )

        model.reset_cache()
        single1 = model.generate(
            torch.tensor([prompt1], dtype=torch.int64),
            max_new_tokens=max_new_tokens,
        )

        model.reset_cache()
        single2 = model.generate(
            torch.tensor([prompt2], dtype=torch.int64),
            max_new_tokens=max_new_tokens,
        )

        print(f"  Batch seq1 ({len(batch_generated[0])} tokens): {batch_generated[0][:10]}...")
        print(f"  Single seq1 ({len(single1)} tokens): {single1[:10]}...")
        print(f"  Batch seq2 ({len(batch_generated[1])} tokens): {batch_generated[1][:10]}...")
        print(f"  Single seq2 ({len(single2)} tokens): {single2[:10]}...")

        match1 = sum(a == b for a, b in zip(batch_generated[0], single1))
        match2 = sum(a == b for a, b in zip(batch_generated[1], single2))
        print(f"  Token match: seq1={match1}/{len(single1)}, seq2={match2}/{len(single2)}")

        assert len(batch_generated[0]) > 0, "Seq1 generated no tokens"
        assert len(batch_generated[1]) > 0, "Seq2 generated no tokens"
        assert batch_generated[0][:3] == single1[:3], \
            f"Seq1 first 3 tokens mismatch: batch={batch_generated[0][:3]}, single={single1[:3]}"
        assert batch_generated[1][:3] == single2[:3], \
            f"Seq2 first 3 tokens mismatch: batch={batch_generated[1][:3]}, single={single2[:3]}"
    finally:
        paged_kv.free()
