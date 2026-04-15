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
    flat_cache = model.create_flat_kv_cache()
    try:
        prompt1 = [151643, 151644, 151645, 1, 2, 3]
        prompt2 = [151643, 151644, 1, 2, 3, 4, 5]

        batch_tokens = model.prefill_batch([prompt1, prompt2], ws, paged_kv)

        flat_cache.reset()
        single_logits_1 = model.prefill(torch.tensor([prompt1], dtype=torch.int64), flat_cache)
        single_token_1 = single_logits_1[0].argmax().item()

        flat_cache.reset()
        single_logits_2 = model.prefill(torch.tensor([prompt2], dtype=torch.int64), flat_cache)
        single_token_2 = single_logits_2[0].argmax().item()

        assert batch_tokens[0] == single_token_1, \
            f"Seq1 prefill token mismatch: batch={batch_tokens[0]}, single={single_token_1}"
        assert batch_tokens[1] == single_token_2, \
            f"Seq2 prefill token mismatch: batch={batch_tokens[1]}, single={single_token_2}"
    finally:
        paged_kv.free()
        flat_cache.free()


def test_batch_prefill_paged_then_decode(glm, qwen3_model, ws):
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

        batch_tokens = model.prefill_batch([prompt1, prompt2], ws, paged_kv)
        paged_kv.update_indptr()

        decode_tokens = model.decode_batch(batch_tokens, ws, paged_kv)

        assert isinstance(decode_tokens[0], int), f"Decode token 0 not int: {decode_tokens[0]}"
        assert isinstance(decode_tokens[1], int), f"Decode token 1 not int: {decode_tokens[1]}"
        print(f"  Paged prefill -> decode tokens: {decode_tokens}")
    finally:
        paged_kv.free()


def test_batch_prefill_append(glm, qwen3_model, ws):
    model = qwen3_model
    cfg = model.cfg
    n_kv = cfg.num_key_value_heads
    hd = cfg.head_dim
    n_layers = cfg.num_hidden_layers
    max_pages = 256

    paged_kv = PagedKVCache(glm, n_kv, hd, n_layers, max_pages, max_batch=1)
    flat_cache = model.create_flat_kv_cache()
    try:
        prompt1 = [151643, 151644, 151645, 1, 2, 3]
        prompt2_suffix = [4, 5, 6, 7]
        full_prompt = prompt1 + prompt2_suffix

        # Method 1: Full paged prefill of combined prompt
        paged_kv.reset(1)
        tokens_full = model.prefill_batch([full_prompt], ws, paged_kv)
        paged_kv.update_indptr()

        # Method 2: Prefill first part, then append second part
        paged_kv.reset(1)
        tokens_first = model.prefill_batch([prompt1], ws, paged_kv)
        paged_kv.update_indptr()
        tokens_append = model.prefill_batch_append([prompt2_suffix], ws, paged_kv)

        # Full prefill reference
        flat_cache.reset()
        single_logits = model.prefill(torch.tensor([full_prompt], dtype=torch.int64), flat_cache)
        single_token = single_logits[0].argmax().item()

        assert tokens_full[0] == single_token, \
            f"Full paged prefill mismatch: paged={tokens_full[0]}, single={single_token}"
        assert tokens_append[0] == tokens_full[0], \
            f"Append prefill mismatch: append={tokens_append[0]}, full={tokens_full[0]}"
        print(f"  Paged append: full={tokens_full[0]}, append={tokens_append[0]}, single={single_token}")
    finally:
        paged_kv.free()
        flat_cache.free()


def test_batch_prefill_truncate_append(glm, qwen3_model, ws):
    model = qwen3_model
    cfg = model.cfg
    n_kv = cfg.num_key_value_heads
    hd = cfg.head_dim
    n_layers = cfg.num_hidden_layers
    max_pages = 256

    paged_kv = PagedKVCache(glm, n_kv, hd, n_layers, max_pages, max_batch=1)
    flat_cache = model.create_flat_kv_cache()
    try:
        prompt1 = [151643, 151644, 151645, 1, 2, 3]
        prompt2_suffix = [4, 5, 6, 7]
        full_prompt = prompt1 + prompt2_suffix

        # Prefill first part, then append, then truncate and append different suffix
        paged_kv.reset(1)
        model.prefill_batch([prompt1], ws, paged_kv)
        paged_kv.update_indptr()
        model.prefill_batch_append([prompt2_suffix], ws, paged_kv)
        paged_kv.update_indptr()

        # Now truncate back to prompt1 length and append prompt2_suffix again
        paged_kv.truncate(0, len(prompt1))
        paged_kv.update_indptr()
        tokens_trunc_append = model.prefill_batch_append([prompt2_suffix], ws, paged_kv)
        paged_kv.update_indptr()

        # Reference: full prefill of combined prompt
        paged_kv2 = PagedKVCache(glm, n_kv, hd, n_layers, max_pages, max_batch=1)
        try:
            paged_kv2.reset(1)
            tokens_full = model.prefill_batch([full_prompt], ws, paged_kv2)
            paged_kv2.update_indptr()

            assert tokens_trunc_append[0] == tokens_full[0], \
                f"Truncate+append mismatch: trunc_append={tokens_trunc_append[0]}, full={tokens_full[0]}"
            print(f"  Truncate+append: trunc_append={tokens_trunc_append[0]}, full={tokens_full[0]}")
        finally:
            paged_kv2.free()

        # Also verify against flat cache reference
        flat_cache.reset()
        single_logits = model.prefill(torch.tensor([full_prompt], dtype=torch.int64), flat_cache)
        single_token = single_logits[0].argmax().item()
        assert tokens_trunc_append[0] == single_token, \
            f"Truncate+append vs single mismatch: trunc_append={tokens_trunc_append[0]}, single={single_token}"
    finally:
        paged_kv.free()
        flat_cache.free()


def test_batch_decode_vs_single(glm, qwen3_model, ws):
    model = qwen3_model
    cfg = model.cfg
    n_kv = cfg.num_key_value_heads
    hd = cfg.head_dim
    n_layers = cfg.num_hidden_layers
    max_pages = 128

    paged_kv = PagedKVCache(glm, n_kv, hd, n_layers, max_pages, max_batch=4)
    flat_cache = model.create_flat_kv_cache()
    try:
        prompt1 = [151643, 151644, 151645, 1, 2, 3]
        prompt2 = [151643, 151644, 1, 2, 3, 4, 5]

        batch_tokens = model.prefill_batch([prompt1, prompt2], ws, paged_kv)

        token1 = batch_tokens[0]
        token2 = batch_tokens[1]

        flat_cache.reset()
        single_logits_1 = model.prefill(torch.tensor([prompt1], dtype=torch.int64), flat_cache)
        single_decode_logits_1 = model.decode(torch.tensor([[single_logits_1[0].argmax().item()]], dtype=torch.int64), flat_cache)
        single_decode_token_1 = single_decode_logits_1[0, 0].argmax().item()

        flat_cache.reset()
        single_logits_2 = model.prefill(torch.tensor([prompt2], dtype=torch.int64), flat_cache)
        single_decode_logits_2 = model.decode(torch.tensor([[single_logits_2[0].argmax().item()]], dtype=torch.int64), flat_cache)
        single_decode_token_2 = single_decode_logits_2[0, 0].argmax().item()

        batch_decode_tokens = model.decode_batch([token1, token2], ws, paged_kv)

        assert batch_decode_tokens[0] == single_decode_token_1, \
            f"Seq1 decode token mismatch: batch={batch_decode_tokens[0]}, single={single_decode_token_1}"
        assert batch_decode_tokens[1] == single_decode_token_2, \
            f"Seq2 decode token mismatch: batch={batch_decode_tokens[1]}, single={single_decode_token_2}"
    finally:
        paged_kv.free()
        flat_cache.free()


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

        batch_tokens = model.prefill_batch([prompt1, prompt2], ws, paged_kv)

        batch_tokens = [
            [batch_tokens[0]],
            [batch_tokens[1]],
        ]

        num_steps = 5
        for step in range(num_steps):
            decode_ids = [t[0] for t in batch_tokens]
            next_tokens = model.decode_batch(decode_ids, ws, paged_kv)

            batch_tokens = [
                [next_tokens[0]],
                [next_tokens[1]],
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
    flat_cache = model.create_flat_kv_cache()
    try:
        prompt1 = [151643, 151644, 151645, 1, 2, 3, 4, 5, 6, 7]
        prompt2 = [151643, 151644, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        batch_generated = model.generate_batch(
            [prompt1, prompt2], ws, paged_kv,
            max_new_tokens=max_new_tokens,
        )

        flat_cache.reset()
        single1 = model.generate(
            torch.tensor([prompt1], dtype=torch.int64), flat_cache,
            max_new_tokens=max_new_tokens,
        )

        flat_cache.reset()
        single2 = model.generate(
            torch.tensor([prompt2], dtype=torch.int64), flat_cache,
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
        flat_cache.free()


def test_cuda_graph_decode(glm, qwen3_model, ws):
    model = qwen3_model
    cfg = model.cfg
    n_kv = cfg.num_key_value_heads
    hd = cfg.head_dim
    n_layers = cfg.num_hidden_layers
    max_pages = 128

    paged_kv = PagedKVCache(glm, n_kv, hd, n_layers, max_pages, max_batch=4)
    try:
        prompt = [151643, 151644, 151645, 1, 2988, 279, 1716, 364]

        # Step 1: Normal decode to get reference tokens
        tokens = model.prefill_batch([prompt], ws, paged_kv)
        state_ref = model.decode_batch_plan([tokens[0]], ws, paged_kv,
                                             enable_cuda_graph=True)
        model.decode_batch_forward(state_ref, ws, paged_kv)
        tokens_ref = model.decode_batch_read(state_ref)
        print(f"  Reference tokens: {tokens_ref}")

        # Step 2: Capture the decode forward pass
        # Note: during capture, kernels are NOT executed — only recorded
        paged_kv.reset(1)
        tokens2 = model.prefill_batch([prompt], ws, paged_kv)
        state = model.decode_batch_plan([tokens2[0]], ws, paged_kv,
                                         enable_cuda_graph=True)
        glm.graph_begin_capture()
        model.decode_batch_forward(state, ws, paged_kv)
        graph = glm.graph_end_capture()
        assert graph is not None and graph != 0, "graph_end_capture returned null"
        graph_exec = glm.graph_instantiate(graph)
        assert graph_exec is not None and graph_exec != 0, "graph_instantiate returned null"

        # Step 3: Replay the captured graph and compare against reference
        paged_kv.reset(1)
        tokens3 = model.prefill_batch([prompt], ws, paged_kv)
        state2 = model.decode_batch_plan([tokens3[0]], ws, paged_kv,
                                          enable_cuda_graph=True)

        glm.graph_launch(graph_exec)
        glm.synchronize()

        tokens_replay = model.decode_batch_read(state2)
        print(f"  Replay tokens:    {tokens_replay}")

        assert tokens_replay == tokens_ref, \
            f"Graph replay mismatch: replay={tokens_replay}, ref={tokens_ref}"

        glm.graph_exec_destroy(graph_exec)
        glm.graph_destroy(graph)
    finally:
        paged_kv.free()


def test_cuda_graph_multi_step_decode(glm, qwen3_model, ws):
    """Capture a decode graph once, replay it multiple times with growing KV cache,
    and verify each step produces the same token as normal decode."""
    model = qwen3_model
    cfg = model.cfg
    n_kv = cfg.num_key_value_heads
    hd = cfg.head_dim
    n_layers = cfg.num_hidden_layers
    max_pages = 128
    num_steps = 10

    paged_kv = PagedKVCache(glm, n_kv, hd, n_layers, max_pages, max_batch=4)
    try:
        prompt = [151643, 151644, 151645, 1, 2988, 279, 1716, 364]

        # Reference: normal decode for num_steps
        paged_kv.reset(1)
        tokens = model.prefill_batch([prompt], ws, paged_kv)
        paged_kv.update_indptr()

        ref_tokens = []
        current = tokens[0]
        for step in range(num_steps):
            state = model.decode_batch_plan([current], ws, paged_kv,
                                             enable_cuda_graph=True)
            model.decode_batch_forward(state, ws, paged_kv)
            current = model.decode_batch_read(state)[0]
            ref_tokens.append(current)
        print(f"  Reference tokens: {ref_tokens}")

        # Graph decode: warmup + capture + multi-step replay
        paged_kv.reset(1)
        tokens = model.prefill_batch([prompt], ws, paged_kv)
        paged_kv.update_indptr()

        # Warmup: one decode step with enable_cuda_graph=True
        current = tokens[0]
        warmup_state = model.decode_batch_plan([current], ws, paged_kv,
                                                enable_cuda_graph=True)
        model.decode_batch_forward(warmup_state, ws, paged_kv)
        current = model.decode_batch_read(warmup_state)[0]
        assert current == ref_tokens[0], \
            f"Warmup mismatch: {current} != {ref_tokens[0]}"

        # Capture the decode forward pass
        state = model.decode_batch_plan([current], ws, paged_kv,
                                         enable_cuda_graph=True)
        glm.graph_begin_capture()
        model.decode_batch_forward(state, ws, paged_kv)
        graph = glm.graph_end_capture()
        assert graph is not None and graph != 0, "graph_end_capture returned null"
        graph_exec = glm.graph_instantiate(graph)
        assert graph_exec is not None and graph_exec != 0, "graph_instantiate returned null"
        glm.graph_destroy(graph)

        # First replay: graph was captured with state for token after warmup
        glm.graph_launch(graph_exec)
        glm.synchronize()
        current = model.decode_batch_read(state)[0]
        graph_tokens = [ref_tokens[0], current]
        assert current == ref_tokens[1], \
            f"Replay step 1 mismatch: {current} != {ref_tokens[1]}"

        # Subsequent replays: plan updates inputs, then replay graph
        for step in range(2, num_steps):
            state = model.decode_batch_plan([current], ws, paged_kv,
                                             enable_cuda_graph=True)
            glm.graph_launch(graph_exec)
            glm.synchronize()
            current = model.decode_batch_read(state)[0]
            graph_tokens.append(current)
            assert current == ref_tokens[step], \
                f"Replay step {step} mismatch: {current} != {ref_tokens[step]}"

        print(f"  Graph tokens:     {graph_tokens}")
        assert graph_tokens == ref_tokens, \
            f"Token sequence mismatch: graph={graph_tokens}, ref={ref_tokens}"

        glm.graph_exec_destroy(graph_exec)
    finally:
        paged_kv.free()
