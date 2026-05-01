import gc
import pytest
import torch
from helpers import GlmOps, has_model_cached
from qwen3_model import Qwen3Model, Qwen3Config
from test_qwen3 import (
    qwen3_model_torch,
    load_qwen3_config,
    load_qwen3_weights,
)

QWEN3_REPO = "Qwen/Qwen3-0.6B"

pytestmark = pytest.mark.skipif(
    not has_model_cached(QWEN3_REPO),
    reason=f"{QWEN3_REPO} not in HF cache"
)


@pytest.fixture(scope="module")
def qwen3_model(glm):
    model = Qwen3Model.from_pretrained(glm, QWEN3_REPO, max_batch=1, max_seq_len=16)
    yield model
    model.free()
    gc.collect(); torch.cuda.empty_cache()


def test_qwen3_forward_vs_reference(qwen3_model, glm):
    model = qwen3_model
    device = torch.device("cuda", glm.device)
    cfg = model.cfg
    B, S = 1, 8

    input_ids = torch.tensor([[151643, 151644, 151645, 1, 2, 3, 4, 5]], dtype=torch.int64, device=device)

    our_logits = model.forward(input_ids)

    weights = load_qwen3_weights(device)
    ref_logits = qwen3_model_torch(input_ids, weights, cfg)

    max_diff = (our_logits - ref_logits.cpu()).abs().max().item()
    mean_diff = (our_logits - ref_logits.cpu()).abs().mean().item()
    print(f"  Max diff: {max_diff:.4f}, Mean diff: {mean_diff:.4f}")

    assert max_diff < 50.0, f"Max diff {max_diff} too large"
    assert mean_diff < 1.0, f"Mean diff {mean_diff} too large"

    our_top5 = our_logits[0, -1].topk(5).indices.tolist()
    ref_top5 = ref_logits[0, -1].topk(5).indices.tolist()
    assert our_top5 == ref_top5, f"Top-5 mismatch: ours={our_top5}, ref={ref_top5}"


def test_qwen3_forward_vs_hf(glm):
    device = torch.device("cuda", glm.device)
    transformers = pytest.importorskip("transformers")
    from transformers import AutoModelForCausalLM

    model = Qwen3Model.from_pretrained(glm, QWEN3_REPO, max_batch=1, max_seq_len=16)

    input_ids = torch.tensor([[151643, 151644, 151645, 1, 2, 3, 4, 5]], dtype=torch.int64, device=device)

    our_logits = model.forward(input_ids)

    model.free()
    del model
    gc.collect(); torch.cuda.empty_cache()

    hf_model = AutoModelForCausalLM.from_pretrained(
        QWEN3_REPO, torch_dtype=torch.bfloat16, device_map=device
    ).eval()

    with torch.no_grad():
        hf_logits = hf_model(input_ids).logits

    max_diff = (our_logits - hf_logits.cpu()).abs().max().item() 
    mean_diff = (our_logits - hf_logits.cpu()).abs().mean().item()
    print(f"  Max diff vs HF: {max_diff:.4f}, Mean diff: {mean_diff:.4f}")

    assert max_diff < 100.0, f"Max diff vs HF {max_diff} too large"
    assert mean_diff < 2.0, f"Mean diff vs HF {mean_diff} too large"

    our_top5 = our_logits[0, -1].topk(5).indices.tolist()
    hf_top5 = hf_logits[0, -1].topk(5).indices.tolist()
    assert our_top5 == hf_top5, f"Top-5 mismatch: ours={our_top5}, hf={hf_top5}"

    del hf_model
    gc.collect(); torch.cuda.empty_cache()


def test_qwen3_forward_multitoken(glm):
    device = torch.device("cuda", glm.device)
    model = Qwen3Model.from_pretrained(glm, QWEN3_REPO, max_batch=1, max_seq_len=16)

    weights = load_qwen3_weights(device)
    cfg = model.cfg

    for S in [1, 4, 8]:
        ids = torch.arange(1, S + 1, dtype=torch.int64, device=device).unsqueeze(0)
        our = model.forward(ids)
        ref = qwen3_model_torch(ids, weights, cfg)
        max_diff = (our - ref.cpu()).abs().max().item()
        mean_diff = (our - ref.cpu()).abs().mean().item()
        print(f"  S={S}: max_diff={max_diff:.4f}, mean_diff={mean_diff:.4f}")
        assert max_diff < 50.0, f"S={S}: max_diff {max_diff} too large"
    our_top1 = our[0, -1].topk(1).indices.item()
    ref_top1 = ref[0, -1].topk(1).indices.item()
    our_top5 = set(our[0, -1].topk(5).indices.tolist())
    ref_top5 = set(ref[0, -1].topk(5).indices.tolist())
    assert our_top1 == ref_top1, f"S={S}: Top-1 mismatch: ours={our_top1}, ref={ref_top1}"
    assert len(our_top5 & ref_top5) >= 4, f"S={S}: Top-5 set mismatch: ours={our_top5}, ref={ref_top5}"

    del weights
    model.free()
    del model
    gc.collect(); torch.cuda.empty_cache()


def test_qwen3_prefill_vs_forward(qwen3_model, glm):
    model = qwen3_model
    device = torch.device("cuda", glm.device)

    input_ids = torch.tensor([[151643, 151644, 151645, 1, 2, 3, 4, 5]], dtype=torch.int64, device=device)

    cache = model.create_flat_kv_cache()
    try:
        prefill_logits = model.prefill(input_ids, cache)
        forward_logits = model.forward(input_ids)

        max_diff = (prefill_logits - forward_logits[0, -1:]).abs().max().item()
        print(f"  Prefill vs forward max diff: {max_diff:.6f}")
        assert max_diff < 0.75, f"Prefill vs forward max diff {max_diff} too large"

        prefill_top5 = prefill_logits[0].topk(5).indices.tolist()
        forward_top5 = forward_logits[0, -1].topk(5).indices.tolist()
        assert prefill_top5 == forward_top5, f"Top-5 mismatch: prefill={prefill_top5}, forward={forward_top5}"
    finally:
        cache.free()


def test_qwen3_prefill_decode(glm):
    device = torch.device("cuda", glm.device)
    model = Qwen3Model.from_pretrained(glm, QWEN3_REPO, max_batch=1, max_seq_len=32)

    weights = load_qwen3_weights(device)
    cfg = model.cfg

    prompt = [151643, 151644, 151645, 1, 2, 3]
    input_ids = torch.tensor([prompt], dtype=torch.int64, device=device)

    cache = model.create_flat_kv_cache()
    try:
        prefill_logits = model.prefill(input_ids, cache)
        assert cache.cache_pos == len(prompt)

        ref_full = qwen3_model_torch(input_ids, weights, cfg)
        max_diff = (prefill_logits - ref_full[0, -1:].cpu()).abs().max().item()
        print(f"  Prefill vs reference max diff: {max_diff:.4f}")
        assert max_diff < 50.0, f"Prefill max diff {max_diff} too large"

        ref_top5 = set(ref_full[0, -1].topk(5).indices.tolist())
        prefill_top5 = set(prefill_logits[0].topk(5).indices.tolist())
        assert len(prefill_top5 & ref_top5) >= 4, f"Top-5 overlap < 4: prefill={prefill_top5}, ref={ref_top5}"

        num_decode_steps = 5
        all_ids = list(prompt)
        decode_logits = None
        for step in range(num_decode_steps):
            if step == 0:
                next_id = prefill_logits[0].argmax().item()
            else:
                next_id = decode_logits[0, 0].argmax().item()
            next_input = torch.tensor([[next_id]], dtype=torch.int64, device=device)

            decode_logits = model.decode(next_input, cache)
            assert cache.cache_pos == len(prompt) + step + 1

            all_ids.append(next_id)

            extended_ids = torch.tensor([all_ids], dtype=torch.int64, device=device)
            ref_extended = qwen3_model_torch(extended_ids, weights, cfg)

            diff = (decode_logits - ref_extended[0, -1:].cpu()).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            print(f"  Decode step {step}: max_diff={max_diff:.4f}, mean_diff={mean_diff:.4f}")

            assert max_diff < 50.0, f"Decode step {step}: max_diff {max_diff} too large"

            decode_top1 = decode_logits[0, 0].topk(1).indices.item()
            ref_top1 = ref_extended[0, -1].topk(1).indices.item()
            assert decode_top1 == ref_top1, \
                f"Decode step {step}: Top-1 mismatch: decode={decode_top1}, ref={ref_top1}"
    finally:
        cache.free()

    del weights
    model.free()
    del model
    gc.collect(); torch.cuda.empty_cache()


def test_qwen3_prefill_decode_vs_hf(glm):
    device = torch.device("cuda", glm.device)
    transformers = pytest.importorskip("transformers")
    from transformers import AutoModelForCausalLM

    prompt = [151643, 151644, 151645, 1, 2, 3]
    input_ids = torch.tensor([prompt], dtype=torch.int64, device=device)

    model = Qwen3Model.from_pretrained(glm, QWEN3_REPO, max_batch=1, max_seq_len=32)
    cache = model.create_flat_kv_cache()
    try:
        prefill_logits = model.prefill(input_ids, cache).cpu()

        num_decode_steps = 5
        all_ids = list(prompt)
        decode_logits_list = []
        decode_logits = None
        for step in range(num_decode_steps):
            if step == 0:
                next_id = prefill_logits[0].argmax().item()
            else:
                next_id = decode_logits[0, 0].argmax().item()
            next_input = torch.tensor([[next_id]], dtype=torch.int64, device=device)
            decode_logits = model.decode(next_input, cache).cpu()
            decode_logits_list.append(decode_logits)
            all_ids.append(next_id)
    finally:
        cache.free()

    model.free()
    del model
    gc.collect(); torch.cuda.empty_cache()

    hf_model = AutoModelForCausalLM.from_pretrained(
        QWEN3_REPO, torch_dtype=torch.bfloat16, device_map=device
    ).eval()

    with torch.no_grad():
        hf_out = hf_model(input_ids)
        hf_prefill_logits = hf_out.logits.cpu()
        hf_past = hf_out.past_key_values

    max_diff = (prefill_logits - hf_prefill_logits[0, -1:].cpu()).abs().max().item()
    print(f"  Prefill vs HF max diff: {max_diff:.4f}")
    assert max_diff < 100.0, f"Prefill vs HF max diff {max_diff} too large"

    for step in range(num_decode_steps):
        next_id = all_ids[len(prompt) + step]
        next_hf_input = torch.tensor([[next_id]], dtype=torch.int64, device=device)
        with torch.no_grad():
            hf_out = hf_model(next_hf_input, past_key_values=hf_past)
            hf_past = hf_out.past_key_values
            hf_decode_logits = hf_out.logits.cpu()

        max_diff = (decode_logits_list[step] - hf_decode_logits).abs().max().item()
        print(f"  Decode step {step} vs HF: max_diff={max_diff:.4f}")
        assert max_diff < 100.0, f"Decode step {step} vs HF: max_diff {max_diff} too large"

        decode_top1 = decode_logits_list[step][0, 0].topk(1).indices.item()
        hf_top1 = hf_decode_logits[0, -1].topk(1).indices.item()
        assert decode_top1 == hf_top1, \
            f"Decode step {step}: Top-1 mismatch: decode={decode_top1}, hf={hf_top1}"

    del hf_model
    gc.collect(); torch.cuda.empty_cache()


def test_qwen3_generate(glm):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(QWEN3_REPO)
    model = Qwen3Model.from_pretrained(glm, QWEN3_REPO, max_batch=1, max_seq_len=128)
    cache = model.create_flat_kv_cache()

    try:
        prompt = "The capital of France is"
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        generated = model.generate(input_ids, cache, max_new_tokens=20, eos_token_ids={151645, 151643})

        text = tokenizer.decode(input_ids[0].tolist() + generated, skip_special_tokens=True)
        assert len(generated) > 0, "generate() produced no tokens"
        assert "Paris" in text, f"Expected 'Paris' in generated text, got: {repr(text)}"
        print(f"  Generated {len(generated)} tokens: {repr(text)}")
    finally:
        cache.free()

    model.free()
    gc.collect(); torch.cuda.empty_cache()


def test_qwen3_generate_tokens(glm):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(QWEN3_REPO)
    model = Qwen3Model.from_pretrained(glm, QWEN3_REPO, max_batch=1, max_seq_len=128)
    cache = model.create_flat_kv_cache()

    try:
        prompt = "The capital of France is"
        input_ids = tokenizer.encode(prompt, return_tensors='pt')

        list_tokens = model.generate(input_ids, cache, max_new_tokens=20, eos_token_ids={151645, 151643})
        gen_tokens = list(model.generate_tokens(input_ids, cache, max_new_tokens=20, eos_token_ids={151645, 151643}))

        assert list_tokens == gen_tokens, f"generate_tokens() mismatch: {list_tokens} vs {gen_tokens}"
        print(f"  Both methods produced {len(gen_tokens)} tokens, identical")
    finally:
        cache.free()

    model.free()
    gc.collect(); torch.cuda.empty_cache()


def test_qwen3_generate_text(glm):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(QWEN3_REPO)
    model = Qwen3Model.from_pretrained(glm, QWEN3_REPO, max_batch=1, max_seq_len=256)
    cache = model.create_flat_kv_cache()

    try:
        text = model.generate_text(
            "The capital of France is", tokenizer, cache,
            max_new_tokens=30, enable_thinking=False,
        )
        print(f"  Generated text: {repr(text)}")
        assert "Paris" in text, f"Expected 'Paris' in generated text, got: {repr(text)}"
    finally:
        cache.free()

    model.free()
    gc.collect(); torch.cuda.empty_cache()


def test_qwen3_flash_vs_bmm_attention(glm):
    model = Qwen3Model.from_pretrained(glm, QWEN3_REPO, max_batch=1, max_seq_len=64)

    input_ids = torch.tensor([[151643, 151644, 151645, 1, 2, 3, 4, 5]], dtype=torch.int64)

    cache = model.create_flat_kv_cache()
    try:
        flash_logits = model.prefill(input_ids, cache)
    finally:
        cache.free()

    bmm_logits = model.forward(input_ids)

    bmm_last = bmm_logits[:, -1, :]
    max_diff = (flash_logits - bmm_last).abs().max().item()
    mean_diff = (flash_logits - bmm_last).abs().mean().item()
    print(f"  Flash vs BMM max diff: {max_diff:.6f}, mean diff: {mean_diff:.6f}")
    assert max_diff < 1.0, f"Flash vs BMM max diff {max_diff} too large"

    flash_top5 = flash_logits[0].topk(5).indices.tolist()
    bmm_top5 = bmm_last[0].topk(5).indices.tolist()
    assert flash_top5 == bmm_top5, f"Top-5 mismatch: flash={flash_top5}, bmm={bmm_top5}"

    model.free()
    gc.collect(); torch.cuda.empty_cache()
