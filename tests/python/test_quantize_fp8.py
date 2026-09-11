"""Blockwise E4M3 quantization and native-FP8 sparse MLA input correctness."""
import math
import pytest
import torch

from test_ds_mla_quant import ref_quantize_ds_mla


@pytest.mark.parametrize('block_size,input_offset,output_offset', [
    (1, 0, 0), (17, 0, 0), (64, 0, 0), (128, 0, 0), (256, 0, 0),
    (128, 1, 0), (128, 0, 1), (128, 1, 1),
])
def test_quantize_fp8(glm, device, block_size, input_offset, output_offset):
    torch.manual_seed(432)
    # An odd number of blocks exercises the final partially occupied warp.
    x = torch.randn(3 * 7 * block_size + input_offset, device=device, dtype=torch.bfloat16)
    x = x[input_offset:].view(3, 7, block_size)
    x[0, 0] = 0
    x[0, 1] = 1e-8
    x[0, 2] = 448
    x[0, 3] = -450
    values = torch.empty(x.numel() + output_offset, device=device, dtype=torch.float8_e4m3fn)
    values = values[output_offset:].view_as(x)
    scales = torch.empty(x.shape[:-1], device=device, dtype=torch.float32)
    torch.cuda.synchronize()
    glm.quantize_fp8(x.data_ptr(), values.data_ptr(), scales.data_ptr(), scales.numel(), block_size)
    torch.cuda.synchronize()
    raw = x.float().abs().amax(-1).clamp_min(1e-4) / 448
    expected_scales = torch.exp2(torch.ceil(torch.log2(raw)))
    expected = (x.float() / expected_scales[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    torch.testing.assert_close(scales, expected_scales, rtol=0, atol=0)
    torch.testing.assert_close(values.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0)


@pytest.mark.parametrize('num_heads', [1, 4, 8, 16, 32, 64, 128])
@pytest.mark.parametrize('decode', [False, True])
def test_sparse_mla_prequantized_q(glm, device, num_heads, decode):
    torch.manual_seed(123)
    h8_prefill = num_heads == 8 and not decode
    valid_lengths = [0, 1, 63, 65, 129, 256, 2048] if h8_prefill else [1, 63, 65, 129, 256]
    tokens, topk, splits = len(valid_lengths), 2048, 32
    kv_tokens = max(valid_lengths)
    q = torch.randn(tokens, num_heads, 512, device=device, dtype=torch.bfloat16)
    q[0, 0] = 0
    rope = torch.randn(tokens, num_heads, 64, device=device, dtype=torch.bfloat16)
    values = torch.empty_like(q, dtype=torch.float8_e4m3fn)
    scales = torch.empty(tokens, num_heads, 4, device=device, dtype=torch.float32)
    ckv = torch.randn(kv_tokens, 512, device=device, dtype=torch.bfloat16)
    kpe = torch.randn(kv_tokens, 64, device=device, dtype=torch.bfloat16)
    kv = ref_quantize_ds_mla(ckv, kpe, 512, 64).reshape(kv_tokens // 64, 64, 656)
    indices = torch.full((tokens, topk), -1, device=device, dtype=torch.int32)
    lengths = torch.tensor(valid_lengths, device=device, dtype=torch.int32)
    for t, length in enumerate(lengths.tolist()):
        indices[t, :length] = torch.randperm(kv_tokens, device=device)[:length].int()
    mid_out = torch.empty(tokens, num_heads, splits, 512, device=device, dtype=torch.bfloat16)
    mid_lse = torch.empty(tokens, num_heads, splits, device=device)
    output = torch.empty_like(q)
    lse = torch.empty(tokens, num_heads, device=device)
    expected = torch.empty_like(output)
    expected_lse = torch.empty_like(lse)
    torch.cuda.synchronize()

    glm.quantize_fp8(q.data_ptr(), values.data_ptr(), scales.data_ptr(), scales.numel(), 128)

    def run(query, out, out_lse, q_scales=None):
        if decode:
            glm.sparse_mla_decode_split_q(
                query.data_ptr(), rope.data_ptr(), kv.data_ptr(), indices.data_ptr(),
                mid_out.data_ptr(), mid_lse.data_ptr(), out.data_ptr(), out_lse.data_ptr(),
                tokens, num_heads, topk, splits, 0.0791, 64 * 656,
                topk_length=lengths.data_ptr(), q_scales=q_scales)
        else:
            glm.sparse_mla_prefill_split_q(
                query.data_ptr(), rope.data_ptr(), kv.data_ptr(), indices.data_ptr(),
                out.data_ptr(), out_lse.data_ptr(), tokens, num_heads, topk, 0.0791, 64 * 656,
                topk_length=lengths.data_ptr(), q_scales=q_scales)

    run(q, expected, expected_lse)
    run(values, output, lse, scales.data_ptr())
    torch.cuda.synchronize()
    # Upstream swapAB uses arbitrary FP32 scales for online BF16 Q, whereas
    # our prequantizer uses power-of-two scales. SG/MG/decode still use the
    # latter online, so exact equivalence is only expected on those paths.
    if decode or num_heads < 64:
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        torch.testing.assert_close(lse, expected_lse, rtol=0, atol=0)

    # Independently evaluate attention on the actual supplied quantized Q/KV.
    # This also covers swapAB, without mistaking different quantizers for an
    # implementation error or merely relaxing the old equivalence tolerance.
    q_ref = (values.float().reshape(tokens, num_heads, 4, 128) * scales[..., None]).flatten(-2)
    packed = kv.reshape(kv_tokens, 656)
    kv_scales = packed[:, 512:528].contiguous().view(torch.float32)
    kv_ref = (packed[:, :512].contiguous().view(torch.float8_e4m3fn).float().reshape(kv_tokens, 4, 128)
              * kv_scales[..., None]).flatten(-2)
    rope_ref = packed[:, 528:].contiguous().view(torch.bfloat16).float()
    for t, length in enumerate(lengths.tolist()):
        if length == 0:
            assert torch.count_nonzero(output[t]) == 0
            assert (lse[t] < -1e20).all()
            continue
        slots = indices[t, :length].long()
        scores = (q_ref[t] @ kv_ref[slots].T + rope[t].float() @ rope_ref[slots].T) * 0.0791
        ref = scores.softmax(-1) @ kv_ref[slots]
        error = (output[t].float() - ref).norm() / ref.norm().clamp_min(1e-6)
        assert error < 0.01, f"quantized attention NRMSE {error.item()}"
        torch.testing.assert_close(lse[t], scores.logsumexp(-1) / math.log(2), rtol=0, atol=0.02)

    if h8_prefill:
        # H16 SG exercises the unpruned, sequential two-pass implementation.
        # Duplicating Q preserves the first eight heads' attention problem.
        q16 = torch.cat((q, q), dim=1)
        rope16 = torch.cat((rope, rope), dim=1)
        out16 = torch.empty_like(q16)
        lse16 = torch.empty(tokens, 16, device=device)
        torch.cuda.synchronize()
        glm.sparse_mla_prefill_split_q(
            q16.data_ptr(), rope16.data_ptr(), kv.data_ptr(), indices.data_ptr(),
            out16.data_ptr(), lse16.data_ptr(), tokens, 16, topk, 0.0791, 64 * 656,
            topk_length=lengths.data_ptr())
        torch.cuda.synchronize()
        ref = out16[:, :8].float()
        assert (output.float() - ref).norm() / ref.norm() < 0.003
        torch.testing.assert_close(lse, lse16[:, :8], rtol=0, atol=0)

    if num_heads in (8, 64):
        glm.graph_begin_capture()
        glm.quantize_fp8(q.data_ptr(), values.data_ptr(), scales.data_ptr(), scales.numel(), 128)
        run(values, output, lse, scales.data_ptr())
        graph = glm.graph_end_capture()
        executable = glm.graph_instantiate(graph)
        try:
            # Replay must read fresh Q and regenerate scales at stable addresses.
            q.mul_(3)
            torch.cuda.synchronize()
            if not decode:
                # Reference quantization is outside the graph and independent
                # of the native quantizer captured above.
                raw = q.float().reshape(tokens, num_heads, 4, 128).abs().amax(-1).clamp_min(1e-4) / 448
                fresh_scales = torch.exp2(torch.ceil(torch.log2(raw)))
                fresh_values = (q.float().reshape(tokens, num_heads, 4, 128)
                                / fresh_scales[..., None]).to(torch.float8_e4m3fn).reshape_as(q)
                torch.cuda.synchronize()
                run(fresh_values, expected, expected_lse, fresh_scales.data_ptr())
            else:
                run(q, expected, expected_lse)
            glm.graph_launch(executable)
            torch.cuda.synchronize()
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            torch.testing.assert_close(lse, expected_lse, rtol=0, atol=0)
        finally:
            glm.graph_exec_destroy(executable)
            glm.graph_destroy(graph)
