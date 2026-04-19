import ctypes
import numpy as np
import pytest

BF16 = 2
I32 = 4


def _bf16_to_f32_ptr(ptr, ops, count):
    buf = ctypes.create_string_buffer(count * BF16)
    ops.lib.glm_d2h(ops.ctx, buf, ctypes.c_void_p(ptr), count * BF16)
    u16 = np.frombuffer(buf.raw, dtype=np.uint16)
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32)


def _f32_to_bf16_bytes(arr):
    u32 = arr.astype(np.float32).view(np.uint32)
    u16 = (u32 >> 16).astype(np.uint16)
    return u16.tobytes()


def test_kv_cache_write_decode_layout(glm):
    n_kv = 4
    hd = 8
    page_size = 4
    num_tokens = 3
    max_pages = 4

    src_k = np.random.randn(num_tokens, n_kv, hd).astype(np.float32)
    src_v = np.random.randn(num_tokens, n_kv, hd).astype(np.float32)

    dst_k = np.zeros((max_pages, n_kv, page_size, hd), dtype=np.float32)
    dst_v = np.zeros((max_pages, n_kv, page_size, hd), dtype=np.float32)

    slot_mapping = [0, 1, 5]

    src_k_gpu = glm.alloc(num_tokens * n_kv * hd * BF16)
    src_v_gpu = glm.alloc(num_tokens * n_kv * hd * BF16)
    dst_k_gpu = glm.alloc(max_pages * n_kv * page_size * hd * BF16)
    dst_v_gpu = glm.alloc(max_pages * n_kv * page_size * hd * BF16)
    slot_gpu = glm.alloc(num_tokens * I32)

    glm.h2d(src_k_gpu, _f32_to_bf16_bytes(src_k.ravel()))
    glm.h2d(src_v_gpu, _f32_to_bf16_bytes(src_v.ravel()))
    glm.h2d(dst_k_gpu, _f32_to_bf16_bytes(dst_k.ravel()))
    glm.h2d(dst_v_gpu, _f32_to_bf16_bytes(dst_v.ravel()))
    glm.h2d(slot_gpu, np.array(slot_mapping, dtype=np.int32).tobytes())

    # Decode layout: [num_tokens, n_kv, hd] for both K and V
    glm.kv_cache_write(
        src_k_gpu, src_v_gpu,
        dst_k_gpu, dst_v_gpu,
        slot_gpu,
        num_tokens, n_kv, hd, page_size,
        n_kv * hd, hd,  # K strides: token_stride=n_kv*hd, head_stride=hd
        n_kv * hd, hd)  # V strides: same layout

    glm.synchronize()

    result_k = _bf16_to_f32_ptr(dst_k_gpu, glm, max_pages * n_kv * page_size * hd).reshape(max_pages, n_kv, page_size, hd)
    result_v = _bf16_to_f32_ptr(dst_v_gpu, glm, max_pages * n_kv * page_size * hd).reshape(max_pages, n_kv, page_size, hd)

    for t, slot in enumerate(slot_mapping):
        page = slot // page_size
        slot_in_page = slot % page_size
        for h in range(n_kv):
            np.testing.assert_allclose(result_k[page, h, slot_in_page], src_k[t, h], atol=1e-2, rtol=1e-2)
            np.testing.assert_allclose(result_v[page, h, slot_in_page], src_v[t, h], atol=1e-2, rtol=1e-2)

    glm.free_buf(src_k_gpu)
    glm.free_buf(src_v_gpu)
    glm.free_buf(dst_k_gpu)
    glm.free_buf(dst_v_gpu)
    glm.free_buf(slot_gpu)


def test_kv_cache_write_prefill_layout(glm):
    n_kv = 4
    hd = 8
    page_size = 4
    num_tokens = 10
    max_pages = 4

    # K is in HND layout [n_kv, num_tokens, hd] (output of fusedNormRope)
    src_k = np.random.randn(n_kv, num_tokens, hd).astype(np.float32)
    # V is in SNHD layout [num_tokens, n_kv, hd] (output of linear projection, no transpose)
    src_v = np.random.randn(num_tokens, n_kv, hd).astype(np.float32)

    dst_k = np.zeros((max_pages, n_kv, page_size, hd), dtype=np.float32)
    dst_v = np.zeros((max_pages, n_kv, page_size, hd), dtype=np.float32)

    slot_mapping = []
    for i in range(num_tokens):
        slot_mapping.append(i)

    src_k_gpu = glm.alloc(n_kv * num_tokens * hd * BF16)
    src_v_gpu = glm.alloc(num_tokens * n_kv * hd * BF16)
    dst_k_gpu = glm.alloc(max_pages * n_kv * page_size * hd * BF16)
    dst_v_gpu = glm.alloc(max_pages * n_kv * page_size * hd * BF16)
    slot_gpu = glm.alloc(num_tokens * I32)

    glm.h2d(src_k_gpu, _f32_to_bf16_bytes(src_k.ravel()))
    glm.h2d(src_v_gpu, _f32_to_bf16_bytes(src_v.ravel()))
    glm.h2d(dst_k_gpu, _f32_to_bf16_bytes(dst_k.ravel()))
    glm.h2d(dst_v_gpu, _f32_to_bf16_bytes(dst_v.ravel()))
    glm.h2d(slot_gpu, np.array(slot_mapping, dtype=np.int32).tobytes())

    # K: HND layout [n_kv, num_tokens, hd] — token_stride=hd, head_stride=num_tokens*hd
    # V: SNHD layout [num_tokens, n_kv, hd] — token_stride=n_kv*hd, head_stride=hd
    glm.kv_cache_write(
        src_k_gpu, src_v_gpu,
        dst_k_gpu, dst_v_gpu,
        slot_gpu,
        num_tokens, n_kv, hd, page_size,
        hd, num_tokens * hd,       # K strides (HND)
        n_kv * hd, hd)             # V strides (SNHD)

    glm.synchronize()

    result_k = _bf16_to_f32_ptr(dst_k_gpu, glm, max_pages * n_kv * page_size * hd).reshape(max_pages, n_kv, page_size, hd)
    result_v = _bf16_to_f32_ptr(dst_v_gpu, glm, max_pages * n_kv * page_size * hd).reshape(max_pages, n_kv, page_size, hd)

    for t, slot in enumerate(slot_mapping):
        page = slot // page_size
        slot_in_page = slot % page_size
        for h in range(n_kv):
            np.testing.assert_allclose(result_k[page, h, slot_in_page], src_k[h, t], atol=1e-2, rtol=1e-2)
            np.testing.assert_allclose(result_v[page, h, slot_in_page], src_v[t, h], atol=1e-2, rtol=1e-2)

    glm.free_buf(src_k_gpu)
    glm.free_buf(src_v_gpu)
    glm.free_buf(dst_k_gpu)
    glm.free_buf(dst_v_gpu)
    glm.free_buf(slot_gpu)


def test_kv_cache_write_prefill_layout_scattered(glm):
    n_kv = 2
    hd = 16
    page_size = 4
    num_tokens = 6
    max_pages = 4

    # K: HND layout [n_kv, num_tokens, hd]
    src_k = np.random.randn(n_kv, num_tokens, hd).astype(np.float32)
    # V: SNHD layout [num_tokens, n_kv, hd]
    src_v = np.random.randn(num_tokens, n_kv, hd).astype(np.float32)

    dst_k = np.zeros((max_pages, n_kv, page_size, hd), dtype=np.float32)
    dst_v = np.zeros((max_pages, n_kv, page_size, hd), dtype=np.float32)

    slot_mapping = [0, 1, 2, 3, 8, 9]

    src_k_gpu = glm.alloc(n_kv * num_tokens * hd * BF16)
    src_v_gpu = glm.alloc(num_tokens * n_kv * hd * BF16)
    dst_k_gpu = glm.alloc(max_pages * n_kv * page_size * hd * BF16)
    dst_v_gpu = glm.alloc(max_pages * n_kv * page_size * hd * BF16)
    slot_gpu = glm.alloc(num_tokens * I32)

    glm.h2d(src_k_gpu, _f32_to_bf16_bytes(src_k.ravel()))
    glm.h2d(src_v_gpu, _f32_to_bf16_bytes(src_v.ravel()))
    glm.h2d(dst_k_gpu, _f32_to_bf16_bytes(dst_k.ravel()))
    glm.h2d(dst_v_gpu, _f32_to_bf16_bytes(dst_v.ravel()))
    glm.h2d(slot_gpu, np.array(slot_mapping, dtype=np.int32).tobytes())

    # K: HND — token_stride=hd, head_stride=num_tokens*hd
    # V: SNHD — token_stride=n_kv*hd, head_stride=hd
    glm.kv_cache_write(
        src_k_gpu, src_v_gpu,
        dst_k_gpu, dst_v_gpu,
        slot_gpu,
        num_tokens, n_kv, hd, page_size,
        hd, num_tokens * hd,
        n_kv * hd, hd)

    glm.synchronize()

    result_k = _bf16_to_f32_ptr(dst_k_gpu, glm, max_pages * n_kv * page_size * hd).reshape(max_pages, n_kv, page_size, hd)
    result_v = _bf16_to_f32_ptr(dst_v_gpu, glm, max_pages * n_kv * page_size * hd).reshape(max_pages, n_kv, page_size, hd)

    for t, slot in enumerate(slot_mapping):
        page = slot // page_size
        slot_in_page = slot % page_size
        for h in range(n_kv):
            np.testing.assert_allclose(result_k[page, h, slot_in_page], src_k[h, t], atol=1e-2, rtol=1e-2)
            np.testing.assert_allclose(result_v[page, h, slot_in_page], src_v[t, h], atol=1e-2, rtol=1e-2)

    non_written_slots = [4, 5, 6, 7, 12, 13, 14, 15]
    for slot in non_written_slots:
        page = slot // page_size
        slot_in_page = slot % page_size
        for h in range(n_kv):
            np.testing.assert_array_equal(result_k[page, h, slot_in_page], 0.0)
            np.testing.assert_array_equal(result_v[page, h, slot_in_page], 0.0)

    glm.free_buf(src_k_gpu)
    glm.free_buf(src_v_gpu)
    glm.free_buf(dst_k_gpu)
    glm.free_buf(dst_v_gpu)
    glm.free_buf(slot_gpu)


def test_kv_cache_write_prefill_layout_multi_seq(glm):
    n_kv = 2
    hd = 16
    page_size = 4
    seq_lens = [5, 3]
    num_tokens = sum(seq_lens)
    max_pages = 4

    # K: HND layout [n_kv, num_tokens, hd]
    src_k = np.random.randn(n_kv, num_tokens, hd).astype(np.float32)
    # V: SNHD layout [num_tokens, n_kv, hd]
    src_v = np.random.randn(num_tokens, n_kv, hd).astype(np.float32)

    dst_k = np.zeros((max_pages, n_kv, page_size, hd), dtype=np.float32)
    dst_v = np.zeros((max_pages, n_kv, page_size, hd), dtype=np.float32)

    pages_seq0 = [0, 1]
    pages_seq1 = [2]
    slot_mapping = []
    for pos in range(seq_lens[0]):
        page_idx = pos // page_size
        offset = pos % page_size
        slot_mapping.append(pages_seq0[page_idx] * page_size + offset)
    for pos in range(seq_lens[1]):
        page_idx = pos // page_size
        offset = pos % page_size
        slot_mapping.append(pages_seq1[page_idx] * page_size + offset)

    src_k_gpu = glm.alloc(n_kv * num_tokens * hd * BF16)
    src_v_gpu = glm.alloc(num_tokens * n_kv * hd * BF16)
    dst_k_gpu = glm.alloc(max_pages * n_kv * page_size * hd * BF16)
    dst_v_gpu = glm.alloc(max_pages * n_kv * page_size * hd * BF16)
    slot_gpu = glm.alloc(num_tokens * I32)

    glm.h2d(src_k_gpu, _f32_to_bf16_bytes(src_k.ravel()))
    glm.h2d(src_v_gpu, _f32_to_bf16_bytes(src_v.ravel()))
    glm.h2d(dst_k_gpu, _f32_to_bf16_bytes(dst_k.ravel()))
    glm.h2d(dst_v_gpu, _f32_to_bf16_bytes(dst_v.ravel()))
    glm.h2d(slot_gpu, np.array(slot_mapping, dtype=np.int32).tobytes())

    # K: HND — token_stride=hd, head_stride=num_tokens*hd
    # V: SNHD — token_stride=n_kv*hd, head_stride=hd
    glm.kv_cache_write(
        src_k_gpu, src_v_gpu,
        dst_k_gpu, dst_v_gpu,
        slot_gpu,
        num_tokens, n_kv, hd, page_size,
        hd, num_tokens * hd,
        n_kv * hd, hd)

    glm.synchronize()

    result_k = _bf16_to_f32_ptr(dst_k_gpu, glm, max_pages * n_kv * page_size * hd).reshape(max_pages, n_kv, page_size, hd)
    result_v = _bf16_to_f32_ptr(dst_v_gpu, glm, max_pages * n_kv * page_size * hd).reshape(max_pages, n_kv, page_size, hd)

    for t, slot in enumerate(slot_mapping):
        page = slot // page_size
        slot_in_page = slot % page_size
        for h in range(n_kv):
            np.testing.assert_allclose(result_k[page, h, slot_in_page], src_k[h, t], atol=1e-2, rtol=1e-2)
            np.testing.assert_allclose(result_v[page, h, slot_in_page], src_v[t, h], atol=1e-2, rtol=1e-2)

    glm.free_buf(src_k_gpu)
    glm.free_buf(src_v_gpu)
    glm.free_buf(dst_k_gpu)
    glm.free_buf(dst_v_gpu)
    glm.free_buf(slot_gpu)


def test_kv_cache_write_decode_and_prefill_produce_same_result(glm):
    n_kv = 2
    hd = 16
    page_size = 4
    num_tokens = 3

    # Decode layout: [num_tokens, n_kv, hd] for both K and V
    data_k = np.random.randn(num_tokens, n_kv, hd).astype(np.float32)
    data_v = np.random.randn(num_tokens, n_kv, hd).astype(np.float32)

    src_decode_k = data_k.copy()
    src_decode_v = data_v.copy()

    # Prefill K layout: [n_kv, num_tokens, hd] (HND, from fusedNormRope)
    src_prefill_k = data_k.transpose(1, 0, 2).copy()
    # Prefill V layout: [num_tokens, n_kv, hd] (SNHD, from linear — no transpose needed)
    src_prefill_v = data_v.copy()

    max_pages = 2
    slot_mapping = [1, 2, 5]

    dst_decode_k = np.zeros((max_pages, n_kv, page_size, hd), dtype=np.float32)
    dst_decode_v = np.zeros((max_pages, n_kv, page_size, hd), dtype=np.float32)
    dst_prefill_k = np.zeros((max_pages, n_kv, page_size, hd), dtype=np.float32)
    dst_prefill_v = np.zeros((max_pages, n_kv, page_size, hd), dtype=np.float32)

    src_decode_k_gpu = glm.alloc(num_tokens * n_kv * hd * BF16)
    src_decode_v_gpu = glm.alloc(num_tokens * n_kv * hd * BF16)
    dst_decode_k_gpu = glm.alloc(max_pages * n_kv * page_size * hd * BF16)
    dst_decode_v_gpu = glm.alloc(max_pages * n_kv * page_size * hd * BF16)

    src_prefill_k_gpu = glm.alloc(n_kv * num_tokens * hd * BF16)
    src_prefill_v_gpu = glm.alloc(num_tokens * n_kv * hd * BF16)
    dst_prefill_k_gpu = glm.alloc(max_pages * n_kv * page_size * hd * BF16)
    dst_prefill_v_gpu = glm.alloc(max_pages * n_kv * page_size * hd * BF16)

    slot_gpu = glm.alloc(num_tokens * I32)

    glm.h2d(src_decode_k_gpu, _f32_to_bf16_bytes(src_decode_k.ravel()))
    glm.h2d(src_decode_v_gpu, _f32_to_bf16_bytes(src_decode_v.ravel()))
    glm.h2d(dst_decode_k_gpu, _f32_to_bf16_bytes(dst_decode_k.ravel()))
    glm.h2d(dst_decode_v_gpu, _f32_to_bf16_bytes(dst_decode_v.ravel()))

    glm.h2d(src_prefill_k_gpu, _f32_to_bf16_bytes(src_prefill_k.ravel()))
    glm.h2d(src_prefill_v_gpu, _f32_to_bf16_bytes(src_prefill_v.ravel()))
    glm.h2d(dst_prefill_k_gpu, _f32_to_bf16_bytes(dst_prefill_k.ravel()))
    glm.h2d(dst_prefill_v_gpu, _f32_to_bf16_bytes(dst_prefill_v.ravel()))

    glm.h2d(slot_gpu, np.array(slot_mapping, dtype=np.int32).tobytes())

    # Decode: both K and V in [num_tokens, n_kv, hd] layout
    glm.kv_cache_write(
        src_decode_k_gpu, src_decode_v_gpu,
        dst_decode_k_gpu, dst_decode_v_gpu,
        slot_gpu,
        num_tokens, n_kv, hd, page_size,
        n_kv * hd, hd,     # K strides (decode)
        n_kv * hd, hd)     # V strides (decode, same layout)

    slot_gpu2 = glm.alloc(num_tokens * I32)
    glm.h2d(slot_gpu2, np.array(slot_mapping, dtype=np.int32).tobytes())

    # Prefill: K in HND [n_kv, num_tokens, hd], V in SNHD [num_tokens, n_kv, hd]
    glm.kv_cache_write(
        src_prefill_k_gpu, src_prefill_v_gpu,
        dst_prefill_k_gpu, dst_prefill_v_gpu,
        slot_gpu2,
        num_tokens, n_kv, hd, page_size,
        hd, num_tokens * hd,    # K strides (prefill, HND)
        n_kv * hd, hd)          # V strides (prefill, SNHD)

    glm.synchronize()

    result_decode_k = _bf16_to_f32_ptr(dst_decode_k_gpu, glm, max_pages * n_kv * page_size * hd).reshape(max_pages, n_kv, page_size, hd)
    result_decode_v = _bf16_to_f32_ptr(dst_decode_v_gpu, glm, max_pages * n_kv * page_size * hd).reshape(max_pages, n_kv, page_size, hd)
    result_prefill_k = _bf16_to_f32_ptr(dst_prefill_k_gpu, glm, max_pages * n_kv * page_size * hd).reshape(max_pages, n_kv, page_size, hd)
    result_prefill_v = _bf16_to_f32_ptr(dst_prefill_v_gpu, glm, max_pages * n_kv * page_size * hd).reshape(max_pages, n_kv, page_size, hd)

    np.testing.assert_allclose(result_prefill_k, result_decode_k, atol=1e-2, rtol=1e-2)
    np.testing.assert_allclose(result_prefill_v, result_decode_v, atol=1e-2, rtol=1e-2)

    glm.free_buf(src_decode_k_gpu)
    glm.free_buf(src_decode_v_gpu)
    glm.free_buf(dst_decode_k_gpu)
    glm.free_buf(dst_decode_v_gpu)
    glm.free_buf(src_prefill_k_gpu)
    glm.free_buf(src_prefill_v_gpu)
    glm.free_buf(dst_prefill_k_gpu)
    glm.free_buf(dst_prefill_v_gpu)
    glm.free_buf(slot_gpu)
    glm.free_buf(slot_gpu2)
