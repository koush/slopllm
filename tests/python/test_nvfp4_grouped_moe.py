"""Tests for glm_nvfp4_mul_mat_id_grouped: grouped NVFP4 MoE with sort-by-expert pipeline.

Compares grouped NVFP4 path against per-entry NVFP4 path and PyTorch reference.
"""

import pytest
import torch
import ctypes
from helpers import GpuBuffer, GpuPtrs, GlmOps
from test_nvfp4 import quantize_weight_nvfp4, dequantize_nvfp4, GROUP_SIZE


class TestNVFP4MulMatIdGrouped:
    def _run_grouped_nvfp4(self, glm, device, batch, K, N, num_experts, topK):
        count = batch * topK
        torch.manual_seed(42)
        input_bf16 = torch.randn(batch, K, dtype=torch.bfloat16, device=device) * 0.5
        expert_ids = torch.randint(0, num_experts, (batch, topK), dtype=torch.int32, device=device)
        expert_ids_flat = expert_ids.reshape(-1)

        weight_ptrs_list = []
        scale_ptrs_list = []
        scale2_ptrs_list = []
        ref_outputs = []

        for e in range(num_experts):
            w_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device=device) * 0.5
            packed, scale, scale_2 = quantize_weight_nvfp4(w_bf16.cpu())
            w_dequant = dequantize_nvfp4(packed, scale, scale_2, (N, K))
            ref_outputs.append(w_dequant)

            fp4_gpu = glm.alloc(N * (K // 2))
            num_k_groups = K // GROUP_SIZE
            scale_gpu = glm.alloc(N * num_k_groups)
            scale2_gpu = glm.alloc(4)

            glm.h2d(fp4_gpu, packed.view(torch.uint8).numpy().ctypes.data_as(ctypes.c_void_p), N * (K // 2))
            glm.h2d(scale_gpu, scale.view(torch.uint8).numpy().ctypes.data_as(ctypes.c_void_p), N * num_k_groups)
            glm.h2d(scale2_gpu, scale_2.numpy().ctypes.data_as(ctypes.c_void_p), 4)

            weight_ptrs_list.append(fp4_gpu)
            scale_ptrs_list.append(scale_gpu)
            scale2_ptrs_list.append(scale2_gpu)

        weight_ptrs = GpuPtrs(weight_ptrs_list, device)
        scale_ptrs = GpuPtrs(scale_ptrs_list, device)
        scale2_ptrs = GpuPtrs(scale2_ptrs_list, device)

        out_per_entry = torch.empty(count, N, dtype=torch.bfloat16, device=device)
        out_grouped = torch.empty(count, N, dtype=torch.bfloat16, device=device)

        glm.nvfp4_mul_mat_id(
            out_per_entry.data_ptr(), input_bf16.data_ptr(),
            weight_ptrs.data_ptr, scale_ptrs.data_ptr, scale2_ptrs.data_ptr,
            expert_ids_flat.data_ptr(),
            topK, count, N, K
        )

        ws_size = glm.grouped_moe_workspace_size(count, N, K, num_experts)
        ws_ptr = glm.alloc(ws_size)

        glm.nvfp4_mul_mat_id_grouped(
            out_grouped.data_ptr(), input_bf16.data_ptr(),
            weight_ptrs.data_ptr, scale_ptrs.data_ptr, scale2_ptrs.data_ptr,
            expert_ids_flat.data_ptr(),
            topK, count, N, K, num_experts,
            ws_ptr
        )
        glm.synchronize()

        ref = torch.zeros(count, N, dtype=torch.float32)
        for i in range(count):
            bid = i // topK
            eid = expert_ids_flat[i].item()
            ref[i] = torch.nn.functional.linear(input_bf16[bid].cpu().float(), ref_outputs[eid].float())

        mean_err = (out_grouped.cpu().float() - ref).abs().mean().item()
        max_err = (out_grouped.cpu().float() - ref).abs().max().item()
        mean_abs_ref = ref.abs().mean().item()

        assert mean_err < mean_abs_ref * 0.2, f"grouped mean error {mean_err:.4f} > 20% of ref {mean_abs_ref:.4f}"
        assert max_err < mean_abs_ref * 4.0, f"grouped max error {max_err:.4f} > 4x ref {mean_abs_ref:.4f}"

        torch.testing.assert_close(out_grouped.cpu().float(), out_per_entry.cpu().float(), atol=1e-3, rtol=1e-3)

        glm.free_buf(ws_ptr)
        for ptrs in [weight_ptrs_list, scale_ptrs_list, scale2_ptrs_list]:
            for p in ptrs:
                glm.free_buf(p)

    def test_nvfp4_grouped_basic(self, glm, device):
        self._run_grouped_nvfp4(glm, device, batch=2, K=128, N=64, num_experts=4, topK=2)

    def test_nvfp4_grouped_single_token(self, glm, device):
        self._run_grouped_nvfp4(glm, device, batch=1, K=256, N=128, num_experts=8, topK=4)

    def test_nvfp4_grouped_multi_batch(self, glm, device):
        self._run_grouped_nvfp4(glm, device, batch=4, K=128, N=64, num_experts=8, topK=3)

    def test_nvfp4_grouped_same_expert(self, glm, device):
        batch = 3
        K = 128
        N = 64
        num_experts = 4
        topK = 2
        count = batch * topK
        torch.manual_seed(42)
        input_bf16 = torch.randn(batch, K, dtype=torch.bfloat16, device=device) * 0.5

        expert_ids = torch.tensor([[0, 1], [0, 2], [0, 3]], dtype=torch.int32, device=device)
        expert_ids_flat = expert_ids.reshape(-1)

        weight_ptrs_list = []
        scale_ptrs_list = []
        scale2_ptrs_list = []
        ref_outputs = []

        for e in range(num_experts):
            w_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device=device) * 0.5
            packed, scale, scale_2 = quantize_weight_nvfp4(w_bf16.cpu())
            w_dequant = dequantize_nvfp4(packed, scale, scale_2, (N, K))
            ref_outputs.append(w_dequant)

            fp4_gpu = glm.alloc(N * (K // 2))
            num_k_groups = K // GROUP_SIZE
            scale_gpu = glm.alloc(N * num_k_groups)
            scale2_gpu = glm.alloc(4)

            glm.h2d(fp4_gpu, packed.view(torch.uint8).numpy().ctypes.data_as(ctypes.c_void_p), N * (K // 2))
            glm.h2d(scale_gpu, scale.view(torch.uint8).numpy().ctypes.data_as(ctypes.c_void_p), N * num_k_groups)
            glm.h2d(scale2_gpu, scale_2.numpy().ctypes.data_as(ctypes.c_void_p), 4)

            weight_ptrs_list.append(fp4_gpu)
            scale_ptrs_list.append(scale_gpu)
            scale2_ptrs_list.append(scale2_gpu)

        from helpers import GpuPtrs
        weight_ptrs = GpuPtrs(weight_ptrs_list, device)
        scale_ptrs = GpuPtrs(scale_ptrs_list, device)
        scale2_ptrs = GpuPtrs(scale2_ptrs_list, device)

        out_grouped = torch.empty(count, N, dtype=torch.bfloat16, device=device)

        ws_size = glm.grouped_moe_workspace_size(count, N, K, num_experts)
        ws_ptr = glm.alloc(ws_size)

        glm.nvfp4_mul_mat_id_grouped(
            out_grouped.data_ptr(), input_bf16.data_ptr(),
            weight_ptrs.data_ptr, scale_ptrs.data_ptr, scale2_ptrs.data_ptr,
            expert_ids_flat.data_ptr(),
            topK, count, N, K, num_experts,
            ws_ptr
        )
        glm.synchronize()

        ref = torch.zeros(count, N, dtype=torch.float32)
        for i in range(count):
            bid = i // topK
            eid = expert_ids_flat[i].item()
            ref[i] = torch.nn.functional.linear(input_bf16[bid].cpu().float(), ref_outputs[eid].float())

        mean_err = (out_grouped.cpu().float() - ref).abs().mean().item()
        max_err = (out_grouped.cpu().float() - ref).abs().max().item()
        mean_abs_ref = ref.abs().mean().item()
        assert mean_err < mean_abs_ref * 0.2, f"mean error {mean_err:.4f} > 20% of ref {mean_abs_ref:.4f}"

        glm.free_buf(ws_ptr)
        for ptrs in [weight_ptrs_list, scale_ptrs_list, scale2_ptrs_list]:
            for p in ptrs:
                glm.free_buf(p)

    def test_nvfp4_grouped_glm51_dims(self, glm, device):
        self._run_grouped_nvfp4(glm, device, batch=2, K=256, N=512, num_experts=8, topK=4)

    def test_nvfp4_grouped_many_tokens_per_expert(self, glm, device):
        # Routes far more than GEMV_ROWS_PER_BLOCK(=8) tokens to a single expert,
        # exercising the M_e > 8 chunked-accumulation path. Compares against the
        # float reference directly (rather than the per-entry kernel) since the
        # two kernels' bf16 accumulation orders can differ by up to 1 ULP.
        batch, K, N, num_experts, topK = 10, 128, 64, 2, 2
        count = batch * topK
        torch.manual_seed(42)
        device_t = device
        input_bf16 = torch.randn(batch, K, dtype=torch.bfloat16, device=device_t) * 0.5
        expert_ids = torch.randint(0, num_experts, (batch, topK), dtype=torch.int32, device=device_t)
        expert_ids_flat = expert_ids.reshape(-1)

        weight_ptrs_list = []
        scale_ptrs_list = []
        scale2_ptrs_list = []
        ref_outputs = []

        for e in range(num_experts):
            w_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device=device_t) * 0.5
            packed, scale, scale_2 = quantize_weight_nvfp4(w_bf16.cpu())
            w_dequant = dequantize_nvfp4(packed, scale, scale_2, (N, K))
            ref_outputs.append(w_dequant)

            fp4_gpu = glm.alloc(N * (K // 2))
            num_k_groups = K // GROUP_SIZE
            scale_gpu = glm.alloc(N * num_k_groups)
            scale2_gpu = glm.alloc(4)

            glm.h2d(fp4_gpu, packed.view(torch.uint8).numpy().ctypes.data_as(ctypes.c_void_p), N * (K // 2))
            glm.h2d(scale_gpu, scale.view(torch.uint8).numpy().ctypes.data_as(ctypes.c_void_p), N * num_k_groups)
            glm.h2d(scale2_gpu, scale_2.numpy().ctypes.data_as(ctypes.c_void_p), 4)

            weight_ptrs_list.append(fp4_gpu)
            scale_ptrs_list.append(scale_gpu)
            scale2_ptrs_list.append(scale2_gpu)

        weight_ptrs = GpuPtrs(weight_ptrs_list, device_t)
        scale_ptrs = GpuPtrs(scale_ptrs_list, device_t)
        scale2_ptrs = GpuPtrs(scale2_ptrs_list, device_t)

        out_grouped = torch.empty(count, N, dtype=torch.bfloat16, device=device_t)

        ws_size = glm.grouped_moe_workspace_size(count, N, K, num_experts)
        ws_ptr = glm.alloc(ws_size)

        glm.nvfp4_mul_mat_id_grouped(
            out_grouped.data_ptr(), input_bf16.data_ptr(),
            weight_ptrs.data_ptr, scale_ptrs.data_ptr, scale2_ptrs.data_ptr,
            expert_ids_flat.data_ptr(),
            topK, count, N, K, num_experts,
            ws_ptr
        )
        glm.synchronize()

        ref = torch.zeros(count, N, dtype=torch.float32)
        for i in range(count):
            bid = i // topK
            eid = expert_ids_flat[i].item()
            ref[i] = torch.nn.functional.linear(input_bf16[bid].cpu().float(), ref_outputs[eid].float())

        # Sanity: this configuration must actually exercise M_e > GEMV_ROWS_PER_BLOCK(=8).
        counts = torch.bincount(expert_ids_flat.cpu(), minlength=num_experts)
        assert counts.max().item() > 8, f"test configuration does not produce M_e > 8 (counts={counts.tolist()})"

        out_f32 = out_grouped.cpu().float()
        mean_err = (out_f32 - ref).abs().mean().item()
        max_err = (out_f32 - ref).abs().max().item()
        mean_abs_ref = ref.abs().mean().item()

        # No row should be left as an untouched zero (the M_e > 8 truncation bug).
        nonzero_rows = (out_f32.abs().sum(dim=1) > 0)
        assert nonzero_rows.all(), f"rows left as zero by truncation: {(~nonzero_rows).nonzero().flatten().tolist()}"

        assert mean_err < mean_abs_ref * 0.2, f"mean error {mean_err:.4f} > 20% of ref {mean_abs_ref:.4f}"
        assert max_err < mean_abs_ref * 4.0, f"max error {max_err:.4f} > 4x ref {mean_abs_ref:.4f}"

        glm.free_buf(ws_ptr)
        for ptrs in [weight_ptrs_list, scale_ptrs_list, scale2_ptrs_list]:
            for p in ptrs:
                glm.free_buf(p)

    def test_nvfp4_grouped_all_same_expert(self, glm, device):
        batch = 4
        K = 128
        N = 64
        num_experts = 4
        topK = 2
        count = batch * topK
        torch.manual_seed(42)
        input_bf16 = torch.randn(batch, K, dtype=torch.bfloat16, device=device) * 0.5

        expert_ids = torch.zeros(batch, topK, dtype=torch.int32, device=device)
        expert_ids_flat = expert_ids.reshape(-1)

        weight_ptrs_list = []
        scale_ptrs_list = []
        scale2_ptrs_list = []
        ref_outputs = []

        for e in range(num_experts):
            w_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device=device) * 0.5
            packed, scale, scale_2 = quantize_weight_nvfp4(w_bf16.cpu())
            w_dequant = dequantize_nvfp4(packed, scale, scale_2, (N, K))
            ref_outputs.append(w_dequant)

            fp4_gpu = glm.alloc(N * (K // 2))
            num_k_groups = K // GROUP_SIZE
            scale_gpu = glm.alloc(N * num_k_groups)
            scale2_gpu = glm.alloc(4)

            glm.h2d(fp4_gpu, packed.view(torch.uint8).numpy().ctypes.data_as(ctypes.c_void_p), N * (K // 2))
            glm.h2d(scale_gpu, scale.view(torch.uint8).numpy().ctypes.data_as(ctypes.c_void_p), N * num_k_groups)
            glm.h2d(scale2_gpu, scale_2.numpy().ctypes.data_as(ctypes.c_void_p), 4)

            weight_ptrs_list.append(fp4_gpu)
            scale_ptrs_list.append(scale_gpu)
            scale2_ptrs_list.append(scale2_gpu)

        from helpers import GpuPtrs
        weight_ptrs = GpuPtrs(weight_ptrs_list, device)
        scale_ptrs = GpuPtrs(scale_ptrs_list, device)
        scale2_ptrs = GpuPtrs(scale2_ptrs_list, device)

        out_grouped = torch.empty(count, N, dtype=torch.bfloat16, device=device)

        ws_size = glm.grouped_moe_workspace_size(count, N, K, num_experts)
        ws_ptr = glm.alloc(ws_size)

        glm.nvfp4_mul_mat_id_grouped(
            out_grouped.data_ptr(), input_bf16.data_ptr(),
            weight_ptrs.data_ptr, scale_ptrs.data_ptr, scale2_ptrs.data_ptr,
            expert_ids_flat.data_ptr(),
            topK, count, N, K, num_experts,
            ws_ptr
        )
        glm.synchronize()

        ref = torch.zeros(count, N, dtype=torch.float32)
        for i in range(count):
            bid = i // topK
            ref[i] = torch.nn.functional.linear(input_bf16[bid].cpu().float(), ref_outputs[0].float())

        mean_err = (out_grouped.cpu().float() - ref).abs().mean().item()
        mean_abs_ref = ref.abs().mean().item()
        assert mean_err < mean_abs_ref * 0.2, f"mean error {mean_err:.4f} > 20% of ref {mean_abs_ref:.4f}"

        glm.free_buf(ws_ptr)
        for ptrs in [weight_ptrs_list, scale_ptrs_list, scale2_ptrs_list]:
            for p in ptrs:
                glm.free_buf(p)
