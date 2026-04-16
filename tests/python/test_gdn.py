import ctypes
import numpy as np
import pytest
import torch

from helpers import GlmOps

ATOL = 1e-2
RTOL = 1e-2
BF16 = 2
F32 = 4


def _bf16_to_f32(data):
    u16 = np.frombuffer(data, dtype=np.uint16)
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32)


def _f32_to_bf16_bytes(arr):
    u32 = arr.astype(np.float32).view(np.uint32)
    u16 = (u32 >> 16).astype(np.uint16)
    return u16.tobytes()


def _upload_bf16(glm, tensor_np):
    bf16_bytes = _f32_to_bf16_bytes(tensor_np)
    ptr = glm.alloc(len(bf16_bytes))
    glm.h2d(ptr, bf16_bytes)
    return ptr


def _upload_f32(glm, tensor_np):
    data = tensor_np.astype(np.float32).tobytes()
    ptr = glm.alloc(len(data))
    glm.h2d(ptr, data)
    return ptr


def _download_bf16(glm, ptr, count):
    nbytes = count * BF16
    buf = ctypes.create_string_buffer(nbytes)
    glm.d2h(buf, ptr, nbytes)
    return _bf16_to_f32(buf.raw).copy()


def _download_f32(glm, ptr, count):
    nbytes = count * F32
    buf = ctypes.create_string_buffer(nbytes)
    glm.d2h(buf, ptr, nbytes)
    return np.frombuffer(buf.raw, dtype=np.float32).copy()


def _to_bf16(t):
    return t.to(torch.bfloat16).to(torch.float32)


def _torch_gdn_recurrent_step(q, k, v, beta, g, state, d_k, d_v):
    B, H, dk = q.shape
    scale = 1.0 / (dk ** 0.5)
    q_norm = q * torch.rsqrt(q.pow(2).sum(dim=-1, keepdim=True) + 1e-8) * scale
    k_norm = k * torch.rsqrt(k.pow(2).sum(dim=-1, keepdim=True) + 1e-8)
    h = state.clone()
    h = h * g.unsqueeze(-1).unsqueeze(-1).exp()
    v_old = (h * k_norm.unsqueeze(-1)).sum(dim=2)
    delta = beta.unsqueeze(-1) * (v - v_old)
    h = h + k_norm.unsqueeze(-1) * delta.unsqueeze(-2)
    o = (h * q_norm.unsqueeze(-1)).sum(dim=2)
    return o, h


def _torch_causal_conv1d(x, weight, kernel_size):
    B, C, S = x.shape
    out = torch.zeros_like(x)
    for c in range(C):
        for t in range(S):
            s = 0.0
            for kk in range(kernel_size):
                xt = t - (kernel_size - 1) + kk
                if xt >= 0:
                    s += weight[c, kk].item() * x[0, c, xt].item()
            out[0, c, t] = torch.nn.functional.silu(torch.tensor(s)).item()
    return out


def _torch_rmsnorm_gated(x, gate, weight, eps):
    rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    normed = x * rms * weight
    return normed * torch.nn.functional.silu(gate)


@pytest.fixture(scope="module")
def glm():
    ops = GlmOps()
    yield ops
    ops.lib.glm_free(ops.ctx)
    ops.ctx = None


class TestGdnRecurrentStep:
    @pytest.mark.parametrize("d_k,d_v", [(128, 128), (64, 64)])
    def test_zero_state(self, glm, d_k, d_v):
        H = 4
        torch.manual_seed(42)
        q = torch.randn(1, H, d_k, dtype=torch.float32)
        k = torch.randn(1, H, d_k, dtype=torch.float32)
        v = torch.randn(1, H, d_v, dtype=torch.float32)
        a_raw = torch.randn(1, H, dtype=torch.float32) * 0.5
        b_raw = torch.randn(1, H, dtype=torch.float32)
        A_log = torch.randn(H, dtype=torch.float32) * 2
        dt_bias = torch.randn(H, dtype=torch.float32) * 0.1
        state = torch.zeros(1, H, d_k, d_v, dtype=torch.float32)

        q_bf = _to_bf16(q)
        k_bf = _to_bf16(k)
        v_bf = _to_bf16(v)
        a_bf = _to_bf16(a_raw)
        b_bf = _to_bf16(b_raw)

        beta = torch.sigmoid(b_bf)
        g = -A_log.float().exp().unsqueeze(0) * torch.nn.functional.softplus(a_bf + dt_bias.unsqueeze(0))

        ref_o, ref_state = _torch_gdn_recurrent_step(q_bf, k_bf, v_bf, beta, g, state, d_k, d_v)

        q_gpu = _upload_bf16(glm, q.reshape(H, d_k).numpy())
        k_gpu = _upload_bf16(glm, k.reshape(H, d_k).numpy())
        v_gpu = _upload_bf16(glm, v.reshape(H, d_v).numpy())
        a_gpu = _upload_bf16(glm, a_raw.reshape(H).numpy())
        b_gpu = _upload_bf16(glm, b_raw.reshape(H).numpy())
        alog_gpu = _upload_f32(glm, A_log.numpy())
        dtb_gpu = _upload_f32(glm, dt_bias.numpy())
        state_gpu = _upload_f32(glm, state.reshape(H, d_k, d_v).numpy())
        out_gpu = glm.alloc(H * d_v * BF16)

        glm.gdn_recurrent_step(out_gpu, state_gpu, q_gpu, k_gpu, v_gpu,
                                a_gpu, b_gpu, alog_gpu, dtb_gpu,
                                H, d_k, d_v)
        glm.synchronize()

        out_np = _download_bf16(glm, out_gpu, H * d_v)
        state_np = _download_f32(glm, state_gpu, H * d_k * d_v)

        out_torch = torch.from_numpy(out_np.reshape(H, d_v))
        state_torch = torch.from_numpy(state_np.reshape(H, d_k, d_v))

        max_out_diff = torch.max(torch.abs(out_torch - ref_o.squeeze(0))).item()
        max_state_diff = torch.max(torch.abs(state_torch - ref_state.squeeze(0))).item()
        atol = 5e-3
        rtol = 1e-2
        assert torch.allclose(out_torch, ref_o.squeeze(0), atol=atol, rtol=rtol), \
            f"Output mismatch: max diff={max_out_diff:.6f}"
        assert torch.allclose(state_torch, ref_state.squeeze(0), atol=atol, rtol=rtol), \
            f"State mismatch: max diff={max_state_diff:.6f}"

        for p in [q_gpu, k_gpu, v_gpu, a_gpu, b_gpu, alog_gpu, dtb_gpu, state_gpu, out_gpu]:
            glm.free_buf(p)

    @pytest.mark.parametrize("d_k,d_v", [(128, 128), (64, 64)])
    def test_nonzero_state(self, glm, d_k, d_v):
        H = 4
        torch.manual_seed(123)
        q = torch.randn(1, H, d_k, dtype=torch.float32)
        k = torch.randn(1, H, d_k, dtype=torch.float32)
        v = torch.randn(1, H, d_v, dtype=torch.float32)
        a_raw = torch.randn(1, H, dtype=torch.float32) * 0.5
        b_raw = torch.randn(1, H, dtype=torch.float32)
        A_log = torch.randn(H, dtype=torch.float32) * 2
        dt_bias = torch.randn(H, dtype=torch.float32) * 0.1
        state = torch.randn(1, H, d_k, d_v, dtype=torch.float32) * 0.1

        q_bf = _to_bf16(q)
        k_bf = _to_bf16(k)
        v_bf = _to_bf16(v)
        a_bf = _to_bf16(a_raw)
        b_bf = _to_bf16(b_raw)

        beta = torch.sigmoid(b_bf)
        g = -A_log.float().exp().unsqueeze(0) * torch.nn.functional.softplus(a_bf + dt_bias.unsqueeze(0))

        ref_o, ref_state = _torch_gdn_recurrent_step(q_bf, k_bf, v_bf, beta, g, state, d_k, d_v)

        q_gpu = _upload_bf16(glm, q.reshape(H, d_k).numpy())
        k_gpu = _upload_bf16(glm, k.reshape(H, d_k).numpy())
        v_gpu = _upload_bf16(glm, v.reshape(H, d_v).numpy())
        a_gpu = _upload_bf16(glm, a_raw.reshape(H).numpy())
        b_gpu = _upload_bf16(glm, b_raw.reshape(H).numpy())
        alog_gpu = _upload_f32(glm, A_log.numpy())
        dtb_gpu = _upload_f32(glm, dt_bias.numpy())
        state_gpu = _upload_f32(glm, state.reshape(H, d_k, d_v).numpy())
        out_gpu = glm.alloc(H * d_v * BF16)

        glm.gdn_recurrent_step(out_gpu, state_gpu, q_gpu, k_gpu, v_gpu,
                                a_gpu, b_gpu, alog_gpu, dtb_gpu,
                                H, d_k, d_v)
        glm.synchronize()

        out_np = _download_bf16(glm, out_gpu, H * d_v)
        state_np = _download_f32(glm, state_gpu, H * d_k * d_v)

        out_torch = torch.from_numpy(out_np.reshape(H, d_v))
        state_torch = torch.from_numpy(state_np.reshape(H, d_k, d_v))

        max_out_diff = torch.max(torch.abs(out_torch - ref_o.squeeze(0))).item()
        max_state_diff = torch.max(torch.abs(state_torch - ref_state.squeeze(0))).item()
        atol = 5e-3
        rtol = 1e-2
        assert torch.allclose(out_torch, ref_o.squeeze(0), atol=atol, rtol=rtol), \
            f"Output mismatch: max diff={max_out_diff:.6f}"
        assert torch.allclose(state_torch, ref_state.squeeze(0), atol=atol, rtol=rtol), \
            f"State mismatch: max diff={max_state_diff:.6f}"

        for p in [q_gpu, k_gpu, v_gpu, a_gpu, b_gpu, alog_gpu, dtb_gpu, state_gpu, out_gpu]:
            glm.free_buf(p)


class TestGdnPrefill:
    @pytest.mark.parametrize("d_k,d_v,S", [(32, 32, 8), (128, 128, 4)])
    def test_basic(self, glm, d_k, d_v, S):
        H = 4
        torch.manual_seed(42)

        q = torch.randn(S, H, d_k, dtype=torch.float32)
        k = torch.randn(S, H, d_k, dtype=torch.float32)
        v = torch.randn(S, H, d_v, dtype=torch.float32)
        a_raw = torch.randn(S, H, dtype=torch.float32) * 0.5
        b_raw = torch.randn(S, H, dtype=torch.float32)
        A_log = torch.randn(H, dtype=torch.float32) * 2
        dt_bias = torch.randn(H, dtype=torch.float32) * 0.1
        state = torch.zeros(1, H, d_k, d_v, dtype=torch.float32)

        q_bf = _to_bf16(q)
        k_bf = _to_bf16(k)
        v_bf = _to_bf16(v)
        a_bf = _to_bf16(a_raw)
        b_bf = _to_bf16(b_raw)

        ref_state = state.clone()
        ref_outputs = []
        for t in range(S):
            q_t = q_bf[t:t+1]
            k_t = k_bf[t:t+1]
            v_t = v_bf[t:t+1]
            a_t = a_bf[t:t+1]
            b_t = b_bf[t:t+1]
            beta_t = torch.sigmoid(b_t)
            g_t = -A_log.float().exp().unsqueeze(0) * torch.nn.functional.softplus(a_t + dt_bias.unsqueeze(0))
            o_t, ref_state = _torch_gdn_recurrent_step(q_t, k_t, v_t, beta_t, g_t, ref_state, d_k, d_v)
            ref_outputs.append(o_t.squeeze(0))
        ref_out = torch.stack(ref_outputs)

        q_gpu = _upload_bf16(glm, q.reshape(S * H, d_k).numpy())
        k_gpu = _upload_bf16(glm, k.reshape(S * H, d_k).numpy())
        v_gpu = _upload_bf16(glm, v.reshape(S * H, d_v).numpy())
        a_gpu = _upload_bf16(glm, a_raw.reshape(S * H).numpy())
        b_gpu = _upload_bf16(glm, b_raw.reshape(S * H).numpy())
        alog_gpu = _upload_f32(glm, A_log.numpy())
        dtb_gpu = _upload_f32(glm, dt_bias.numpy())
        state_gpu = _upload_f32(glm, state.reshape(H, d_k, d_v).numpy())
        out_gpu = glm.alloc(S * H * d_v * BF16)

        glm.gdn_prefill(out_gpu, state_gpu, q_gpu, k_gpu, v_gpu,
                         a_gpu, b_gpu, alog_gpu, dtb_gpu,
                         S, H, d_k, d_v)
        glm.synchronize()

        out_np = _download_bf16(glm, out_gpu, S * H * d_v)
        state_np = _download_f32(glm, state_gpu, H * d_k * d_v)

        out_torch = torch.from_numpy(out_np.reshape(S, H, d_v))
        state_torch = torch.from_numpy(state_np.reshape(H, d_k, d_v))

        max_out_diff = torch.max(torch.abs(out_torch - ref_out)).item()
        max_state_diff = torch.max(torch.abs(state_torch - ref_state.squeeze(0))).item()
        assert max_out_diff < 0.02, f"Output mismatch: max diff={max_out_diff:.6f}"
        assert max_state_diff < 0.02, f"State mismatch: max diff={max_state_diff:.6f}"

        for p in [q_gpu, k_gpu, v_gpu, a_gpu, b_gpu, alog_gpu, dtb_gpu, state_gpu, out_gpu]:
            glm.free_buf(p)


class TestCausalConv1d:
    def test_prefill(self, glm):
        conv_dim = 64
        kernel_size = 4
        S = 16
        torch.manual_seed(42)

        x = torch.randn(1, conv_dim, S, dtype=torch.float32)
        weight = torch.randn(conv_dim, kernel_size, dtype=torch.float32)

        ref_out = _torch_causal_conv1d(x, weight, kernel_size)

        x_flat = x.squeeze(0).contiguous().numpy()
        w_flat = weight.numpy()
        x_gpu = _upload_bf16(glm, x_flat)
        w_gpu = _upload_bf16(glm, w_flat)
        out_gpu = glm.alloc(conv_dim * S * BF16)
        cs_gpu = glm.alloc(conv_dim * (kernel_size - 1) * BF16)

        glm.causal_conv1d(out_gpu, cs_gpu, x_gpu, w_gpu, conv_dim, S, kernel_size)
        glm.synchronize()

        out_np = _download_bf16(glm, out_gpu, conv_dim * S)
        out_torch = torch.from_numpy(out_np.reshape(conv_dim, S))

        atol = max(ATOL, 0.02)
        rtol = max(RTOL, 0.02)
        ref_flat = ref_out.squeeze(0)
        assert torch.allclose(out_torch, ref_flat, atol=atol, rtol=rtol), \
            f"Conv1d prefill mismatch: max diff={torch.max(torch.abs(out_torch - ref_flat)):.6f}"

        for p in [x_gpu, w_gpu, out_gpu, cs_gpu]:
            glm.free_buf(p)

    def test_decode_update(self, glm):
        conv_dim = 64
        kernel_size = 4
        torch.manual_seed(42)

        weight = torch.randn(conv_dim, kernel_size, dtype=torch.float32)

        conv_state = torch.randn(conv_dim, kernel_size - 1, dtype=torch.float32) * 0.1
        x_new = torch.randn(conv_dim, dtype=torch.float32)

        ref_out = torch.zeros(conv_dim, dtype=torch.float32)
        for c in range(conv_dim):
            s = weight[c, kernel_size - 1].item() * x_new[c].item()
            for kk in range(kernel_size - 1):
                s += weight[c, kk].item() * conv_state[c, kk].item()
            ref_out[c] = torch.nn.functional.silu(torch.tensor(s)).item()

        cs_gpu = _upload_bf16(glm, conv_state.numpy())
        x_gpu = _upload_bf16(glm, x_new.numpy())
        w_gpu = _upload_bf16(glm, weight.numpy())
        out_gpu = glm.alloc(conv_dim * BF16)

        glm.causal_conv1d_update(out_gpu, cs_gpu, x_gpu, w_gpu, conv_dim, kernel_size)
        glm.synchronize()

        out_np = _download_bf16(glm, out_gpu, conv_dim)
        out_torch = torch.from_numpy(out_np)

        cs_updated_np = _download_bf16(glm, cs_gpu, conv_dim * (kernel_size - 1))
        cs_updated_torch = torch.from_numpy(cs_updated_np.reshape(conv_dim, kernel_size - 1))

        ref_state_updated = conv_state.clone()
        for c in range(conv_dim):
            for kk in range(kernel_size - 2):
                ref_state_updated[c, kk] = conv_state[c, kk + 1]
            ref_state_updated[c, kernel_size - 2] = x_new[c]

        atol = max(ATOL, 0.02)
        rtol = max(RTOL, 0.02)
        assert torch.allclose(out_torch, ref_out, atol=atol, rtol=rtol), \
            f"Conv1d update output mismatch: max diff={torch.max(torch.abs(out_torch - ref_out)):.6f}"
        assert torch.allclose(cs_updated_torch, ref_state_updated, atol=atol, rtol=rtol), \
            f"Conv1d update state mismatch: max diff={torch.max(torch.abs(cs_updated_torch - ref_state_updated)):.6f}"

        for p in [cs_gpu, x_gpu, w_gpu, out_gpu]:
            glm.free_buf(p)


class TestRmsnormGated:
    def test_basic(self, glm):
        dim = 128
        batch = 16
        eps = 1e-6
        torch.manual_seed(42)

        x = torch.randn(batch, dim, dtype=torch.float32)
        gate = torch.randn(batch, dim, dtype=torch.float32)
        weight = torch.randn(dim, dtype=torch.float32)

        ref = _torch_rmsnorm_gated(x, gate, weight, eps)

        x_gpu = _upload_bf16(glm, x.numpy())
        gate_gpu = _upload_bf16(glm, gate.numpy())
        w_gpu = _upload_bf16(glm, weight.numpy())
        out_gpu = glm.alloc(batch * dim * BF16)

        glm.rmsnorm_gated(out_gpu, x_gpu, gate_gpu, w_gpu, eps, dim, batch)
        glm.synchronize()

        out_np = _download_bf16(glm, out_gpu, batch * dim)
        out_torch = torch.from_numpy(out_np.reshape(batch, dim))

        atol = max(ATOL, 0.02)
        rtol = max(RTOL, 0.02)
        assert torch.allclose(out_torch, ref, atol=atol, rtol=rtol), \
            f"RMSNorm gated mismatch: max diff={torch.max(torch.abs(out_torch - ref)):.6f}"

        for p in [x_gpu, gate_gpu, w_gpu, out_gpu]:
            glm.free_buf(p)


class TestQkvSplit:
    def test_basic(self):
        """Test QKV split from [convDim, S] to [S, H, d_k], [S, H, d_k], [S, H, d_v]"""
        glm = GlmOps()
        try:
            S = 8
            H = 4
            d_k = 16
            d_v = 16
            k_total = H * d_k  # 64
            v_total = H * d_v  # 64
            conv_dim = 2 * k_total + v_total  # 192

            # Create QKV input in [convDim, S] layout (channel-first, non-interleaved)
            # Channels 0..k_total-1 are Q for all heads
            # Channels k_total..2*k_total-1 are K for all heads
            # Channels 2*k_total..conv_dim-1 are V for all heads
            qkv_np = np.random.randn(conv_dim, S).astype(np.float32)
            qkv_gpu = _upload_bf16(glm, qkv_np)

            q_size = S * H * d_k
            k_size = S * H * d_k
            v_size = S * H * d_v
            q_gpu = glm.alloc(q_size * BF16)
            k_gpu = glm.alloc(k_size * BF16)
            v_gpu = glm.alloc(v_size * BF16)

            glm.qkv_split(q_gpu, k_gpu, v_gpu, qkv_gpu, S, H, d_k, d_v)
            glm.synchronize()

            q_np = _download_bf16(glm, q_gpu, q_size).reshape(S, H, d_k)
            k_np = _download_bf16(glm, k_gpu, k_size).reshape(S, H, d_k)
            v_np = _download_bf16(glm, v_gpu, v_size).reshape(S, H, d_v)

            # Build reference from qkv_np (non-interleaved layout)
            q_ref = np.zeros((S, H, d_k), dtype=np.float32)
            k_ref = np.zeros((S, H, d_k), dtype=np.float32)
            v_ref = np.zeros((S, H, d_v), dtype=np.float32)
            for t in range(S):
                for h in range(H):
                    for d in range(d_k):
                        q_ref[t, h, d] = qkv_np[h * d_k + d, t]
                        k_ref[t, h, d] = qkv_np[k_total + h * d_k + d, t]
                    for d in range(d_v):
                        v_ref[t, h, d] = qkv_np[2 * k_total + h * d_v + d, t]

            q_torch = torch.from_numpy(q_np)
            k_torch = torch.from_numpy(k_np)
            v_torch = torch.from_numpy(v_np)
            q_ref_torch = torch.from_numpy(q_ref)
            k_ref_torch = torch.from_numpy(k_ref)
            v_ref_torch = torch.from_numpy(v_ref)

            assert torch.allclose(q_torch, q_ref_torch, atol=ATOL, rtol=RTOL), \
                f"Q mismatch: max diff={torch.max(torch.abs(q_torch - q_ref_torch)):.6f}"
            assert torch.allclose(k_torch, k_ref_torch, atol=ATOL, rtol=RTOL), \
                f"K mismatch: max diff={torch.max(torch.abs(k_torch - k_ref_torch)):.6f}"
            assert torch.allclose(v_torch, v_ref_torch, atol=ATOL, rtol=RTOL), \
                f"V mismatch: max diff={torch.max(torch.abs(v_torch - v_ref_torch)):.6f}"

        finally:
            for p in [qkv_gpu, q_gpu, k_gpu, v_gpu]:
                glm.free_buf(p)
            del glm
