import torch
import pytest
import ctypes
import numpy as np
from helpers import GlmOps

NUM_GPUS = min(torch.cuda.device_count(), 2)
SKIP_REASON = f"Requires 1 GPU, found {torch.cuda.device_count()}"


def bf16_bytes_to_tensor(raw_bytes, count):
    u8 = np.frombuffer(raw_bytes[:count * 2], dtype=np.uint8)
    u16 = u8.view(np.uint16)
    return torch.from_numpy(u16.copy()).view(torch.bfloat16)


@pytest.mark.skipif(NUM_GPUS < 1, reason=SKIP_REASON)
class TestSumPointers:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ops = GlmOps(device_id=0)
        yield
        del self.ops
        torch.cuda.empty_cache()

    def _sum_pointers(self, tensors_bf16, dtype=9):
        count = tensors_bf16[0].shape[0]
        N = len(tensors_bf16)

        gpu_ptrs = []
        for t in tensors_bf16:
            ptr = self.ops.alloc(count * (4 if dtype == 7 else 2))
            if dtype == 7:
                cpu = t.cpu().float().numpy().astype(np.float32).tobytes()
            else:
                cpu = t.cpu().view(torch.int16).numpy().tobytes()
            self.ops.h2d(ptr, cpu)
            gpu_ptrs.append(ptr)

        out_ptr = gpu_ptrs[0]

        ptr_args = [ctypes.c_void_p(int(p)) for p in gpu_ptrs] + [ctypes.c_void_p(0)] * (8 - N)

        self.ops.lib.glm_sum_pointers.restype = None
        self.ops.lib.glm_sum_pointers.argtypes = (
            [ctypes.c_void_p] +  # ctx
            [ctypes.c_void_p] * 8 +  # p0..p7
            [ctypes.c_void_p, ctypes.c_int, ctypes.c_int64, ctypes.c_int]  # out, N, numel, dtype
        )
        self.ops.lib.glm_sum_pointers(
            self.ops.ctx,
            *ptr_args,
            ctypes.c_void_p(int(out_ptr)), N, count, dtype,
        )

        self.ops.synchronize()

        out_bytes = ctypes.create_string_buffer(count * (4 if dtype == 7 else 2))
        self.ops.d2h(out_bytes, out_ptr, count * (4 if dtype == 7 else 2))

        if dtype == 7:
            return torch.from_numpy(np.frombuffer(out_bytes.raw, dtype=np.float32)[:count]).float()
        return bf16_bytes_to_tensor(out_bytes.raw, count)

        for p in gpu_ptrs[1:] + [gpu_ptrs[0]]:
            self.ops.free_buf(p)

    def test_single_tensor(self):
        a = torch.arange(256, dtype=torch.float32).bfloat16()
        result = self._sum_pointers([a])
        expected = a
        assert torch.equal(result, expected)

    def test_two_tensors(self):
        a = torch.arange(512, dtype=torch.float32).bfloat16()
        b = torch.arange(512, dtype=torch.float32).bfloat16() * 2
        expected = (a.float() + b.float()).bfloat16()
        result = self._sum_pointers([a, b])
        assert torch.equal(result, expected)

    def test_three_tensors(self):
        a = torch.arange(512, dtype=torch.float32).bfloat16()
        b = torch.arange(512, dtype=torch.float32).bfloat16() * 0.5
        c = torch.arange(512, dtype=torch.float32).bfloat16() * 3
        expected = (a.float() + b.float() + c.float()).bfloat16()
        result = self._sum_pointers([a, b, c])
        assert torch.equal(result, expected)

    def test_many_tensors(self):
        tensors = [(torch.arange(128, dtype=torch.float32).bfloat16() * (i + 1)) for i in range(8)]
        expected_sum = sum(t.float() for t in tensors).bfloat16()
        result = self._sum_pointers(tensors)
        assert torch.equal(result, expected_sum)

    def test_max_tensors(self):
        tensors = [(torch.arange(64, dtype=torch.float32).bfloat16() * (i + 1)) for i in range(8)]
        expected_sum = sum(t.float() for t in tensors).bfloat16()
        result = self._sum_pointers(tensors)
        assert torch.equal(result, expected_sum)

    def test_negative_values(self):
        a = (torch.arange(256, dtype=torch.float32) - 128).bfloat16()
        b = -(torch.arange(256, dtype=torch.float32) - 128).bfloat16()
        result = self._sum_pointers([a, b])
        expected = torch.zeros(256, dtype=torch.bfloat16)
        assert torch.equal(result, expected)

    def test_f32(self):
        a = torch.arange(256, dtype=torch.float32)
        b = torch.arange(256, dtype=torch.float32) * 2
        c = torch.arange(256, dtype=torch.float32) * 3
        result = self._sum_pointers([a.bfloat16(), b.bfloat16(), c.bfloat16()], dtype=9)
        expected = (a + b + c).bfloat16()
        assert torch.equal(result, expected)

    def test_f32_native(self):
        a = torch.arange(256, dtype=torch.float32)
        b = torch.arange(256, dtype=torch.float32) * 2
        c = torch.arange(256, dtype=torch.float32) * 3
        result = self._sum_pointers([a, b, c], dtype=7)
        expected = a + b + c
        assert torch.allclose(result, expected, atol=1e-6), f"max err = {(result - expected).abs().max():.8f}"
