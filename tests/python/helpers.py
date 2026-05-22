import ctypes
import os
import os
import torch

LIB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "build", "Release", "libglm_ops.so")

ATOL = 1e-2
RTOL = 1e-2


class ModelNotFoundError(Exception):
    pass


def get_model_path(repo_id: str) -> str:
    hf_home = os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
    model_dir = os.path.join(hf_home, "hub", f"models--{repo_id.replace('/', '--')}")
    if not os.path.isdir(model_dir):
        raise ModelNotFoundError(
            f'Model "{repo_id}" not found in HuggingFace cache.\n'
            f"  Looked at: {model_dir}\n"
            f"  HF_HOME: {hf_home}\n"
            f"  Download with: huggingface-cli download {repo_id}"
        )
    ref_path = os.path.join(model_dir, "refs", "main")
    if not os.path.isfile(ref_path):
        raise ModelNotFoundError(
            f'No refs/main found for "{repo_id}" — model may not be fully downloaded.\n'
            f"  Expected: {ref_path}"
        )
    commit_hash = open(ref_path).read().strip()
    snapshot_dir = os.path.join(model_dir, "snapshots", commit_hash)
    if not os.path.isdir(snapshot_dir):
        raise ModelNotFoundError(
            f'Snapshot directory not found for "{repo_id}".\n'
            f"  Expected: {snapshot_dir}\n"
            f"  Commit hash: {commit_hash}"
        )
    return snapshot_dir


def has_model_cached(repo_id: str) -> bool:
    try:
        get_model_path(repo_id)
        return True
    except ModelNotFoundError:
        return False

class GpuBuffer:
    def __init__(self, ops, ptr, size):
        self._ops = ops
        self._ptr = ptr
        self._size = size
        self._freed = False

    @property
    def ptr(self):
        if self._freed:
            raise RuntimeError("Cannot access freed GpuBuffer")
        return self._ptr

    @property
    def size(self):
        return self._size

    def free(self):
        if self._freed:
            return
        self._ops.free_buf(self._ptr)
        self._freed = True

    def __del__(self):
        if not self._freed:
            self.free()


class GpuPtrs:
    """Array of GPU pointers stored on device. Used for mul_mat_id weight pointer arrays."""
    def __init__(self, ptrs, device):
        import torch
        self._ptrs_tensor = torch.tensor(ptrs, dtype=torch.int64, device=device)

    @property
    def data_ptr(self):
        return self._ptrs_tensor.data_ptr()


class MmapFile:
    def __init__(self, ops, ptr, path, size):
        self._ops = ops
        self._ptr = ptr
        self._path = path
        self._size = size
        self._closed = False

    @property
    def ptr(self):
        if self._closed:
            raise RuntimeError("Cannot access closed MmapFile")
        return self._ptr

    @property
    def size(self):
        return self._size

    @property
    def path(self):
        return self._path

    def close(self):
        if self._closed:
            return
        self._ops.mmap_close(self._ptr, self._size)
        self._closed = True

    def __del__(self):
        if not self._closed:
            self.close()


class GlmOps:
    def __init__(self, lib_path=None, device_id=0):
        if lib_path is None:
            lib_path = LIB_PATH
        self.lib = ctypes.CDLL(lib_path)
        self._setup_signatures()
        self.ctx = self.lib.glm_init(device_id)
        if self.ctx is None or self.ctx == 0:
            raise RuntimeError(f"glm_init failed on device {device_id}")
        self.device = device_id

    def _setup_signatures(self):
        self.lib.glm_init.restype = ctypes.c_void_p
        self.lib.glm_init.argtypes = [ctypes.c_int]

        self.lib.glm_free.restype = None
        self.lib.glm_free.argtypes = [ctypes.c_void_p]

        self.lib.glm_alloc.restype = ctypes.c_void_p
        self.lib.glm_alloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t]

        self.lib.glm_free_buf.restype = None
        self.lib.glm_free_buf.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

        self.lib.glm_h2d.restype = None
        self.lib.glm_h2d.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]

        self.lib.glm_d2h.restype = None
        self.lib.glm_d2h.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]

        self.lib.glm_rmsnorm.restype = None
        self.lib.glm_rmsnorm.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_float, ctypes.c_int, ctypes.c_int
        ]

        self.lib.glm_silu_and_mul.restype = None
        self.lib.glm_silu_and_mul.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int
        ]

        self.lib.glm_linear.restype = None
        self.lib.glm_linear.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]

        self.lib.glm_embedding.restype = None
        self.lib.glm_embedding.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int
        ]

        self.lib.glm_layernorm.restype = None
        self.lib.glm_layernorm.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_float, ctypes.c_int, ctypes.c_int
        ]

        self.lib.glm_relu.restype = None
        self.lib.glm_relu.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int
        ]

        self.lib.glm_sigmoid.restype = None
        self.lib.glm_sigmoid.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int
        ]

        self.lib.glm_softmax.restype = None
        self.lib.glm_softmax.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int
        ]

        self.lib.glm_causal_mask.restype = None
        self.lib.glm_causal_mask.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int
        ]

        self.lib.glm_fill.restype = None
        self.lib.glm_fill.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float, ctypes.c_int
        ]

        self.lib.glm_gather.restype = None
        self.lib.glm_gather.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]

        self.lib.glm_scatter_scalar.restype = None
        self.lib.glm_scatter_scalar.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float,
            ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]

        self.lib.glm_cat_last_dim.restype = None
        self.lib.glm_cat_last_dim.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]

        self.lib.glm_masked_fill.restype = None
        self.lib.glm_masked_fill.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_float, ctypes.c_int
        ]

        self.lib.glm_index_add.restype = None
        self.lib.glm_index_add.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int
        ]

        self.lib.glm_rotary_embedding.restype = None
        self.lib.glm_rotary_embedding.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]

        self.lib.glm_apply_rotary_pos_emb.restype = None
        self.lib.glm_apply_rotary_pos_emb.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_bool
        ]

        self.lib.glm_apply_rotary_pos_emb_partial.restype = None
        self.lib.glm_apply_rotary_pos_emb_partial.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_bool
        ]

        self.lib.glm_topk.restype = None
        self.lib.glm_topk.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]

        self.lib.glm_bmm.restype = None
        self.lib.glm_bmm.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_float, ctypes.c_float,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]

        self.lib.glm_rope_transpose.restype = None
        self.lib.glm_rope_transpose.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_bool
        ]

        self.lib.glm_mla_v_expand.restype = None
        self.lib.glm_mla_v_expand.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int
        ]

        self.lib.glm_scale.restype = None
        self.lib.glm_scale.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_float, ctypes.c_int
        ]

        self.lib.glm_add.restype = None
        self.lib.glm_add.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int
        ]

        self.lib.glm_add_broadcast.restype = None
        self.lib.glm_add_broadcast.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int
        ]

        self.lib.glm_row_scale_add.restype = None
        self.lib.glm_row_scale_add.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int
        ]

        self.lib.glm_expand_dim1.restype = None
        self.lib.glm_expand_dim1.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]

        self.lib.glm_expand_dim1_strided.restype = None
        self.lib.glm_expand_dim1_strided.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int
        ]

        self.lib.glm_transpose_4d.restype = None
        self.lib.glm_transpose_4d.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]

        self.lib.glm_mul.restype = None
        self.lib.glm_mul.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_int
        ]

        self.lib.glm_mul_broadcast.restype = None
        self.lib.glm_mul_broadcast.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int
        ]

        self.lib.glm_reduce_sum.restype = None
        self.lib.glm_reduce_sum.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int
        ]

        self.lib.glm_index_select.restype = None
        self.lib.glm_index_select.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int
        ]

        self.lib.glm_rotate_input_ids.restype = None
        self.lib.glm_rotate_input_ids.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int
        ]

        self.lib.glm_arange.restype = None
        self.lib.glm_arange.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]

        self.lib.glm_max.restype = None
        self.lib.glm_max.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]

        self.lib.glm_memcpy.restype = None
        self.lib.glm_memcpy.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int
        ]

        self.lib.glm_synchronize.restype = None
        self.lib.glm_synchronize.argtypes = [ctypes.c_void_p]

        self.lib.glm_mmap_open.restype = ctypes.c_void_p
        self.lib.glm_mmap_open.argtypes = [ctypes.c_char_p]

        self.lib.glm_mmap_close.restype = None
        self.lib.glm_mmap_close.argtypes = [ctypes.c_void_p, ctypes.c_uint64]

        self.lib.glm_mmap_load.restype = None
        self.lib.glm_mmap_load.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64]

        self.lib.glm_flash_prefill.restype = None
        self.lib.glm_flash_prefill.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_float,
        ]

        self.lib.glm_flash_decode.restype = None
        self.lib.glm_flash_decode.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_float,
        ]

        self.lib.glm_alloc_pinned.restype = ctypes.c_void_p
        self.lib.glm_alloc_pinned.argtypes = [ctypes.c_size_t]

        self.lib.glm_free_pinned.restype = None
        self.lib.glm_free_pinned.argtypes = [ctypes.c_void_p]

        self.lib.glm_batch_decode_plan.restype = None
        self.lib.glm_batch_decode_plan.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_bool,
        ]

        self.lib.glm_batch_decode_run.restype = None
        self.lib.glm_batch_decode_run.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_float,
        ]

        self.lib.glm_batch_decode_plan.restype = None
        self.lib.glm_batch_prefill_paged_plan.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_int32,
        ]

        self.lib.glm_batch_prefill_paged_run.restype = None
        self.lib.glm_batch_prefill_paged_run.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_int32, ctypes.c_int32,
            ctypes.c_int32, ctypes.c_float,
        ]

        self.lib.glm_mla_prefill_plan.restype = None
        self.lib.glm_mla_prefill_plan.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_bool,
            ctypes.c_uint32, ctypes.c_uint32,  # cp_world_size, cp_rank
        ]

        self.lib.glm_mla_prefill_run.restype = None
        self.lib.glm_mla_prefill_run.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_int, ctypes.c_float,
            ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_void_p,  # lse (optional, nullptr to skip)
            ctypes.c_uint32, ctypes.c_uint32,  # cp_world_size, cp_rank
        ]

        self.lib.glm_mla_decode_plan.restype = None
        self.lib.glm_mla_decode_plan.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_uint32, ctypes.c_bool,
            ctypes.c_uint32, ctypes.c_uint32,
        ]

        self.lib.glm_mla_decode_run.restype = None
        self.lib.glm_mla_decode_run.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_uint32, ctypes.c_float,
            ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_void_p,  # lse (optional, nullptr to skip)
        ]

        self.lib.glm_mla_kv_cache_append.restype = None
        self.lib.glm_mla_kv_cache_append.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_size_t, ctypes.c_size_t,
            ctypes.c_uint32, ctypes.c_uint32,  # cp_rank, cp_world_size
        ]

        self.lib.glm_decode_step.restype = None
        self.lib.glm_decode_step.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_int32,
        ]

        self.lib.glm_graph_begin_capture.restype = None
        self.lib.glm_graph_begin_capture.argtypes = [ctypes.c_void_p]

        self.lib.glm_graph_end_capture.restype = ctypes.c_void_p
        self.lib.glm_graph_end_capture.argtypes = [ctypes.c_void_p]

        self.lib.glm_graph_instantiate.restype = ctypes.c_void_p
        self.lib.glm_graph_instantiate.argtypes = [ctypes.c_void_p]

        self.lib.glm_graph_launch.restype = None
        self.lib.glm_graph_launch.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

        self.lib.glm_graph_exec_update.restype = ctypes.c_int
        self.lib.glm_graph_exec_update.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

        self.lib.glm_graph_destroy.restype = None
        self.lib.glm_graph_destroy.argtypes = [ctypes.c_void_p]

        self.lib.glm_graph_exec_destroy.restype = None
        self.lib.glm_graph_exec_destroy.argtypes = [ctypes.c_void_p]

        self.lib.glm_kv_cache_write.restype = None
        self.lib.glm_kv_cache_write.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_uint32, ctypes.c_uint32,
        ]

        self.lib.glm_fp8_linear_decode.restype = None
        self.lib.glm_fp8_linear_decode.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]

        self.lib.glm_nvfp4_linear_decode.restype = None
        self.lib.glm_nvfp4_linear_decode.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]

        self.lib.glm_gdn_recurrent_step.restype = None
        self.lib.glm_gdn_recurrent_step.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]

        self.lib.glm_gdn_prefill.restype = None
        self.lib.glm_gdn_prefill.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]

        self.lib.glm_causal_conv1d.restype = None
        self.lib.glm_causal_conv1d.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]

        self.lib.glm_causal_conv1d_update.restype = None
        self.lib.glm_causal_conv1d_update.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
        ]

        self.lib.glm_rmsnorm_gated.restype = None
        self.lib.glm_rmsnorm_gated.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_float, ctypes.c_int, ctypes.c_int,
        ]

        self.lib.glm_fused_add_rmsnorm.restype = None
        self.lib.glm_fused_add_rmsnorm.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_float, ctypes.c_int, ctypes.c_int,
        ]

        self.lib.glm_fused_norm_rope.restype = None
        self.lib.glm_fused_norm_rope.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_float, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_bool
        ]

        self.lib.glm_gate_sigmoid_mul.restype = None
        self.lib.glm_gate_sigmoid_mul.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]

        self.lib.glm_context_parallel_merge.restype = None
        self.lib.glm_context_parallel_merge.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]

        self.lib.glm_context_parallel_merge_heads.restype = None
        self.lib.glm_context_parallel_merge_heads.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]

        self.lib.glm_p2p_enable_peer_access.restype = ctypes.c_int
        self.lib.glm_p2p_enable_peer_access.argtypes = [
            ctypes.c_void_p, ctypes.c_int,
        ]

        self.lib.glm_p2p_create_instance.restype = ctypes.c_void_p
        self.lib.glm_p2p_create_instance.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ]

        self.lib.glm_p2p_destroy_instance.restype = None
        self.lib.glm_p2p_destroy_instance.argtypes = [ctypes.c_void_p]

        self.lib.glm_p2p_get_flag_ptr.restype = ctypes.c_void_p
        self.lib.glm_p2p_get_flag_ptr.argtypes = [ctypes.c_void_p]

        self.lib.glm_p2p_set_max_bytes.restype = None
        self.lib.glm_p2p_set_max_bytes.argtypes = [ctypes.c_void_p, ctypes.c_size_t]

        self.lib.glm_p2p_set_peers.restype = None
        self.lib.glm_p2p_set_peers.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p),
        ]

        self.lib.glm_p2p_allreduce.restype = None
        self.lib.glm_p2p_allreduce.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int,
        ]

        self.lib.glm_p2p_rmsnorm.restype = None
        self.lib.glm_p2p_rmsnorm.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_float, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]

        self.lib.glm_p2p_cp_merge.restype = None
        self.lib.glm_p2p_cp_merge.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]

        self.lib.glm_p2p_cp_merge_heads.restype = None
        self.lib.glm_p2p_cp_merge_heads.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]

        self.lib.glm_row_normalize.restype = None
        self.lib.glm_row_normalize.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_float, ctypes.c_int, ctypes.c_int, ctypes.c_bool,
        ]

        self.lib.glm_group_mask_mul.restype = None
        self.lib.glm_group_mask_mul.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]

        self.lib.glm_expert_scale.restype = None
        self.lib.glm_expert_scale.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]

        self.lib.glm_mul_mat_id.restype = None
        self.lib.glm_mul_mat_id.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]

        self.lib.glm_nvfp4_mul_mat_id.restype = None
        self.lib.glm_nvfp4_mul_mat_id.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]

        self.lib.glm_scatter_add_rows.restype = None
        self.lib.glm_scatter_add_rows.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p,
        ]

    def __del__(self):
        if hasattr(self, 'ctx') and self.ctx:
            self.lib.glm_free(self.ctx)
            self.ctx = None

    def _ptr(self, t):
        if isinstance(t, ctypes.c_void_p):
            return t
        if isinstance(t, int):
            return ctypes.c_void_p(t)
        return ctypes.c_void_p(t.data_ptr())

    def alloc(self, size):
        ptr = self.lib.glm_alloc(self.ctx, size)
        if ptr is None or ptr == 0:
            raise RuntimeError(f"glm_alloc failed for size {size}")
        return ptr.value if hasattr(ptr, 'value') else ptr

    def free_buf(self, ptr):
        self.lib.glm_free_buf(self.ctx, ctypes.c_void_p(ptr))

    def h2d(self, gpu_ptr, cpu_data, size=None):
        if size is None:
            size = len(cpu_data)
        self.lib.glm_h2d(self.ctx, ctypes.c_void_p(gpu_ptr), cpu_data, size)

    def d2h(self, cpu_buf, gpu_ptr, size):
        self.lib.glm_d2h(self.ctx, cpu_buf, ctypes.c_void_p(gpu_ptr), size)

    def synchronize(self):
        self.lib.glm_synchronize(self.ctx)

    def mmap_open(self, path):
        encoded = path.encode('utf-8') if isinstance(path, str) else path
        ptr = self.lib.glm_mmap_open(encoded)
        if ptr is None or ptr == 0:
            raise RuntimeError(f"glm_mmap_open failed for {path}")
        raw_ptr = ptr.value if hasattr(ptr, 'value') else ptr
        size = os.path.getsize(path if isinstance(path, str) else path.decode('utf-8'))
        return MmapFile(self, raw_ptr, path, size)

    def mmap_close(self, ptr, size):
        self.lib.glm_mmap_close(ctypes.c_void_p(ptr), ctypes.c_uint64(size))

    def mmap_load(self, gpu_dst, mmap_ptr, offset, nbytes):
        self.lib.glm_mmap_load(self.ctx, ctypes.c_void_p(gpu_dst),
                               ctypes.c_void_p(mmap_ptr),
                               ctypes.c_uint64(offset), ctypes.c_uint64(nbytes))

    def upload_tensor(self, tensor):
        assert tensor.is_cuda, "Tensor must be on CUDA"
        return ctypes.c_void_p(tensor.data_ptr())

    def arange(self, out, start, step, count):
        self.lib.glm_arange(self.ctx, self._ptr(out), start, step, count)

    def argmax(self, out_index, input, dim, batch=1):
        tmp_vals = self.alloc(batch * 2)
        self.max(tmp_vals, out_index, input, dim, batch, offset=0)
        self.free_buf(tmp_vals)

    def max(self, out_values, out_indices, input, dim, batch=1, offset=0):
        self.lib.glm_max(self.ctx, self._ptr(out_values), self._ptr(out_indices), self._ptr(input), dim, batch, offset)

    def memcpy(self, dst, src, bytes, kind=4):
        self.lib.glm_memcpy(self.ctx, self._ptr(dst), self._ptr(src), bytes, kind)

    def rmsnorm(self, output, input, weight, eps, dim, batch):
        self.lib.glm_rmsnorm(
            self.ctx,
            self._ptr(output),
            self._ptr(input),
            self._ptr(weight),
            ctypes.c_float(eps), dim, batch
        )

    def silu_and_mul(self, output, gate, up, intermediate, batch):
        self.lib.glm_silu_and_mul(
            self.ctx,
            self._ptr(output),
            self._ptr(gate),
            self._ptr(up),
            intermediate, batch
        )

    def linear(self, output, input, weight, batch, n, k):
        self.lib.glm_linear(
            self.ctx,
            self._ptr(output),
            self._ptr(input),
            self._ptr(weight),
            batch, n, k
        )

    def embedding(self, output, table, ids, hidden, seq_len):
        self.lib.glm_embedding(
            self.ctx,
            self._ptr(output),
            self._ptr(table),
            self._ptr(ids),
            hidden, seq_len
        )

    def layernorm(self, output, input, weight, bias, eps, dim, batch):
        bias_ptr = self._ptr(bias) if bias is not None else ctypes.c_void_p(0)
        self.lib.glm_layernorm(
            self.ctx,
            self._ptr(output),
            self._ptr(input),
            self._ptr(weight),
            bias_ptr,
            ctypes.c_float(eps), dim, batch
        )

    def relu(self, output, input, n):
        self.lib.glm_relu(
            self.ctx,
            self._ptr(output),
            self._ptr(input),
            n
        )

    def sigmoid(self, output, input, n):
        self.lib.glm_sigmoid(
            self.ctx,
            self._ptr(output),
            self._ptr(input),
            n
        )

    def softmax(self, output, input, mask, dim, batch):
        mask_ptr = self._ptr(mask) if mask is not None else ctypes.c_void_p(0)
        self.lib.glm_softmax(
            self.ctx,
            self._ptr(output),
            self._ptr(input),
            mask_ptr,
            dim, batch
        )

    def causal_mask(self, output, seq_len):
        self.lib.glm_causal_mask(
            self.ctx,
            self._ptr(output),
            seq_len
        )

    def fill(self, output, value, n):
        self.lib.glm_fill(
            self.ctx,
            self._ptr(output),
            ctypes.c_float(value),
            n
        )

    def gather(self, output, input, indices, k, in_dim, batch, elem_size=2):
        self.lib.glm_gather(
            self.ctx,
            self._ptr(output),
            self._ptr(input),
            self._ptr(indices),
            k, in_dim, batch, elem_size
        )

    def scatter_scalar(self, output, indices, value, k, out_dim, batch):
        self.lib.glm_scatter_scalar(
            self.ctx,
            self._ptr(output),
            self._ptr(indices),
            ctypes.c_float(value),
            k, out_dim, batch
        )

    def cat_last_dim(self, output, a, b, a_last_dim, b_last_dim, outer):
        self.lib.glm_cat_last_dim(
            self.ctx,
            self._ptr(output),
            self._ptr(a),
            self._ptr(b),
            a_last_dim, b_last_dim, outer
        )

    def masked_fill(self, output, input, mask, value, n):
        self.lib.glm_masked_fill(
            self.ctx,
            self._ptr(output),
            self._ptr(input),
            self._ptr(mask),
            ctypes.c_float(value),
            n
        )

    def index_add(self, output, indices, values, n_indices, dim):
        self.lib.glm_index_add(
            self.ctx,
            self._ptr(output),
            self._ptr(indices),
            self._ptr(values),
            n_indices, dim
        )

    def rotary_embedding(self, cos_out, sin_out, inv_freq, position_ids, dim_half, batch, seq_len):
        if hasattr(position_ids, 'data_ptr'):
            pos_ids_i32 = position_ids.to(torch.int32)
            pos_ptr = self._ptr(pos_ids_i32)
        else:
            pos_ptr = self._ptr(position_ids)
        self.lib.glm_rotary_embedding(
            self.ctx,
            self._ptr(cos_out),
            self._ptr(sin_out),
            self._ptr(inv_freq),
            pos_ptr,
            dim_half, batch, seq_len
        )

    def apply_rotary_pos_emb(self, output, x, cos, sin, rope_dim, n_heads, seq_len, batch, unsqueeze_dim, interleaved=False):
        self.lib.glm_apply_rotary_pos_emb(
            self.ctx,
            self._ptr(output),
            self._ptr(x),
            self._ptr(cos),
            self._ptr(sin),
            rope_dim, n_heads, seq_len, batch, unsqueeze_dim, interleaved
        )

    def apply_rotary_pos_emb_partial(self, output, x, cos, sin, rope_dim, head_dim, n_heads, seq_len, batch, unsqueeze_dim, interleaved=False):
        self.lib.glm_apply_rotary_pos_emb_partial(
            self.ctx,
            self._ptr(output),
            self._ptr(x),
            self._ptr(cos),
            self._ptr(sin),
            rope_dim, head_dim, n_heads, seq_len, batch, unsqueeze_dim, interleaved
        )

    def topk(self, out_values, out_indices, input, k, dim, batch, offset=0):
        self.lib.glm_topk(
            self.ctx,
            self._ptr(out_values),
            self._ptr(out_indices),
            self._ptr(input),
            k, dim, batch, offset
        )

    def bmm(self, C, A, B, alpha, beta, batch, M, N, K, transA=0, transB=0):
        self.lib.glm_bmm(
            self.ctx,
            self._ptr(C),
            self._ptr(A),
            self._ptr(B),
            ctypes.c_float(alpha), ctypes.c_float(beta),
            batch, M, N, K, transA, transB
        )

    def ropeTranspose(self, output, input, cos, sin, rope_dim, head_dim, n_heads, seq_len, batch, in_stride, interleaved=False):
        self.lib.glm_rope_transpose(
            self.ctx,
            self._ptr(output), self._ptr(input),
            self._ptr(cos) if cos is not None else ctypes.c_void_p(0),
            self._ptr(sin) if sin is not None else ctypes.c_void_p(0),
            rope_dim, head_dim, n_heads, seq_len, batch, in_stride, interleaved
        )

    def mlaVExpand(self, result, attn_out, v_proj, kv_lora_rank, v_head_dim, n_heads, seq_len, batch, attn_n_heads=None, head_offset=0):
        if attn_n_heads is None:
            attn_n_heads = n_heads
        self.lib.glm_mla_v_expand(
            self.ctx,
            self._ptr(result), self._ptr(attn_out), self._ptr(v_proj),
            kv_lora_rank, v_head_dim, n_heads, seq_len, batch,
            attn_n_heads, head_offset
        )

    def scale(self, output, input, scale, n):
        self.lib.glm_scale(
            self.ctx,
            self._ptr(output),
            self._ptr(input),
            ctypes.c_float(scale),
            n
        )

    def add(self, output, a, b, n):
        self.lib.glm_add(
            self.ctx,
            self._ptr(output),
            self._ptr(a),
            self._ptr(b),
            n
        )

    def add_broadcast(self, output, a, b, dim, rows):
        self.lib.glm_add_broadcast(
            self.ctx,
            self._ptr(output),
            self._ptr(a),
            self._ptr(b),
            dim,
            rows
        )

    def row_scale_add(self, output, input, scales, rows, dim):
        self.lib.glm_row_scale_add(
            self.ctx,
            self._ptr(output),
            self._ptr(input),
            self._ptr(scales),
            rows, dim
        )

    def expand_dim1(self, output, input, dim1_out, dim1_in, seq_len, head_dim, batch):
        self.lib.glm_expand_dim1(
            self.ctx,
            self._ptr(output),
            self._ptr(input),
            dim1_out, dim1_in, seq_len, head_dim, batch
        )

    def expand_dim1_strided(self, output, input, dim1_out, dim1_in, seq_len, head_dim, batch, head_stride):
        self.lib.glm_expand_dim1_strided(
            self.ctx,
            self._ptr(output),
            self._ptr(input),
            dim1_out, dim1_in, seq_len, head_dim, batch, head_stride
        )

    def transpose_4d(self, output, input, d0, d1, d2, d3, p0, p1, p2, p3):
        self.lib.glm_transpose_4d(
            self.ctx,
            self._ptr(output),
            self._ptr(input),
            d0, d1, d2, d3, p0, p1, p2, p3
        )

    def mul(self, output, a, b, n):
        self.lib.glm_mul(
            self.ctx,
            self._ptr(output),
            self._ptr(a),
            self._ptr(b),
            n
        )

    def mul_broadcast(self, output, a, b, dim, rows):
        self.lib.glm_mul_broadcast(
            self.ctx,
            self._ptr(output),
            self._ptr(a),
            self._ptr(b),
            dim,
            rows
        )

    def reduce_sum(self, output, input, rows, cols):
        self.lib.glm_reduce_sum(
            self.ctx,
            self._ptr(output),
            self._ptr(input),
            rows, cols
        )

    def index_select(self, output, src, indices, dim, k):
        self.lib.glm_index_select(
            self.ctx,
            self._ptr(output),
            self._ptr(src),
            self._ptr(indices),
            dim, k
        )

    def rotate_input_ids(self, output_ids, input_ids, qo_indptr, new_tokens, batch_size):
        self.lib.glm_rotate_input_ids(
            self.ctx,
            self._ptr(output_ids),
            self._ptr(input_ids),
            self._ptr(qo_indptr),
            self._ptr(new_tokens),
            batch_size
        )

    def flash_prefill(self, q, k, v, o, tmp,
                      qo_len, kv_len,
                      num_qo_heads, num_kv_heads, head_dim,
                      q_stride_n, q_stride_h,
                      kv_stride_n, kv_stride_h,
                      v_stride_n, v_stride_h,
                      mask_mode, kv_layout, sm_scale):
        self.lib.glm_flash_prefill(
            self.ctx,
            self._ptr(q), self._ptr(k), self._ptr(v), self._ptr(o), self._ptr(tmp),
            qo_len, kv_len,
            num_qo_heads, num_kv_heads, head_dim,
            q_stride_n, q_stride_h,
            kv_stride_n, kv_stride_h,
            v_stride_n, v_stride_h,
            mask_mode, kv_layout, ctypes.c_float(sm_scale)
        )

    def flash_decode(self, q, k, v, o, tmp,
                     kv_len,
                     num_qo_heads, num_kv_heads, head_dim,
                     q_stride_n, q_stride_h,
                     kv_stride_n, kv_stride_h,
                     sm_scale):
        self.lib.glm_flash_decode(
            self.ctx,
            self._ptr(q), self._ptr(k), self._ptr(v), self._ptr(o), self._ptr(tmp),
            kv_len,
            num_qo_heads, num_kv_heads, head_dim,
            q_stride_n, q_stride_h,
            kv_stride_n, kv_stride_h,
            ctypes.c_float(sm_scale)
        )

    def alloc_pinned(self, nbytes):
        ptr = self.lib.glm_alloc_pinned(nbytes)
        if ptr is None or ptr == 0:
            raise RuntimeError(f"glm_alloc_pinned failed for size {nbytes}")
        return ptr.value if hasattr(ptr, 'value') else ptr

    def free_pinned(self, ptr):
        self.lib.glm_free_pinned(ctypes.c_void_p(ptr))

    def batch_decode_plan(self, float_ws, float_ws_size,
                          int_ws, pinned_int_ws, int_ws_size,
                          plan_info, indptr_h,
                          batch_size, num_qo_heads, num_kv_heads, page_size,
                          enable_cuda_graph=False):
        self.lib.glm_batch_decode_plan(
            self.ctx,
            ctypes.c_void_p(float_ws), ctypes.c_size_t(float_ws_size),
            ctypes.c_void_p(int_ws), ctypes.c_void_p(pinned_int_ws), ctypes.c_size_t(int_ws_size),
            ctypes.c_void_p(plan_info),
            ctypes.c_void_p(indptr_h),
            ctypes.c_uint32(batch_size),
            ctypes.c_uint32(num_qo_heads), ctypes.c_uint32(num_kv_heads),
            ctypes.c_uint32(page_size),
            ctypes.c_bool(enable_cuda_graph)
        )

    def batch_decode_run(self, q, o, k_data, v_data,
                         indices, indptr_d, last_page_len,
                         float_ws, int_ws, plan_info,
                         batch_size,
                         num_qo_heads, num_kv_heads,
                         head_dim, page_size, sm_scale):
        self.lib.glm_batch_decode_run(
            self.ctx,
            ctypes.c_void_p(q), ctypes.c_void_p(o),
            ctypes.c_void_p(k_data), ctypes.c_void_p(v_data),
            ctypes.c_void_p(indices), ctypes.c_void_p(indptr_d), ctypes.c_void_p(last_page_len),
            ctypes.c_void_p(float_ws), ctypes.c_void_p(int_ws),
            ctypes.c_void_p(plan_info),
            ctypes.c_uint32(batch_size),
            ctypes.c_uint32(num_qo_heads), ctypes.c_uint32(num_kv_heads),
            ctypes.c_uint32(head_dim), ctypes.c_uint32(page_size),
            ctypes.c_float(sm_scale)
        )

    def batch_prefill_paged_plan(self, float_ws, float_ws_size,
                                  int_ws, pinned_int_ws, int_ws_size,
                                  plan_info, qo_indptr_h, paged_kv_indptr_h,
                                  total_qo_rows, batch_size,
                                  num_qo_heads, num_kv_heads,
                                  head_dim, page_size, mask_mode):
        self.lib.glm_batch_prefill_paged_plan(
            self.ctx,
            ctypes.c_void_p(float_ws), ctypes.c_size_t(float_ws_size),
            ctypes.c_void_p(int_ws), ctypes.c_void_p(pinned_int_ws), ctypes.c_size_t(int_ws_size),
            ctypes.c_void_p(plan_info),
            ctypes.c_void_p(qo_indptr_h), ctypes.c_void_p(paged_kv_indptr_h),
            ctypes.c_uint32(total_qo_rows), ctypes.c_uint32(batch_size),
            ctypes.c_uint32(num_qo_heads), ctypes.c_uint32(num_kv_heads),
            ctypes.c_uint32(head_dim),             ctypes.c_uint32(page_size),
            ctypes.c_int32(mask_mode)
        )

    def batch_prefill_paged_run(self, q, o,
                                 k_data, v_data,
                                 indices, indptr_d, last_page_len,
                                 float_ws, int_ws, q_indptr_d,
                                 plan_info,
                                 total_qo_rows, batch_size,
                                 num_qo_heads, num_kv_heads, head_dim,
                                 page_size,
                                 q_stride_n, q_stride_h,
                                 mask_mode, sm_scale):
        self.lib.glm_batch_prefill_paged_run(
            self.ctx,
            ctypes.c_void_p(q), ctypes.c_void_p(o),
            ctypes.c_void_p(k_data), ctypes.c_void_p(v_data),
            ctypes.c_void_p(indices), ctypes.c_void_p(indptr_d), ctypes.c_void_p(last_page_len),
            ctypes.c_void_p(float_ws), ctypes.c_void_p(int_ws),
            ctypes.c_void_p(q_indptr_d),
            ctypes.c_void_p(plan_info),
            ctypes.c_uint32(total_qo_rows), ctypes.c_uint32(batch_size),
            ctypes.c_uint32(num_qo_heads), ctypes.c_uint32(num_kv_heads), ctypes.c_uint32(head_dim),
            ctypes.c_uint32(page_size),
            ctypes.c_int32(q_stride_n), ctypes.c_int32(q_stride_h),
            ctypes.c_int32(mask_mode),             ctypes.c_float(sm_scale)
        )

    def mla_prefill_plan(self, float_ws, float_ws_size,
                         int_ws, pinned_int_ws, int_ws_size,
                         plan_info, qo_indptr_h, kv_indptr_h, kv_len_h,
                         batch_size, num_heads, head_dim_o,
                         causal=False, cp_world_size=0, cp_rank=0):
        self.lib.glm_mla_prefill_plan(
            self.ctx,
            ctypes.c_void_p(float_ws), ctypes.c_size_t(float_ws_size),
            ctypes.c_void_p(int_ws), ctypes.c_void_p(pinned_int_ws), ctypes.c_size_t(int_ws_size),
            ctypes.c_void_p(plan_info),
            ctypes.c_void_p(qo_indptr_h), ctypes.c_void_p(kv_indptr_h), ctypes.c_void_p(kv_len_h),
            ctypes.c_uint32(batch_size), ctypes.c_uint32(num_heads), ctypes.c_uint32(head_dim_o),
            ctypes.c_bool(causal),
            ctypes.c_uint32(cp_world_size), ctypes.c_uint32(cp_rank)
        )

    def mla_prefill_run(self, q_nope, q_pe, ckv_data, kpe_data,
                        kv_indices, o, float_ws, int_ws, plan_info,
                        num_heads, page_size, mask_mode, sm_scale,
                        q_nope_stride_n, q_nope_stride_h,
                        q_pe_stride_n, q_pe_stride_h,
                        ckv_stride_page, ckv_stride_n,
                        kpe_stride_page, kpe_stride_n,
                        o_stride_n, o_stride_h,
                        head_dim_ckv, head_dim_kpe,
                        lse=None,
                        cp_world_size=0, cp_rank=0):
        self.lib.glm_mla_prefill_run(
            self.ctx,
            ctypes.c_void_p(q_nope), ctypes.c_void_p(q_pe),
            ctypes.c_void_p(ckv_data), ctypes.c_void_p(kpe_data),
            ctypes.c_void_p(kv_indices),
            ctypes.c_void_p(o),
            ctypes.c_void_p(float_ws), ctypes.c_void_p(int_ws),
            ctypes.c_void_p(plan_info),
            ctypes.c_uint32(num_heads), ctypes.c_uint32(page_size),
            ctypes.c_int(mask_mode), ctypes.c_float(sm_scale),
            ctypes.c_uint32(q_nope_stride_n), ctypes.c_uint32(q_nope_stride_h),
            ctypes.c_uint32(q_pe_stride_n), ctypes.c_uint32(q_pe_stride_h),
            ctypes.c_uint32(ckv_stride_page), ctypes.c_uint32(ckv_stride_n),
            ctypes.c_uint32(kpe_stride_page), ctypes.c_uint32(kpe_stride_n),
            ctypes.c_uint32(o_stride_n), ctypes.c_uint32(o_stride_h),
            ctypes.c_uint32(head_dim_ckv), ctypes.c_uint32(head_dim_kpe),
            ctypes.c_void_p(lse),
            ctypes.c_uint32(cp_world_size), ctypes.c_uint32(cp_rank),
        )

    def mla_decode_plan(self, float_ws, float_ws_size,
                        int_ws, pinned_int_ws, int_ws_size,
                        plan_info, indptr_h,
                        batch_size, num_qo_heads, page_size,
                        enable_cuda_graph=False,
                        head_dim_ckv=512, head_dim_kpe=64):
        self.lib.glm_mla_decode_plan(
            self.ctx,
            ctypes.c_void_p(float_ws), ctypes.c_size_t(float_ws_size),
            ctypes.c_void_p(int_ws), ctypes.c_void_p(pinned_int_ws), ctypes.c_size_t(int_ws_size),
            ctypes.c_void_p(plan_info),
            ctypes.c_void_p(indptr_h),
            ctypes.c_uint32(batch_size), ctypes.c_uint32(num_qo_heads),
            ctypes.c_uint32(page_size), ctypes.c_bool(enable_cuda_graph),
            ctypes.c_uint32(head_dim_ckv), ctypes.c_uint32(head_dim_kpe)
        )

    def mla_decode_run(self, q_nope, q_pe, ckv_data, kpe_data,
                       indices, indptr_d, last_page_len,
                       o, float_ws, int_ws, plan_info,
                       batch_size, num_qo_heads, page_size, sm_scale,
                       head_dim_ckv=512, head_dim_kpe=64,
                       lse=None):
        self.lib.glm_mla_decode_run(
            self.ctx,
            ctypes.c_void_p(q_nope), ctypes.c_void_p(q_pe),
            ctypes.c_void_p(ckv_data), ctypes.c_void_p(kpe_data),
            ctypes.c_void_p(indices), ctypes.c_void_p(indptr_d), ctypes.c_void_p(last_page_len),
            ctypes.c_void_p(o),
            ctypes.c_void_p(float_ws), ctypes.c_void_p(int_ws),
            ctypes.c_void_p(plan_info),
            ctypes.c_uint32(batch_size), ctypes.c_uint32(num_qo_heads),
            ctypes.c_uint32(page_size), ctypes.c_float(sm_scale),
            ctypes.c_uint32(head_dim_ckv), ctypes.c_uint32(head_dim_kpe),
            ctypes.c_void_p(lse)
        )

    def mla_kv_cache_append(self, ckv_data, kpe_data,
                            indices, indptr, last_page_len,
                            append_ckv, append_kpe,
                            batch_indices, positions,
                            nnz, page_size,
                            head_dim_ckv, head_dim_kpe,
                            append_ckv_stride_n, append_kpe_stride_n,
                            cp_rank=0, cp_world_size=1):
        self.lib.glm_mla_kv_cache_append(
            self.ctx,
            ctypes.c_void_p(ckv_data), ctypes.c_void_p(kpe_data),
            ctypes.c_void_p(indices), ctypes.c_void_p(indptr), ctypes.c_void_p(last_page_len),
            ctypes.c_void_p(append_ckv), ctypes.c_void_p(append_kpe),
            ctypes.c_void_p(batch_indices), ctypes.c_void_p(positions),
            ctypes.c_uint32(nnz), ctypes.c_uint32(page_size),
            ctypes.c_uint32(head_dim_ckv), ctypes.c_uint32(head_dim_kpe),
            ctypes.c_size_t(append_ckv_stride_n), ctypes.c_size_t(append_kpe_stride_n),
            ctypes.c_uint32(cp_rank), ctypes.c_uint32(cp_world_size)
        )

    def decode_step(self, position_ids, last_page_len, slot_mapping,
                     indptr, indices, page_size, batch_size,
                     cp_world_size=1, cp_rank=0, steps=1):
        self.lib.glm_decode_step(
            self.ctx,
            ctypes.c_void_p(position_ids), ctypes.c_void_p(last_page_len),
            ctypes.c_void_p(slot_mapping),
            ctypes.c_void_p(indptr), ctypes.c_void_p(indices),
            ctypes.c_uint32(page_size), ctypes.c_uint32(batch_size),
            ctypes.c_uint32(cp_world_size), ctypes.c_uint32(cp_rank),
            ctypes.c_int32(steps)
        )

    def graph_begin_capture(self):
        self.lib.glm_graph_begin_capture(self.ctx)

    def graph_end_capture(self):
        ptr = self.lib.glm_graph_end_capture(self.ctx)
        return ptr.value if hasattr(ptr, 'value') else ptr

    def graph_instantiate(self, graph):
        ptr = self.lib.glm_graph_instantiate(ctypes.c_void_p(graph))
        return ptr.value if hasattr(ptr, 'value') else ptr

    def graph_launch(self, graph_exec):
        self.lib.glm_graph_launch(ctypes.c_void_p(graph_exec), self.ctx)

    def graph_exec_update(self, graph_exec, graph):
        return self.lib.glm_graph_exec_update(ctypes.c_void_p(graph_exec), ctypes.c_void_p(graph))

    def graph_destroy(self, graph):
        self.lib.glm_graph_destroy(ctypes.c_void_p(graph))

    def graph_exec_destroy(self, graph_exec):
        self.lib.glm_graph_exec_destroy(ctypes.c_void_p(graph_exec))

    def kv_cache_write(self, src_k, src_v, dst_k, dst_v, slot_mapping,
                        batch_size, n_kv, hd, page_size,
                        src_k_token_stride, src_k_head_stride,
                        src_v_token_stride, src_v_head_stride):
        self.lib.glm_kv_cache_write(
            self.ctx,
            ctypes.c_void_p(src_k), ctypes.c_void_p(src_v),
            ctypes.c_void_p(dst_k), ctypes.c_void_p(dst_v),
            ctypes.c_void_p(slot_mapping),
            ctypes.c_uint32(batch_size), ctypes.c_uint32(n_kv),
            ctypes.c_uint32(hd), ctypes.c_uint32(page_size),
            ctypes.c_uint32(src_k_token_stride), ctypes.c_uint32(src_k_head_stride),
            ctypes.c_uint32(src_v_token_stride), ctypes.c_uint32(src_v_head_stride)
        )

    def fp8_linear_decode(self, bf16_out, bf16_input, fp8_weight, weight_scale, m, n, k):
        self.lib.glm_fp8_linear_decode(
            self.ctx,
            self._ptr(bf16_out), self._ptr(bf16_input),
            self._ptr(fp8_weight), self._ptr(weight_scale),
            m, n, k
        )

    def nvfp4_linear_decode(self, bf16_out, bf16_input, fp4_weight, weight_scale, weight_scale_2, m, n, k):
        self.lib.glm_nvfp4_linear_decode(
            self.ctx,
            self._ptr(bf16_out), self._ptr(bf16_input),
            self._ptr(fp4_weight), self._ptr(weight_scale), self._ptr(weight_scale_2),
            m, n, k
        )

    def gdn_recurrent_step(self, output, state, qkv, a_raw, b_raw, A_log, dt_bias,
                            num_heads, d_k, d_v, batch_size=1, state_stride=None, qkv_ch_stride=None, qkv_seq_stride=None):
        if state_stride is None:
            state_stride = num_heads * d_k * d_v
        conv_dim = num_heads * (2 * d_k + d_v)
        if qkv_ch_stride is None:
            qkv_ch_stride = batch_size  # default: [convDim, batch] layout
        if qkv_seq_stride is None:
            qkv_seq_stride = 1  # default: [convDim, batch] layout
        self.lib.glm_gdn_recurrent_step(
            self.ctx,
            self._ptr(output), self._ptr(state),
            self._ptr(qkv),
            self._ptr(a_raw), self._ptr(b_raw),
            self._ptr(A_log), self._ptr(dt_bias),
            num_heads, d_k, d_v,
            batch_size, state_stride, qkv_ch_stride, qkv_seq_stride
        )

    def gdn_prefill(self, output, state, qkv, a_raw, b_raw, A_log, dt_bias,
                     cu_seqlens, total_seq_len, num_heads, d_k, d_v, batch_size=1, state_stride=None, qkv_ch_stride=None, qkv_seq_stride=None):
        if state_stride is None:
            state_stride = num_heads * d_k * d_v
        if qkv_ch_stride is None:
            qkv_ch_stride = total_seq_len  # default: [convDim, total_seq_len] layout
        if qkv_seq_stride is None:
            qkv_seq_stride = 1  # default: [convDim, total_seq_len] layout
        self.lib.glm_gdn_prefill(
            self.ctx,
            self._ptr(output), self._ptr(state),
            self._ptr(qkv),
            self._ptr(a_raw), self._ptr(b_raw),
            self._ptr(A_log), self._ptr(dt_bias),
            self._ptr(cu_seqlens),
            total_seq_len, num_heads, d_k, d_v,
            batch_size, state_stride, qkv_ch_stride, qkv_seq_stride
        )

    def causal_conv1d(self, output, conv_state, input, weight, cu_seqlens, conv_dim, total_seq_len, kernel_size, batch_size=1, conv_state_stride=None, ch_stride=None, seq_stride=1):
        cs_ptr = self._ptr(conv_state) if conv_state is not None else ctypes.c_void_p(0)
        if conv_state_stride is None:
            conv_state_stride = conv_dim * (kernel_size - 1)
        if ch_stride is None:
            ch_stride = total_seq_len  # default: [conv_dim, total_seq_len] layout
        self.lib.glm_causal_conv1d(
            self.ctx,
            self._ptr(output), cs_ptr,
            self._ptr(input), self._ptr(weight),
            self._ptr(cu_seqlens),
            conv_dim, total_seq_len, kernel_size,
            batch_size, conv_state_stride, ch_stride, seq_stride
        )

    def causal_conv1d_update(self, output, conv_state, input, weight, conv_dim, kernel_size, batch_size=1, conv_state_stride=None):
        if conv_state_stride is None:
            conv_state_stride = conv_dim * (kernel_size - 1)
        self.lib.glm_causal_conv1d_update(
            self.ctx,
            self._ptr(output), self._ptr(conv_state),
            self._ptr(input), self._ptr(weight),
            conv_dim, kernel_size,
            batch_size, conv_state_stride
        )

    def rmsnorm_gated(self, output, input, gate, weight, eps, dim, batch):
        self.lib.glm_rmsnorm_gated(
            self.ctx,
            self._ptr(output),
            self._ptr(input),
            self._ptr(gate),
            self._ptr(weight),
            ctypes.c_float(eps), dim, batch
        )

    def fused_add_rmsnorm(self, output, residual, input_a, input_b, weight, eps, dim, batch):
        self.lib.glm_fused_add_rmsnorm(
            self.ctx,
            self._ptr(output),
            self._ptr(residual),
            self._ptr(input_a),
            self._ptr(input_b),
            self._ptr(weight),
            ctypes.c_float(eps), dim, batch
        )

    def fused_norm_rope(self, output, input_tensor, weight, cos, sin, eps, rope_dim, head_dim, n_heads, seq_len, batch, in_stride=None, interleaved=False):
        if in_stride is None:
            in_stride = head_dim
        self.lib.glm_fused_norm_rope(
            self.ctx,
            self._ptr(output),
            self._ptr(input_tensor),
            self._ptr(weight),
            self._ptr(cos),
            self._ptr(sin),
            ctypes.c_float(eps), rope_dim, head_dim, n_heads, seq_len, batch, in_stride, interleaved
        )

    def gate_sigmoid_mul(self, attn_out, gate_interleaved, batch_seq, num_heads, head_dim):
        self.lib.glm_gate_sigmoid_mul(
            self.ctx,
            self._ptr(attn_out),
            self._ptr(gate_interleaved),
            batch_seq, num_heads, head_dim
        )

    def row_normalize(self, output, input, scale, rows, cols, normalize=True):
        self.lib.glm_row_normalize(
            self.ctx,
            self._ptr(output),
            self._ptr(input),
            ctypes.c_float(scale), rows, cols, ctypes.c_bool(normalize)
        )

    def group_mask_mul(self, scores, group_mask, num_experts, experts_per_group, n_group, batch):
        self.lib.glm_group_mask_mul(
            self.ctx,
            self._ptr(scores),
            self._ptr(group_mask),
            num_experts, experts_per_group, n_group, batch
        )

    def expert_scale(self, output, weights, indices, expert_id, top_k, batch):
        self.lib.glm_expert_scale(
            self.ctx,
            self._ptr(output),
            self._ptr(weights),
            self._ptr(indices),
            expert_id, top_k, batch
        )

    def mul_mat_id(self, output, input, weight_ptrs, expert_ids, top_k, count, N, K):
        self.lib.glm_mul_mat_id(
            self.ctx,
            self._ptr(output),
            self._ptr(input),
            ctypes.c_void_p(weight_ptrs),
            self._ptr(expert_ids),
            top_k, count, N, K
        )

    def nvfp4_mul_mat_id(self, output, input, weight_ptrs, scale_ptrs, scale2_ptrs, expert_ids, top_k, count, N, K):
        self.lib.glm_nvfp4_mul_mat_id(
            self.ctx,
            self._ptr(output),
            self._ptr(input),
            ctypes.c_void_p(weight_ptrs),
            ctypes.c_void_p(scale_ptrs),
            ctypes.c_void_p(scale2_ptrs),
            self._ptr(expert_ids),
            top_k, count, N, K
        )

    def scatter_add_rows(self, out, input, scales, top_k, dim, num_rows, workspace):
        self.lib.glm_scatter_add_rows(
            self.ctx,
            self._ptr(out),
            self._ptr(input),
            self._ptr(scales),
            top_k, dim, num_rows,
            self._ptr(workspace)
        )

    def context_parallel_merge(self, partial_v_outs, partial_lses, num_shards,
                               merged_v_out, merged_lse,
                               batch_size, num_heads, v_head_dim):
        v_ptrs = (ctypes.c_void_p * num_shards)(*[ctypes.c_void_p(int(p)) for p in partial_v_outs])
        lse_ptrs = (ctypes.c_void_p * num_shards)(*[ctypes.c_void_p(int(p)) for p in partial_lses])
        self.lib.glm_context_parallel_merge(
            self.ctx,
            v_ptrs, lse_ptrs,
            num_shards,
            ctypes.c_void_p(int(merged_v_out)),
            ctypes.c_void_p(int(merged_lse)) if merged_lse is not None else ctypes.c_void_p(0),
            batch_size, num_heads, v_head_dim
        )

    def context_parallel_merge_heads(self, partial_v_outs, partial_lses, num_shards,
                                     merged_v_out, merged_lse,
                                     batch_size, num_heads, shard_n_heads, head_offset, input_n_heads, v_head_dim):
        v_ptrs = (ctypes.c_void_p * num_shards)(*[ctypes.c_void_p(int(p)) for p in partial_v_outs])
        lse_ptrs = (ctypes.c_void_p * num_shards)(*[ctypes.c_void_p(int(p)) for p in partial_lses])
        self.lib.glm_context_parallel_merge_heads(
            self.ctx,
            v_ptrs, lse_ptrs,
            num_shards,
            ctypes.c_void_p(int(merged_v_out)),
            ctypes.c_void_p(int(merged_lse)) if merged_lse is not None else ctypes.c_void_p(0),
            batch_size, num_heads, shard_n_heads, head_offset, input_n_heads, v_head_dim
        )

    def p2p_enable_peer_access(self, peer_device):
        return self.lib.glm_p2p_enable_peer_access(self.ctx, peer_device)

    def p2p_create_instance(self, my_rank, world_size):
        return self.lib.glm_p2p_create_instance(self.ctx, my_rank, world_size)

    def p2p_destroy_instance(self, inst):
        self.lib.glm_p2p_destroy_instance(inst)

    def p2p_get_flag_ptr(self, inst):
        return self.lib.glm_p2p_get_flag_ptr(inst)

    def p2p_set_max_bytes(self, inst, max_bytes):
        self.lib.glm_p2p_set_max_bytes(inst, max_bytes)

    def p2p_set_peers(self, inst, peer_data_ptrs, peer_flag_ptrs, world_size):
        data_arr = (ctypes.c_void_p * world_size)(*[ctypes.c_void_p(int(p)) for p in peer_data_ptrs])
        flag_arr = (ctypes.c_void_p * world_size)(*[ctypes.c_void_p(int(p)) for p in peer_flag_ptrs])
        self.lib.glm_p2p_set_peers(self.ctx, inst, data_arr, flag_arr)

    def p2p_allreduce(self, inst, inp, out, count, dtype=9):
        self.lib.glm_p2p_allreduce(self.ctx, inst, self._ptr(inp), self._ptr(out), count, dtype)

    def p2p_rmsnorm(self, inst, input_ptr, weight_ptr, output_ptr, eps, shard_dim, full_dim, batch):
        self.lib.glm_p2p_rmsnorm(
            self.ctx, inst,
            ctypes.c_void_p(int(input_ptr)),
            ctypes.c_void_p(int(weight_ptr)),
            ctypes.c_void_p(int(output_ptr)),
            ctypes.c_float(eps), shard_dim, full_dim, batch
        )

    def p2p_cp_merge(self, inst, my_v_out, my_lse, merged_v_out, merged_lse,
                     num_shards, batch_size, num_heads, v_head_dim):
        self.lib.glm_p2p_cp_merge(
            self.ctx, inst,
            ctypes.c_void_p(int(my_v_out)),
            ctypes.c_void_p(int(my_lse)),
            ctypes.c_void_p(int(merged_v_out)),
            ctypes.c_void_p(int(merged_lse)) if merged_lse is not None else ctypes.c_void_p(0),
            num_shards, batch_size, num_heads, v_head_dim
        )

    def p2p_cp_merge_heads(self, inst, my_v_out, my_lse, merged_v_out, merged_lse,
                           num_shards, batch_size, num_heads, shard_n_heads, head_offset, input_n_heads, v_head_dim):
        self.lib.glm_p2p_cp_merge_heads(
            self.ctx, inst,
            ctypes.c_void_p(int(my_v_out)),
            ctypes.c_void_p(int(my_lse)),
            ctypes.c_void_p(int(merged_v_out)),
            ctypes.c_void_p(int(merged_lse)) if merged_lse is not None else ctypes.c_void_p(0),
            num_shards, batch_size, num_heads, shard_n_heads, head_offset, input_n_heads, v_head_dim
        )
