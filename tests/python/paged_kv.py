import ctypes
import numpy as np

BF16 = 2  # bytes per bfloat16
I32 = 4   # bytes per int32
BATCH_FLOAT_WS_SIZE = 128 * 1024 * 1024  # 128MB float workspace
BATCH_INT_WS_SIZE = 8 * 1024 * 1024  # 8MB int workspace (GPU)
BATCH_PINNED_INT_WS_SIZE = 8 * 1024 * 1024  # 8MB pinned host int workspace
PAGE_SIZE = 16
DECODE_PLAN_INFO_SIZE = 10  # int64s
PREFILL_PLAN_INFO_SIZE = 15  # int64s


class PagedKVCache:
    def __init__(self, glm, n_kv, hd, n_layers, max_pages, max_batch, page_size=PAGE_SIZE):
        self.glm = glm
        self.n_kv = n_kv
        self.hd = hd
        self.n_layers = n_layers
        self.max_pages = max_pages
        self.max_batch = max_batch
        self.page_size = page_size
        self.page_stride = n_kv * page_size * hd * BF16
        self.k_data = [glm.alloc(max_pages * n_kv * page_size * hd * BF16) for _ in range(n_layers)]
        self.v_data = [glm.alloc(max_pages * n_kv * page_size * hd * BF16) for _ in range(n_layers)]
        self.indices = glm.alloc(max_pages * I32)
        self.indptr_d = glm.alloc((max_batch + 1) * I32)
        self.last_page_len = glm.alloc(max_batch * I32)
        self.indptr_h = glm.alloc_pinned((max_batch + 1) * I32)
        self.last_page_len_h = glm.alloc_pinned(max_batch * I32)
        self.slot_mapping = glm.alloc(max_batch * I32)
        self.slot_mapping_h = glm.alloc_pinned(max_batch * I32)
        self.num_pages_used = 0
        self.seq_pages = []
        self.seq_kv_lens = []

    def free(self):
        glm = self.glm
        for ptr in self.k_data:
            glm.free_buf(ptr)
        for ptr in self.v_data:
            glm.free_buf(ptr)
        glm.free_buf(self.indices)
        glm.free_buf(self.indptr_d)
        glm.free_buf(self.last_page_len)
        glm.free_pinned(self.indptr_h)
        glm.free_pinned(self.last_page_len_h)
        glm.free_buf(self.slot_mapping)
        glm.free_pinned(self.slot_mapping_h)
        self.k_data = []
        self.v_data = []

    def reset(self, batch_size):
        assert batch_size <= self.max_batch, \
            f"batch_size {batch_size} exceeds max_batch {self.max_batch}"
        self.num_pages_used = 0
        self.seq_pages = [[] for _ in range(batch_size)]
        self.seq_kv_lens = [0] * batch_size

    def alloc_pages(self, seq_idx, num_tokens):
        page_size = self.page_size
        num_new_pages = (num_tokens + page_size - 1) // page_size
        start_page = self.num_pages_used
        self.num_pages_used += num_new_pages
        self.seq_pages[seq_idx].extend(range(start_page, start_page + num_new_pages))
        self.seq_kv_lens[seq_idx] += num_tokens
        return start_page, num_new_pages

    def alloc_prefill_pages(self, seq_idx, seq_len):
        page_size = self.page_size
        num_pages = (seq_len + page_size - 1) // page_size
        start_page = self.num_pages_used
        self.num_pages_used += num_pages
        self.seq_pages[seq_idx] = list(range(start_page, start_page + num_pages))
        self.seq_kv_lens[seq_idx] = seq_len
        return start_page, num_pages

    def alloc_append_pages(self, seq_idx, num_new_tokens):
        page_size = self.page_size
        current_len = self.seq_kv_lens[seq_idx]
        current_page_count = len(self.seq_pages[seq_idx])
        new_total_len = current_len + num_new_tokens
        new_page_count = (new_total_len + page_size - 1) // page_size
        num_new_pages = new_page_count - current_page_count
        start_page = self.num_pages_used
        for i in range(num_new_pages):
            self.seq_pages[seq_idx].append(start_page + i)
        self.num_pages_used += num_new_pages
        self.seq_kv_lens[seq_idx] = new_total_len
        return start_page, num_new_pages

    def alloc_decode_token(self, seq_idx):
        kv_len = self.seq_kv_lens[seq_idx]
        page_size = self.page_size
        page_idx_in_seq = kv_len // page_size
        if page_idx_in_seq >= len(self.seq_pages[seq_idx]):
            new_page = self.num_pages_used
            self.num_pages_used += 1
            self.seq_pages[seq_idx].append(new_page)
        self.seq_kv_lens[seq_idx] = kv_len + 1
        abs_page = self.seq_pages[seq_idx][page_idx_in_seq]
        slot_in_page = kv_len % page_size
        return abs_page, slot_in_page

    def update_indptr(self):
        batch_size = len(self.seq_pages)
        indptr = [0] * (batch_size + 1)
        for i in range(batch_size):
            indptr[i + 1] = indptr[i] + len(self.seq_pages[i])
        all_indices = []
        for i in range(batch_size):
            all_indices.extend(self.seq_pages[i])
        indices_np = np.array(all_indices, dtype=np.int32)
        indptr_np = np.array(indptr, dtype=np.int32)
        last_page_len_list = []
        for i in range(batch_size):
            kv_len = self.seq_kv_lens[i]
            remainder = kv_len % self.page_size
            last_page_len_list.append(remainder if remainder != 0 else self.page_size if kv_len > 0 else 0)
        last_page_len_np = np.array(last_page_len_list, dtype=np.int32)
        ctypes.memset(self.indptr_h, 0, (batch_size + 1) * I32)
        ctypes.memset(self.last_page_len_h, 0, batch_size * I32)
        ctypes.memmove(self.indptr_h, indptr_np.tobytes(), (batch_size + 1) * I32)
        ctypes.memmove(self.last_page_len_h, last_page_len_np.tobytes(), batch_size * I32)
        self.glm.h2d(self.indices, indices_np.tobytes())
        self.glm.h2d(self.indptr_d, indptr_np.tobytes())
        self.glm.h2d(self.last_page_len, last_page_len_np.tobytes())

    def update_slot_mapping(self, write_locations, page_size):
        batch_size = len(write_locations)
        slot_mapping_np = np.array([
            abs_page * page_size + slot_in_page
            for abs_page, slot_in_page in write_locations
        ], dtype=np.int32)
        ctypes.memmove(self.slot_mapping_h, slot_mapping_np.tobytes(), batch_size * I32)
        self.glm.h2d(self.slot_mapping, slot_mapping_np.tobytes())


class WorkspaceBuffers:
    def __init__(self, glm):
        self.glm = glm
        self.float_ws = glm.alloc(BATCH_FLOAT_WS_SIZE)
        self.int_ws = glm.alloc(BATCH_INT_WS_SIZE)
        self.pinned_int_ws = glm.alloc_pinned(BATCH_PINNED_INT_WS_SIZE)
        self.decode_plan_info = glm.alloc_pinned(DECODE_PLAN_INFO_SIZE * 8)
        self.prefill_plan_info = glm.alloc_pinned(PREFILL_PLAN_INFO_SIZE * 8)

    def free(self):
        glm = self.glm
        glm.free_buf(self.float_ws)
        glm.free_buf(self.int_ws)
        glm.free_pinned(self.pinned_int_ws)
        glm.free_pinned(self.decode_plan_info)
        glm.free_pinned(self.prefill_plan_info)
        self.float_ws = 0
        self.int_ws = 0
