import ctypes

BF16 = 2  # bytes per bfloat16


class FlatKVCache:
    def __init__(self, glm, n_kv, hd, n_layers, max_batch, max_seq_len):
        self.glm = glm
        self.n_kv = n_kv
        self.hd = hd
        self.n_layers = n_layers
        self.max_batch = max_batch
        self.max_seq_len = max_seq_len
        self.cache_pos = 0
        self.k_data = [glm.alloc(max_batch * n_kv * max_seq_len * hd * BF16) for _ in range(n_layers)]
        self.v_data = [glm.alloc(max_batch * n_kv * max_seq_len * hd * BF16) for _ in range(n_layers)]

    def free(self):
        glm = self.glm
        for ptr in self.k_data:
            glm.free_buf(ptr)
        for ptr in self.v_data:
            glm.free_buf(ptr)
        self.k_data = []
        self.v_data = []

    def reset(self):
        self.cache_pos = 0
