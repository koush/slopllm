"""Microbenchmark the production GLM-5.1 prefill indexer pipeline.

This exercises one GPU's query-sharded work with flat replicated K, matching
the 8-GPU CP prefill path without loading model weights.

Examples:
  GLM_GPU=0 python bench_indexer_prefill.py
  GLM_INDEXER_PREFILL_CONFIG=q64_k192_w8_q2 python bench_indexer_prefill.py --kv-lens 131072
  nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    -o ../../scratchpad/indexer \
    python bench_indexer_prefill.py --profile --kv-lens 131072
"""

import argparse
import math
import os
import time

import torch

from helpers import GlmOps, pack_indexer_k


N_HEADS = 32
HEAD_DIM = 128
TOPK = 2048
GLOBAL_PAGE_SIZE = 64


def parse_lengths(value):
    if ":" not in value:
        return [int(part) for part in value.split(",")]
    parts = [int(part) for part in value.split(":")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("range must be START:STOP:STEP")
    start, stop, step = parts
    if step <= 0 or stop < start:
        raise argparse.ArgumentTypeError("range requires STOP >= START and STEP > 0")
    return list(range(start, stop + 1, step))


def padded_kv_len(kv_len):
    return max(1024, 1 << math.ceil(math.log2(kv_len)))


class IndexerCase:
    def __init__(self, device, kv_len, chunk_size, world_size, rank, topk):
        if chunk_size % world_size != 0:
            raise ValueError("chunk size must be divisible by world size")
        if not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world size)")
        if kv_len < chunk_size:
            raise ValueError("KV length must include at least the current chunk")

        self.kv_len = kv_len
        self.max_kv = padded_kv_len(kv_len)
        self.chunk_size = chunk_size
        self.local_q = chunk_size // world_size
        self.rank = rank
        self.q_global_start = rank * self.local_q
        self.topk = topk
        # A gathered CP shard retains the effective per-GPU page dimension.
        self.page_size = GLOBAL_PAGE_SIZE // world_size

        torch.manual_seed(1234)
        self.q = torch.randn(
            self.local_q, N_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device
        )
        self.k_data = pack_indexer_k(torch.randn(
            self.max_kv // self.page_size,
            self.page_size,
            HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        ))
        self.weights = torch.rand(
            self.local_q, N_HEADS, dtype=torch.bfloat16, device=device
        )

        num_pages = (kv_len + GLOBAL_PAGE_SIZE - 1) // GLOBAL_PAGE_SIZE
        last_page_len = kv_len - (num_pages - 1) * GLOBAL_PAGE_SIZE
        self.page_indices = torch.arange(num_pages, dtype=torch.int32, device=device)
        self.page_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=device)
        self.last_page_len = torch.tensor([last_page_len], dtype=torch.int32, device=device)
        # Production keeps the global query range while each rank passes only
        # its local Q slice and identifies that slice through q_global_start.
        self.qo_indptr = torch.tensor([0, chunk_size], dtype=torch.int32, device=device)
        self.kv_token_indptr = torch.tensor([0, kv_len], dtype=torch.int32, device=device)
        self.global_last_page_len = self.last_page_len.clone()

        self.out_idx = torch.empty(self.local_q, topk, dtype=torch.int32, device=device)
        self.out_scores = torch.empty(
            self.local_q, topk, dtype=torch.bfloat16, device=device
        )
        self.scores = torch.empty(
            self.local_q, self.max_kv, dtype=torch.bfloat16, device=device
        )
        self.row_len = torch.empty(self.local_q, dtype=torch.int32, device=device)
        self.coarse_hist = torch.empty(
            self.local_q, 1024, dtype=torch.int32, device=device
        )
        self.fine_hist = torch.empty(
            self.local_q, 64, dtype=torch.int32, device=device
        )
        self.meta = torch.empty(self.local_q, 4, dtype=torch.int32, device=device)

    def launch(self, glm):
        glm.indexer_score_topk_prefill(
            self.out_idx,
            self.out_scores,
            self.q,
            self.k_data,
            self.weights,
            self.page_indices,
            self.page_indptr,
            self.last_page_len,
            self.qo_indptr,
            HEAD_DIM**-0.5,
            self.local_q,
            N_HEADS,
            HEAD_DIM,
            self.page_size,
            self.topk,
            True,
            self.scores,
            self.row_len,
            self.max_kv,
            self.coarse_hist,
            self.fine_hist,
            self.meta,
            1,
            q_global_start=self.q_global_start,
            global_last_page_len=self.global_last_page_len,
            kv_token_indptr=self.kv_token_indptr,
        )

    def validate(self):
        prefix = self.kv_len - self.chunk_size
        expected_first = prefix + self.q_global_start + 1
        expected_last = expected_first + self.local_q - 1
        actual = self.row_len[[0, -1]].cpu().tolist()
        if actual != [expected_first, expected_last]:
            raise RuntimeError(
                f"unexpected row lengths {actual}, expected {[expected_first, expected_last]}"
            )

    @property
    def scored_pairs(self):
        prefix = self.kv_len - self.chunk_size + self.q_global_start + 1
        return self.local_q * (2 * prefix + self.local_q - 1) // 2


def run_timed(glm, case, warmup, iterations):
    for _ in range(warmup):
        case.launch(glm)
    glm.synchronize()
    case.validate()

    start = time.perf_counter()
    for _ in range(iterations):
        case.launch(glm)
    glm.synchronize()
    return (time.perf_counter() - start) / iterations


def run_profile(glm, case, warmup, iterations):
    for _ in range(warmup):
        case.launch(glm)
    glm.synchronize()
    case.validate()

    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(iterations):
        case.launch(glm)
    glm.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kv-lens", type=parse_lengths, default=parse_lengths("8192:131072:8192"))
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--rank", type=int, default=7, help="query-shard rank; last rank is the critical causal shard")
    parser.add_argument("--topk", type=int, default=TOPK)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--profile", action="store_true", help="capture the largest KV case with CUDA profiler API")
    parser.add_argument("--profile-iterations", type=int, default=1)
    args = parser.parse_args()

    gpu = int(os.environ.get("GLM_GPU", "0"))
    torch.cuda.set_device(gpu)
    device = torch.device(f"cuda:{gpu}")
    glm = GlmOps(device_id=gpu)
    config = os.environ.get("GLM_INDEXER_PREFILL_CONFIG", "auto")

    print(
        f"Indexer prefill | GPU {gpu} | config={config} | chunk={args.chunk_size} "
        f"| world={args.world_size} | rank={args.rank} | topk={args.topk}"
    )
    print("  kv_len   max_kv    latency   chunk tok/s   scored Gpair/s")

    lengths = [max(args.kv_lens)] if args.profile else args.kv_lens
    total_elapsed = 0.0
    for kv_len in lengths:
        case = IndexerCase(
            device, kv_len, args.chunk_size, args.world_size, args.rank, args.topk
        )
        if args.profile:
            run_profile(glm, case, args.warmup, args.profile_iterations)
            print(f"  {kv_len:6d}  {case.max_kv:7d}    captured {args.profile_iterations} iteration(s)")
        else:
            elapsed = run_timed(glm, case, args.warmup, args.iterations)
            total_elapsed += elapsed
            chunk_rate = args.chunk_size / elapsed
            pair_rate = case.scored_pairs / elapsed / 1e9
            print(
                f"  {kv_len:6d}  {case.max_kv:7d}  {elapsed * 1e3:8.3f} ms"
                f"  {chunk_rate:11.0f}  {pair_rate:15.3f}"
            )
        del case
        torch.cuda.empty_cache()

    if not args.profile and len(lengths) > 1:
        total_tokens = len(lengths) * args.chunk_size
        print(
            f"  aggregate: {total_elapsed * 1e3:.3f} ms/layer, "
            f"{total_tokens / total_elapsed:.0f} chunk tok/s"
        )


if __name__ == "__main__":
    main()
