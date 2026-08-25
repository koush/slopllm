"""Microbenchmark the production-shaped GLM-5.1 decode indexer pipeline.

This runs one synthetic context-parallel KV shard on one GPU. K packing and
rank-major merge candidates are built before timing; no model weights or peer
GPUs are required.

Examples:
  GLM_GPU=0 python bench_indexer_decode.py
  python bench_indexer_decode.py --batch 1,8 --kv-lens 32768,131072 --cp-world-size 8
  nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    -o ../../scratchpad/indexer_decode \
    python bench_indexer_decode.py --profile --stage all
"""

import argparse
import math
import os
import statistics
import time

import torch

from helpers import GlmOps, pack_indexer_k


N_HEADS = 32
HEAD_DIM = 128
GLOBAL_PAGE_SIZE = 64
TOPK_SCRATCH_I32 = 1056
ALL_STAGES = ("score", "merge", "sort", "paged", "flat")


def parse_matrix(value):
    values = []
    for item in value.split(","):
        if ":" not in item:
            values.append(int(item))
            continue
        parts = [int(part) for part in item.split(":")]
        if len(parts) != 3:
            raise argparse.ArgumentTypeError("range must be START:STOP:STEP")
        start, stop, step = parts
        if step <= 0 or stop < start:
            raise argparse.ArgumentTypeError(
                "range requires STOP >= START and STEP > 0"
            )
        values.extend(range(start, stop + 1, step))
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("matrix values must be positive")
    return values


def parse_stages(value):
    if value == "all":
        return ALL_STAGES
    stages = tuple(dict.fromkeys(value.split(",")))
    invalid = set(stages) - set(ALL_STAGES)
    if invalid:
        raise argparse.ArgumentTypeError(
            f"unknown stage(s) {','.join(sorted(invalid))}; choose from "
            f"all,{','.join(ALL_STAGES)}"
        )
    return stages


def padded_kv_len(kv_len):
    return max(1024, 1 << math.ceil(math.log2(kv_len)))


class IndexerDecodeCase:
    def __init__(self, device, batch, global_kv_len, world_size, topk):
        if GLOBAL_PAGE_SIZE % world_size != 0:
            raise ValueError("CP world size must divide the global page size (64)")

        self.batch = batch
        self.global_kv_len = global_kv_len
        self.world_size = world_size
        self.rank = 0
        self.topk = topk
        self.page_size = GLOBAL_PAGE_SIZE // world_size

        num_pages = (global_kv_len + GLOBAL_PAGE_SIZE - 1) // GLOBAL_PAGE_SIZE
        global_last = global_kv_len - (num_pages - 1) * GLOBAL_PAGE_SIZE
        local_last = (global_last + world_size - 1 - self.rank) // world_size
        self.local_kv_len = (num_pages - 1) * self.page_size + local_last
        # Production allocates capacity-sized decode scratch, which is never
        # narrower than the configured top-k even for a short live sequence.
        self.max_kv = max(topk, padded_kv_len(self.local_kv_len))
        self.num_splits = min(256, max(1, (self.max_kv + 255) // 256))

        torch.manual_seed(1234)
        self.q = torch.randn(
            batch, N_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device
        )
        self.weights = torch.rand(
            batch, N_HEADS, dtype=torch.bfloat16, device=device
        )

        # Packing is deliberately part of case setup, not any timed stage.
        total_pages = batch * num_pages
        unpacked_k = torch.randn(
            total_pages,
            self.page_size,
            HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        )
        self.k_data = pack_indexer_k(unpacked_k)
        del unpacked_k

        self.page_indices = torch.arange(
            total_pages, dtype=torch.int32, device=device
        )
        self.page_indptr = torch.arange(
            0, total_pages + 1, num_pages, dtype=torch.int32, device=device
        )
        self.last_page_len = torch.full(
            (batch,), local_last, dtype=torch.int32, device=device
        )
        self.global_last_page_len = torch.full(
            (batch,), global_last, dtype=torch.int32, device=device
        )
        self.qo_indptr = torch.arange(
            batch + 1, dtype=torch.int32, device=device
        )
        self.batch_indices = torch.arange(
            batch, dtype=torch.int32, device=device
        )
        self.kv_token_indptr = torch.arange(
            0,
            (batch + 1) * global_kv_len,
            global_kv_len,
            dtype=torch.int32,
            device=device,
        )

        self.local_idx = torch.empty(batch, topk, dtype=torch.int32, device=device)
        self.local_scores = torch.empty(
            batch, topk, dtype=torch.bfloat16, device=device
        )
        self.scores = torch.empty(
            batch, self.max_kv, dtype=torch.bfloat16, device=device
        )
        self.row_len = torch.empty(batch, dtype=torch.int32, device=device)
        self.hist = torch.empty(
            batch, TOPK_SCRATCH_I32, dtype=torch.int32, device=device
        )
        self.meta = torch.empty(batch, 4, dtype=torch.int32, device=device)

        # Model the post-AllGather buffers: each rank contributes one contiguous
        # topk-wide chunk, so candidates are rank-major within every row.
        candidate_dim = topk * world_size
        self.candidate_idx = torch.full(
            (batch, candidate_dim), -1, dtype=torch.int32, device=device
        )
        self.candidate_scores = torch.full(
            (batch, candidate_dim),
            float("-inf"),
            dtype=torch.bfloat16,
            device=device,
        )
        for rank in range(world_size):
            positions = torch.arange(
                rank, global_kv_len, world_size, dtype=torch.int32, device=device
            )[:topk]
            count = positions.numel()
            start = rank * topk
            self.candidate_idx[:, start:start + count] = positions
            self.candidate_scores[:, start:start + count] = torch.rand(
                batch, count, dtype=torch.bfloat16, device=device
            )

        self.merge_values = torch.empty(
            batch, topk, dtype=torch.bfloat16, device=device
        )
        self.merge_offsets = torch.empty(
            batch, topk, dtype=torch.int32, device=device
        )
        self.merge_idx = torch.empty(batch, topk, dtype=torch.int32, device=device)
        self.sort_idx = torch.empty_like(self.merge_idx)
        self.sort_values = torch.empty_like(self.merge_values)
        self.paged_slots = torch.empty_like(self.merge_idx)
        self.flat_slots = torch.empty_like(self.merge_idx)
        self.paged_length = torch.empty(batch, dtype=torch.int32, device=device)
        self.flat_length = torch.empty(batch, dtype=torch.int32, device=device)

    def score(self, glm):
        glm.indexer_score_topk_v2(
            self.local_idx,
            self.local_scores,
            self.q,
            self.k_data,
            self.weights,
            self.page_indices,
            self.page_indptr,
            self.last_page_len,
            self.qo_indptr,
            HEAD_DIM**-0.5,
            self.batch,
            N_HEADS,
            HEAD_DIM,
            self.page_size,
            self.topk,
            False,
            self.scores,
            self.row_len,
            self.hist,
            self.meta,
            self.max_kv,
            self.num_splits,
            cp_world_size=self.world_size if self.world_size > 1 else 0,
            cp_rank=self.rank,
            global_last_page_len=self.global_last_page_len,
        )

    def merge(self, glm):
        candidate_dim = self.topk * self.world_size
        glm.topk_from_scores(
            self.merge_offsets,
            self.merge_values,
            self.candidate_scores,
            None,
            self.hist,
            self.meta,
            self.batch,
            candidate_dim,
            self.topk,
            min(256, max(1, (candidate_dim + 255) // 256)),
        )
        glm.gather(
            self.merge_idx,
            self.candidate_idx,
            self.merge_offsets,
            self.topk,
            candidate_dim,
            self.batch,
            elem_size=4,
        )

    def sort(self, glm):
        glm.sort_topk_by_index(
            self.sort_idx, self.sort_values, self.batch, self.topk
        )

    def paged(self, glm):
        glm.topk_to_slots(
            self.paged_slots,
            self.merge_idx,
            self.page_indices,
            self.page_indptr,
            self.last_page_len,
            self.batch_indices,
            self.batch,
            self.topk,
            GLOBAL_PAGE_SIZE,
            cp_world_size=self.world_size if self.world_size > 1 else 0,
            cp_rank=self.rank,
            topk_length=self.paged_length,
        )

    def flat(self, glm):
        glm.topk_to_slots(
            self.flat_slots,
            self.merge_idx,
            self.page_indices,
            self.page_indptr,
            self.global_last_page_len,
            self.batch_indices,
            self.batch,
            self.topk,
            GLOBAL_PAGE_SIZE,
            cp_world_size=1,
            topk_length=self.flat_length,
            kv_token_indptr=self.kv_token_indptr,
        )

    def prepare(self, glm):
        self.score(glm)
        self.merge(glm)
        self.sort_idx.copy_(self.merge_idx)
        self.sort_values.copy_(self.merge_values)
        self.sort(glm)
        self.paged(glm)
        self.flat(glm)
        glm.synchronize()

    def validate(self):
        expected_local = min(self.topk, self.local_kv_len)
        if not torch.all(self.row_len == self.local_kv_len).item():
            raise RuntimeError("score stage returned an unexpected local row length")
        local = self.local_idx[self.local_idx >= 0]
        if local.numel() != self.batch * expected_local:
            raise RuntimeError("score stage returned an unexpected valid top-k count")
        if local.numel() and (
            torch.any(local >= self.global_kv_len).item()
            or torch.any(local.remainder(self.world_size) != self.rank).item()
        ):
            raise RuntimeError("score stage returned an invalid CP-remapped index")

        gathered = torch.gather(
            self.candidate_idx, 1, self.merge_offsets.to(torch.int64)
        )
        if not torch.equal(gathered, self.merge_idx):
            raise RuntimeError("candidate gather does not match merge offsets")
        valid_count = min(self.topk, self.global_kv_len)
        expected_paged_lengths = (
            (self.merge_idx >= 0)
            & (self.merge_idx.remainder(self.world_size) == self.rank)
        ).sum(dim=1, dtype=torch.int32)
        if not torch.equal(self.paged_length, expected_paged_lengths):
            raise RuntimeError("paged slot conversion returned an unexpected length")
        expected_lengths = torch.full_like(self.flat_length, valid_count)
        if not torch.equal(self.flat_length, expected_lengths):
            raise RuntimeError("flat slot conversion returned an unexpected length")
        sorted_valid = self.sort_idx[:, :valid_count]
        if valid_count > 1 and torch.any(sorted_valid[:, 1:] < sorted_valid[:, :-1]).item():
            raise RuntimeError("sort stage did not order valid indices")
        flat_expected = self.merge_idx + self.kv_token_indptr[:-1, None]
        if not torch.equal(self.flat_slots[:, :valid_count], flat_expected[:, :valid_count]):
            raise RuntimeError("flat slot conversion does not match token offsets")


def time_stage(glm, launch, warmup, iterations, repeats):
    for _ in range(warmup):
        launch(glm)
    glm.synchronize()

    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(iterations):
            launch(glm)
        glm.synchronize()
        samples.append((time.perf_counter() - start) / iterations)
    return statistics.median(samples)


def profile_stages(glm, case, stages, warmup, iterations):
    for stage in stages:
        launch = getattr(case, stage)
        for _ in range(warmup):
            launch(glm)
    glm.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(iterations):
        for stage in stages:
            getattr(case, stage)(glm)
    glm.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch", "--batch-sizes", dest="batch", type=parse_matrix,
        default=parse_matrix("1,8,32"),
    )
    parser.add_argument(
        "--kv-lens", type=parse_matrix, default=parse_matrix("8192,32768,131072"),
        help="global KV lengths",
    )
    parser.add_argument(
        "--cp-world-size", "--cp-world-sizes", dest="cp_world_size",
        type=parse_matrix, default=parse_matrix("8"),
    )
    parser.add_argument("--topk", type=parse_matrix, default=parse_matrix("2048"))
    parser.add_argument("--stage", type=parse_stages, default=ALL_STAGES)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--gpu", type=int, default=int(os.environ.get("GLM_GPU", "0")))
    parser.add_argument(
        "--profile", action="store_true",
        help="capture the largest matrix case with the CUDA profiler API",
    )
    parser.add_argument("--profile-iterations", type=int, default=1)
    args = parser.parse_args()

    if min(args.warmup, args.iterations, args.repeats) < 1:
        parser.error("warmup, iterations, and repeats must be positive")
    if args.profile_iterations < 1:
        parser.error("profile iterations must be positive")

    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    glm = GlmOps(device_id=args.gpu)
    stages = args.stage

    print(
        f"Indexer decode | GPU {args.gpu} | stages={','.join(stages)} "
        f"| warmup={args.warmup} | iterations={args.iterations} "
        f"| repeats={args.repeats}"
    )
    print(
        "  batch  global_kv  local_kv  world  topk"
        + "".join(f"  {stage:>10}" for stage in stages)
    )

    matrix = [
        (batch, kv_len, world_size, topk)
        for batch in args.batch
        for kv_len in args.kv_lens
        for world_size in args.cp_world_size
        for topk in args.topk
    ]
    if args.profile:
        matrix = [max(matrix, key=lambda case: math.prod(case))]

    validated = False
    for batch, kv_len, world_size, topk in matrix:
        case = IndexerDecodeCase(device, batch, kv_len, world_size, topk)
        case.prepare(glm)
        if not validated:
            case.validate()
            validated = True

        prefix = (
            f"  {batch:5d}  {kv_len:9d}  {case.local_kv_len:8d}"
            f"  {world_size:5d}  {topk:4d}"
        )
        if args.profile:
            profile_stages(
                glm, case, stages, args.warmup, args.profile_iterations
            )
            print(prefix + f"  captured {args.profile_iterations} iteration(s)")
        else:
            timings = []
            for stage in stages:
                elapsed = time_stage(
                    glm,
                    getattr(case, stage),
                    args.warmup,
                    args.iterations,
                    args.repeats,
                )
                timings.append(f"  {elapsed * 1e3:8.3f} ms")
            print(prefix + "".join(timings))

        del case
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
