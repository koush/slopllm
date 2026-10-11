#pragma once

// Decode-only route metadata. No activation copies and no host readback.
constexpr int MOE_HYBRID_MAX_ROUTES = 512;
constexpr int MOE_HYBRID_MAX_EXPERTS = 256;
constexpr int MOE_HYBRID_TILE_ROWS = 16;
constexpr int MOE_HYBRID_MIN_MMA_ROWS = 2;

// The FP32 partial-output array immediately follows this metadata.
struct alignas(128) MoeHybridPlan {
    int cuda_count;
    int mma_count;
    int offsets[MOE_HYBRID_MAX_EXPERTS + 1];
    int routes[MOE_HYBRID_MAX_ROUTES];
    // (expert, tile index). Eligible experts stay entirely on MMA; divide
    // their rows evenly across ceil(rows/16) tiles, each at least two rows.
    int2 mma_tasks[MOE_HYBRID_MAX_ROUTES];
    int min_rows;
    // Compact sparse-expert routes for the wide-output, short-K down worker.
    int cuda_routes[MOE_HYBRID_MAX_ROUTES];
};
