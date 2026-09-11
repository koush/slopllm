// FlashInfer's fallback wave-filling policy, evaluated over useful local work
// rather than the allocated slot capacity. The kernel still receives its full
// numSplits allocation and device-side topkLength; this only changes grouping.
export function sparseMlaChunksPerBlock(numQueries: number, numHeads: number, localTopkBound: number, smCount: number): number {
  const chunks = Math.max(1, Math.ceil(localTopkBound / 64));
  const headGroups = Math.ceil(numHeads / 16);
  let best = 1;
  let bestGap = Infinity;
  for (let cpb = 1; cpb <= chunks; cpb++) {
    const active = numQueries * headGroups * Math.ceil(chunks / cpb);
    const waves = Math.ceil(active / smCount);
    if (waves > 3) continue;
    // Integer idle-SM count avoids floating-point tie-breaking differences.
    const gap = waves * smCount - active;
    if (gap <= bestGap) {
      bestGap = gap;
      best = cpb;
    }
  }
  return best;
}
