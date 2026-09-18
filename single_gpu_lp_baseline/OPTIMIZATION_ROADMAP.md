# Quality optimization roadmap

Keep the implementation order below when improving the single-GPU label
propagation partitioner:

1. Restore the best feasible state after an unsuccessful global cycle and
   diversify the next structural-field perturbation.
2. Replace target-only quotas with net-flow capacity scheduling and
   capacity-neutral swaps between partitions.
3. Restrict refinement to an active boundary and use gain buckets with
   incremental exact-gain updates.
4. Stay single-level: evaluate connected-group moves on the original graph,
   exact joint gains, bounded negative-gain exploration, and best-prefix rollback.
   Do not introduce contraction or an uncoarsening hierarchy.
5. After quality stabilizes, reduce host/device synchronization, specialize
   high-degree kernels for larger `k`, and evaluate CUDA Graph or persistent
   scheduling.

Quality milestones on Products with `k=4` and maximum vertex imbalance 1.10:

- Current: cut ratio 0.0445111.
- Near-term single-level target: below 0.040.
- Single-level quality target: approach Jet's current observed 0.032653253.

2026-09-08 update: the CPU reference plus GPU initialization reached
0.034821539 (Products k=4, upper ratio 1.10), independently verified.
Four-start GPU search reached 0.037196084, identical across three repeats.
These are higher-cost searches, not acceleration results. See
QUALITY_RESULTS_20260908.md. Cycle rollback alone did not consistently help;
keep it optional. Next priorities are reducing initialization sensitivity and
implementing bounded connected-group proposals/commit on GPU, with equal-budget
comparisons to the existing search. Large 4096-vertex proposals were too expensive
in the CPU reference; do not assume larger groups are automatically useful.

Multi-k follow-up (QUALITY_RESULTS_V2.md): Products k=4 reached 0.034751185,
but further local refinement contributed only 0.20% vs the prior best. Products
k=8 remains 23.52% above Jet in cut count, whereas k=16 is 4.06% above Jet.
Evaluate all k=2,4,8,16,32 on Products and LiveJournal for every quality policy.
Group exchange, redirect, external-label GPU re-entry, seed distance powers,
and recursive initialization are implemented as optional research experiments.
Do not enable all by default: several ablations regress quality or add large
cost for small gains. Next policy selection must use equal-compute comparisons
and focus on initial layout and coordinated region movement across k.

Regional follow-up (QUALITY_REGIONAL_RESULTS.md): implemented structural-group
LP candidates, joint binary regional cuts, mass-biased cuts, residual-closure
balancing, and continuous GPU soft-label candidates. Products k=8 reached
0.057662997, still 15.77% above Jet's cut count and requiring an additional
420.70 CPU seconds on that path. This does not meet the requested broad quality
breakthrough. Other k and LiveJournal were tested and independently validated.
The effective regional operator includes min-cut and must not be presented as
pure LP. Keep the reference separate from the default GPU runtime.
