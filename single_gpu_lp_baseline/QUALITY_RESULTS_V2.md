# Multi-k single-level quality study, 2026-09-08

The scope is Products and com-LiveJournal, each at k=2,4,8,16,32, with final
max vertex load <= floor(1.10*n/k). There is no graph contraction. Recursive
bisection was evaluated as a separate initialization experiment and is not the
default. Quality remains the priority; CPU reference stages are still costly.

## Best observed feasible results

These are the best of the tested configurations, not one uniform algorithm's
results. The provenance of every candidate is in `selection.json` and the
selected binary labels are linked in `best_observed.csv`.

| Graph | k | Cut edges | Cut ratio | Jet ratio | Extra cut vs Jet |
|---|---:|---:|---:|---:|---:|
| products | 2 | 979,266 | 0.01583061 | 0.01411689 | 12.14% |
| products | 4 | 2,149,674 | 0.03475119 | 0.03265325 | 6.42% |
| products | 8 | 3,805,836 | 0.06152436 | 0.04980862 | 23.52% |
| products | 16 | 4,711,696 | 0.07616830 | 0.07319443 | 4.06% |
| products | 32 | 6,513,774 | 0.10530032 | 0.09119717 | 15.46% |
| com-LiveJournal | 2 | 2,124,794 | 0.06126647 | 0.04895144 | 25.16% |
| com-LiveJournal | 4 | 3,838,926 | 0.11069188 | 0.10409017 | 6.34% |
| com-LiveJournal | 8 | 4,997,105 | 0.14408690 | 0.13985002 | 3.03% |
| com-LiveJournal | 16 | 6,637,306 | 0.19138058 | 0.16664031 | 14.85% |
| com-LiveJournal | 32 | 8,147,438 | 0.23492384 | 0.20503322 | 14.58% |

All ten were independently recomputed against the full original symmetric CSR,
including label count/range and the final upper-capacity bound. Ratios count each
undirected cut edge once and divide by the number of undirected edges.

Products k=4 improves only 0.20% vs the previous 2,154,026-edge result. There is
no new broad quality breakthrough here. Products k=16 improves from the prior
GPU policy's 0.08491982 to 0.07616830, but it now includes more search and CPU work.
Remaining gaps are strongly graph/k dependent; k=4 is not a sufficient benchmark.

## Implemented and evaluated

1. `GROUP_EXCHANGE=1`: propose a connected group even if its target is full.
   Temporarily apply its moves, choose reverse donor moves, evaluate the actual
   joint gain with incremental neighbor counts, and commit only a profitable,
   capacity-feasible transaction. Otherwise roll back every move. Donor pools
   are bounded heuristic candidates; the exchange is not an exact global optimum.
2. `GROUP_REDIRECT=1`: optional donor moves to any partition with available
   capacity, using a lazy gain queue updated when adjacent labels change.
   A Products k=8 plateau test improved 3,829,894 to 3,827,112 cut edges in
   34.70 additional CPU seconds, less than the benefit of GPU re-entry below.
3. `INITIAL_PARTITION`: load binary int32 labels for GPU search. This permits
   field exploration after CPU refinement while preserving the best feasible
   incumbent. Initial labels may be infeasible, but no infeasible final result
   is accepted. Products k=8 improved 3,829,894 to 3,805,836 on GPU re-entry.
4. Deeper GPU fields (64 rounds, 40 cycles), tested with the same policy at
   every k on both graphs. Helpful for some cases, worse for others.
5. `SEED_DISTANCE_POWER=2/3`: increase the role of seed separation relative to
   degree. Tested on both graphs at every k; no selected best came from this
   ablation. Default remains 1.
6. Recursive two-way LP initialization: exact intermediate balance performed
   poorly. Relaxed intermediate balance plus full-original-graph GPU repair
   also usually lost to direct k-way LP; its feasible Products k=32 output
   was slightly better. Intermediate labels with accumulated imbalance are
   not counted as valid final outputs.

The core group ablation used identical input labels for exchange off/on,
group size 128, 2 passes, patience 10000, seed 42. In Products the exchange
usually saved a few thousand extra cut edges. On LiveJournal k=4/8/16/32 it
slightly worsened the final result vs ordinary group search, despite accepting
only profitable individual transactions: search trajectories differ. Thus
exchange and redirect remain optional, not new defaults.

## Time and practical interpretation

CPU refinements in the multi-k ablations took approximately 28–158 seconds per
run. Exchange often increased runtime substantially for little extra gain. The
new best Products k=4 used an additional 53.27 CPU seconds after the previous
best pipeline, saving only 4,352 cut edges. This is not a viable speed/quality
tradeoff for a final system. Parallel CPU experiments and shared-node execution
make these diagnostic timings unsuitable for a formal speed comparison.

The next quality work should address robust initial layouts and coordinated
region movement across k, using explicit equal-compute-budget baselines.
Current evidence does not justify indefinitely increasing refinement rounds or
claiming that single-level LP has reached Jet quality on all cases. Before
architectural speed work, choose a bounded, repeatable search policy whose
quality gains persist across the full k grid; then move its profitable batch
operations to GPU and remove the sequential CPU reference from the runtime path.

## Validation and reproduction

- 60 random symmetric graph tests in each of three modes (ordinary groups,
  reciprocal exchange, multi-target redirect), including full-capacity cases.
- GPU external-initialization tests cover exact label passthrough, exact cut,
  invalid labels, and rejection when no feasible result exists.
- Default GPU output remains byte-identical to the previous baseline.
- Full-CSR independent validation of all ten selected results.

Artifacts: `../results/single_gpu_lp/quality_20260908_v2/`.
`summary.csv` is the controlled exchange ablation; `deep/summary.csv`,
`geometry/summary.csv`, and `resume/summary.csv` record the other policies.
`best_observed.csv`, `selection.json`, and `independent_validation.jsonl` provide
the final comparison and audit trail. Recursive initialization intermediates
remain in their separate experiment directories for inspection.

Build/test:

```bash
cmake --build build-gh200 -j4
python3 test_quality_refine.py
GROUP_EXCHANGE=1 python3 test_quality_refine.py
GROUP_EXCHANGE=1 GROUP_REDIRECT=1 python3 test_quality_refine.py
python3 test_gpu_initial.py
```

Run optional group exchange with the standard wrapper:

```bash
python3 run_quality.py products 8 --reference-refine --group-exchange \
  --output /tmp/lp-products-k8-new
```

Resume a known partition on GPU:

```bash
python3 run_quality.py products 8 --seeds 0 --initial-partition /absolute/input.parts \
  --field-rounds 32 --cycles 10 --verify --output /tmp/lp-products-k8-resume-new
```

Output directories must not exist. For exact study configurations, use the
experiment scripts and per-run logs; wrapper defaults use three 256-vertex
group passes and therefore are not identical to the two-pass 128-vertex ablation.
