# Structural-group, regional-cut, and soft-label experiments

This study implements new search operators, rather than only changing the
existing LP round counts. It has **not achieved a broad quality breakthrough**.
All reported final partitions obey the original upper load bound 1.10, without
a lower load constraint. No Jet or METIS partition is used as initialization.

## Implementations

- `community_seed.cpp`: size-constrained structural label propagation independent
  of k, followed by whole-group partition-label moves on the original CSR.
  No contracted CSR or uncoarsening hierarchy is built. This did not improve
  the existing Products k=8 best result, even after GPU refinement.
- `region_refine.cpp`: build a two-label boundary region, solve its binary
  labeling jointly by min-cut, repair capacity using exact marginal gains,
  and commit only a strictly profitable feasible transaction. Other labels
  and vertices outside the region are fixed. Every pass independently recomputes
  the complete cut, and checkpoints the partition.
- Optional `REGION_LAGRANGE_STEPS`: bias binary cuts by label mass, then search
  for a more capacity-compatible solution before the discrete repair step.
- Optional `REGION_RESIDUAL_BALANCE`: compute strongly connected components of
  the residual flow graph and greedily select admissible residual closures to
  improve capacity compliance while retaining a minimum cut of that flow
  problem. Residual closure is explicitly checked. This selection is heuristic;
  it does not solve arbitrary cardinality-constrained min-cut exactly.
- `quality_soft_labels.py`: GPU sparse diffusion, soft labels, annealed
  softmax, and dual projection of expected partition loads. Existing GPU
  refinement enforces the final discrete capacity. This is a candidate
  generator, not a monotonic improvement operator.

Regional min-cut is a different algorithmic ingredient from pure LP. The
quality gains below must not be described as a pure label-propagation result,
and they are not evidence of paper-level novelty. Original defaults remain intact.

## Final quality

| Graph | k | Previous ratio | New ratio | Jet ratio |
|---|---:|---:|---:|---:|
| products | 2 | 0.01583061 | 0.01562130 | 0.01411689 |
| products | 4 | 0.03475119 | 0.03419954 | 0.03265325 |
| products | 8 | 0.06152436 | 0.05766300 | 0.04980862 |
| products | 16 | 0.07616830 | 0.07554915 | 0.07319443 |
| products | 32 | 0.10530032 | 0.10358242 | 0.09119717 |
| com-LiveJournal | 2 | 0.06126647 | 0.06123962 | 0.04895144 |
| com-LiveJournal | 4 | 0.11069188 | 0.11030415 | 0.10409017 |
| com-LiveJournal | 8 | 0.14408690 | 0.14353421 | 0.13985002 |
| com-LiveJournal | 16 | 0.19138058 | 0.18852751 | 0.16664031 |
| com-LiveJournal | 32 | 0.23492384 | 0.23017870 | 0.20503322 |

Exact recomputed values, cut counts, provenance, and loads are in
`../results/single_gpu_lp/quality_community/best_observed.csv` and
`validated_results.json`; use those values for quantitative work.

The uniform regional ablation uses region size 65536, four passes, seed 42,
no Lagrange or residual balancing, and a 120-second soft budget. It starts
from each V2 selected partition. Products k=8 additionally used the exploratory
chain below; therefore its best row has a larger search budget than other rows.

## Products k=8 trajectory and cost

| Stage | Undirected cut | Ratio | Additional CPU seconds |
|---|---:|---:|---:|
| V2 incumbent | 3,805,836 | 0.06152436 | — |
| 4096-vertex regions, four passes | 3,719,661 | 0.0601313 | 14.7335 |
| 65536-vertex regions, eight passes | 3,621,903 | 0.0585509 | 56.8351 |
| Eight-step mass bias search | 3,581,057 | 0.0578906 | 205.336 |
| Residual minimum-cut balancing | 3,566,976 | 0.0576630 | 143.792 |

This is a 6.28% reduction relative to the V2 cut, with 15.77% more cut edges
than Jet remaining. The additional chain alone costs about 420.70 CPU seconds,
excluding the earlier incumbent search, graph I/O, and discarded trials.
Uniform four-pass refinements took approximately 18–64 additional CPU seconds.
These are shared-node diagnostic timings, not speedup claims or equal-time comparisons.

Budgets are checked between groups of 128 region attempts and can overrun
their nominal duration; actual elapsed times above are used instead of the
requested budgets. No region or transaction is interrupted midway by that check.
The `cpu_seconds` column in `best_observed.csv` is the last reference stage's
time; it is not end-to-end time. For the intensive k=8 result use the complete
420.70-second chain above, plus the prior incumbent's construction cost.

## Negative results retained

- Structural-group projection at block limit 256 gave ratio 0.070988; subsequent
  GPU refinement reached 0.0632772, worse than the incumbent 0.06152436.
- Soft-label diffusion with row-degree exponents 0, 0.5, 1 gave final ratios
  0.06162318, 0.06011360, 0.06043816 from the V2 incumbent. The diffusion itself
  took about 1.2 seconds, plus roughly 3.95 seconds of diagnostic GPU refinement.
  It did not beat the regional-search result.
- A longer, cooler soft-label run starting at 0.0578906 worsened to 0.06105068.
  This candidate is not promoted.
- Temporarily allowing 1.20 imbalance reduced the cut to 0.0551754, but this
  is **infeasible under the experiment's 1.10 constraint**. GPU repair returned
  a feasible 0.0592607, worse than the strict regional incumbent. The loose
  partition is excluded from all final tables.

## Validation and usage

Five planted-graph structural-group tests passed. Regional search passed 60
random graph cases in each of its ordinary, mass-bias, and residual-balance
modes, checking full cut, capacity and non-regression. GPU candidate refinement
used incremental-cut verification. Every final selected graph/k pair was
independently recomputed from the original symmetric CSR using NumPy.

Build and tests:

```bash
cmake --build build-gh200 -j4
python3 test_community_seed.py
python3 test_region_refine.py
REGION_LAGRANGE_STEPS=8 python3 test_region_refine.py
REGION_RESIDUAL_BALANCE=1 REGION_LAGRANGE_STEPS=8 python3 test_region_refine.py
```

Regional search positional arguments:

```text
region_refine indptr.bin indices.bin input.parts output.parts k region_size passes seed
```

Optional environment variables: `REGION_SECONDS` (soft time budget),
`REGION_LAGRANGE_STEPS` (default 0), `REGION_RESIDUAL_BALANCE` (default off),
`REGION_MAX_RATIO` (default 1.10; values above 1.10 produce exploration candidates,
not necessarily valid final partitions). Input and output labels are binary int32.

The implementation caps region size at 65536. These programs are research
references; they do not replace the GPU baseline or claim a deployable runtime.
