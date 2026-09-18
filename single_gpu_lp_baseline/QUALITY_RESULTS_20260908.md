# Single-level quality experiments, 2026-09-08

All runs use the original symmetric CSR, k-way labels, and only an upper vertex
load bound of floor(1.10*n/k). No coarse graph or hierarchy is constructed. Jet
labels are never used as input. Edge cuts below count undirected edges once.

## Implemented

- `SEARCH_SEED`: reproducible randomized degree/distance seed scores on GPU.
  Zero preserves the original seed rule and default baseline behavior.
- `RESTORE_BEST_CYCLE`: optional restoration of best feasible labels before
  the next structural-field cycle. Off by default; ablations did not show
  consistent gains.
- `quality_refine.cpp`: CPU reference for exact incremental single-vertex gains,
  capacity-neutral reciprocal swaps (including the adjacency correction),
  bounded negative-gain exploration, best feasible prefix rollback, and connected
  group moves. Groups are temporary search candidates in the original CSR.
  Each pass verifies incremental cut against a full graph scan.
- `run_quality.py`: fixed seeds 0,1,2,3, 32 field rounds and 10 cycles; selects
  the best feasible candidate and charges the sum of all four GPU searches.
  Optional `--reference-refine` runs the CPU reference separately.

## Products k=4 quality ladder

| Configuration | Undirected cut | Ratio | Timing |
|---|---:|---:|---|
| Original GPU baseline | 2,753,414 | 0.044511122 | median 0.290161 s |
| Baseline + CPU swaps/prefix search | 2,704,552 | 0.0437212 | CPU refinement 4.117 s |
| Baseline + CPU groups of up to 128 | 2,658,541 | 0.0429774 | CPU refinement 147.612 s |
| Fixed four-start GPU policy | 2,300,913 | 0.037196084 | median total GPU search 4.350720 s |
| Four-start winner + groups of up to 256 | 2,217,297 | 0.035844365 | additional CPU refinement 46.0552 s |
| Tuned seed 2, field=64, cycles=40 | 2,247,850 | 0.0363383 | GPU search 7.21983 s in pipeline repeat |
| Tuned deeper search + groups of up to 256 | 2,154,026 | 0.034821539 | additional CPU refinement 43.5481 s in pipeline repeat |
| Jet historical comparison | 2,019,898 | 0.032653253 | historical 0.606586 s |

Best cut reduction vs original baseline: 21.77%. The best result still has
6.64% more cut edges than Jet. The mixed GPU+CPU path takes roughly 57 seconds
of algorithm time for the chosen configuration in the initial experiment (the
complete pipeline repeat took 50.76793 s algorithm time and 51.81735 s including
I/O), excluding the cost of finding
that configuration. This is a quality prototype, not a speed improvement.
Timing samples were not a controlled exclusive-node benchmark.

## Fixed four-start policy across datasets and k

| Dataset | k | Baseline ratio | Four-start ratio | Baseline s | Four-start total GPU s |
|---|---:|---:|---:|---:|---:|
| products | 2 | 0.0316679 | 0.016443942 | 0.304430 | 6.501790 |
| products | 4 | 0.0445111 | 0.037196084 | 0.290161 | 4.350720 |
| products | 8 | 0.0756378 | 0.063838071 | 0.630379 | 13.238999 |
| products | 16 | 0.0997043 | 0.084919817 | 0.804650 | 16.112410 |
| products | 32 | 0.121246 | 0.108556648 | 1.623400 | 35.921930 |
| com-LiveJournal | 4 | 0.130809 | 0.116361466 | 0.292599 | 2.878805 |

Products k=4 was repeated three times with identical final cut for each method;
its times above are medians. Other rows are single runs. These results compare
different compute budgets, and the policy was selected using Products. They do
not establish equal-budget superiority or generalization to all graph families.

## Validation and artifacts

- 60 small random symmetric graph runs checked exact cuts, capacity, and
  non-regression, including full-capacity swaps and groups with available slack.
- GPU ablations used `INCREMENTAL_CUT_VERIFY=1`; reported benchmark timings
  disabled that expensive diagnostic.
- Independent NumPy full-CSR recomputation confirmed baseline, the fixed GPU
  winner, and both improved reference partitions. Best partition loads are
  [673482, 593554, 549576, 632417], max/mean=1.0999984075.
- Original baseline output was reproduced after the CUDA source changes.
- A group_size=4096/loss_limit=4096 trial was stopped after more than 180 s
  without finishing a pass. It produced no final result and is not included.

Artifacts are under `../results/single_gpu_lp/quality_20260908/`:
`benchmark/summary.csv`, `init_sweep/summary.csv`, `seed_sweep/summary.csv`,
`deep_sweep/summary.csv`, `independent_validation.jsonl`, `final_validation.jsonl`.
Best partition: `deep_group.parts`. It is binary int32 with one label per vertex.
The complete `run_quality.py` reproduction is in `best_pipeline/`; its output
`best.parts` is byte-identical to `deep_group.parts`. `best_pipeline/metrics.json`
records all final pipeline times.
GPU-only reproducible candidate: `benchmark/products_k4_r0/best.parts`.

## Reproduce

Build GPU and reference executables using the existing GH200 build directory:

```bash
cmake --build build-gh200 -j4
python3 test_quality_refine.py
python3 run_quality.py products 4 --output /tmp/lp-products-quality-new
```

For the best quality configuration (Products-tuned):

```bash
python3 run_quality.py products 4 --seeds 2 --field-rounds 64 --cycles 40 \
  --reference-refine --output /tmp/lp-products-deep-new
```

Output directories must not exist. For independent verification, use a Python
environment with NumPy and run `verify_quality.py CSR_DIRECTORY K PARTS...`.
The reference requires a symmetric, loop-free graph and binary int32 labels.

Next research work: amortize initialization through better structured seeding;
make small connected-group proposals parallel; select conflict-safe profitable
group batches under net-flow capacity constraints; compare against equal-time
multistart LP. Mere seed tuning or a CPU FM implementation is not a novelty claim.
