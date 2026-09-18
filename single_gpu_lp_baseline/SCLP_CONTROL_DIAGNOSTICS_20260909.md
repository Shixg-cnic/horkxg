# SCLP quality-control diagnostics (2026-09-09)

## Scope

This iteration changed no affinity or admission formula. It added a configurable
two-hop trigger, unrestricted-vs-role gain diagnostics, late-level progress
counters, cluster-weight percentiles, and non-overlapping hierarchy timing.
Large outputs remain under `single_gpu_lp_baseline`; source changes remain in
the `homework` repository.

All quality experiments use `k=4`, maximum load ratio `1.10`, four SCLP rounds,
and the unchanged Jet initialization/projection/refinement half. Cuts below are
undirected cuts reported on the original graph. Positive relative differences
mean SCLP has more cut edges than Jet.

## A. Two-hop trigger ablation (`beta=64`, `seed=0`)

| graph | threshold 0.50 | threshold 0.60 | threshold 0.70 | off |
|---|---:|---:|---:|---:|
| products | 2,179,926 | **2,126,129** | 2,129,986 | 2,245,583 |
| LiveJournal | 3,544,983 | 3,540,538 | 3,689,202 | **3,522,126** |

Natural LP already contracts the first products layer to `0.529n` and the first
LiveJournal layer to `0.571n`. Raising the trigger to `0.60` therefore skips the
first-layer forced pairing in both cases. It improves products by 2.47% and
LiveJournal by 0.13% relative to 0.50. Completely disabling fallback helps
LiveJournal by 0.65%, but hurts products by 3.01%. Threshold 0.70 is particularly
bad for LiveJournal. This is evidence that fallback should be conditional, but
not evidence for a graph-specific selector. `0.60` was fixed before the beta and
multi-seed experiments.

## B. Role and late-level diagnostics

On the first products layer (`beta=64`, threshold `0.60`):

| metric | value |
|---|---:|
| unrestricted positive-gain vertices | 2,400,608 |
| vertices losing gain to role restriction | 1,246,910 |
| summed blocked raw gain | 1,246,910 |
| natural LP contraction ratio | 0.528916 |
| singleton with favorite / pairable singleton | 706,031 / 488,678 |
| weight p50 / p90 / p99 / max | 1 / 3 / 9 / 538 |
| capacity `U`; max / `U` | 9,567; 0.0562 |

Thus first-layer quality is not limited by capacity. Mover/receiver isolation
removes many stronger choices, although this diagnostic is an upper bound: it
does not claim those unrestricted moves could all be synchronously committed.

The final attempted levels tell a different story:

| graph/config | attempted ratio | positive gain | role blocked | capacity reject ratio | p99 / U |
|---|---:|---:|---:|---:|---:|
| products, beta 256 | 0.9099 | 11,253 | 8,577 | 90.58% | 2,006 / 2,392 |
| LiveJournal, beta 256 | 0.9125 | 1,763 | 1,334 | 99.23% | 3,905 / 3,905 |
| products, beta 64 | 0.9185 | 8,582 | 5,955 | 87.37% | 323 / 9,567 |
| LiveJournal, beta 64 | 0.9248 | 450 | 328 | 99.33% | 15,587 / 15,623 |

There are still positive-gain vertices. Products is constrained by both role
separation and admission; LiveJournal's deep levels are overwhelmingly
capacity-saturated. The observed stop near 60k products vertices is therefore
not simply “no useful merge exists”. Aggressive two-hop is not the justified
fix because it ignores direct adjacency and did not give stable quality gains.

## C. Beta ablation (`threshold=0.60`, `seed=0`)

| beta | products cut | LiveJournal cut |
|---:|---:|---:|
| 64 | 2,126,129 | 3,540,538 |
| 128 | 2,230,579 | **3,469,154** |
| 160 | 2,278,128 | 3,530,417 |
| 256 | **2,075,266** | 3,541,543 |

No beta wins both graphs at seed 0. Beta 256 was selected before multi-seed
validation because it improved products by 2.39% over beta 64 while changing
LiveJournal by only +0.03%. This selection is not installed as a new default.

## Five-seed paired comparison (`beta=256`, threshold=0.60`)

| graph | Jet mean ± sample std | SCLP mean ± sample std | mean cut ratios (Jet / SCLP) | SCLP vs Jet means | mean paired delta ± sample std |
|---|---:|---:|---:|---:|---:|
| products | 2,098,514 ± 83,343 | 2,122,398 ± 66,099 | 3.3924% / 3.4310% | +1.14% | +1.35% ± 6.98% |
| LiveJournal | 3,504,674 ± 96,592 | 3,517,338 ± 23,895 | 10.1054% / 10.1419% | +0.36% | +0.42% ± 2.60% |

SCLP wins some paired seeds and loses others; seed 0 substantially overstates
its products advantage. These results do not establish a stable quality win.
The mean SCLP aggregate-plus-contraction times were 2.58 s (products) and
2.49 s (LiveJournal). Mean complete coarsening-process times were 31.76 s and
30.23 s respectively; LiveJournal seed 3 was an I/O/system outlier at 42.72 s.
The latter times include graph reading, snapshots, hierarchy export, and checks,
and are not presented as pure GPU kernel time.

All ten SCLP final partition files were independently scanned against the full
original symmetric CSR. Recomputed cuts exactly equal the table inputs, every
label is in `[0,4)`, and every run satisfies Jet's integer capacity convention.
The maximum observed load ratios are `1.10000004` for products and `1.10000045`
for LiveJournal; the tiny excess over decimal 1.10 is solely the documented
integer `floor(1.10 * ceil(n/k))` convention.

## Decision

Keep SCLP as the active experimental coarsener, but do not claim it beats Jet or
change the default beta based on seed 0. The next evidence-driven mechanism, if
continued, should address late capacity saturation and the quality cost of
mover/receiver scheduling while preserving deterministic synchronous semantics.
Do not add stronger two-hop, degree penalties, confidence, or hot/cold rules at
this point.

Machine-readable rows are in `SCLP_CONTROL_RESULTS_20260909.csv`. Raw logs and
hierarchies are under:

- `build-gh200/experiments/results/sclp_twohop_ablation`
- `build-gh200/experiments/results/sclp_beta_ablation`
- `build-gh200/experiments/results/sclp_multiseed_b256_th060`
- `build-gh200/experiments/results/sclp_diagnostics_probe`
