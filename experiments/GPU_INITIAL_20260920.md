# GPU initialization: speed-first, single implementation

Subsequent coarsening-only engineering results are recorded in
[COARSEN_RECORD_RESERVE_20260920.md](COARSEN_RECORD_RESERVE_20260920.md).
The timings below describe the earlier baseline, not the latest performance.

Products only, k=4, seed=0, imbalance=1.10, verify=false, GH200.
No changes to SCLP decisions, plain LP, Pair Escape, or uncoarsening rules.

## Retained changes

- `contract()` moves the already computed aggregate vertex weights into the
  coarse graph instead of allocating and copying another identical array.
- `initial_partition_gpu()` performs deterministic, weighted spectral recursive
  bisection on the device CSR. It uses fixed 64-step lazy diffusion per split,
  weighted prefix balance checks, and vertex-ID tie breaking. No host CSR copy,
  METIS call, random/BFS alternative, or additional refinement pass is hidden
  inside this backend. Only small control/reduction results return to the host.
- Select it with the trailing `--gpu-initial` flag on `gpart_partition_standard`
  or `gpart_partition_big`. METIS remains the default/reference backend.
- Standard/Big share one CUDA implementation. The input device CSR must already
  be validated. Infeasible ordering/capacity and weight overflow are reported,
  not silently repaired with another initializer.

## Results

The following are **independent per-column medians**, not a synthetic run.
All times include stage allocation/destruction. Algorithm time excludes input
load/preparation and final output; full wall time includes them.

| Version | Coarsen s | Initial s | Uncoarsen s | Algorithm s | Full wall s | Final cut |
|---|---:|---:|---:|---:|---:|---:|
| Saved pre-turn baseline, rerun | 0.822909376 | 0.110149888 | 0.237054880 | 1.170437664 | 1.812236480 | 2405065 |
| Weight ownership transfer + METIS | 0.817809024 | 0.110019648 | 0.236846560 | 1.164519328 | 1.782787680 | 2405065 |
| Weight ownership transfer + GPU initial | 0.827297632 | 0.011522592 | 0.241558624 | 1.080354016 | 1.702006400 | 2484676 |

GPU initialization is about 9.56x faster than METIS here. Algorithm time falls
about 7.7% versus the saved baseline; final cut rises about 3.31%. This is a
measured tradeoff, not a quality improvement or a cross-graph conclusion.
The small coarsening difference is not evidence of a major acceleration.

GPU final maximum part weight: 673392; imbalance: 1.099851410.
All three GPU outputs are byte-identical:
`b820c9b7c19a5fa6a58a62644ecf67eaafa99fdbd6de5c6ef138e9d3e594deaf`.
The METIS path retains cut 2405065, maximum weight 668666, pair accepted 1283,
and SHA `d114b7a4901c1be3d43b90fcc533b0a675b27d396c9c79dd5d12ebd5a182b162`.

Standard/Big uncoarsening regressions passed. GPU initializer tests passed for
both types: repeatability, weighted disconnected communities, isolated vertices,
non-power-of-two part count, infeasible heavy-vertex rejection, and Big weights
above 32 bits. The GPU tests were rerun after removing the extra LP variant.
Each measured run was guarded by GPU-process/utilization checks. No foreign
compute process was observed. Idle memory was about 2.9 GiB, not the historical
299 MiB; traces retain that distinction.

## Complete run rows

Columns: run, coarsen, initial, uncoarsen, algorithm, full wall (seconds).

```text
baseline
1 0.820925760 0.109471488 0.236652416 1.167050880 1.812236480
2 0.866702016 0.110149888 0.241330560 1.218184064 1.861998944
3 0.822909376 0.110472128 0.237054880 1.170437664 1.794645888

weight move + METIS
1 0.838807584 0.110109088 0.236688896 1.185606688 1.800899616
2 0.817809024 0.109862656 0.236846560 1.164519328 1.782787680
3 0.816113376 0.110019648 0.237118112 1.163252192 1.772007968

weight move + GPU initial
1 0.833489920 0.011569152 0.241796032 1.086856864 1.699294336
2 0.827297632 0.011495904 0.241558624 1.080354016 1.711676064
3 0.822992352 0.011522592 0.241515040 1.076031456 1.702006400
```

Raw logs, GPU traces, partitions and JSON are in `/tmp/gpart-ab-suY2pw/`:
`final-0920-baseline-*`, `move-0920-*`, and `simple-0920-gpu-*`.

## Rejected and removed experiments

Async allocation, retaining the async pool during coarsen, managed/prefetched
scratch, ATS system scratch and explicit huge-page scratch did not improve
overall performance. Their respective coarsening medians were approximately
0.855, 0.833, 0.838, 1.272 and 1.530 s. None remains in the source.

Adding an extra four-round plain LP after GPU initialization produced cut
2488774 and initial time about 0.0146 s: worse cut and slower initialization
than the simpler version. That pass and its additional public wrapper were
removed. `refine.cu` and `refine.hpp` are unchanged by this turn.

## Bottleneck and next boundary

A representative baseline run spends about 0.035 s in affinity and 0.020 s in
contraction sorting, versus about 0.82 s for full coarsening. Workspace setup
alone is about 0.22 s; CSR build includes further allocations. The remainder
also includes teardown and other host/runtime work, and should not all be
attributed to one kernel without a separate measurement.

The packed full-edge key/value double buffers still occupy about 2.77 GiB for
Products. The next engineering target is reducing that storage footprint and
redundant operations, not accumulating more initialization/refinement policies.
Neither 0.4 s algorithm time nor a large new coarsening speedup was achieved.
