# Coarsening: remove cross-edge allocation atomics

Subsequent refinement-only performance results:
[REFINE_CUT_20260921.md](REFINE_CUT_20260921.md).

Products only, k=4, seed=0, imbalance=1.10, verify=false, GH200.
Baseline is the previously validated packed-record + hierarchy-reserve version.
Both timed versions use the existing GPU initializer. No SCLP decision,
affinity/gain, random/tie, admission, initialization or refinement rule changed.

## Analysis and retained implementation

An external Nsight Systems CUDA trace of the baseline measured coarsening at
0.375115 s. Approximately 0.28 s of runtime API duration in the coarsening
interval was cudaMalloc. This is allocation-call wall duration, not GPU kernel
execution time or a claim that all allocation cost can be removed. Whole-process
cudaMalloc time was 0.566 s, including input preparation and other stages; that
whole-process number must NOT be attributed entirely to coarsening.

The remaining cross-edge kernel used one globally contended atomicAdd for each
warp chunk. Its cumulative compact time was about 12 ms. The retained kernel
writes each input edge directly to its own record position. Internal edges use
UINT64_MAX, which is outside the valid structural-key range in both checked
packed and legacy layouts. Radix sorting places these invalid records last;
lower_bound excludes them before reduce. The resulting CSR still contains only
cross edges in exactly the same (source, target) order with the same integer
weight sums.

**Sorting input changes:** all input entries, including invalid markers, now
enter contraction sort; it is no longer a cross-edge-only sort. On Products,
94–99% of input entries are cross edges, so the extra sort work is small. This
does not establish a performance improvement on graphs with many internal
edges. There is no threshold, alternative policy, or new tuning parameter.
The existing compact_seconds field now measures record encoding; sort_seconds
includes locating the invalid suffix. Logs report sorted_entries and
cross_fraction explicitly.

A count -> prefix -> write variant was tried first: compact median
12.2473 -> 10.1506 ms, coarsening 0.371852448 -> 0.370867136 s, algorithm
0.625482784 -> 0.628992128 s. Its extra pass/scratch was not worth retaining.
That kernel, its prefix logic, and its temporary runner script were deleted.
The untracked runner cannot be restored through Git; raw results remain.
Only the simpler sentinel implementation and permanent correctness tests remain.

## Final A/B: independent per-column medians

Three independent processes before, then three after, same Release compiler,
CUDA and flags. Every process had idle-GPU preflight and runtime monitoring.
No foreign compute PID was observed. Idle memory today was 4541 MiB; the
monitor accepted up to 5000 MiB of idle memory but rejected other compute PIDs
and sustained utilization. Slow rows were not discarded.

| Seconds | Before | After |
|---|---:|---:|
| Compact / encode | 0.012125700 | 0.004361470 |
| Contraction sort | 0.011902100 | 0.012627500 |
| Contraction total | 0.124804000 | 0.117292000 |
| Coarsen | 0.374012864 | 0.365428896 |
| Initial partition | 0.011460096 | 0.011490080 |
| Uncoarsen | 0.244090688 | 0.242682592 |
| Algorithm total | 0.629652544 | 0.619671936 |
| Full wall | 1.275268832 | 1.246002016 |

Compact/encode falls 64.0%; coarsening falls 2.3%; algorithm median falls 1.6%.
This is a small improvement, not a new large speedup. Unmodified-stage and
input timing variation should not be credited to this kernel change.
The algorithm remains above the 0.4 s target. Allocation is still dominant.

Complete run rows, seconds (not assembled from separate medians):

```text
version run coarsen     initial     uncoarsen   algorithm   full_wall
before  1   .371759648  .011460096  .242141344  .625362336  1.258266912
before  2   .428798464  .011422368  .245888224  .686110304  1.330129888
before  3   .374012864  .011546848  .244090688  .629652544  1.275268832
after   1   .376420064  .011451520  .243599072  .631472032  1.251211104
after   2   .365428896  .011558944  .242682592  .619671936  1.243681728
after   3   .362420896  .011490080  .241616576  .615529024  1.246002016
```

## Correctness and artifacts

- Standard/Big uncoarsening tests passed. Added an independent CPU contraction
  reference covering duplicate edges, self-loops, empty rows, a non-warp-aligned
  vertex count, an all-internal-edge graph, scratch reuse, and Big weights above
  32 bits. Existing packed/legacy hierarchy and hub tests also passed.
- All six timed GPU-initial outputs have cut 2484676, maximum part weight
  673392, imbalance 1.099851410 and SHA256
  `b820c9b7c19a5fa6a58a62644ecf67eaafa99fdbd6de5c6ef138e9d3e594deaf`.
- A separate fixed Products METIS regression retains cut 2405065, maximum
  part weight 668666, imbalance 1.092132433, pair accepted 1283 and SHA256
  `d114b7a4901c1be3d43b90fcc533b0a675b27d396c9c79dd5d12ebd5a182b162`.
  This was one correctness run, not a METIS performance A/B.
- No additional dataset, seed or k was run. No profiler instrumentation was
  added to production code. Changes this turn are confined to src/coarsen.cu,
  tests/test_uncoarsen.cu and this report/link.

Raw logs, traces, partitions and complete JSON are in temporary storage
`/tmp/gpart-ab-suY2pw/`: `sentinel-0921-results.json`, `sentinel-0921-before-*`,
`sentinel-0921-after-*`, `sentinel-0921-test-*`, `sentinel-0921-metis*`.
The unsuccessful variant is recorded in `prefix-0921-results.json` and its
logs; the external profile is `coarse-0921-profile.nsys-rep` and its SQLite
export. Profiling results were not used as timed A/B rows.
