# Cross-stage scratch reuse: production result

## Stream-ordering experiment completed: rejected (2026-09-23)

The candidate removed the allocator's three explicit waits, preserving the
final session synchronization. Standard/Big and stream-ordered allocation
memcheck passed (0 errors); all 12 Products outputs matched the reference SHA.
However, two opposite-order A/B batches did not reproduce a performance gain:
old → new medians 0.514232128 → 0.498967264 s in the first batch;
0.506869312 → 0.614389440 s in the reverse-order batch. No foreign GPU process
was observed. The three waits have been restored; no candidate speedup is
claimed. The asynchronous free/reuse regression test is retained.

Complete run rows and logs: [STREAM_ORDER_REJECTED_20260923.md](STREAM_ORDER_REJECTED_20260923.md).

Products only, k=4, seed=0, imbalance=1.10, verify=false, GPU initializer.
Baseline is cooperative-cut + sentinel-contraction. No algorithm, parameter,
tie, admission, rollback or verification rule changes.

## Implementation and timing boundary

`include/detail/scratch_pool.hpp` provides a private CUDA memory pool session
and a scratch allocator. Only the two edge-record sort buffers in coarsening
and the refinement workspace use it. Graphs, partitions, and public API vector
types remain unchanged. Outside a session, standalone APIs retain ordinary
allocation. The pool never changes the device's global/default pool settings.

Coarsening releases its large scratch buffers into the pool. Refinement reuses
that backing memory rather than requesting fresh storage. Shared ownership
keeps a pool alive if a scratch vector outlives its session. Allocation returns
only when ready; free conservatively synchronizes before releasing storage.

The app creates the pool AFTER coarsen_start, not before timing. It explicitly
synchronizes, trims the pool and destroys the session BEFORE free_end. Thus
allocation, pool setup and final reclamation remain in algorithm_total_seconds.
Measured final cleanup rises from about 1 microsecond to about 3 milliseconds;
that cost is included, not hidden. No input load/preparation/output is added to
the algorithm timing boundary.

## Exploratory result versus final result

A repository-external link-wrapper experiment initially used the default pool
for every cudaMalloc/free. It passed Standard/Big and partition SHA gates; its
three-run algorithm median was 0.484605888 s (baseline 0.562956352 s).
It retained the default pool until process exit and was broader than the final
implementation, so this is NOT the production performance result. The wrapper
source, object, test binaries, executable and runner were removed after the
explicit scoped implementation passed. They were untracked and are not
recoverable through Git. Raw experiment logs/JSON/partitions remain.

## Final clean-GPU A/B

Three independent processes per version, before then after, same compiler,
CUDA and Release flags. Preflight and process monitoring guarded every run.
The c5g7_vac workload appeared during compilation; tests/benchmarks waited for
it to exit. No foreign compute PID was observed during these measured runs.
Idle memory was about 4541 MiB, not 299 MiB.

Independent per-column medians, seconds (not a synthetic run):

| Stage | Before | After |
|---|---:|---:|
| Coarsen | 0.363858208 | 0.366302592 |
| Initial partition | 0.011520096 | 0.011578688 |
| Uncoarsen | 0.186409952 | 0.116032448 |
| Hierarchy/pool cleanup | 0.000001056 | 0.003064512 |
| Algorithm total | 0.562066240 | 0.497210176 |
| Supplemental full wall | 1.185592864 | 1.123999328 |

Uncoarsening decreases 37.8%, algorithm total 11.5%. Coarsening did not improve;
its first-use allocation cost remains. Against historical Jet 0.58036 s this
would be about 1.17x, NOT the requested 1.5x. The historical-value target is
0.386907 s, leaving about 110 ms to remove. Jet must be remeasured on the same
machine before asserting a final comparative speedup.

Complete measured rows, seconds:

```text
version run coarsen     initial     uncoarsen   cleanup     algorithm   full_wall
before  1   .363858208  .011590304  .186409952  .000001280  .561860288  1.193653632
before  2   .364387648  .011520096  .186156960  .000001056  .562066240  1.182630304
before  3   .363332352  .011427616  .190772864  .000000992  .565534048  1.185592864
after   1   .371880640  .011527328  .116032448  .003035808  .502476480  1.148852768
after   2   .366302592  .011578688  .115978816  .003064512  .496924864  1.117824032
after   3   .366273344  .011592480  .116273536  .003070528  .497210176  1.123999328
```

## Correctness

- Standard/Big tests pass, including independent CPU cut/contraction checks,
  packed/legacy hierarchy comparisons and strict/fast regression paths.
- New lifetime checks cover nested sessions, restoration of the outer/default
  session, move assignment across allocators, live storage after finish(), and
  resizing that storage after its session ends.
- All six GPU-initial outputs retain cut 2484676, maximum weight 673392,
  imbalance 1.099851410, SHA256
  `b820c9b7c19a5fa6a58a62644ecf67eaafa99fdbd6de5c6ef138e9d3e594deaf`.
- Separate Products METIS regression retains cut 2405065, maximum weight
  668666, imbalance 1.092132433, pair accepted 1283 and SHA256
  `d114b7a4901c1be3d43b90fcc533b0a675b27d396c9c79dd5d12ebd5a182b162`.
  This is a correctness run, not a METIS three-run timing comparison.
- Final build has no allocator host/device annotation warnings; diff check
  passes. No additional dataset, seed or k was run.

Raw data in temporary storage `/tmp/gpart-ab-suY2pw/`:
`scratch-pool-0921-results.json`, `scratch-pool-0921-before-*`,
`scratch-pool-0921-after-*`, `scratch-pool-0921-test-*`,
`scratch-pool-0921-metis*`. Exploratory results: `pipeline-pool-0921-results.json`
and its logs. Baseline binary remains `pre-scratch-pool-0921-standard`.
