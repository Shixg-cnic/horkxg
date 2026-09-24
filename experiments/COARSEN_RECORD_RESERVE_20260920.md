# Coarsening: packed records and hierarchy capacity

Follow-up kernel results: [COARSEN_SENTINEL_20260921.md](COARSEN_SENTINEL_20260921.md).
The measurements below remain the historical packed-record/reserve A/B.

Products, k=4, seed=0, imbalance=1.10, verify=false, GH200.
This change preserves coarsening decisions and partition output. Both A/B
binaries use the existing GPU initializer; no additional quality tradeoff was
introduced here. Release compiler/CUDA configuration is unchanged.

## Retained implementation

1. Pack source, target and weight in one 64-bit contraction/affinity record
   when conservative accumulated incident-weight bounds prove it safe.
   Sort only the structural key bits, preserving source/target ordering.
   Keep the original wider representation as the required overflow fallback.
   Products edge scratch falls from about 2.77 GiB to 1.84 GiB.
2. Reserve the hierarchy graph and mapping containers before filling them.
   Thrust device-vector moves are not noexcept in this environment, so
   std::vector growth otherwise deep-copies existing GPU graphs and mappings.
   These hidden copies explain much of the previously unattributed hierarchy
   loop time; it was not all workspace teardown or kernel time.

No allocator experiments or new partitioning rules are retained. The five
temporary async/pool/managed/ATS/huge-page experiment runner scripts were also
deleted from the external temporary directory. They were untracked and cannot
be restored with Git; their measurement logs remain.

## Clean-GPU A/B

Each process was checked before launch and monitored during execution. No
foreign compute process was observed. Idle memory was about 2.9 GiB, not
299 MiB. Three independent processes per version, before then after.
The slower second after-run is retained, not discarded.

These are independent per-column medians, not one synthetic run:

| Stage, seconds | Before | After |
|---|---:|---:|
| Coarsen | 0.829820960 | 0.373921632 |
| Initial partition | 0.011462048 | 0.011592416 |
| Uncoarsen | 0.240464416 | 0.242598080 |
| Algorithm total | 1.081483040 | 0.627701632 |
| Full wall | 1.699348160 | 1.248105568 |

Coarsening is 2.22x faster (54.9% less time); algorithm time falls 42.0%.
Full wall includes input loading/preparation and output. The 0.4 s algorithm
target has NOT been reached.

Complete stage rows, seconds:

```text
version run coarsen     initial     uncoarsen   algorithm   wall
before  1   .840013056  .011234464  .240464416  1.091713280 1.699348160
before  2   .829820960  .011475264  .240185728  1.081483040 1.707891872
before  3   .822397504  .011462048  .241592288  1.075452896 1.687263680
after   1   .368646464  .011532928  .242598080   .622778944 1.245841024
after   2   .518699776  .011627936  .264614624   .794943648 1.583470304
after   3   .373921632  .011592416  .242186176   .627701632 1.248105568
```

Full raw results, commands, GPU traces and output partitions remain in
`/tmp/gpart-ab-suY2pw/`: `reserve-ab-results.json`, `reserve-before-*`,
`reserve-after-*`. This is temporary storage, not a committed artifact archive.

## Correctness

- All six A/B outputs: cut 2484676, maximum part weight 673392,
  imbalance 1.099851410, SHA256
  `b820c9b7c19a5fa6a58a62644ecf67eaafa99fdbd6de5c6ef138e9d3e594deaf`.
- Separate fixed Products METIS regression: cut 2405065, maximum part weight
  668666, imbalance 1.092132433, pair accepted 1283, SHA256
  `d114b7a4901c1be3d43b90fcc533b0a675b27d396c9c79dd5d12ebd5a182b162`.
  Its single-run algorithm time was 0.719458464 s (not a median).
- Standard/Big uncoarsening tests passed. Every hierarchy mapping and coarse
  CSR/weight array is compared against the legacy representation, including
  hub sorting, duplicate edges, self-loops and Big weights requiring fallback.
- Packed-record CPU ordering/reduction and overflow-bound tests passed,
  including UBSan. Hierarchy capacity regression checks passed.

A representative final run has device-input/workspace preparation 0.149108 s,
aggregation 0.0965935 s and contraction 0.124693 s. Allocation/preparation
remains material; this report does not attribute it to a compute kernel.
