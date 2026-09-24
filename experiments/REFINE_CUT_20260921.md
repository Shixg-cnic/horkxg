# Cooperative refinement cut reduction

Follow-up production memory reuse results:
[SCRATCH_POOL_20260921.md](SCRATCH_POOL_20260921.md).

Products only: k=4, seed=0, imbalance=1.10, verify=false. Baseline is the
sentinel-contraction version. Both timed binaries use the GPU initializer.
Only src/refine.cu and permanent tests in tests/test_uncoarsen.cu changed
in this step; no coarsening, initialization, LP, Pair or rollback rule changed.

## Implementation

The old cut kernel assigned one thread to each vertex and scanned its whole
adjacency serially. An earlier external CUDA trace showed about 86 ms cumulative
execution in this kernel. The replacement assigns a warp to each vertex,
reduces exact uint64 directed-edge contributions within the warp, then combines
eight warp sums in shared memory before one block-level atomic accumulation.
Tail warps contribute zero and participate in the barrier. The empty graph
case returns zero without launching a zero-block kernel.

Every existing cut calculation, including commit/rollback checks and strict
projection verification, remains. No cached-cut shortcut or new strategy was
introduced. The serial kernel was replaced, not retained behind a switch.
The only additional storage is 64 bytes of shared memory per block.

## Clean-GPU A/B

A foreign Python compute process was observed initially. No benchmark ran
until it exited. Each subsequent test/run had idle preflight and process
monitoring. Idle memory was 4541 MiB; no foreign compute PID was observed
during the measurements. Same Release compiler/CUDA/flags, three independent
processes per version, before then after. All original rows are retained.

Independent per-column medians (seconds), not a synthetic run:

| Stage | Before | After |
|---|---:|---:|
| Coarsen | 0.364100224 | 0.368134400 |
| Initial partition | 0.011558304 | 0.011451584 |
| Uncoarsen | 0.241869856 | 0.188534208 |
| Algorithm total | 0.617529664 | 0.566464384 |

Uncoarsening decreases 22.1%; algorithm total decreases 8.3%. Coarsening was
unchanged, and its timing variation must not be credited to this change.
The algorithm boundary is device input ready -> partition on device, including
hierarchy cleanup. It excludes input load/preparation and output download/I/O.
Supplementary full-wall medians are 1.241600960 -> 1.189201024 s, not the main
Jet comparison metric. The 0.4 s algorithm target is not reached. The historical
Jet total of 0.58036 s was not remeasured in this A/B and is not evidence of a
statistically stable win over Jet.

Complete rows (seconds):

```text
version run coarsen     initial     uncoarsen   algorithm   full_wall
before  1   .363749376  .011644224  .245481088  .620875712  1.234864512
before  2   .364511168  .011551392  .241212288  .617276416  1.242407392
before  3   .364100224  .011558304  .241869856  .617529664  1.241600960
after   1   .363113248  .011451584  .188534208  .563100832  1.187759584
after   2   .368134400  .011495008  .186833632  .566464384  1.189201024
after   3   .376259680  .011359040  .190812576  .578432832  1.264926432
```

## Correctness

- Standard/Big uncoarsening tests passed. Added an independent CPU cut oracle
  with a hub, duplicate edges, isolated vertex, partial final block, two fixed
  balanced labelings, and Big weights above 32 bits. With zero refinement
  rounds, both reported cuts must equal CPU cut and labels must be unchanged.
- All six GPU-initial benchmark outputs: cut 2484676, maximum part weight
  673392, imbalance 1.099851410, SHA256
  `b820c9b7c19a5fa6a58a62644ecf67eaafa99fdbd6de5c6ef138e9d3e594deaf`.
- Separate fixed Products METIS regression: cut 2405065, maximum part weight
  668666, imbalance 1.092132433, pair accepted 1283, SHA256
  `d114b7a4901c1be3d43b90fcc533b0a675b27d396c9c79dd5d12ebd5a182b162`.
  It is a correctness run, not a three-run METIS performance comparison.
- No new dataset, seed or k; no runtime diagnostic branch added. Existing
  decisions and verification are intact. git diff --check passed.

Full logs, GPU traces, partitions and commands remain in temporary storage
`/tmp/gpart-ab-suY2pw/`: `cut-0921-results.json`, `cut-0921-before-*`,
`cut-0921-after-*`, `cut-0921-test-*`, and `cut-0921-metis*`.
