# Coarsening pool extensions: rejected (2026-09-22)

No algorithm or parameter changes. Fixed Products k=4 seed=0 imbalance=1.10, GPU initializer; same Release build. Baseline is the existing scoped scratch-pool version. GPU preflight checked no compute processes and idle utilization; idle memory was 5072 MiB (guard limit 5400 MiB), with foreign-process monitoring during every run. Input load/preparation and final output excluded from algorithm timing; allocations and final pool cleanup included.

## Decisions

1. Pooling graph neighbor/edge-weight arrays: rejected and removed. Median algorithm time 0.498362688 → 0.521316480 s; CSR build remained about 0.096 s. Public graph vector types and initializer pointer helpers restored.
2. Pooling all coarsening workspace arrays: rejected and removed. Apparent median 0.519808736 → 0.511112960 s is not reliable evidence of improvement: the baseline group includes a 0.688250144 s outlier; the earlier same-session baseline median was 0.498362688 s. In run 1, aggregate increased from 0.0943807 to 0.102719 s and contraction 0.117394 to 0.120473 s. No reliable improvement in the targeted stages. Only the previous two edge-record pooled buffers remain.

Do not promote the apparent second-group median change as a speedup. No new production optimization is retained. Temporary variant runners were deleted; untracked source is not Git-recoverable, but all measurements and output partitions remain.

## Correctness

Both candidates passed Standard/Big uncoarsening tests, including previous empty/identity CSR tests. All twelve timed Products outputs have cut 2484676, max weight 673392, imbalance 1.09985141, SHA256 `b820c9b7c19a5fa6a58a62644ecf67eaafa99fdbd6de5c6ef138e9d3e594deaf`. This is the current GPU-initializer baseline, not the older METIS result 2405065.

## Complete measured rows

Seconds. Median rows are independent per-column statistics, not synthetic runs. All raw fields/commands/SHA are retained in JSON.

### graphpool

| Variant/run | Load | Prepare | Coarsen | Initial | Uncoarsen | Cleanup | D2H | Output | Algorithm | Full wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| before 1 | 0.194125888 | 0.424029344 | 0.368119936 | 0.011399392 | 0.115726112 | 0.003090144 | 0.000504736 | 0.001863424 | 0.498335872 | 1.118870144 |
| before 2 | 0.192850528 | 0.439200384 | 0.393773472 | 0.011494048 | 0.116034304 | 0.003110432 | 0.000513024 | 0.001830528 | 0.524412544 | 1.158818208 |
| before 3 | 0.193250208 | 0.432072192 | 0.368035552 | 0.011587872 | 0.115770240 | 0.002968736 | 0.000522976 | 0.001796576 | 0.498362688 | 1.126017888 |
| before median | 0.193250208 | 0.432072192 | 0.368119936 | 0.011494048 | 0.115770240 | 0.003090144 | 0.000513024 | 0.001830528 | 0.498362688 | 1.126017888 |
| after 1 | 0.189771904 | 0.486277984 | 0.555147744 | 0.011332288 | 0.115921312 | 0.005051136 | 0.000560832 | 0.001894816 | 0.687452768 | 1.365969344 |
| after 2 | 0.188908576 | 0.429066272 | 0.369102560 | 0.011668384 | 0.114520960 | 0.004853440 | 0.000518656 | 0.001810432 | 0.500145888 | 1.120461760 |
| after 3 | 0.195694880 | 0.426116192 | 0.390131072 | 0.011544032 | 0.114546240 | 0.005094752 | 0.000602496 | 0.001893280 | 0.521316480 | 1.145635200 |
| after median | 0.189771904 | 0.429066272 | 0.390131072 | 0.011544032 | 0.114546240 | 0.005051136 | 0.000560832 | 0.001893280 | 0.521316480 | 1.145635200 |

### workspacepool

| Variant/run | Load | Prepare | Coarsen | Initial | Uncoarsen | Cleanup | D2H | Output | Algorithm | Full wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| before 1 | 0.190971968 | 0.448882624 | 0.369247296 | 0.011730912 | 0.116536512 | 0.003081056 | 0.000560160 | 0.001924896 | 0.500596352 | 1.142947040 |
| before 2 | 0.188800256 | 0.472900320 | 0.557716288 | 0.011498976 | 0.115997536 | 0.003037056 | 0.000526208 | 0.001830624 | 0.688250144 | 1.352318720 |
| before 3 | 0.191427328 | 0.424214016 | 0.387933632 | 0.011578880 | 0.117393152 | 0.002902528 | 0.000544096 | 0.001924672 | 0.519808736 | 1.137930464 |
| before median | 0.190971968 | 0.448882624 | 0.387933632 | 0.011578880 | 0.116536512 | 0.003037056 | 0.000544096 | 0.001924672 | 0.519808736 | 1.142947040 |
| after 1 | 0.194596000 | 0.428165568 | 0.378550944 | 0.011989536 | 0.116498464 | 0.004150592 | 0.000512448 | 0.001864352 | 0.511190080 | 1.136338784 |
| after 2 | 0.194514848 | 0.427223040 | 0.378569792 | 0.011939552 | 0.116431616 | 0.004171744 | 0.000529664 | 0.001860544 | 0.511112960 | 1.135250432 |
| after 3 | 0.194564928 | 0.426050624 | 0.375154400 | 0.011864480 | 0.116980576 | 0.004098976 | 0.000512832 | 0.001853568 | 0.508098688 | 1.131091040 |
| after median | 0.194564928 | 0.427223040 | 0.378550944 | 0.011939552 | 0.116498464 | 0.004150592 | 0.000512832 | 0.001860544 | 0.511112960 | 1.135250432 |

## Raw evidence

Restored production implementation was rebuilt successfully and passed Standard/Big tests plus one guarded Products SHA regression. This final run is a correctness check, not a new three-run performance median. `git diff --check` passed. Both experimental runners and the final temporary check script were removed; logs, JSON, partitions and baseline binary are retained.

Directory `/tmp/gpart-ab-suY2pw/`: `graphpool-0922-results.json`, `workspacepool-0922-results.json`; corresponding `{name}-0922-{before,after}-{1,2,3}.{log,gpu.jsonl,part}`; `{name}-0922-test-{standard,big}.{log,gpu.jsonl}`. Restored baseline build and verification logs use `restored-0922-*`.
