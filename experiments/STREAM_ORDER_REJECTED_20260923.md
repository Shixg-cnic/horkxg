# Stream-ordering synchronization removal: rejected

Fixed Products k=4 seed=0 imbalance=1.10, GPU initializer, same build flags. Algorithm time excludes load/preparation/output and includes final pool cleanup. GPU idle preflight and foreign-process monitoring guarded every measured run. No foreign compute PID was observed. Different batch medians must not be cherry-picked or combined into an invented run.

The candidate removed three allocator waits and relied on default-stream ordering. Standard/Big tests and Compute Sanitizer memcheck (`--track-stream-ordered-races all --error-exitcode 99`) passed with zero errors. The initial sanitizer monitor raced with child process exit; its monitor was corrected and both checks rerun successfully. This was a harness issue, not a reported CUDA memory error.

All 12 measured outputs: SHA256 b820c9b7c19a5fa6a58a62644ecf67eaafa99fdbd6de5c6ef138e9d3e594deaf, cut 2484676, maximum part weight 673392, imbalance 1.09985141.

The first batch appeared faster, but the reverse-order batch did not reproduce the result. Observed variability is not explained solely by the absence of other GPU compute processes. No causal speedup or universal regression is claimed. The candidate was removed and the prior allocator waits restored. The asynchronous copy/free/reallocate safety test remains.

## streamorder-0923 (old then new)

Seconds. Median rows are separate per-column statistics.

| Variant/run | Load | Prepare | Coarse | Initial | Uncoarse | Cleanup | D2H | Output | Algorithm | Wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| before 1 | 0.182096960 | 0.480264832 | 0.419149728 | 0.011628704 | 0.115720416 | 0.002951520 | 0.000564864 | 0.001834144 | 0.549450624 | 1.214219616 |
| before 2 | 0.181109504 | 0.435831616 | 0.384052992 | 0.011657312 | 0.115556192 | 0.002965056 | 0.000534016 | 0.001821888 | 0.514232128 | 1.133536544 |
| before 3 | 0.185380544 | 0.441905056 | 0.370337440 | 0.011508768 | 0.116286144 | 0.002894176 | 0.000426912 | 0.001716064 | 0.501026784 | 1.130462336 |
| before median | 0.182096960 | 0.441905056 | 0.384052992 | 0.011628704 | 0.115720416 | 0.002951520 | 0.000534016 | 0.001821888 | 0.514232128 | 1.133536544 |
| after 1 | 0.186293792 | 0.436623968 | 0.369495456 | 0.011467936 | 0.115076672 | 0.002926976 | 0.000561088 | 0.001815360 | 0.498967264 | 1.124270496 |
| after 2 | 0.187741376 | 0.431762880 | 0.369567744 | 0.011526848 | 0.116943584 | 0.003002720 | 0.000523104 | 0.001811520 | 0.501041440 | 1.122895616 |
| after 3 | 0.189091136 | 0.426034912 | 0.363864672 | 0.011571456 | 0.115632096 | 0.002861696 | 0.000510208 | 0.001855808 | 0.493930144 | 1.111430560 |
| after median | 0.187741376 | 0.431762880 | 0.369495456 | 0.011526848 | 0.115632096 | 0.002926976 | 0.000523104 | 0.001815360 | 0.498967264 | 1.122895616 |

## streamorder-confirm (new then old)

Seconds. Median rows are separate per-column statistics.

| Variant/run | Load | Prepare | Coarse | Initial | Uncoarse | Cleanup | D2H | Output | Algorithm | Wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| after 1 | 0.184533248 | 0.471409152 | 0.483897696 | 0.011644096 | 0.115844416 | 0.003002976 | 0.000546784 | 0.001859872 | 0.614389440 | 1.272747744 |
| after 2 | 0.185362976 | 0.491359392 | 0.514955552 | 0.011687264 | 0.116053184 | 0.002947456 | 0.000531136 | 0.001826208 | 0.645643712 | 1.324730848 |
| after 3 | 0.191478208 | 0.427159552 | 0.365235968 | 0.011494624 | 0.115957152 | 0.002973600 | 0.000489504 | 0.001815552 | 0.495661600 | 1.116612832 |
| after median | 0.185362976 | 0.471409152 | 0.483897696 | 0.011644096 | 0.115957152 | 0.002973600 | 0.000531136 | 0.001826208 | 0.614389440 | 1.272747744 |
| before 1 | 0.190378272 | 0.422102816 | 0.365280160 | 0.011653088 | 0.115533760 | 0.002936224 | 0.000491872 | 0.001872384 | 0.495403776 | 1.110257120 |
| before 2 | 0.189932672 | 0.425674528 | 0.377516768 | 0.011320736 | 0.115132416 | 0.002899104 | 0.000508192 | 0.001856544 | 0.506869312 | 1.124848064 |
| before 3 | 0.193471776 | 0.427009472 | 0.383840512 | 0.011586272 | 0.117824608 | 0.002938272 | 0.000532608 | 0.001784160 | 0.516189920 | 1.138995136 |
| before median | 0.190378272 | 0.425674528 | 0.377516768 | 0.011586272 | 0.115533760 | 0.002936224 | 0.000508192 | 0.001856544 | 0.506869312 | 1.124848064 |

## Artifacts

Restored production code was rebuilt and passed Standard/Big tests, including
the retained asynchronous-lifetime case, and a final guarded Products SHA
regression. This final run is not an additional performance median.

Full JSON, per-run `.log`, `.gpu.jsonl`, and `.part` files are retained under `/tmp/gpart-ab-suY2pw/`, with the batch prefixes above. Checked sanitizer logs are `streamorder-0923-sanitizer-checked-{standard,big}.log`. Temporary runner scripts were removed; logs and partitions were not removed. Restored build/check logs use `streamorder-restored-*`.
