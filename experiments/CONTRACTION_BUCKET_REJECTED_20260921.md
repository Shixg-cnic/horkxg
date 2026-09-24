# Source-bucketed 32-bit contraction: rejected (2026-09-21)

Products k=4 seed=0 imbalance=1.10, GPU initializer. Same build configuration; three consecutive runs per version, guarded GPU preflight and workload monitoring. Algorithm time excludes input load/preparation and output, includes scratch-pool lifetime/cleanup.

The candidate replaced global 64-bit records with source buckets and segmented 32-bit sorting when safe. It passed Standard/Big tests and all six partition SHA checks, but regressed performance. Experimental kernels, dispatch, include and temporary runner have been removed; prior scratch-pool production implementation restored. Identity/empty-CSR regression tests retained. Earlier unrelated work is preserved.

| Variant/run | Coarsen | Initial | Uncoarsen | Cleanup | Algorithm total | Wall |
|---|---:|---:|---:|---:|---:|---:|
| before 1 | 0.367748928 | 0.011648352 | 0.116296064 | 0.003032032 | 0.498725952 | 1.123832000 |
| before 2 | 0.376367520 | 0.011459072 | 0.115941088 | 0.002977376 | 0.506745312 | 1.139571584 |
| before 3 | 0.366654304 | 0.011534112 | 0.116002144 | 0.003052352 | 0.497243136 | 1.128005664 |
| before median (per column) | 0.367748928 | 0.011534112 | 0.116002144 | 0.003032032 | 0.498725952 | 1.128005664 |
| after 1 | 0.384507456 | 0.011614976 | 0.116015712 | 0.001527392 | 0.513665824 | 1.134517408 |
| after 2 | 0.385316832 | 0.012442400 | 0.116280320 | 0.001578912 | 0.515618720 | 1.155567808 |
| after 3 | 0.387747360 | 0.012397376 | 0.116409184 | 0.001538304 | 0.518092512 | 1.142726496 |
| after median (per column) | 0.385316832 | 0.012397376 | 0.116280320 | 0.001538304 | 0.515618720 | 1.142726496 |

Coarsening regressed 4.78%; algorithm total regressed 3.39%. Narrow records applied only at levels 0, 1, 6; remaining contraction levels required the wide fallback. Smaller record size did not yield a net speedup. Final workspace capacity is not peak device memory.

Run 1 stage evidence (before → candidate): contraction compact 0.00439251 → 0.050639 s; sort 0.012653 → 0.0381194 s; total contraction 0.117449 → 0.206835 s. Device-input stage fell 0.153605 → 0.0482439 s after removing eager scratch reservation, but work moved into the algorithm rather than disappearing. The full coarsening measurement is the deciding comparison.

All six runs: cut=2484676, max part weight=673392, imbalance=1.09985141, SHA256=b820c9b7c19a5fa6a58a62644ecf67eaafa99fdbd6de5c6ef138e9d3e594deaf.

Full raw timings, commands and SHA: `/tmp/gpart-ab-suY2pw/segment-0921-results.json`; per-run logs and GPU monitoring: `segment-0921-{before,after}-{1,2,3}.{log,gpu.jsonl}` in the same directory. Partitions are retained. Removed runner was temporary/untracked; raw evidence remains recoverable in these files.

After reverting, a foreign GPU process (`./run/c5g7_vac`, PID 3846148, 4212 MiB) appeared. No further GPU validation or timing was launched under interference. Restored code build result is recorded in `segment-restored-build.log`; the new identity/empty-CSR cases passed both types before removal of the rejected candidate, but have not yet been rerun on restored code.
