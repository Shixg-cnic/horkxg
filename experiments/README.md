# 实验工具

- `test_sclp_small.py`：生成含自环、平行边、孤立点和断开分量的小图，独立检查
  hierarchy 的容量、覆盖、对称性和投影切边守恒。
- `test_sclp_repeatability.py`：同 seed 重复运行并比较 hierarchy SHA256。
- `verify_partition.py`：从原始 symmetric CSR 独立复算最终 cut、负载和容量。
- `run_gpu_lp_jet_compare.sh`：运行 Jet 原粗化与固定参数 SCLP + 同一 Jet 后半段。
- `run_sclp_merge_diagnostics.sh`：用 Jet 与 METIS reference partition 运行
  mover/receiver 和临时 priority-oriented 诊断版，逐 accepted proposal 记录
  `Acur/A1/A2/weighted_degree/target_weight/capacity` 与两套 oracle loss。
- `analyze_merge_diagnostics.py`：汇总每层 weighted purity、oracle bad/loss，按
  normalized gain、best-second margin、support、degree、target/capacity 做十桶统计；
  MR/priority 的精确集合比较限定在 `level=0, round=0`，避免把已经分叉的 coarse
  vertex id 当成同一对象。

固定参数是 `beta=256 / rounds=4 / two-hop=0.60`，脚本不提供参数搜索入口。
默认二进制用 CUDA events 输出每层 `affinity/admission/two-hop/compact/contraction`
五项 `ml_gpu_timing`，并在结束时输出 `ml_gpu_timing_total`；这些计时不依赖
`SCLP_DIAGNOSTICS`，benchmark 可配合 `ML_SKIP_HIERARCHY_EXPORT=1` 使用。
诊断环境当前没有 ParMETIS/MPI，因此第二套 oracle 使用 serial METIS 5.2.1 并明确
标记为 `metis`。`it-2004` 在未改动的 `13e2e75` production contraction 中也会超过
GH200 的 96 GiB HBM；脚本对该图只收集完整的 level-0 proposal 诊断，并在结果目录
写入 `scope.txt`，不会为了跑通而暗改 contraction 或 coarsening rule。
