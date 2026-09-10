# SCLP GPU 粗化

本目录只保留 SCLP 主线：`src/main.cu` 负责层次驱动、校验和 Jet hierarchy
导出，`src/sclp.cu` 负责聚合、admission 和 GPU contraction，`src/graph.cpp`
负责读取 64-bit CSR。

参数固定为 `beta=256`、最多 `4` 轮、two-hop 触发阈值 `0.60`。这些值是
编译期常量，不再读取环境变量，也不按数据集或 k 搜索参数。

每轮从冻结的 cluster 快照计算精确 affinity。低度点由 warp `match_any`
归并相同邻簇；中度点使用每 warp shared-memory hash；只有度数大于 256 的 hub
把紧凑边集送入全局 radix sort/reduce。首选正 gain proposal 按
`(target, -gain, -tie, vertex)` 组成 CUB 自定义 radix key，并在 valid proposal
压紧后排序。容量拒绝的点可按同一快照尝试一次 second-best 正 gain receiver。

contraction 先以 warp ballot 压紧跨簇边，只有 cross-edge entries 进入
sort/reduce。所有 admission 仍使用按 target 的 segmented prefix sum，结果不依赖
CUDA atomic 到达顺序。

构建与小图验证：

```bash
cmake -S . -B build-gh200 -DCMAKE_BUILD_TYPE=Release
cmake --build build-gh200 -j
GPU_LP_STRICT_VERIFY=1 SCLP_DIAGNOSTICS=1 \
  python3 experiments/test_sclp_small.py
```

固定 Jet 后半段对照：

```bash
KS=4 SEED=0 experiments/run_gpu_lp_jet_compare.sh products com-LiveJournal
```
