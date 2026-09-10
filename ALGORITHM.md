# SCLP GPU 粗化

本目录只保留 SCLP 主线：`src/main.cu` 负责层次驱动、校验和 Jet hierarchy
导出，`src/sclp.cu` 负责聚合、admission 和 GPU contraction，`src/graph.cpp`
负责读取 64-bit CSR。

参数固定为 `beta=256`、最多 `4` 轮、two-hop 触发阈值 `0.60`。这些值是
编译期常量，不再读取环境变量，也不按数据集或 k 搜索参数。
层次驱动的最低粗图尺度相应设为 `beta*k`；接近该尺度后的高容量拒绝率属于
cluster 逐渐填满的预期现象，不再试图向不可达的 `8*k` cutoff 收缩。

每轮从冻结的 cluster 快照计算精确 affinity。低度点由 warp `match_any`
归并相同邻簇；中度点使用每 warp shared-memory hash；只有度数大于 256 的 hub
把紧凑边集送入全局 radix sort/reduce。首选正 gain proposal 按
`(target, -gain, -tie, vertex)` 组成 CUB 自定义 radix key，并在 valid proposal
压紧后排序。容量拒绝不再尝试 second-best，避免把当前层的次优正 gain 永久固化
进 coarse hierarchy。

two-hop fallback 的配对规则保持不变。曾测试按 pair 端点 hash 选择全局 merge
budget，但 Products k=4 的最终 cut 从 `2075165` 恶化到 `2144221`，因此没有纳入
当前主线；现有 budget 的 target-ID ordering bias 仍是明确的待处理项。

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
