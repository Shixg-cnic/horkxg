# 目标引导多源前沿粗化：首轮实现与验证

## 实现范围

代码位于 sibling `partition/homework`，本目录仅保存实验报告和产物。新增
`multilevel_lp ... frontier` 后端，实现确定性冷热种子、GPU 多源前沿增长、
分段前缀容量准入、紧急种子、两轮边界 relabel，以及 device-resident GPU
sort/reduce contraction。没有调用自研原图划分、BASC、METIS 或 CPU refinement；
完整层次交给现有 `jet_gpu_lp_import`，使用同一 Jet 初始化、投影与 refinement。

默认配置为 `r=8`、capacity slack=0.20、hot ratio=0.50、confidence=0.60、
postpone=2、boundary rounds=2。后三项在首轮实验中固定，没有逐案例调参。

## 正确性与重复性

- 严格小图测试通过，覆盖孤立点、断开分量、平行边、自环和非单位粗点权重。
- 每层检查映射覆盖、编号范围、点权守恒、簇容量、粗 CSR 对称性、平行边合并和
  随机标签投影切边守恒。
- `FRONTIER_DIAGNOSTICS=1` 下，边界 relabel 的完整重算切边下降量与接受收益一致。
- kim2/k=4/seed=0 三次均生成 7 层、最粗保留层 41 点，hierarchy SHA256 均为
  `79da4890347d0fa4e46b9ceaf15ab0709bb228938ad3544f1316acf997de08a2`；
  三次完整粗化时间约 1.72、1.65、1.82 秒。
- 诊断曾在第二层发现 top-4 候选支持度不能直接作为精确迁移收益。当前实现只用
  top-4 选择目标，提交前对该目标重新完整扫描邻边；修正后所有层的完整切边下降
  均与接受收益相等。以下正式结果均来自修正后的实现。

重复性产物：
`build-gh200/experiments/results/frontier_repeatability_current_v2/`。

## products/k=4/seed=0 首轮结果

默认配置首层计划粗点 306,129，实际粗点 1,318,067，其中 emergency seed
1,011,938 个；最终导出 7 层，最粗保留层 78,516 点，停止原因是下一层收缩率
0.914 超过 0.90 阈值。GPU aggregate+contraction 核心约 6.83 秒，层次导出
约 38.66 秒，总计约 46.40 秒。相同 Jet 后半段结果如下；frontier 的切边由 Jet
在原图层完整计算，容量比也由后端报告：

| 粗化路径 | 最终无向切边 | 切边比例 | 最大负载比 | 容量合法 |
|---|---:|---:|---:|---:|
| Jet 原粗化（近期对应运行） | 2,158,447 | 约 3.49% | 合法 | 是 |
| device BASC | 2,210,789 | 3.5739% | 合法 | 是 |
| frontier 默认（收益修正版） | 2,215,379 | 3.58134% | 1.091607 | 是 |

frontier 产物：
`build-gh200/experiments/results/frontier_corrected/products/k4/seed0/`。

为判断失败是否只是容量过紧，预先做了一个上界诊断：slack=4.0、hot ratio=0.8，
其余不变。该探测来自精确收益修正前，仅用于判断 seed/capacity 覆盖：首层实际
粗点降至 538,178、emergency seed 降至 232,049。其旧端到端切边不再作为当前
实现的正式质量数字；由于当时已比默认路径更差，不据此采用该参数，也不继续扫参。

探测产物：
`build-gh200/experiments/results/frontier_capacity4_hot08/products/k4/seed0/`。

## 当前结论

实现层面，frontier 已能正确替换粗化器并把完整多层映射交给 Jet；它不是“GPU
收缩算不动”。kim2 可以稳定深入到 44 个粗点。products 的主要问题是冻结的连通
增长与局部簇容量在 hub/leaf 区域形成屏障，导致大量 emergency singleton，因而
`ceil(n/8)` 只是目标而未被实际控制。

放宽簇容量能明显减少紧急种子，却没有改善相同 Jet 后半段的最终质量。现有证据
因此不支持把该方法称为质量突破，也不支持立即扩展全量 k 和三个 seed。若继续，
最需要验证的是增长期间的容量再分配/边界释放能否在不破坏确定性和容量正确性的
前提下减少 emergency seed；在此之前继续调 seed 比例、轮数或容量参数意义有限。
