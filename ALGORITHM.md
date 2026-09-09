# 多层次 GPU-LP 算法说明

## 项目边界

本项目是单 GPU、多层次图粗化实验，不实现多层递归二分，也不调用 METIS、
Jet 或 CPU 最小割来生成自研粗化层。Jet 只作为固定后半段对照：读取自研
输出的完整加权层次，在最粗层初始化并执行原生 projection/refinement。

内部 `multilevel_lp` 维护：

- 64-bit CSR offset 和顶点权重；
- 每层 `fine_to_coarse` 映射；
- 加权粗 CSR，合并平行边并删除簇内自环；
- 每层的顶点/边收缩比例、内部边比例、aggregate 分布和阶段耗时。

每层会检查映射范围与覆盖、簇重量守恒、容量上限、粗图双向边权一致性，
并用随机粗标签验证投影前后的加权切边守恒。设置
`GPU_LP_STRICT_VERIFY=1` 时，会额外逐边检查反向 CSR。

## LP 粗化

默认 `coarsen_method=lp`。每层从单点簇开始，在 GPU 上执行两轮容量约束的
邻居标签聚合，然后在 host 上构造加权 CSR。当前 host contraction 是质量验证
优先版本，尚未作为性能结论。

## BASC 粗化

`coarsen_method=basc` 提供 Bounded Anchor-Support Coarsening 研究路径：

1. 基于确定性 priority 的局部最大值选 anchor；
2. 为顶点保留 K 个 anchor 候选，支持 K=1/2/4；
3. 用常数状态 support 近似邻域社区连接；
4. 按 4 个置信度桶，以 64-bit atomicCAS 进行容量预留；
5. 对未分配点做一轮 aggregate expansion；
6. 剩余点退化为 singleton，再压缩 aggregate ID。

第一版固定 alpha、gamma、lambda、beta、置信度桶数和 expansion 阈值，不按
单个案例调参。BASC 目前是可选研究后端，不改变默认 LP 路径。

## 可复核实验

小图正确性测试：

```bash
GPU_LP_STRICT_VERIFY=1 python3 experiments/test_basc_small.py
```

Jet 后半段对照和结果表：

```bash
METHOD=basc BASC_K=2 STOP_RATIO=0.85 MAX_LEVELS=24 \
  experiments/run_gpu_lp_jet_compare.sh products com-LiveJournal
```

完整结果与当前限制见
[`GPU_LP_JET_COARSEN_EXPERIMENT_20260909.md`](GPU_LP_JET_COARSEN_EXPERIMENT_20260909.md)。
