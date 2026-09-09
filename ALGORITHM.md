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

### Device-resident GPU 路径

`coarsen_method=basc_gpu` 是同一 BASC 规则的性能实现：当前层 CSR、顶点权重
和 aggregate 映射留在 GPU 上跨层传递；anchor 选举、候选生成、support、同步
expansion、根压缩和聚类重量累加使用 CUDA，粗图收缩使用 GPU key sort/reduce，
并在 GPU 上生成粗 CSR。每个顶点的 CSR 扫描采用一个 warp，K 候选的 support
匹配也逐槽检查，不再把 K>1 简化为只看第一个候选。

为了交给 Jet importer，仍会在每层做一次 host snapshot 并最终导出完整层次；
这部分不属于 GPU contraction 核心，日志分别记录
`ml_gpu_basc_core_seconds`、`ml_gpu_basc_snapshot_total_seconds` 和
`ml_hierarchy_export_seconds`。products/k=4/seed=0 的一次 smoke 显示核心约
3--4 秒，而完整层次导出约 55--61 秒，当前端到端瓶颈是 Jet 层次快照/导出。

当前 admission 使用 GPU atomicCAS 做容量预留，因此同一 seed 的运行尚未保证
bitwise repeatability；kim2/k=4/seed=0 的三次测试产生了不同层数和 hierarchy
SHA256。该事实由 `experiments/test_basc_repeatability.py --method basc_gpu`
记录，不把 seed=0 当作确定性证明。后续若需要严格复现实验，应增加 GPU 分段
前缀容量预留，而不是继续依赖 atomic 竞争顺序。

## 目标引导多源前沿粗化

`coarsen_method=frontier` 实现独立于 BASC 的 target-guided seeded frontier
路径。每层目标粗点数为 `max(C_min, ceil(n/r))`，默认 `r=8`。算法先按确定性
priority 选取约一半 degree-local-maximum 热种子，再从其一跳未覆盖区域选择冷
种子。所有 seed 从单点簇出发，GPU warp 按当前前沿扫描 CSR，按真实邻接标签
权重和剩余容量评分；低置信候选最多推迟两轮。

同一轮 proposal 基于冻结标签快照生成。proposal 按目标簇、分数降序和顶点号
排序，使用 segmented prefix sum 做确定性容量 admission，提交后才生成下一轮
前沿。无法从已有簇到达的剩余区域以确定性的局部极大点产生 emergency seed，
保证孤立点和断开分量终止。增长完成后执行两轮短边界 relabel；同轮迁移点采用
非邻接局部竞争，并再次用分段前缀和控制目标容量，因此被接受迁移的收益可以相加。

当前层图、映射、权重以及 sort/reduce contraction 全部驻留 GPU。每层 host
snapshot 仅用于完整层次导出给 Jet；设置 `ML_SKIP_HIERARCHY_EXPORT=1` 可在性能
诊断时跳过最终文件写入，但正式 Jet 对照不能使用该选项。关键配置为：

```bash
FRONTIER_CONTRACTION_FACTOR=8   # r
FRONTIER_CAPACITY_SLACK=0.20    # 簇容量松弛
FRONTIER_HOT_RATIO=0.50         # 热种子比例
FRONTIER_DIAGNOSTICS=1          # 逐轮计数和边界收益核验
FRONTIER_VERIFY=1               # 每层完整结构检查
```

默认参数不按图名或 k 特化。相同 seed 的确定性来自稳定排序和分段前缀 admission，
不依赖 CUDA atomic 的到达顺序。严格小图覆盖自环、平行边、孤立点、断开分量、
非单位点权和投影切边守恒；边界阶段还会比较完整重算切边下降量与接受收益之和。

该版本目前有一个明确限制：目标粗点数是规划目标，不是无条件保证。在幂律图中，
冻结增长加局部簇容量会形成空间屏障，未分配区域需要 emergency seed，实际粗点数
可能明显高于目标。products/k=4/seed=0 已观察到该现象；增大容量松弛虽减少紧急
种子，但固定 Jet 后半段的最终切边变差，因此没有把探测参数改成默认值，也没有
据此宣称质量改善。数值结果保存在 sibling `single_gpu_lp_baseline` 的实验报告和
产物目录，本代码仓库不存放大型实验输出。

## 可复核实验

小图正确性测试：

```bash
GPU_LP_STRICT_VERIFY=1 python3 experiments/test_basc_small.py
```

测试两条 BASC 路径：

```bash
GPU_LP_STRICT_VERIFY=1 BASC_TEST_METHOD=basc \
  python3 experiments/test_basc_small.py
GPU_LP_STRICT_VERIFY=1 BASC_TEST_METHOD=basc_gpu \
  python3 experiments/test_basc_small.py
GPU_LP_STRICT_VERIFY=1 FRONTIER_DIAGNOSTICS=1 BASC_TEST_METHOD=frontier \
  python3 experiments/test_basc_small.py
python3 experiments/test_basc_repeatability.py --method basc_gpu \
  --indptr ../dataset/Gpartition_dataset/Sym_CSR/kim2/kim2_sym_indptr.bin \
  --indices ../dataset/Gpartition_dataset/Sym_CSR/kim2/kim2_sym_indices.bin \
  --parts 4 --seed 0 --basc-k 2
```

Jet 后半段对照和结果表：

```bash
METHOD=basc_gpu BASC_K=2 STOP_RATIO=0.90 MAX_LEVELS=24 \
  experiments/run_gpu_lp_jet_compare.sh products com-LiveJournal
METHOD=frontier FRONTIER_CONTRACTION_FACTOR=8 \
  FRONTIER_CAPACITY_SLACK=0.20 FRONTIER_HOT_RATIO=0.50 \
  experiments/run_gpu_lp_jet_compare.sh products
```

完整结果和大型产物保存在 sibling `single_gpu_lp_baseline` 项目中。
