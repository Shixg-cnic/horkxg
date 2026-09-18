# single_gpu_lp_baseline

## 主线与研究实验的边界（2026-09-08 整理）

当前主算法定义以 [ALGORITHM.md](ALGORITHM.md) 为准，实验分类见
[experiments/README.md](experiments/README.md)。下文的质量报告是历史实验，
其中“目前最好”仅指对应报告当时的实验结果，不代表默认主算法。

唯一默认运行入口是 `./run.sh <dataset> <k>`，实际执行 C++/CUDA
`single_gpu_lp_baseline`，不依赖 Python，也不调用 CPU 精修器。
默认构建仅包含主算法；研究工具需显式设置 `-DBUILD_RESEARCH_TOOLS=ON`。
历史脚本暂保留原路径以免破坏实验复现，不作为主算法的必需步骤。
本次整理不改变搜索逻辑或默认参数，也不声称因此提升质量。

区域联合搜索实验见 [QUALITY_REGIONAL_RESULTS.md](QUALITY_REGIONAL_RESULTS.md)。
Products 八分从 0.06152436 降到约 0.057663；该结果加入了 CPU 局部最小割，
不能归为纯 LP 或 GPU 加速成果，也尚未实现普遍的质量质变。

后续多份数对照见 [QUALITY_RESULTS_V2.md](QUALITY_RESULTS_V2.md)：覆盖 Products 和
LiveJournal 的 2/4/8/16/32 份，包含容量协同小组交换、GPU 继续搜索及独立结果校验。
Products 四分目前最好为 0.034751185；这些多配置择优结果不能当成统一策略的速度结果。

2026-09-08 单层质量实验见 [QUALITY_RESULTS_20260908.md](QUALITY_RESULTS_20260908.md)。
新增可复现的 GPU 多起点搜索和可选 CPU 小组精修参考实现；原有默认参数不变。
Products 四分目前最好切边比例为 0.034821539（GPU+CPU，非纯 GPU 加速结果），
固定四起点 GPU 版本为 0.037196084。质量与完整搜索耗时需一起比较。

干净的单 GPU 标签传播基线。这个项目只保留一张 GPU 上的完整 CSR 和标签传播搜索，不包含 multilevel、coarsening、matching、uncoarsening、MPI、NCCL、halo 或分布式 quota。

流程为：图距离分离 seed、容量约束的多源距离增长初始化、结构场扩散、field 投影、balance、descent，以及搜索结束时的 best-feasible recorder。与 DAC/G-kway 一致，平衡约束只限制每个目标分区的最大点数。

固定基线参数：

```text
k=4
MAX_VERTEX_RATIO=1.10
GLOBAL_CYCLES=5
GROW_ROUNDS=30
REFINE_ROUNDS=50
SEEDS_PER_PART=1
FIELD_ROUNDS=8
BALANCE_ROUNDS=5
POLISH_ROUNDS=20
FEASIBLE_RECORDER=1
OSCILLATION_GUARD=0
MINIMAL_BALANCE_REPAIR=0
INCREMENTAL_CUT=1
INCREMENTAL_CUT_MAX_MOVED_RATIO=0.05
CACHED_NEIGHBOR_COUNTS=1
NEIGHBOR_COUNT_DELTA_MAX_MOVED_RATIO=0.15
STRUCTURAL_WARP_DEGREE=64
```

运行：

```bash
cd /cnic/work/shixg/GNN/partition/single_gpu_lp_baseline
CUDA_VISIBLE_DEVICES=0 ./run.sh products 4
CUDA_VISIBLE_DEVICES=0 ./run.sh com-LiveJournal 4
```

默认划分结果直接写入项目目录：`products_k4.parts` 或 `com-LiveJournal_k4.parts`。也可以指定输出路径：

```bash
OUT_FILE=/tmp/products_k4.parts CUDA_VISIBLE_DEVICES=2 ./run.sh products 4
```

`MAX_VERTEX_RATIO=1.10` 表示每个分区点数最多为平均点数的 1.10 倍，不设置最小点数约束。
输出中的 `vertex_imb` 是最大点数/平均点数，`vertex_min_ratio` 仅作为诊断信息，`feasible=1` 表示最大点数满足约束。
可选的 GPU 两点交换默认关闭：`PAIR_EXCHANGE=0`、`PAIR_EXCHANGE_ROUNDS=2`、
`PAIR_TOP_TARGETS=2`、`PAIR_BUCKET_LIMIT=32`、`PAIR_VERIFY=0`。它只在当前已满足最终
1.10 容量约束时运行；候选基于同一标签快照，交换端点可以相邻但不同交换之间不能
共享端点或存在跨端点边。实现、正确性测试和固定种子对照见
`ALGORITHM.md`、`test_pair_exchange.py` 与 `PAIR_EXCHANGE_RESULTS_20260908.md`。
`OSCILLATION_GUARD` 默认关闭；当前近似检测只用于消融实验，不能替代精确标签状态检测。
搜索恢复最佳可行解后，`POLISH_ROUNDS` 轮 conflict-aware 标签传播只并行提交互不相邻的正增益顶点，用于稳定降低最终 cut。`MINIMAL_BALANCE_REPAIR=1` 是仅迁出超载点数的消融实验，默认关闭。

增量 cut 会记录实际移动点的旧标签，小批移动只扫描受影响边，大批移动自动退回完整扫描；`INCREMENTAL_CUT_VERIFY=1` 可逐轮用完整 cut 校验。完整 cut 对高阶点使用 warp，增量 cut 先压紧实际移动点，再由一个 warp 扫描一个移动点。邻居标签计数缓存让 proposal 直接读取 `count[v][part]`，移动比例较小时用同样的压紧 warp 路径维护，较大时整图重建。距离初始化采用精确 frontier BFS，每个前沿点由一个 warp 展开。seed 使用的局部度数极大点只预计算一次，不再在每次选 seed 时重复扫描。`k=4` 时，高阶点在结构扩散和邻居计数重建中也走独立 warp 路径，以减少幂律图的线程负载不均。

## Products 基准（GH200，k=4）

```text
配置: MAX_VERTEX_RATIO=1.10, REFINE_ROUNDS=50, POLISH_ROUNDS=20
directed edge cut: 5,506,828
edge cut ratio: 0.0445111
vertex imbalance (max/avg): 1.099998
best observed partition time: 0.374 s（低占用窗口单次实测）
final exclusive-GPU median: pending
```

相同搜索轨迹的旧实现为 `0.0445655 / 6.047 s`。当前版本缓存阶段 cut、增加 conflict-aware polish，并为常用的 `k=2/4/8/16/32` 编译定长 proposal/结构扩散 kernel；其他分区数使用最多 32 路的通用 kernel。邻居标签计数缓存把 products 上 proposal 的 profiler 总时间从约 `1.45 s` 降至 `0.010 s`；高阶点 warp 路径把结构扩散 kernel 总时间从约 `380.6 ms` 降至 `175.3 ms`。精确的 warp-frontier BFS 降至约 `21.4 ms`；压紧后的 warp 增量计数维护约为 `12.1 ms`，此前逐点版本约为 `135.6 ms`；完整与增量 cut 的合计 kernel 时间在相邻 profiler 中从约 `239.6 ms` 降至 `105.8 ms`。由于节点上的 GPU 可能同时被其他进程占用，跨时段墙钟数字只作为参考，正式实验应在独占 GPU 上重复多次并报告中位数。
