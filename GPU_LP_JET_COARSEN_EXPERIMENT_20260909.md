# GPU-LP 粗化 + Jet 后半段对照

这是一个隔离粗化质量的研究入口，不是默认 `single_gpu_lp_baseline` 流程，
也不把 BASC 或 GPU-LP 粗化宣称为已成立的算法创新。

## Jet 本地接口审计

本地 Jet 的标准 GPU 类型定义在
[`header/jet_defs.h`](../sotas/Jet-Partitioner/header/jet_defs.h)：

- `value_t`, `ordinal_t`, `edge_offset_t` 均为 `int32_t`；标准
  `matrix_t` 是 `KokkosSparse::CrsMatrix<value_t, ordinal_t, Device, void,
  edge_offset_t>`。
- `wgt_vt` 是设备端 `Kokkos::View<int32_t*, Device>`，`part_vt` 是设备端
  `Kokkos::View<int*, Device>`。
- Jet 另有 `big_matrix_t`/`biggest_matrix_t`，但这次导出的 products 和
  LiveJournal 层次在标准 Jet 的 32-bit 点号、边偏移、边权范围内。

原版调用路径在 [`src/partitioner.hpp`](../sotas/Jet-Partitioner/src/partitioner.hpp)：

```text
load_metis_graph
  -> contracter::generate_coarse_graphs
  -> initial_partitioner::metis_init(coarsest graph, coarse vertex weights)
  -> uncoarsener::uncoarsen
       -> jet_refiner::jet_refine(coarsest -> finest)
       -> project(next fine vertex <- current coarse map)
```

`coarse_level_triple` 在 [`src/contract.hpp`](../sotas/Jet-Partitioner/src/contract.hpp)
中拥有当前层 `matrix_t mtx`、当前层 `wgt_vt vtx_w`、以及
`interp_mtx.map`。列表按 fine→coarse 存放；对于 coarse 层 `l+1`，其 map
长度是 fine 层 `l` 的点数，方向是 `fine_vertex -> coarse_vertex`。
`uncoarsen` 从 `cg_list.back()` 开始，refine 当前层后用该 map 投影到更细层。
所有 Kokkos View 是引用计数的设备视图，列表元素持有它们的所有权，直到
uncoarsening 结束。

原版停止阈值也在 `partitioner.hpp`：通常 `k*8`，大 k 时至少 1024，
并要求 coarse 点数不低于 `cutoff/4`。Jet 的最粗初始化确实是 METIS，
不是随机标签；原版每次运行的旧 coarsener seed 是 wall-clock。

## A/B 接入

自研层次生成器是 [`src/multilevel_lp.cu`](src/multilevel_lp.cu) 的
`multilevel_lp` 研究 executable。它现在的默认 `lp` 路径只做：

```text
输入加权 CSR -> GPU-LP cluster moves (两轮) -> host weighted contraction
             -> 下一层，直到固定停止规则 -> Jet binary hierarchy
```

它不再进行原图单层初始化、按原标签投票、每层自研 refinement 或自研最终
polish。每层会检查映射范围和覆盖、逐簇重量、总重量守恒、粗 CSR 无自环、
反向边权一致，以及随机粗标签投影后的切边守恒。默认是线性形状/切边检查；
设置 `GPU_LP_STRICT_VERIFY=1` 会额外逐边二分检查反向 CSR，适合小图或 smoke
test，不混入大图默认计时。

Jet 导入后端是 [`app/gpu_lp_import.cpp`](../sotas/Jet-Partitioner/app/gpu_lp_import.cpp)：
它读取每一层 CSR、边权、顶点权和 map，直接调用 Jet 的 `metis_init` 与
`uncoarsener::uncoarsen`。因此 B 不是只把最粗图送进原图 refinement，
而是真正将完整层次交给 Jet。

```text
A: Jet 原版 coarsener -> Jet METIS init -> Jet projection/refinement
B: 自研 GPU-LP coarsener -> Jet METIS init -> Jet projection/refinement
```

`jet_config` 增加可选第 6 行 seed；少于 6 行时保持 Jet 的 wall-clock 默认行为。
修改前的 Jet 源码和可运行二进制备份在
`/cnic/work/shixg/GNN/partition/backups/20260909_gpu_lp_jet_coarsen/`。

## seed=0 首轮六组结果

这是 products、com-LiveJournal 的 k=4/8/32；A/B 使用同一 Jet 配置和输入。
切边与容量由 [`experiments/verify_partition.py`](experiments/verify_partition.py)
从原始 Sym_CSR 独立重算；Jet 的整数容量约定是
`floor(1.10 * ceil(n/k))`，原始比例也同时保留在 JSON 中。

| 图 | k | A cut | A 比例 | B cut | B 比例 | B-A | B 层数/最粗点 | B 粗化秒 | B Jet 后半段秒 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| products | 4 | 1,991,611 | 0.032195972 | 2,270,970 | 0.036712032 | -14.027% | 9 / 105,526 | 60.45 | 4.23 |
| products | 8 | 3,075,295 | 0.049714583 | 3,519,338 | 0.056892891 | -14.439% | 9 / 105,449 | 59.32 | 4.21 |
| products | 32 | 5,661,289 | 0.091519228 | 6,092,159 | 0.098484583 | -7.611% | 9 / 105,484 | 65.18 | 3.83 |
| LiveJournal | 4 | 3,473,755 | 0.100162512 | 3,448,232 | 0.099426580 | +0.735% | 25 / 5,667 | 92.78 | 6.57 |
| LiveJournal | 8 | 4,862,752 | 0.140212955 | 4,827,177 | 0.139187183 | +0.732% | 25 / 5,897 | 87.72 | 5.70 |
| LiveJournal | 32 | 7,145,936 | 0.206046454 | 7,215,180 | 0.208043040 | -0.969% | 25 / 5,846 | 91.03 | 3.58 |

这六组已经足以说明：B 可以正确替换粗化并接通相同 Jet 后半段，但当前
停止规则不是同等粗化规模。Jet A 的最粗层在这几组是 29/49/218 点，而
旧 B 在 LiveJournal 因 `max_levels=24` 停在约 5.8k 点；因此这张表不能解释
为“同等粗化深度下 GPU-LP 聚类质量”结论。products 的 B 质量明显退化，
LiveJournal 的 k=4/8 只出现约 0.7% 的小幅改善，k=32 退化。

## BASC 第一版

根据方案文档，`multilevel_lp` 新增 `coarsen_method=basc`，默认 K=2，
同时支持 K=1/4。第一版固定：`alpha=.25`、`gamma=.5`、`
lambda=.35`、`beta=2`、4 个置信度桶、1 次 expansion、`
theta=.25`。anchor、top-K candidate、support、4 桶 admission、expansion
和 singleton fallback 在 CUDA 中执行；aggregate 权重用 64-bit CAS。
粗 CSR 仍先用 host sort/reduce，便于与层次校验隔离。

命令格式为：

```text
multilevel_lp indptr indices k hierarchy.out 1.10 seed stop_ratio basc 2 max_levels
```

[`experiments/test_basc_small.py`](experiments/test_basc_small.py) 已通过 K=1/2/4
小图测试，覆盖自环、平行边、孤立点、断开分量、容量、映射覆盖、粗 CSR 对称性
和随机标签切边守恒。kim2/k=4/K=2 的端到端 smoke 也已由 Jet importer 完成，
独立原图 cut=20,814，容量合法。

products/k=4/seed=0 的 BASC K=2 smoke：6 层、最粗 190,778 点，自研粗化
49.3 秒，接 Jet 后 cut=2,191,432。它优于同一轮旧 GPU-LP B 的 2,270,970，
但仍明显差于 A 的 1,991,611；粗化阶段也不是免费。因此目前只能称为一个
有质量信号的可选研究后端，不能称为突破。

## BASC K=1/2/4 首轮消融

随后在同一批次重新运行 A，并固定 `STOP_RATIO=0.85`、`MAX_LEVELS=24`、
`seed=0`，只改变 BASC 的候选数 K。A 与每个 BASC 输出均由原始 CSR 独立
重算切边和容量；表中百分比定义为 `(A cut - BASC cut) / A cut`，正数表示
BASC 更好。

| 图 | k | Jet A | BASC K=1 | BASC K=2 | BASC K=4 |
|---|---:|---:|---:|---:|---:|
| products | 4 | 2,154,817 | 2,220,938 (-3.07%) | 2,133,529 (+0.99%) | 2,290,708 (-6.31%) |
| products | 8 | 3,146,029 | 3,475,619 (-10.48%) | 3,495,104 (-11.10%) | 3,574,198 (-13.61%) |
| products | 32 | 5,672,104 | 6,213,516 (-9.55%) | 6,200,259 (-9.31%) | 6,410,468 (-13.02%) |
| LiveJournal | 4 | 3,576,569 | 3,605,784 (-0.82%) | 3,621,474 (-1.26%) | 3,613,104 (-1.02%) |
| LiveJournal | 8 | 4,800,350 | 4,996,567 (-4.09%) | 4,873,886 (-1.53%) | 4,903,840 (-2.16%) |
| LiveJournal | 32 | 7,151,968 | 7,338,960 (-2.61%) | 7,343,802 (-2.68%) | 7,384,568 (-3.25%) |

18 个 BASC 输出和 6 个 A 输出均通过独立容量检查。BASC K=2 的粗化总时间
约为 products 50.6--56.3 秒、LiveJournal 50.4--55.7 秒；K=4 在
LiveJournal/k=32 达到 75.3 秒。粗 CSR 仍由 host sort/reduce 构造，整个层次
生成流程累计约 13--15 秒，是当前主要成本之一。

这轮结果不支持“增大 K 就改善质量”：K=2 仅在 products/k=4 小幅胜出，
K=1/4 也没有跨图跨 k 的稳定改善。因此 BASC 暂保留为正确可运行的研究后端，
不进入默认主线；后续若继续，应先做固定最粗规模的公平对照，而不是继续调 K。

## 运行入口和未完成项

[`experiments/run_gpu_lp_jet_compare.sh`](experiments/run_gpu_lp_jet_compare.sh)
支持 `METHOD=lp|basc`、`BASC_K=1|2|4`、`STOP_RATIO` 和 `MAX_LEVELS`。
旧 LP 六组的日志、层次、partition、独立 JSON 和
[`seed0_summary.csv`](build-gh200/experiments/results/gpu_lp_jet_compare/seed0_summary.csv)
均保留。

尚未把 k=2/16、seed=1/2 的全套 A/B 或 BASC K 消融写成正式结论；这些应在
先固定深度/目标规模规则后再运行。下一步若继续，优先使用固定 `MAX_LEVELS=200`
或固定粗点目标做“同等粗化规模”组，不能按最终 cut 逐案例调整停止参数；然后再
决定 BASC 是否值得扩展 contraction 的 GPU sort/reduce。当前 Jet 后端的输入读取、
层次加载、METIS 初始化、Jet refinement 和独立校验时间仍分开记录，没有把输入输出
时间伪装成 kernel 时间。
