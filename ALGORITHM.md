# 当前 GPU 图划分算法

更新日期：2026-10-09。仓库只保留当前主线；与 Jet / DAC G-kway 的阶段对比、
方法来源和已有实验结果见 [ALGORITHM_COMPARISON.md](ALGORITHM_COMPARISON.md)。

## 执行入口

`app/partition_main.cu` 的调用顺序为：

```text
load_weighted_graph / make_device_weighted
  → coarsen                      src/coarsen.cu
  → initial_partition_gpu        src/initial_partition_gpu.cu
  → uncoarsen                    src/uncoarsen.cu
    → refine_hierarchy_device    src/refine.cu
      → run_incremental_refinement  src/refine_incremental.cuh
```

输入准备后，各层 CSR、投影映射和划分标签保留在 GPU 上。
CPU 调度层次及初始化的递归任务；主程序不调用 METIS 或 Jet 的算法。
`app/coarsen_main.cu` 只导出粗化层次，供独立检查使用。

## 粗化

生产入口固定使用 `beta=max(64,ceil(512/k))`，每层最多 4 轮 SCLP。
簇容量为 `max(最大顶点重量,ceil(total_weight/(beta*k)))`；
粗图 cutoff 为 `max(32,beta*k)`。库接口允许指定簇容量系数，默认 256，
独立粗化检查使用该库默认值；完整划分入口明确设置上述生产参数。

每轮冻结簇标签，累计顶点到邻簇的连接权重。低度点通过 warp `match_any`
归并邻簇，中度点使用 shared-memory hash，高度点使用紧凑边集的排序归约。
哈希角色将簇分成 mover/receiver，选取进入 receiver 的正收益迁移。
候选按目标簇、收益及 tie 排序，通过重量前缀限制目标容量。

排序且无重复邻点的 CSR，每层首轮的邻簇均为单点，直接按边权选择目标，
保持相同的 tie 和容量规则。未排序或有平行边时使用完整 affinity 路径。

LP 后，零度顶点按重量前缀放入容量 bin，完整落在同一 bin 的顶点合并，
跨边界顶点单独保留。剩余簇比例大于 0.60 时，two-hop 将共享 favorite 的
单点配对，受容量和合并预算限制。
收缩时丢弃簇内边，排序归约重复粗边并生成加权 CSR。

## 初始化分

最粗图采用固定 64 步 lazy diffusion 的 GPU 递归谱二分：
`y=(x+D⁻¹Ax)/2`。每个任务按剩余分区数分配左右目标重量。
哈希向量由固定 seed 启动，每轮按子图加权度计算均值和方差，移除常数分量
并归一化；零度点使用质量一，保持自己的扩散值。

每个 warp 处理一个顶点，扩散同时归约下一轮 moments，度数在任务内复用。
GPU 排序谱坐标，以顶点 ID 打破平局，再按顶点重量前缀选择目标附近的边界。
检查左右容量后递归，最终在 GPU 累计负载并下载检查。
这是近似谱方向，没有调用特征值求解器或线性方程求解器。

固定步数对低谱隙图可能不足，加权顶点也可能让当前排序找不到可行边界。
此时明确报错，不自动回退到 CPU 初始化。

## 投影与细化

先细化最粗图，再逐层投影。顶点数达到上次细化图的两倍，或到达原图时，
才重新建立缓存并细化；其余层只投影，保持 cut 和分区重量。

细化使用 `n*k` 的 uint64 邻接分区权重缓存。从缓存生成迁移候选，允许有限
负收益探索：无权边图阈值 0.25，加权边图阈值 0.75。
按初始收益和顶点 ID 的优先级修正相邻提案收益，允许相邻顶点一起迁移。
扫描实际迁移点的邻边，融合更新邻点缓存和整批精确 cut；随后提交标签和负载。

探索允许临时超载。再平衡按收益/顶点重量分入 50 个桶、每桶 16 个 section，
先限制超载源的迁出重量，再按目标余量接纳整个顶点。
一次修复只从超载源移到有余量的目标；无法修复时恢复最佳可行 checkpoint。

每层最多 32 个 LP 阶段，连续 3 个阶段没有显著可行改进时停止。
显著改进阈值为 0.1%；最终恢复该层切边最少且满足容量的状态。
探索、优先级修正、增量缓存、收益桶及最佳状态保留借鉴 Jet，
双侧准入、层次选择及缓存/cut 融合是本项目的具体实现改动。

## 容量、验证与整理记录

初始化和细化共用 `include/partition_capacity.hpp`，严格容量为
`floor(alpha*total_weight/k)`，默认 `alpha=1.03`。
分桶原子排名受 GPU 调度影响，因此同 seed 的 cut 可能有小幅波动。
`--verify` 独立复算每批 cut、重建 affinity 缓存，并检查层次和投影守恒。
构建和运行方式见 [README.md](README.md)。

2026-10-09 整理：删除旧正收益 LP/pair 细化、排序式再平衡、未采用的 swap、
多方向谱扩散、MinHash 排序等试验分支及旧消融目录；当前生产规则保持上述版本。
细化接口默认参数统一为生产配置，日志只报告实际执行的细化阶段。
清理前性能数据在对比文档中标为参考，本次整理不宣称提升质量或速度。
