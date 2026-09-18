# GPU size-constrained LP 粗化：第一版结果

> 本文保留最初“强制异 cluster 迁移”版本的历史结果。正 gain、hash tie、
> mover/receiver 和 pair-only two-hop 修正版见
> `SCLP_SEMANTICS_V2_20260909.md`；当前代码行为以修正版为准。

## 配置和实现

实现代码位于 sibling `partition/homework`，实验产物保存在本目录。新增独立
`coarsen_method=sclp`，没有复用 BASC/frontier 的候选规则。固定配置：

```text
parts=4
seed=0
SCLP_BETA=64
SCLP_ROUNDS=4
per-level target contraction ratio=0.5
hierarchy stop contraction ratio=0.9
maximum load ratio for Jet=1.10
```

每层 singleton 初始化；精确计算加权邻接 cluster affinity；proposal 按
`(target,-affinity,vertex)` 排序并以 segmented prefix sum 做确定性容量准入；
同步提交。低、中、高度点分别由 thread-per-vertex、warp-per-vertex 和
CTA-per-vertex 生成 affinity 键，GPU sort/reduce 完成精确聚合。若 4 轮仍未达到
0.5 收缩率，则对拥有相同 favorite 的 singleton 做一次简单 two-hop grouping。
粗图构建沿用 device-resident GPU contraction。

## 正确性和重复性

- 严格小图通过映射覆盖、粗点权重、容量、加权 CSR 双向一致、平行边合并、自环
  删除和随机粗标签投影切边守恒检查。
- products 和 LiveJournal 的每个实际保留层均输出
  `map=ok weights=ok csr=ok projection_cut=ok`。
- kim2 使用 `stop_ratio=1.0,max_levels=1` 强制导出非平凡第一层，三次 hierarchy
  SHA256 均为
  `0b049cfc7ca77de308cdb043b18eaf15eaaa993b64f0dec2db66c612d1273633`。
  该层从 456,976 点收缩到 444,217 点，证明比较的不是仅含原图的空 hierarchy。

重复性产物：
`build-gh200/experiments/results/sclp_repeatability_kim2_one_level/`。

## k=4、seed=0 质量结果

两条 SCLP hierarchy 都交给同一个 `jet_gpu_lp_import`，最终切边由 Jet 在原图层
完整计算。切边比例统一除以原图无向边数。

| 图 | Jet 原粗化 cut | SCLP cut | SCLP 比例 | 相对 Jet | 最大负载比 |
|---|---:|---:|---:|---:|---:|
| products | 2,158,447 | 2,218,136 | 3.58579% | +2.77% | 1.099999 |
| com-LiveJournal | 3,473,755 | 4,166,454 | 12.0136% | +19.94% | 1.100000 |

### products 层次

| level | fine | proposed coarse | ratio | LP accepted | capacity rejected | singleton | two-hop merged |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2,449,029 | 421,262 | 0.1720 | 2,400,608 | 0 | 245,680 | 0 |
| 1 | 421,262 | 135,912 | 0.3226 | 369,149 | 3,692 | 101,254 | 0 |
| 2 | 135,912 | 76,697 | 0.5643 | 314,524 | 31,730 | 70,406 | 3,415 |

下一候选层为 75,326/76,697=0.9821，按 0.9 停止规则未写入 hierarchy。最终
保留 4 层，GPU aggregate+contraction 核心 2.42 秒，hierarchy 导出 15.58 秒，
粗化总时间 18.81 秒；Jet hierarchy load/init/refinement 约
0.79/0.16/0.89 秒。

产物：`build-gh200/experiments/results/sclp_k4_seed0/products/`。

### LiveJournal 层次

| level | fine | proposed coarse | ratio | LP accepted | capacity rejected | singleton | two-hop merged |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3,997,962 | 1,044,288 | 0.2612 | 3,997,962 | 0 | 503,389 | 0 |
| 1 | 1,044,288 | 308,659 | 0.2956 | 1,034,643 | 9,645 | 178,359 | 0 |
| 2 | 308,659 | 130,389 | 0.4224 | 290,909 | 17,750 | 92,778 | 0 |
| 3 | 130,389 | 65,646 | 0.5035 | 421,979 | 79,827 | 54,705 | 9,962 |

下一候选层比例为 0.9599，未保留。最终 5 层，GPU 核心 2.38 秒，导出 12.50
秒，粗化总时间 15.41 秒；Jet load/init/refinement 约 0.65/0.38/0.28 秒。

产物：`build-gh200/experiments/results/sclp_k4_seed0/com-LiveJournal/`。

## 结论

第一版已经满足 device-resident、确定性 batched admission、严格容量和 weighted
cut conservation 要求。在两张目标幂律图上，前几层收缩率稳定且无需 two-hop；
但最终质量没有超过 Jet 原粗化，LiveJournal 明显退化，因此暂不扩展到其他 k 或
多 seed，也不把它替换成主后端。

kim2 的诊断还显示：同步 singleton LP 可能出现大量互换/环迁移——四轮 proposal
全部接受但只收缩 2.8%。这不是 atomic capacity contention，而是同步迁移本身的
结构问题。后续若继续，应先讨论如何确定性地打破这类迁移环，以及为什么
LiveJournal 的早期聚合损伤后半段恢复；不应先添加 hot/cold、confidence 或不断
调整 beta。
