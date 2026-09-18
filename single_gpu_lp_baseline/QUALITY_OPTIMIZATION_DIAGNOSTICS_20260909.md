# GPU LP 质量优化诊断（2026-09-09）

本轮目标是寻找跨图、跨份数的单层 GPU 标签传播质量改进。所有正式比较均使用
`products`、`com-LiveJournal`，`k=4,8,32`，`SEARCH_SEED=0`，最大负载比例
`1.10`，无最小分区点数。Jet 只作为历史质量参照，未用于初始化。

## 当前固定基线

```ini
ENABLE_FIELD=0
PAIR_EXCHANGE=0
BLOCK_LP=0
SEARCH_SEED=0
MAX_VERTEX_RATIO=1.10
GLOBAL_CYCLES=5
GROW_ROUNDS=30
REFINE_ROUNDS=50
BALANCE_ROUNDS=5
POLISH_ROUNDS=20
FEASIBLE_RECORDER=1
RESTORE_BEST_CYCLE=0
OSCILLATION_GUARD=0
MINIMAL_BALANCE_REPAIR=0
INCREMENTAL_CUT=1
CACHED_NEIGHBOR_COUNTS=1
```

六个案例的初始标签 FNV-1a 摘要与此前固定 seed=0 结果一致：

| 图 | k=4 | k=8 | k=32 |
| --- | ---: | ---: | ---: |
| products | 7167236614275041421 | 8731398766608079354 | 6011621667199981082 |
| com-LiveJournal | 17148409824319034585 | 15717862114113566433 | 13209025086608924007 |

## 基线结果

切边和容量由独立完整 CSR 重算；切边是无向切边。完整输出、日志和配置位于
[`experiments/results/quality_opt_20260909_baseline`](experiments/results/quality_opt_20260909_baseline)。

| 图 | k | 无向切边 | 切边比例 | 最大负载比 | 完整时间(s) | 相对历史 Jet 比例 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| products | 4 | 3,067,004 | 0.0495806 | 1.099998 | 0.215847 | +51.84% |
| products | 8 | 4,710,837 | 0.0761544 | 1.088026 | 0.300724 | +52.89% |
| products | 32 | 7,677,754 | 0.1241170 | 1.099995 | 0.633217 | +36.10% |
| com-LiveJournal | 4 | 4,505,868 | 0.1299225 | 1.099942 | 0.298268 | +24.82% |
| com-LiveJournal | 8 | 6,190,944 | 0.1785101 | 1.099998 | 0.418689 | +27.64% |
| com-LiveJournal | 32 | 10,059,155 | 0.2900464 | 1.099992 | 0.749692 | +41.46% |

Jet 比例来自历史独立报告 [`QUALITY_RESULTS_V2.md`](QUALITY_RESULTS_V2.md)，不是本轮
运行结果。六个输出均满足容量约束、标签范围和完整 CSR 切边校验。

## 诊断证据

在 CUDA 主程序中增加了有限的每轮字段：候选数、按目标容量配额数、容量淘汰数、
实际迁移数、单点评分和完整批次切边变化。没有新增算法阶段或无界数组。

基线五个全局周期的累计诊断如下。`bad` 是该算子完整批次切边上升的轮数；
`interaction` 是真实双向切边下降量减去 `2 * 已提交单点评分和`，只对 descent
具有直接单点收益含义，balance 的 score 是负载压力排序分数。

| 图 | k | balance 候选/容量淘汰 | descent 候选/容量淘汰 | descent bad/250 | descent interaction |
| --- | ---: | ---: | ---: | ---: | ---: |
| products | 4 | 12,100,591 / 4,971,871 | 4,700,828 / 2,631,558 | 28 | −11,895,664 |
| products | 8 | 16,659,649 / 9,265,938 | 7,822,548 / 5,107,879 | 33 | −29,018,982 |
| products | 32 | 20,424,402 / 12,877,948 | 7,039,409 / 4,109,497 | 36 | −13,217,826 |
| com-LiveJournal | 4 | 30,771,173 / 18,270,574 | 30,009,087 / 24,335,197 | 62 | −1,935,428 |
| com-LiveJournal | 8 | 39,797,796 / 27,217,461 | 87,782,165 / 81,520,510 | 16 | −3,263,650 |
| com-LiveJournal | 32 | 42,235,655 / 29,262,233 | 35,342,258 / 27,200,075 | 39 | −28,500,468 |

这回答了首个质量损失问题：普通 descent 不是没有正收益候选，而是大量候选受目标容量配额淘汰，
保留下来的同步批次仍会发生负交互；只看单点评分会高估批次收益。balance 每案都有 12–43
百万候选，其中约 41%–71% 被容量配额淘汰，且 balance 可能显著增加切边，随后再由 descent 修复。

## 单假设实验

每个实验都从固定基线代码状态开始，仍为六案、seed=0、五周期；没有串联 field、pair 或 block。

1. **全部相邻候选冲突过滤。** 直接复用现有 polish 的保守过滤作用于普通 descent。六案全部变差，
   products-k4 从 3,067,004 增至 3,852,901；假设失败，已撤回。
2. **仅过滤负交互边。** 保留同源同目标的正协同，只消除精确二点交互为负的相邻候选。只有
   LiveJournal-k32 略有改善，其余五案变差；仍不稳定，已撤回。
3. **所有真实变差 descent 批次回滚。** CUDA 回滚迁移标签、负载和邻居计数，只保留完整切边不升的
   descent 批次。第一版 4/6 略有改善，但 products-k8/k32 变差；限制为“只在 descent 起点已可行”
   后仍只有 LiveJournal-k4/k8 略有改善，products-k8/k32 和 LiveJournal-k32 变差，未保留。
4. **最小超载修复（已有消融）。** 仅允许从超载源区迁出。六案全部明显变差且更慢，不进入主线。

上述尝试的临时日志目录为：

```text
/tmp/lp_interaction_filter_20260908
/tmp/lp_conflict_descent_20260908
/tmp/lp_rollback_worsening_20260909
/tmp/lp_rollback_feasible_20260909
/tmp/lp_min_balance_20260909
```

各失败变体的独立最终无向切边如下；它们都使用同一初始标签摘要。

| 图/k | baseline | all-adj filter | negative-edge filter | rollback-all | rollback-feasible | minimal-balance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| products/4 | 3,067,004 | 3,852,901 | 3,104,834 | 3,060,093 | 3,067,193 | 4,130,376 |
| products/8 | 4,710,837 | 5,697,481 | 4,979,777 | 4,732,914 | 4,822,798 | 7,455,636 |
| products/32 | 7,677,754 | 9,757,410 | 7,847,871 | 7,702,450 | 7,710,352 | 9,699,525 |
| com-LiveJournal/4 | 4,505,868 | 5,383,902 | 4,672,001 | 4,504,510 | 4,499,408 | 5,922,325 |
| com-LiveJournal/8 | 6,190,944 | 6,892,840 | 6,470,868 | 6,180,553 | 6,186,988 | 7,680,106 |
| com-LiveJournal/32 | 10,059,155 | 10,382,477 | 10,003,976 | 10,057,872 | 10,070,787 | 11,341,773 |

## 结论

当前没有足够证据支持新的质量主算法。已恢复并确认固定 baseline 为当前最佳主线；新增代码只保留
有限诊断输出，不保留失败的过滤、回滚或容量分支。下一步若继续研究，应针对“带容量净流的候选
调度”提出一个单独、可验证的假设，而不是继续增加后处理或循环次数。

可复现实验入口：
[`experiments/run_quality_diagnostics.sh`](experiments/run_quality_diagnostics.sh)。例如：

```bash
bash experiments/run_quality_diagnostics.sh \
  experiments/results/quality_opt_20260909_baseline
```

脚本拒绝覆盖已有输出；共享 GPU 时通过 `CUDA_VISIBLE_DEVICES` 指定设备，不终止其他任务。
