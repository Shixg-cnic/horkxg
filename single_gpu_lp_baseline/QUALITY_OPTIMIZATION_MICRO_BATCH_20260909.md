# Descent 微批次质量实验（2026-09-09）

## 假设与实现

固定无场基线后，日志显示普通 descent 的单点评分和同步批次真实切边变化有明显差距。
本轮只改变 descent 的提交调度：每个目标分区每次最多提交 256 个候选，提交后立即维护
邻居标签计数，再重新生成下一批候选；每个原 descent 轮最多重复 8 次。field、balance、
初始化、最终 polish 和容量上限没有改变，`PAIR_EXCHANGE=0`、`BLOCK_LP=0`。

开关为：

```ini
DESCENT_MICRO_BATCH=0       # 默认关闭，保持基线
DESCENT_MICRO_ROUNDS=8
```

`DESCENT_MICRO_BATCH=256` 是本轮唯一实验配置。微批次使用 GPU 候选/提交和现有邻居计数
缓存；每个微批次的精确增量切边先在主程序中累计，`INCREMENTAL_CUT_VERIFY=1` 时再用
完整 CSR 核验。没有新增 CPU 划分器或后处理器。

## 三 seed 全范围结果

每个单元是 seed=0,1,2 的平均无向切边；时间是完整主程序搜索时间，不含读图和写盘。
相对切边为 `(micro / baseline - 1) * 100%`，负值表示改善。

| 图 | k | 原始 LP 切边 | 微批次切边 | 相对切边 | 原始时间(s) | 微批次时间(s) | 时间倍数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| products | 2 | 1,652,134 | 1,624,695 | -1.66% | 0.169 | 0.851 | 5.0x |
| products | 4 | 2,944,887 | 2,786,187 | -5.39% | 0.224 | 0.939 | 4.2x |
| products | 8 | 4,624,749 | 4,605,057 | -0.43% | 0.300 | 0.991 | 3.3x |
| products | 16 | 6,248,900 | 6,042,994 | -3.30% | 0.409 | 1.081 | 2.6x |
| products | 32 | 7,806,956 | 7,689,626 | -1.50% | 0.643 | 1.976 | 3.1x |
| com-LiveJournal | 2 | 2,549,114 | 2,656,919 | +4.23% | 0.262 | 2.335 | 8.9x |
| com-LiveJournal | 4 | 4,467,315 | 4,598,718 | +2.94% | 0.320 | 2.777 | 8.7x |
| com-LiveJournal | 8 | 6,546,769 | 6,366,115 | -2.76% | 0.410 | 2.674 | 6.5x |
| com-LiveJournal | 16 | 8,245,608 | 7,867,023 | -4.59% | 0.502 | 2.279 | 4.5x |
| com-LiveJournal | 32 | 10,117,131 | 9,554,096 | -5.57% | 0.762 | 2.911 | 3.8x |

products 五个 k 都是平均改善；LiveJournal 的 k=8/16/32 稳定改善，k=2/4 稳定退化。
因此这是一项有条件的质量改进，不是对所有图和份数都安全的默认替换。seed 级别结果、
日志和完整输出由实验脚本生成；本轮临时运行目录为：

```text
/tmp/lp_baseline_seeds_20260909
/tmp/lp_descent_micro_seeds_20260909
```

可复现实验入口：

```bash
bash experiments/run_descent_micro_experiment.sh \
  experiments/results/quality_opt_20260909_descent_micro
```

脚本会拒绝覆盖已有结果，运行 products/com-LiveJournal 的 k=2,4,8,16,32，seed=0,1,2，
分别产生原始 LP 和微批次结果，并调用 `verify_quality.py` 独立核验标签、容量和完整 CSR
切边。

## 正确性与决策

- `test_descent_micro.py` 通过；小图开启 `INCREMENTAL_CUT_VERIFY=1` 时，程序切边与完整
  CSR 重算一致。
- field、block、pair、外部初始化回归测试均通过。
- 关闭 `DESCENT_MICRO_BATCH` 时，products/k=4 输出与固定基线逐字一致。
- 所有 30 个实验输出均通过完整 CSR 容量和标签范围检查。

本轮保留微批次作为“质量优先候选”，但默认仍为 0，因为 LiveJournal 的 k=2/4 会退化，
且时间增加约 2.6–8.9 倍。下一步若继续，应先研究为什么低 k 的 LiveJournal 需要较大的
同步批次，再决定是否设计一个有证据支撑的调度规则；不把本轮结果包装为全域质量质变，
也不叠加 pair、block 或 CPU 后处理。
