# SCLP 迁移语义修正：k=4 首轮对照

## 修改假设

保持 `SCLP_BETA=64`、`SCLP_ROUNDS<=4`、精确 GPU affinity、segmented prefix
admission 和 GPU contraction 不变，只修正五项更新语义：

1. 计算 `gain=A_best_receiver-A_current`，仅允许严格正 gain；
2. affinity 平局使用 `(vertex,target,seed,level,round)` 的确定性 hash；
3. 每批 cluster 分为 mover/receiver，仅允许 mover→receiver；
4. admission 从 raw affinity 排序改成 raw gain 排序；
5. 普通 LP 在约 0.5--0.6 收缩停止，two-hop 改为 pair-only，并受补到 0.5 的
   全局 merge budget 限制。

调试模式在每个 LP batch 前后完整重算当前 cluster cut。products、LiveJournal
及小图的所有 batch 都满足实际 cut 不增加；自环从 current affinity 中排除。

## 首层行为变化

| 图 | 旧 SCLP 首层比例 | 正 gain LP 后 | pair fallback 后 |
|---|---:|---:|---:|
| products | 0.1720 | 0.5289 | 0.5000 |
| com-LiveJournal | 0.2612 | 0.5706 | 0.5000 |

products 首轮 LP 预测 gain 1,153,698，完整重算实际 gain 1,281,328；LiveJournal
分别为 1,716,798 和 1,863,688。receiver 固定后没有出现旧版的 reciprocal label
交换，也没有再发生单层 4--6 倍的过度收缩。

## 固定 Jet 后半段质量

切边比例以原图无向边数为分母。Jet 原粗化数字沿用同配置 seed=0 对照。

| 图 | 旧 SCLP cut | 语义修正版 cut | 相对旧版改善 | Jet 原粗化 | 修正版距 Jet |
|---|---:|---:|---:|---:|---:|
| products | 2,218,136 | 2,179,926 | 1.72% | 2,158,447 | +1.00% |
| com-LiveJournal | 4,166,454 | 3,544,983 | 14.92% | 3,473,755 | +2.05% |

对应切边比例为 products 3.52402%、LiveJournal 10.22163%；最终最大负载比分别
为 1.049208 和 1.100000，均满足 Jet 配置的 1.10 上限规则。

## 层次和时间

products 保留 7 层，点数为：

```text
2,449,029 → 1,224,515 → 612,258 → 306,129 → 153,065 → 76,855 → 60,530
```

下一候选层比例 0.925，按 0.90 停止规则拒绝。GPU aggregate+contraction 核心
4.09 秒，层次导出 28.12 秒，粗化总时间 33.21 秒；Jet load/init/refinement
约 1.41/0.087/0.056 秒。

LiveJournal 保留 16 层，前十层基本按 0.5 缩小，之后在正 gain、容量或可配对
singleton 不足时自然放缓，最粗保留层 431 点。下一候选层比例 0.923，未写入。
GPU 核心 3.60 秒，层次导出 21.02 秒，粗化总时间 25.28 秒；Jet
load/init/refinement 约 1.15/0.049/0.226 秒。

产物目录：

```text
build-gh200/experiments/results/sclp_semantics_v2/products/k4/seed0/
build-gh200/experiments/results/sclp_semantics_v2/com-LiveJournal/k4/seed0/
```

## 正确性与重复性

- 两张大图每个保留层均通过 map、点权、容量、CSR 和 weighted projection cut
  conservation 检查。
- kim2 非平凡第一层从 456,976 收缩到 228,488，三次 hierarchy SHA256 均为
  `c9a959d9c4c9cf53e0871c4b4521b297864cbec2d55e013dd504fb1d9e519a50`。
- kim2 旧版四轮接受 1,827,904 次却只收缩 2.8%；修正版一轮 LP 加 291 个
  two-hop pair 即准确到 0.5，直接验证 mover/receiver 消除了主要交换问题。

重复性产物：
`build-gh200/experiments/results/sclp_semantics_v2_repeatability_kim2/`。

## 结论

结果支持“主要损失来自迁移语义而非 beta 参数”的假设，尤其 LiveJournal 的 14.9%
改善非常显著。不过当前两个案例仍分别落后 Jet 约 1.0% 和 2.1%，尚不能宣称替代
Jet 粗化，也没有理由立即加入更多评分机制。本轮没有扫描 beta、rounds 或其他
参数；下一步应先分析剩余差距出现在哪些层，再决定是否值得继续。
