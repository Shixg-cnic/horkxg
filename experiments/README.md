# 实验工具

- `test_basc_small.py`：生成带自环、平行边、孤立点和断开分量的小图，检查
  BASC K=1/2/4 的层次合法性和随机投影切边守恒。
- `verify_partition.py`：从原始 Symmetric CSR 独立重算无向切边、负载和容量。
- `run_gpu_lp_jet_compare.sh`：运行 Jet 原版粗化 A，以及自研层次加 Jet 后半段 B。
- `summarize_gpu_lp_jet.py`：汇总既有 LP 对照目录；BASC 消融的原始日志保留在
  `build-gh200/experiments/results/`。

脚本假定本项目、`dataset` 和 `sotas` 是 `partition` 下的兄弟目录。
