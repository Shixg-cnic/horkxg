# 实验工具

- `test_sclp_small.py`：生成含自环、平行边、孤立点和断开分量的小图，独立检查
  hierarchy 的容量、覆盖、对称性和投影切边守恒。
- `test_sclp_repeatability.py`：同 seed 重复运行并比较 hierarchy SHA256。
- `verify_partition.py`：从原始 symmetric CSR 独立复算最终 cut、负载和容量。
- `run_gpu_lp_jet_compare.sh`：运行 Jet 原粗化与固定参数 SCLP + 同一 Jet 后半段。

固定参数是 `beta=256 / rounds=4 / two-hop=0.60`，脚本不提供参数搜索入口。
