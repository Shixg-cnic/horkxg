# 结果校验与基线对照

`verify_partition.py` 从原始对称 CSR 独立复算切边、分区负载、容量和标签范围。
本项目输出为二进制 int32 标签，Jet 文本输出使用 `--partition-format text`。

```bash
python3 tools/verify_partition.py \
  --indptr indptr.bin --indices indices.bin --parts 4 --partition result.part
```

`compare_jet.py` 交替运行当前完整划分与 Jet，每个配置使用独立进程，匹配
严格整数容量，记录二进制 SHA256、全部 cut/时间和范围。计时结束后再进行
CPU 独立校验，结果写入被忽略的 `results/`。工具需要用户提供已构建的 Jet，
主项目编译不依赖 Jet。

```bash
CUDA_VISIBLE_DEVICES=0 python3 tools/compare_jet.py \
  --indptr ../dataset/Gpartition_dataset/Sym_CSR/products/products_sym_indptr.bin \
  --indices ../dataset/Gpartition_dataset/Sym_CSR/products/products_sym_indices.bin \
  --jet-graph ../dataset/Gpartition_dataset/Sym_CSR/products/products_sym.metis \
  --jet ../sotas/.build-deps/jet-5090/build-jet/app/jet \
  --output results/products-jet --parts 4 8 --repeats 5
```

比较使用 `algorithm_total_seconds`，排除加载、输入上传和最终输出。
研究目标是质量接近或优于 Jet，同时缩短完整算法时间；已记录的结果见
[算法对比](../ALGORITHM_COMPARISON.md)，不能以单次最差 Jet 输出作为基准。
