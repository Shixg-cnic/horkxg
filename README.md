# GPU 多层图划分

当前唯一划分流程：**GPU SCLP 粗化 → GPU 递归加权谱二分初始化 → GPU 增量分桶细化**。
支持 `2 ≤ k ≤ 32`，默认最大负载比 `1.03`。主程序不链接 METIS、Jet 或 Kokkos。
递归任务和阶段调度由 CPU 控制，图、标签及主要计算保留在 GPU 上。

## 代码入口

| 位置 | 内容 |
|---|---|
| `app/partition_main.cu` | 完整划分入口与阶段计时 |
| `src/coarsen.cu` | SCLP 聚合、容量准入、two-hop、CSR 收缩 |
| `src/initial_partition_gpu.cu` | 固定 64 步扩散、加权递归二分 |
| `src/refine.cu` | 细化工作区、层次投影和执行调度 |
| `src/refine_incremental.cuh` | 迁移候选、优先级修正、增量更新、双侧分桶再平衡 |
| `include/` | 图类型、GPU 层次、容量约束和接口 |
| `tests/` | CTest 及合成图正确性测试 |
| `tools/` | 最终结果校验和 Jet 对照工具 |
| `graph_untils/` | 数据转换、对称化和划分分析工具 |

算法细节见 [ALGORITHM.md](ALGORITHM.md)，三个阶段与 Jet / DAC G-kway 的对比及
已记录的实验结果见 [ALGORITHM_COMPARISON.md](ALGORITHM_COMPARISON.md)。

## 编译

需要 CMake ≥ 3.24、C++17 和支持目标 GPU 的 CUDA 编译器。RTX 5090 示例：

```bash
cmake -S . -B build-5090 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build-5090 -j6
CUDA_VISIBLE_DEVICES=0 ctest --test-dir build-5090 --output-on-failure
```

未指定架构时使用 `native`。算法库由同宽度的驱动和测试复用。
构建产物只留在本地，`build*/`、对象文件、静态/动态库、数据和结果目录均由
[.gitignore](.gitignore) 排除。

## 使用

```bash
CUDA_VISIBLE_DEVICES=0 build-5090/gpart_partition_standard \
  indptr.bin indices.bin 4 partition.part
python3 tools/verify_partition.py \
  --indptr indptr.bin --indices indices.bin --parts 4 --partition partition.part
```

参数：`indptr indices k output [imbalance_ratio=1.03] [seed=0] [--verify]`。
`--verify` 启用内部层次、投影、cut 和增量缓存校验；性能测试关闭该开关。

输入为无向对称的二进制 CSR，offsets 和邻点 ID 均为本机字节序 `int64`，
输入顶点和边权重均为一。输出为每顶点一个 `int32` 分区 ID。
`standard` 在 GPU 上使用 32 位 offsets/权重，`gpart_partition_big` 使用 64 位
offsets/权重；两者的顶点 ID 和输出格式一致。
容量为 `floor(imbalance_ratio * total_vertex_weight / k)`。
容量不可行或谱排序无法产生可行二分时，程序明确报错。

`algorithm_total_seconds` 包含粗化、初始化、投影细化及层次清理，排除加载、
输入准备/上传和最终下载/文件输出；`wall_total_seconds` 包含完整进程工作。

## 验证与分析

```bash
CUDA_VISIBLE_DEVICES=0 python3 tests/synthetic_suite.py --verify
CUDA_VISIBLE_DEVICES=0 python3 tests/test_sclp_small.py
cmake -S graph_untils -B graph_untils/build
cmake --build graph_untils/build -j6
```

合成套件默认测试三类图、`k=2,4,8,32`、standard/big、每配置两次，独立复算 cut
和负载。生成数据与日志写到被忽略的 `results/`。细化桶内原子排名可能造成同 seed
结果变化，重复测试检查正确性，不要求标签逐字节一致。

`gpart_coarsen_standard` / `gpart_coarsen_big` 仅用于粗化检查和层次导出；
它们不是另一套完整划分算法。参数为
`indptr indices k hierarchy [seed=0] [stop_ratio=0.90] [max_levels=24] [--verify]`。
数据工具说明见 [graph_untils/README.md](graph_untils/README.md)，真实图对照见
[tools/README.md](tools/README.md)。
