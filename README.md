# GPU-LP 多层次划分实验项目

这是从 `single_gpu_lp_baseline` 独立抽出的多层次研究项目，当前只包含：

```text
输入 Symmetric CSR
    -> GPU-LP / BASC 聚类粗化
    -> 加权 CSR 层次
    -> Jet 原生最粗层初始化、投影和 refinement（外部项目）
```

它不包含原来的单层 LP 主程序、结构场、两点交换、连通块实验或 85G 构建
目录。Jet 源码仍使用同级目录中的
`../sotas/Jet-Partitioner`，数据使用 `../dataset/Gpartition_dataset`。

## 构建

```bash
cd /cnic/work/shixg/GNN/partition/homework
PATH=/usr/local/cuda/bin:$PATH cmake -S . -B build-gh200 \
  -DCMAKE_BUILD_TYPE=Release
PATH=/usr/local/cuda/bin:$PATH cmake --build build-gh200 --target multilevel_lp -j2
```

运行小图严格校验：

```bash
GPU_LP_STRICT_VERIFY=1 python3 experiments/test_basc_small.py
```

## 多层次粗化入口

```text
multilevel_lp indptr.bin indices.bin parts hierarchy.out \
  [max_vertex_ratio] [seed] [stop_contraction_ratio] \
  [coarsen_method=lp|basc] [basc_k=1|2|4] [max_levels]
```

默认 `coarsen_method=lp`，BASC 是可选研究后端。例如：

```bash
build-gh200/multilevel_lp \
  ../dataset/Gpartition_dataset/Sym_CSR/products/products_sym_indptr.bin \
  ../dataset/Gpartition_dataset/Sym_CSR/products/products_sym_indices.bin \
  4 /tmp/products_k4_basc.hierarchy 1.10 0 0.85 basc 2 24
```

完整 Jet 对照入口见
[`experiments/run_gpu_lp_jet_compare.sh`](experiments/run_gpu_lp_jet_compare.sh)。
它只把完整层次交给 Jet importer，不把自研最粗层初始化或自研 polish 混入
后半段。

## 当前结论

接口、层次映射、权重守恒、粗图边权合并、随机投影切边守恒和最终容量均已
通过小图及大图独立校验。BASC K=1/2/4 的首轮六案例消融尚未显示跨图、跨
份数的稳定质量提升，因此默认主线保持 `lp`，不把 BASC 宣称为已成立的
算法创新。实验细节见
[`GPU_LP_JET_COARSEN_EXPERIMENT_20260909.md`](GPU_LP_JET_COARSEN_EXPERIMENT_20260909.md)。

每次修改都应先执行构建和小图校验，再在 Git 中提交；远程 GitHub 仓库地址
由项目维护者配置到本仓库的 `origin`。
