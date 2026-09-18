# graph_untils

Generate a small random graph in the same binary CSR layout used by `kim2`.

For a graph named `toy`, the tool writes four files:

- `toy_coo.txt`: readable COO text, one edge per line: `src dst value`
- `toy_indptr.bin`: CSR row pointer, `int64_t`, length `num_vertices + 1`
- `toy_indices.bin`: CSR column indices, `int64_t`, length `num_edges`
- `toy_values.bin`: CSR edge values, `float`, length `num_edges`

`--edges` means CSR adjacency entries. By default, the graph is directed and
contains no duplicate edges and no self loops.

## Build

```bash
cmake -S graph_untils -B graph_untils/build
cmake --build graph_untils/build
```

## Example

```bash
./graph_untils/build/generate_graph \
  --vertices 4 \
  --edges 8 \
  --name v4_e8 \
  --output-dir ../../data/v4_e8 \
  --seed 1
```

The generated files can be fed into the single-card partition prototype:

```bash
single_gpu_csr/build/cpu_weighted_lp \
  --indptr graph_untils/out/toy_indptr.bin \
  --indices graph_untils/out/toy_indices.bin \
  --parts 2 \
  --iters 4
```

## Analyze a partition result

`analyze_partition` reads one partition result txt file and the original CSR graph,
then reports a compact graph partition quality table.

The partition txt format is one part id per line, matching vertex ids `0..N-1`.
For the original graph, pass either `--graph-dir DIR --name NAME` for the usual
`NAME_indptr.bin`, `NAME_indices.bin`, and optional `NAME_values.bin` files, or
pass the CSR files explicitly.

```bash
./graph_untils/build/analyze_partition \
  data/kim2/kim2_4_cpu.txt \
  --graph-dir data/kim2 \
  --name kim2 \
  --parts 4
```

You can analyze multiple partition files for the same graph in one run:

```bash
./graph_untils/build/analyze_partition \
  data/kim2/kim2_4_cpu.txt \
  data/kim2/kim2_4_gpu.txt \
  --graph-dir data/kim2 \
  --name kim2 \
  --parts 4
```

When multiple partition files are provided, the tool prints one comparison table
with one row per partition file.

Explicit CSR paths also work:

```bash
./graph_untils/build/analyze_partition \
  --partition data/kim2/kim2_4_cpu.txt \
  --partition data/kim2/kim2_4_gpu.txt \
  --indptr data/kim2/kim2_indptr.bin \
  --indices data/kim2/kim2_indices.bin \
  --values data/kim2/kim2_values.bin \
  --parts 4
```

Default reported metrics:

- `edge_cut`: whole-graph cut edge count. For directed CSR, every cross-part adjacency entry is counted once.
- `edge_cut_ratio`: `edge_cut / directed_edges`; lower is better.
- `vertex_imb`: max part vertex count divided by the average vertex count; closer to `1.0` is better.
- `edge_imb`: max part edge-set size divided by the average edge-set size; closer to `1.0` is better. `|E(pi)|` follows Slota et al. Section 2.1 and counts edges with at least one endpoint in the part.
- `max_local_cut`: largest per-part cut edge count, matching `max_i |C(G, pi_i)|`; lower is better.

Use `--balance-ratio` and `--edge-balance-ratio` to set allowed imbalance thresholds.
Use `--detail` to print per-part debug metrics such as internal edges,
incoming/outgoing cut edges, weighted cuts, and per-part deviations.

## Convert original CSR to processed symmetric CSR

`csr_symmetrize` converts an int64 CSR graph into an unweighted symmetric
undirected CSR graph for partitioners that assume undirected adjacency. It
removes self-loops, maps each directed edge `u -> v` to the undirected key
`{min(u,v), max(u,v)}`, deduplicates those keys, and writes both adjacency
directions in CSR form.

The dataset naming convention is:

- `dataset/NAME/NAME_{indptr,indices}.bin`: original CSR;
- `dataset/process_data/NAME/NAME_{indptr,indices}.bin`: processed symmetric CSR;
- `dataset/process_data/NAME/NAME.metis`: undirected METIS input.

No `_sym` suffix is used. For an input directory following `dataset/NAME`, the
processed output directory and output name are inferred automatically:

```bash
./graph_untils/build/csr_symmetrize \
  data/kim2 \
  --memory-mb 4096
```

This writes `data/process_data/kim2/kim2_indptr.bin` and
`data/process_data/kim2/kim2_indices.bin`.

For large graphs, the converter uses external sort runs instead of keeping all
edges in memory. The main memory knob is `--memory-mb`; temporary files are
written under `--tmp-dir` or `OUTPUT_DIR/.csr_symmetrize_tmp`.

Explicit CSR paths also work:

```bash
./graph_untils/build/csr_symmetrize \
  --indptr data/kim2/kim2_indptr.bin \
  --indices data/kim2/kim2_indices.bin \
  --output-dir data/process_data/kim2 \
  --output-name kim2 \
  --tmp-dir /scratch/kim2_tmp \
  --memory-mb 8192
```

Output files:

- `kim2_indptr.bin`: `int64_t`, length `num_vertices + 1`
- `kim2_indices.bin`: `int64_t`, length `2 * unique_undirected_edges`

For very large graphs, put `--tmp-dir` on a fast local SSD or scratch filesystem
with enough free space. Temporary data can be several times the final CSR size
because the tool sorts undirected and directed edge runs.


## Convert CSR to METIS adjacency

`csr_to_metis` converts the binary CSR directory layout into a 1-indexed
undirected METIS adjacency file for PuLP/XtraPuLP style baselines.

The converter always treats every CSR entry `u -> v` as an undirected candidate
edge `{min(u,v), max(u,v)}`. It removes self-loops, deduplicates repeated or
reciprocal edges, and writes symmetric METIS adjacency. The METIS header edge
count is therefore the number of unique undirected non-self-loop edges.

For a processed directory such as `data/process_data/papers100M` containing:

- `papers100M_indptr.bin`
- `papers100M_indices.bin`
- `papers100M_values.bin` optional, ignored by this unweighted converter

run:

```bash
./graph_untils/build/csr_to_metis data/process_data/papers100M
```

The graph name defaults to the directory basename. You can also pass explicit CSR paths:

```bash
./graph_untils/build/csr_to_metis \
  --indptr data/process_data/papers100M/papers100M_indptr.bin \
  --indices data/process_data/papers100M/papers100M_indices.bin \
  --output data/process_data/papers100M/papers100M.metis
```

This default is correct for all three common inputs:

- symmetric undirected CSR, where reciprocal entries are deduplicated;
- one-sided storage of an undirected graph, where each input edge becomes one
  METIS edge;
- general directed CSR, where the baseline graph is the undirected projection.

## Convert SNAP and Matrix Market edge lists to CSR

`edge_list_to_csr` converts plain or gzip-compressed SNAP edge lists and
Matrix Market coordinate files into the repository's int64 CSR layout. It uses
two input scans and writes the large indices array through a memory-mapped file,
so it does not retain all edge pairs in memory.

For a SNAP file whose identifiers are not contiguous, use `--remap-ids`. The
additional `NAME_vertex_ids.bin` file maps each new dense vertex ID back to its
original ID.

```bash
./graph_untils/build/edge_list_to_csr com-lj.ungraph.txt.gz \
  --vertices 3997962 \
  --remap-ids \
  --output-dir dataset/com-LiveJournal \
  --output-name com-LiveJournal
```

Matrix Market files are detected automatically and their 1-based coordinates
are converted to 0-based CSR identifiers:

```bash
./graph_untils/build/edge_list_to_csr com-Friendster.mtx \
  --output-dir dataset/com-Friendster \
  --output-name com-Friendster
```

The converter preserves each listed edge once and removes self-loops by
default. Run `csr_symmetrize` afterward to construct the processed bidirectional
CSR used by the partitioner.
