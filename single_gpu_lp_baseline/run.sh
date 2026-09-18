#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${CUDACXX:-}" ]]; then
    if [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/nvcc" ]]; then
        export CUDACXX="$CONDA_PREFIX/bin/nvcc"
    elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
        export CUDACXX=/usr/local/cuda/bin/nvcc
    else
        echo "nvcc not found; set CUDACXX to the CUDA compiler" >&2
        exit 1
    fi
fi
export CUDA_HOME=${CUDA_HOME:-$(cd "$(dirname "$CUDACXX")/.." && pwd)}
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 <dataset-name> [parts]" >&2
    exit 2
fi

DATASET=$1
PARTS=${2:-4}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DATA_ROOT=${DATA_ROOT:-/cnic/work/shixg/GNN/partition/dataset/Gpartition_dataset/Sym_CSR}
DIR="$DATA_ROOT/$DATASET"
INDPTR="$DIR/${DATASET}_sym_indptr.bin"
INDICES="$DIR/${DATASET}_sym_indices.bin"
OUT=${OUT_FILE:-"$ROOT/${DATASET}_k${PARTS}.parts"}
BIN="$ROOT/build/single_gpu_lp_baseline"
if [[ -z "${TARGET_GPU_ARCH:-}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    TARGET_GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader \
        | head -n 1 | tr -d '.')
fi
TARGET_GPU_ARCH=${TARGET_GPU_ARCH:-80}

if [[ ! -f "$INDPTR" || ! -f "$INDICES" ]]; then
    echo "missing symmetric CSR: $INDPTR or $INDICES" >&2
    exit 1
fi

cmake -S "$ROOT" -B "$ROOT/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="$TARGET_GPU_ARCH" \
    -DBUILD_RESEARCH_TOOLS=OFF
cmake --build "$ROOT/build" --target single_gpu_lp_baseline -j
mkdir -p "$(dirname "$OUT")"

"$BIN" "$INDPTR" "$INDICES" "$PARTS" "$OUT" \
    "${GROW_ROUNDS:-30}" "${REFINE_ROUNDS:-50}" "${SEEDS_PER_PART:-1}" \
    "${MAX_VERTEX_RATIO:-1.10}"
