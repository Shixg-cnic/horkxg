#!/usr/bin/env bash
set -euo pipefail

# Paired comparison:
#   A: Jet's own seeded coarsener + Jet initialization/uncoarsening.
#   B: self GPU-LP hierarchy + the same Jet initialization/uncoarsening.
# The hierarchy is deliberately retained because its layer statistics are part
# of the experiment, not an intermediate that may be silently discarded.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SELF_BUILD="${SELF_BUILD:-${ROOT}/build-gh200}"
JET_ROOT="${JET_ROOT:-${ROOT}/../sotas/Jet-Partitioner}"
JET_BUILD="${JET_BUILD:-${JET_ROOT}/build-gh200-kokkos4}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/../dataset/Gpartition_dataset/Sym_CSR}"
METIS_ROOT="${METIS_ROOT:-${ROOT}/../dataset/Gpartition_dataset_metis}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/../single_gpu_lp_baseline/build-gh200/experiments/results/gpu_lp_jet_compare}"
RATIO="${RATIO:-1.10}"
SEED="${SEED:-0}"
STOP_RATIO="${STOP_RATIO:-0.90}"
METHOD="${METHOD:-lp}"
BASC_K="${BASC_K:-2}"
MAX_LEVELS="${MAX_LEVELS:-24}"
KS="${KS:-4 8 32}"
FRONTIER_CONTRACTION_FACTOR="${FRONTIER_CONTRACTION_FACTOR:-8}"
FRONTIER_CAPACITY_SLACK="${FRONTIER_CAPACITY_SLACK:-0.20}"
FRONTIER_HOT_RATIO="${FRONTIER_HOT_RATIO:-0.50}"
SCLP_BETA="${SCLP_BETA:-64}"
SCLP_ROUNDS="${SCLP_ROUNDS:-4}"

case "$METHOD" in
    lp|basc|basc_gpu|frontier|sclp) ;;
    *) echo "METHOD must be lp, basc, basc_gpu, frontier, or sclp" >&2; exit 2 ;;
esac

if [[ "$#" -gt 0 ]]; then
    DATASETS=("$@")
else
    DATASETS=(products com-LiveJournal)
fi

read -r -a PARTS_LIST <<< "$KS"
for dataset in "${DATASETS[@]}"; do
    case "$dataset" in
        products|com-LiveJournal) ;;
        *) echo "unsupported dataset: ${dataset}" >&2; exit 2 ;;
    esac
    indptr="${DATA_ROOT}/${dataset}/${dataset}_sym_indptr.bin"
    indices="${DATA_ROOT}/${dataset}/${dataset}_sym_indices.bin"
    metis="${METIS_ROOT}/${dataset}/${dataset}.metis"
    if [[ "$METHOD" == "basc" ]]; then
        method_tag="basc_k${BASC_K}"
    elif [[ "$METHOD" == "basc_gpu" ]]; then
        method_tag="basc_gpu_k${BASC_K}"
    elif [[ "$METHOD" == "frontier" ]]; then
        frontier_slack_tag="${FRONTIER_CAPACITY_SLACK//./p}"
        frontier_hot_tag="${FRONTIER_HOT_RATIO//./p}"
        method_tag="frontier_r${FRONTIER_CONTRACTION_FACTOR}_s${frontier_slack_tag}_h${frontier_hot_tag}"
    elif [[ "$METHOD" == "sclp" ]]; then
        method_tag="sclp_b${SCLP_BETA}_r${SCLP_ROUNDS}"
    else
        method_tag="lp"
    fi
    for k in "${PARTS_LIST[@]}"; do
        if [[ "$k" != "2" && "$k" != "4" && "$k" != "8" &&
              "$k" != "16" && "$k" != "32" ]]; then
            echo "unsupported k in KS: ${k}" >&2
            exit 2
        fi
        run_dir="${OUT_ROOT}/${dataset}/k${k}/seed${SEED}/${method_tag}"
        mkdir -p "$run_dir"
        config="${run_dir}/jet_config.txt"
        printf '0\n%s\n1\n%s\n0\n%s\n' "$k" "$RATIO" "$SEED" > "$config"

        echo "=== ${dataset} k=${k} seed=${SEED} A=jet ==="
        "${JET_BUILD}/app/jet" "$metis" "$config" \
            "${run_dir}/A_jet.part" "${run_dir}/A_jet_metrics.json" \
            > "${run_dir}/A_jet.log" 2>&1

        echo "=== ${dataset} k=${k} seed=${SEED} B=${METHOD} tag=${method_tag} ==="
        "${SELF_BUILD}/multilevel_lp" "$indptr" "$indices" "$k" \
            "${run_dir}/B_${METHOD}.hierarchy" "$RATIO" "$SEED" "$STOP_RATIO" \
            "$METHOD" "$BASC_K" "$MAX_LEVELS" \
            > "${run_dir}/B_${METHOD}_coarsen.log" 2>&1
        test -s "${run_dir}/B_${METHOD}.hierarchy"
        "${JET_BUILD}/app/jet_gpu_lp_import" \
            "${run_dir}/B_${METHOD}.hierarchy" "$config" \
            "${run_dir}/B_${METHOD}.part" "${run_dir}/B_${METHOD}_metrics.json" \
            > "${run_dir}/B_${METHOD}.log" 2>&1
    done
done
