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
MAX_LEVELS="${MAX_LEVELS:-24}"
KS="${KS:-4 8 32}"

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
    method_tag="sclp_b256_r4_th0p60_no_second_best"
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

        echo "=== ${dataset} k=${k} seed=${SEED} B=sclp tag=${method_tag} ==="
        "${SELF_BUILD}/multilevel_lp" "$indptr" "$indices" "$k" \
            "${run_dir}/B_sclp.hierarchy" "$RATIO" "$SEED" "$STOP_RATIO" \
            sclp 2 "$MAX_LEVELS" \
            > "${run_dir}/B_sclp_coarsen.log" 2>&1
        test -s "${run_dir}/B_sclp.hierarchy"
        "${JET_BUILD}/app/jet_gpu_lp_import" \
            "${run_dir}/B_sclp.hierarchy" "$config" \
            "${run_dir}/B_sclp.part" "${run_dir}/B_sclp_metrics.json" \
            > "${run_dir}/B_sclp.log" 2>&1
    done
done
