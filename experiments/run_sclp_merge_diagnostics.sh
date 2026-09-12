#!/usr/bin/env bash
set -euo pipefail

# Frozen diagnostic experiment: beta=256, rounds=4, two-hop=0.60,
# second-best off, k=4, seed=0. No coarsening rule is tuned here.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${BUILD:-${ROOT}/build-gh200}"
JET_ROOT="${JET_ROOT:-${ROOT}/../sotas/Jet-Partitioner}"
JET_BUILD="${JET_BUILD:-${JET_ROOT}/build-gh200-kokkos4}"
METIS_ROOT="${METIS_ROOT:-${ROOT}/../sotas/metis-5.2.1-arm64}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/../dataset/Gpartition_dataset/Sym_CSR}"
GRAPH_ROOT="${GRAPH_ROOT:-${ROOT}/../dataset/Gpartition_dataset_metis}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/../single_gpu_lp_baseline/build-gh200/experiments/results/sclp_merge_diagnostics}"

if [[ "$#" -gt 0 ]]; then
    DATASETS=("$@")
else
    DATASETS=(products com-LiveJournal it-2004 kim2)
fi

for dataset in "${DATASETS[@]}"; do
    case "$dataset" in
        products|com-LiveJournal|it-2004|kim2) ;;
        *) echo "unsupported dataset: ${dataset}" >&2; exit 2 ;;
    esac
    indptr="${DATA_ROOT}/${dataset}/${dataset}_sym_indptr.bin"
    indices="${DATA_ROOT}/${dataset}/${dataset}_sym_indices.bin"
    graph="${GRAPH_ROOT}/${dataset}/${dataset}.metis"
    dataset_out="${OUT_ROOT}/${dataset}"
    references="${dataset_out}/references"
    mkdir -p "$references"
    config="${references}/jet_config.txt"
    printf '0\n4\n1\n1.10\n0\n0\n' > "$config"

    jet_part="${references}/jet.part"
    if [[ ! -s "$jet_part" ]]; then
        "${JET_BUILD}/app/jet" "$graph" "$config" \
            "$jet_part" "${references}/jet_metrics.json" \
            > "${references}/jet.log" 2>&1
    fi

    metis_link="${references}/${dataset}.metis"
    metis_part="${metis_link}.part.4"
    if [[ ! -s "$metis_part" ]]; then
        ln -sfn "$graph" "$metis_link"
        LD_LIBRARY_PATH="${METIS_ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
            "${METIS_ROOT}/bin/gpmetis" -seed=0 "$metis_link" 4 \
            > "${references}/metis.log" 2>&1
    fi

    # The unmodified 13e2e75 contraction needs more than the 96 GiB HBM on
    # it-2004 after level 0 (the production binary fails at the same point).
    # Preserve the algorithm and collect the level-0 proposal comparison by
    # stopping before contraction instead of silently changing contraction.
    level0_only=0
    if [[ "$dataset" == it-2004 ]]; then
        level0_only=1
        cat > "${dataset_out}/scope.txt" <<'EOF'
scope=level-0-only
reason=13e2e75 production and diagnostic binaries both exhaust 96 GiB HBM in the first it-2004 contraction
EOF
    fi

    for mode in mr priority; do
        mode_out="${dataset_out}/${mode}"
        mkdir -p "$mode_out"
        if [[ "$mode" == mr ]]; then
            binary="${BUILD}/sclp_diag_mr"
        else
            binary="${BUILD}/sclp_diag_priority"
        fi
        prefix="${mode_out}/accepted"
        hierarchy="${mode_out}/sclp.hierarchy"
        if [[ "$level0_only" == 1 ]]; then
            SCLP_DIAG_PREFIX="$prefix" \
            SCLP_DIAG_JET_PART="$jet_part" \
            SCLP_DIAG_METIS_PART="$metis_part" \
            SCLP_DIAGNOSTICS=1 ML_SKIP_HIERARCHY_EXPORT=1 \
                "$binary" "$indptr" "$indices" 4 "$hierarchy" \
                1.10 0 0.000001 sclp 2 24 \
                > "${mode_out}/coarsen.log" 2>&1
        else
            SCLP_DIAG_PREFIX="$prefix" \
            SCLP_DIAG_JET_PART="$jet_part" \
            SCLP_DIAG_METIS_PART="$metis_part" \
            SCLP_DIAGNOSTICS=1 \
                "$binary" "$indptr" "$indices" 4 "$hierarchy" \
                1.10 0 0.90 sclp 2 24 \
                > "${mode_out}/coarsen.log" 2>&1
            "${JET_BUILD}/app/jet_gpu_lp_import" \
                "$hierarchy" "$config" "${mode_out}/sclp.part" \
                "${mode_out}/sclp_metrics.json" \
                > "${mode_out}/jet_import.log" 2>&1
            # Retain records and partitions, but discard the reproducible
            # hierarchy because it can occupy tens of GB.
            rm -f "$hierarchy"
        fi
    done

    python3 "${ROOT}/experiments/analyze_merge_diagnostics.py" \
        --mr-prefix "${dataset_out}/mr/accepted" \
        --priority-prefix "${dataset_out}/priority/accepted" \
        --output "${dataset_out}/analysis.json" \
        > "${dataset_out}/analysis.log"
done
