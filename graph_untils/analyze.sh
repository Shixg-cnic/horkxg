#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <partition-file> <dataset-name>" >&2
  echo "Example: $0 results/papers100M.parts papers100M" >&2
  exit 1
fi

PARTITION_FILE=$1
DATASET_NAME=$2

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ANALYZER="${SCRIPT_DIR}/build/analyze_partition"
GRAPH_DIR="/home/shixg/gpart/dataset/${DATASET_NAME}"

PARTS=4
VERTEX_BALANCE_RATIO=1.10
EDGE_BALANCE_RATIO=1.10

if [[ ! -x "$ANALYZER" ]]; then
  echo "error: analyzer not found: $ANALYZER" >&2
  exit 1
fi
if [[ ! -f "$PARTITION_FILE" ]]; then
  echo "error: partition file not found: $PARTITION_FILE" >&2
  exit 1
fi
if [[ ! -f "${GRAPH_DIR}/${DATASET_NAME}_indptr.bin" ||
      ! -f "${GRAPH_DIR}/${DATASET_NAME}_indices.bin" ]]; then
  echo "error: original CSR not found under: $GRAPH_DIR" >&2
  exit 1
fi

ANALYZE_INPUT=$PARTITION_FILE
TEMP_PARTITION=""
cleanup() {
  if [[ -n "$TEMP_PARTITION" ]]; then
    rm -f -- "$TEMP_PARTITION"
  fi
}
trap cleanup EXIT

# V5 writes int32 binary labels, while XtraPuLP commonly writes text labels.
if ! LC_ALL=C grep -Iq . "$PARTITION_FILE"; then
  TEMP_PARTITION=$(mktemp "${TMPDIR:-/tmp}/gpart-parts.XXXXXX.txt")
  od -An -v -td4 "$PARTITION_FILE" | tr -s '[:space:]' '\n' | sed '/^$/d' \
    > "$TEMP_PARTITION"
  ANALYZE_INPUT=$TEMP_PARTITION
  echo "Detected binary int32 partition file; converted temporarily for analysis."
fi

"$ANALYZER" "$ANALYZE_INPUT" \
  --graph-dir "$GRAPH_DIR" \
  --name "$DATASET_NAME" \
  --parts "$PARTS" \
  --balance-ratio "$VERTEX_BALANCE_RATIO" \
  --edge-balance-ratio "$EDGE_BALANCE_RATIO"
