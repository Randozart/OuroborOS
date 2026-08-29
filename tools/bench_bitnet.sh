#!/usr/bin/env bash
# Single-node BitNet benchmark: prompt-processing + generation tok/s.
# Usage: tools/bench_bitnet.sh [model.gguf] [n_threads]
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${1:-$HOME/Desktop/Projects/bitnet-2b-tq1_0.gguf}"
# Default to PHYSICAL cores: 8 hw-threads measured 5.8x slower (OpenMP oversubscription)
PHYS="$(awk -F: '/cpu cores/ {print $2; exit}' /proc/cpuinfo | tr -d ' ')"
THREADS="${2:-${PHYS:-4}}"

if [ ! -f "$MODEL" ]; then
    echo "model not found: $MODEL" >&2
    exit 1
fi

cargo build --quiet -p bitnet-rs --tests
echo "== BitNet benchmark: $(basename "$MODEL") | threads=$THREADS =="
BITNET_MODEL="$MODEL" OURO_N_THREADS="$THREADS" \
    cargo test --quiet -p bitnet-rs --test model_test test_bitnet_benchmark -- --ignored --nocapture
