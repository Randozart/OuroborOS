#!/usr/bin/env bash
# M2 bridge benchmark: llama.cpp CUDA tensor-split on local GPUs vs CPU baseline.
# Establishes the performance bar the OuroborOS pipeline must eventually beat.
#
# Prereqs (user hands, once):
#   pacman -S cuda                                    # nvcc 12.8
#   (later, for the 1070 Ti) nvidia-580xx driver branch
#
# Usage: tools/m2_bridge.sh [model.gguf ...]
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS=("${@:-}")
if [ -z "${MODELS}" ]; then
    MODELS=(
        "$HOME/Downloads/Qwen3.8-9B-Q6_K.gguf"
        "$HOME/Downloads/Qwen3.8-27B-Q3_K_M.gguf"
        "$HOME/Desktop/Projects/bitnet-2b-tq1_0.gguf"
    )
fi

# Arch/CachyOS keeps the toolkit under /opt/cuda
if ! command -v nvcc >/dev/null; then
    if [ -x /opt/cuda/bin/nvcc ]; then
        export PATH="/opt/cuda/bin:$PATH"
    else
        echo "nvcc not found. Install the CUDA toolkit first:"
        echo "  sudo pacman -S cuda"
        exit 1
    fi
fi

BUILD=bitnet-cpp/build-cuda
ARCH="${CUDA_ARCH:-86}"   # 3060 = sm_86. Add 61 once r580 lands (never 52: CUDA 12.8 warns but compiles)

echo "== configuring GGML_CUDA=$ARCH =="
cmake -S bitnet-cpp -B "$BUILD" \
    -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
    -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc \
    -DCMAKE_BUILD_TYPE=Release \
    -DBITNET_ARM_TL1=OFF -DBITNET_X86_TL2=OFF >/dev/null
cmake --build "$BUILD" -j"$(nproc)" --config Release >/dev/null 2>&1 || {
    echo "build produced errors; showing tail:"
    cmake --build "$BUILD" -j"$(nproc)" 2>&1 | tail -15
    exit 1
}

BENCH=$(find "$BUILD" -name 'llama-bench' -type f | head -1)
CLI=$(find "$BUILD" -name 'llama-cli' -type f | head -1)
[ -n "$BENCH" ] || { echo "llama-bench not built (fork targets?)"; ls "$BUILD/bin" 2>/dev/null; exit 1; }

echo
echo "== nvidia-smi =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

{
    echo
    echo "### M2 bridge results (CUDA, $(date +%F))"
    echo
    echo "| model | device | m | t/s (pp|tg) |"
    echo "|-------|--------|---|--------------|"
} >> PLAN.md

for M in "${MODELS[@]}"; do
    [ -f "$M" ] || { echo "skip (absent): $M"; continue; }
    for NGL in 0 99; do
        DEV=$([ "$NGL" = 0 ] && echo cpu || echo "cuda($ARCH)")
        echo "== $(basename "$M") | $DEV =="
        OUT=$("$BENCH" -m "$M" -ngl "$NGL" -m 512 -p 64,512 -n 32,128 -fa 0 -r 2 2>/dev/null | grep -E "pp|tg" | tail -2 || true)
        echo "$OUT"
        PP=$(echo "$OUT" | grep "| pp" | awk -F'|' '{print $8}' | tr -d ' ' | head -1)
        TG=$(echo "$OUT" | grep "| tg" | awk -F'|' '{print $8}' | tr -d ' ' | tail -1)
        echo "| $(basename "$M") | $DEV | $NGL | ${PP:-?} | ${TG:-?} |" >> PLAN.md
    done
done

echo
echo "Results appended to PLAN.md (section 16.3 region)."
