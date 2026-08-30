#!/usr/bin/env bash
# Watchdog: when the CUDA build (nohup'd) finishes, run the benchmark phase.
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=/tmp/cuda_build2.log
until ! pgrep -f "build-cuda" >/dev/null 2>&1 && ! pgrep -x cmake >/dev/null 2>&1 && ! pgrep -x nvcc >/dev/null 2>&1; do
    if grep -qE "Error [0-9]+" "$LOG" 2>/dev/null; then
        echo "BUILD FAILED — see $LOG" > /tmp/m2_status.txt
        exit 1
    fi
    sleep 30
done
if [ ! -e bitnet-cpp/build-cuda/bin/llama-bench ] && [ ! -e bitnet-cpp/build-cuda/llama-bench ]; then
    BENCH=$(find bitnet-cpp/build-cuda -name llama-bench -type f 2>/dev/null | head -1)
    [ -n "$BENCH" ] || { echo "BUILD DONE but no llama-bench binary" > /tmp/m2_status.txt; exit 1; }
fi
echo "build complete — benchmarking" 
export PATH=/opt/cuda/bin:$PATH
bash tools/m2_bridge.sh > /tmp/m2_bench.log 2>&1 && echo "M2 BRIDGE COMPLETE — PLAN.md updated, see /tmp/m2_bench.log" > /tmp/m2_status.txt || echo "bench failed, see /tmp/m2_bench.log" > /tmp/m2_status.txt
