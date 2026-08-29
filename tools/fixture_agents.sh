#!/usr/bin/env bash
#
# fixture_agents.sh — Launch N agent fixtures on localhost for testing.
#
# Usage:
#   ./tools/fixture_agents.sh [count]    # default: 3 agents
#
# Then run the shell with:
#   cargo run --bin ouro-shell -- --nodes 127.0.0.1:9501,127.0.0.1:9502,127.0.0.1:9503

set -euo pipefail

COUNT="${1:-3}"
BASE_PORT=9501
PIDS=()

cleanup() {
    echo ""
    echo "Shutting down agents..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    echo "All agents stopped."
}

trap cleanup EXIT INT TERM

echo "Starting $COUNT agents on ports $BASE_PORT-$((BASE_PORT + COUNT - 1))..."
echo ""

for i in $(seq 0 $((COUNT - 1))); do
    PORT=$((BASE_PORT + i))
    OURO_PORT="$PORT" cargo run --bin ouro-agent 2>&1 | sed "s/^/  [n$((i+1)):$PORT] /" &
    PIDS+=($!)
done

sleep 2

echo ""
echo "Agents ready. Connect with:"
echo "  cargo run --bin ouro-shell -- --nodes 127.0.0.1:9501,127.0.0.1:9502,127.0.0.1:9503"
echo ""
echo "Or test manually:"
echo "  echo ping | nc 127.0.0.1 9501"
echo "  echo telemetry | nc 127.0.0.1 9501"
echo ""

# Keep running until Ctrl+C
echo "Press Ctrl+C to stop."
wait
