#!/usr/bin/env bash
# OurobourOS CI gate. Fast tier always; parity tier on --heavy.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== build =="
cargo build --workspace --bin ouro-agent

echo "== clippy (gates: no warnings) =="
cargo clippy --workspace --all-targets -- -D warnings

echo "== fast tier =="
cargo test --workspace

if [ "${1:-}" = "--heavy" ]; then
    echo "== parity ladder (nightly tier) =="
    cargo build --release --bin ouro-agent
    OURO_AGENT_BIN="$PWD/target/release/ouro-agent" \
        cargo test --release --workspace -- --ignored --nocapture
fi
echo "CI GREEN"
