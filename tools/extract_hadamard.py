#!/usr/bin/env python3
"""Extract PrismML Hadamard fold metadata from a GGUF into hadamard.json.

Reads prism.hadamard.* KV pairs (block_size, sign_widths, sign_values) and
writes {"block_size", "sign_widths", "signs"} JSON next to the BMTS shards
(cluster/src/infer/qwen35.rs HadamardCfg consumes it via Card::load_dir).
"""
import json
import struct
import sys

from shard_model import read_kv, read_str, skip_kv_value


def extract(path):
    with open(path, "rb") as f:
        f.read(4)  # magic "GGUF"
        f.read(4)  # version
        f.read(8)  # n_tensors
        n_kv = struct.unpack("<Q", f.read(8))[0]

        kv = {}
        for _ in range(n_kv):
            key = read_str(f)
            vt = struct.unpack("<I", f.read(4))[0]
            if key.startswith("prism.hadamard"):
                kv[key] = read_kv(f, vt)
            else:
                skip_kv_value(f, vt)

    version = kv.get("prism.hadamard.version")
    if version != 1:
        raise SystemExit(f"unsupported prism.hadamard.version {version}")
    widths = kv["prism.hadamard.sign_widths"]
    values = kv["prism.hadamard.sign_values"]
    if sum(widths) != len(values):
        raise SystemExit(f"sign widths {widths} sum != len {len(values)}")
    for v in values:
        if v not in (1, 0xFFFFFFFF):
            raise SystemExit(f"bad sign value {v}")
    return {
        "block_size": kv["prism.hadamard.block_size"],
        "sign_widths": widths,
        "signs": [-1 if v == 0xFFFFFFFF else 1 for v in values],
        "gdn_v_grouped": bool(kv.get("prism.hadamard.gdn_v_grouped", 0)),
    }


def main():
    gguf = sys.argv[1]
    out_dirs = sys.argv[2:]
    had = extract(gguf)
    n_neg = sum(1 for s in had["signs"] if s < 0)
    print(f"block_size={had['block_size']} widths={had['sign_widths']} "
          f"signs={len(had['signs'])} (-1 x{n_neg})")
    for d in out_dirs:
        out = f"{d}/hadamard.json"
        with open(out, "w") as f:
            json.dump(had, f)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
