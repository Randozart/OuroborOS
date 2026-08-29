#!/usr/bin/env python3
"""
shard_model.py — Split a BitNet GGUF model across N nodes for pipeline parallelism.

Usage:
    python3 tools/shard_model.py <model.gguf> <num_nodes> [--output-dir <dir>]

Output:
    <output_dir>/shard_<i>.bmts — binary shard per node (BMTS format below)
    <output_dir>/shard_map.json — layer-to-node mapping metadata

BMTS v1 layout (all little-endian):
    magic:    u32  0x4F55524F ("OURO")
    version:  u16  1
    node:     u16  node index (1-based)
    n_tensors: u32
    meta_len: u32
    meta:     JSON [{name, shape, dtype, offset, length}] (offset = within data section)
    data:     concatenated tensor bytes
"""

import argparse
import json
import os
import struct
import sys

GGUF_MAGIC = 0x46554747  # "GGUF"
BMTS_MAGIC = 0x4F55524F  # "OURO"
BMTS_VERSION = 1


def read_str(f):
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8")


def skip_kv_value(f, t):
    if t in (0, 1):
        f.read(1)
    elif t in (2, 3):
        f.read(2)
    elif t in (4, 5, 6):
        f.read(4)
    elif t == 7:
        f.read(1)
    elif t == 8:
        read_str(f)
    elif t == 9:
        at = struct.unpack("<I", f.read(4))[0]
        al = struct.unpack("<Q", f.read(8))[0]
        for _ in range(al):
            skip_kv_value(f, at)
    elif t in (10, 11, 12):
        f.read(8)
    else:
        raise ValueError(f"unknown KV type {t}")


def parse_gguf(path):
    """Return (tensors, alignment). tensors = list of dicts with name/shape/dtype/offset."""
    with open(path, "rb") as f:
        magic = struct.unpack("<I", f.read(4))[0]
        if magic != GGUF_MAGIC:
            raise ValueError(f"not GGUF: {path}")
        f.read(4)  # version
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]

        alignment = 32
        for _ in range(n_kv):
            key = read_str(f)
            vt = struct.unpack("<I", f.read(4))[0]
            if key == "general.alignment":
                alignment = struct.unpack("<I", f.read(4))[0]
            else:
                skip_kv_value(f, vt)

        tensors = []
        for _ in range(n_tensors):
            name = read_str(f)
            nd = struct.unpack("<I", f.read(4))[0]
            shape = [struct.unpack("<Q", f.read(8))[0] for _ in range(nd)]
            dtype = struct.unpack("<I", f.read(4))[0]
            offset = struct.unpack("<Q", f.read(8))[0]
            tensors.append({"name": name, "shape": shape, "dtype": dtype, "offset": offset})

        return tensors, alignment


def tensor_data_ranges(path, tensors):
    """Absolute file ranges: last tensor ends at EOF. Returns dict name -> (abs_offset, length)."""
    # data section start = file header end + padding; derive from minimum tensor offset:
    # offsets in GGUF are relative to data section start.
    with open(path, "rb") as f:
        f.read(4)
        f.read(4)
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n_kv):
            read_str(f)
            vt = struct.unpack("<I", f.read(4))[0]
            skip_kv_value(f, vt)
        for _ in range(n_tensors):
            read_str(f)
            nd = struct.unpack("<I", f.read(4))[0]
            f.read(8 * nd)
            f.read(4)
            f.read(8)
        data_start = f.tell()
        # aligned up
        align = 32
        if data_start % align:
            data_start += align - (data_start % align)

    file_size = os.path.getsize(path)
    ranges = {}
    by_rel = sorted(tensors, key=lambda t: t["offset"])
    for i, t in enumerate(by_rel):
        abs_off = data_start + t["offset"]
        if i + 1 < len(by_rel):
            length = by_rel[i + 1]["offset"] - t["offset"]
        else:
            length = file_size - abs_off
        ranges[t["name"]] = (abs_off, length)
    return ranges


def classify(tensors, num_nodes):
    """Split tensors into layer groups, distributed round-robin contiguous blocks."""
    layer_map = {}
    non_layer = []
    for t in tensors:
        if "blk." in t["name"]:
            try:
                layer = int(t["name"].split("blk.")[1].split(".")[0])
                layer_map.setdefault(layer, []).append(t)
            except (IndexError, ValueError):
                non_layer.append(t)
        else:
            non_layer.append(t)

    layers = sorted(layer_map)
    if not layers:
        raise SystemExit("no blk.N tensors found — cannot shard")

    per = max(1, len(layers) // num_nodes)
    groups = []
    for i in range(num_nodes):
        lo = i * per
        hi = (i + 1) * per if i < num_nodes - 1 else len(layers)
        take = [lay for j, lay in enumerate(layers) if lo <= j < hi]
        ts = []
        for lay in take:
            ts.extend(layer_map[lay])
        if i == 0:
            ts = [t for t in non_layer if "token_embd" in t["name"]] + ts
        if i == num_nodes - 1:
            ts += [t for t in non_layer if "token_embd" not in t["name"]]
        groups.append((take, ts))
    return groups


def write_bmts(out_path, node_idx, tensors, ranges, src_path):
    """Write a BMTS v1 shard file."""
    meta = []
    data_sections = []
    local_off = 0
    with open(src_path, "rb") as src:
        for t in tensors:
            abs_off, length = ranges[t["name"]]
            src.seek(abs_off)
            blob = src.read(length)
            data_sections.append(blob)
            meta.append({
                "name": t["name"],
                "shape": t["shape"],
                "dtype": t["dtype"],
                "offset": local_off,
                "length": length,
            })
            local_off += length

    meta_json = json.dumps(meta).encode("utf-8")
    with open(out_path, "wb") as out:
        out.write(struct.pack("<IHHII", BMTS_MAGIC, BMTS_VERSION, node_idx, len(tensors), len(meta_json)))
        out.write(meta_json)
        for blob in data_sections:
            out.write(blob)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("model")
    ap.add_argument("nodes", type=int)
    ap.add_argument("--output-dir", default="shards")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        sys.exit(f"model not found: {args.model}")

    os.makedirs(args.output_dir, exist_ok=True)
    tensors, _ = parse_gguf(args.model)
    ranges = tensor_data_ranges(args.model, tensors)
    groups = classify(tensors, args.nodes)

    shard_map = {"model": os.path.basename(args.model), "nodes": []}
    total = 0
    for i, (layer_ids, ts) in enumerate(groups):
        path = os.path.join(args.output_dir, f"shard_{i+1}.bmts")
        write_bmts(path, i + 1, ts, ranges, args.model)
        size = os.path.getsize(path)
        total += size
        entry = {
            "node": i + 1,
            "file": path,
            "layers": layer_ids,
            "tensors": len(ts),
            "bytes": size,
        }
        shard_map["nodes"].append(entry)
        print(f"  node {i+1}: layers {layer_ids[0] if layer_ids else '-'}..{layer_ids[-1] if layer_ids else '-'} "
              f"{len(ts)} tensors -> {size/1e6:.1f} MB")

    with open(os.path.join(args.output_dir, "shard_map.json"), "w") as f:
        json.dump(shard_map, f, indent=2)
    print(f"total shard bytes: {total/1e6:.1f} MB (source {os.path.getsize(args.model)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
