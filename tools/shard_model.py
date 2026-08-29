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


def read_kv(f, t):
    import json as _json
    sz = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8,13:2,14:2}
    if t == 8:
        return read_str(f)
    if t == 9:
        at = struct.unpack("<I", f.read(4))[0]
        al = struct.unpack("<Q", f.read(8))[0]
        return [read_kv(f, at) for _ in range(al)]
    fmt = {2:"<H",3:"<h",4:"<i",5:"<I",6:"<f",7:"<B",10:"<Q",11:"<q",12:"<d"}
    return struct.unpack(fmt[t], f.read(sz[t]))[0]


def build_model_card(kv):
    """Extract a family-parameterized card. Understands qwen35 + bitnet/llama."""
    arch = kv.get("general.architecture", "unknown")
    p = {k[len(arch)+1:]: v for k, v in kv.items() if k.startswith(arch + ".")}
    n_layer = p.get("block_count", 0)
    card = {
        "architecture": arch,
        "n_layer": n_layer,
        "n_embd": p.get("embedding_length", 0),
        "n_head": p.get("attention.head_count", 0),
        "n_head_kv": p.get("attention.head_count_kv", 0),
        "n_ff": p.get("feed_forward_length", 0),
        "n_vocab": kv.get("__vocab__") or p.get("vocab_size", 0),
        "eps": p.get("attention.layer_norm_rms_epsilon", 1e-5),
        "rope_base": p.get("rope.freq_base", 10000.0),
        "n_rot": p.get("rope.dimension_count", 0),
        "head_dim": p.get("attention.key_length", 0),
        "head_v_dim": p.get("attention.value_length", 0),
        "full_attention_interval": p.get("full_attention_interval", 1),
        "nextn": p.get("nextn_predict_layers", 0),
        "ssm": {
            "conv_kernel": p.get("ssm.conv_kernel", 0),
            "d_state": p.get("ssm.state_size", 0),
            "n_k_heads": p.get("ssm.group_count", 0),
            "n_v_heads": p.get("ssm.time_step_rank", 0),
            "d_inner": p.get("ssm.inner_size", 0),
        },
    }
    return card


def parse_gguf(path):
    """Return (tensors, alignment, card). card = model-card dict from GGUF KV."""
    with open(path, "rb") as f:
        magic = struct.unpack("<I", f.read(4))[0]
        if magic != GGUF_MAGIC:
            raise ValueError(f"not GGUF: {path}")
        f.read(4)  # version
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]

        alignment = 32
        kv = {}
        for _ in range(n_kv):
            key = read_str(f)
            vt = struct.unpack("<I", f.read(4))[0]
            if key == "general.alignment":
                alignment = struct.unpack("<I", f.read(4))[0]
            else:
                try:
                    v = read_kv(f, vt)
                    if key == "tokenizer.ggml.tokens":
                        kv["__vocab__"] = len(v)
                    else:
                        kv[key] = v
                except Exception:
                    skip_kv_value(f, vt)
        card = build_model_card(kv)

        tensors = []
        for _ in range(n_tensors):
            name = read_str(f)
            nd = struct.unpack("<I", f.read(4))[0]
            shape = [struct.unpack("<Q", f.read(8))[0] for _ in range(nd)]
            dtype = struct.unpack("<I", f.read(4))[0]
            offset = struct.unpack("<Q", f.read(8))[0]
            tensors.append({"name": name, "shape": shape, "dtype": dtype, "offset": offset})

        return tensors, alignment, card


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
    tensors, _, card = parse_gguf(args.model)

    keep_layers = card["n_layer"] - card.get("nextn", 0)
    def keep(t):
        n = t["name"]
        if n.startswith("v.") or ".v_" in n or n.startswith("model.visual"):
            return False
        if n.startswith("blk."):
            try:
                return int(n.split(".")[1]) < keep_layers
            except ValueError:
                return True
        if "nextn" in n:
            return False
        return True
    dropped = [t["name"] for t in tensors if not keep(t)]
    tensors = [t for t in tensors if keep(t)]
    if dropped:
        print(f"filtered {len(dropped)} non-text tensors (vision/nextn)")
    card["keep_layers"] = keep_layers

    ranges = tensor_data_ranges(args.model, tensors)
    groups = classify(tensors, args.nodes)

    shard_map = {"model": os.path.basename(args.model), "model_card": card, "nodes": []}
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
    with open(os.path.join(args.output_dir, "model.json"), "w") as f:
        json.dump(card, f, indent=2)
    print(f"total shard bytes: {total/1e6:.1f} MB (source {os.path.getsize(args.model)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
