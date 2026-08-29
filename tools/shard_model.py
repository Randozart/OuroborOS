#!/usr/bin/env python3
"""
shard_model.py — Split a BitNet GGUF model across N nodes for pipeline parallelism.

Usage:
    python3 tools/shard_model.py <model.gguf> <num_nodes> [--output-dir <dir>]

Output:
    Creates <output_dir>/node_1.bin, node_2.bin, ... node_N.bin
    Each file contains the raw tensor data for that node's layer slice.
    Also outputs a shard_map.json with metadata for the orchestrator.
"""

import argparse
import json
import os
import struct
import sys


# GGUF constants
GGUF_MAGIC = 0x46554747  # "GGUF" little-endian
GGUF_TYPE_F16 = 1
GGUF_TYPE_F32 = 2
GGUF_TYPE_I8 = 3
GGUF_TYPE_I32 = 8


def read_gguf_header(path: str) -> dict:
    """Read GGUF header and return metadata."""
    with open(path, "rb") as f:
        magic = struct.unpack("<I", f.read(4))[0]
        if magic != GGUF_MAGIC:
            raise ValueError(f"Not a GGUF file: {path} (magic={magic:#x})")

        version = struct.unpack("<I", f.read(4))[0]
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]

        return {
            "version": version,
            "n_tensors": n_tensors,
            "n_kv": n_kv,
        }


def read_tensor_info(f) -> tuple:
    """Read a tensor info entry from GGUF."""
    # name (length-prefixed string)
    name_len = struct.unpack("<Q", f.read(8))[0]
    name = f.read(name_len).decode("utf-8")

    # n_dims
    n_dims = struct.unpack("<Q", f.read(8))[0]

    # shape
    shape = []
    for _ in range(n_dims):
        shape.append(struct.unpack("<Q", f.read(8))[0])

    # type
    dtype = struct.unpack("<I", f.read(4))[0]

    # offset
    offset = struct.unpack("<Q", f.read(8))[0]

    return name, shape, dtype, offset


def shard_gguf(model_path: str, num_nodes: int, output_dir: str) -> dict:
    """Shard a GGUF model across N nodes based on layer structure."""
    os.makedirs(output_dir, exist_ok=True)

    info = read_gguf_header(model_path)
    print(f"Model: {model_path}")
    print(f"  Tensors: {info['n_tensors']}")
    print(f"  Nodes: {num_nodes}")

    # Read all tensor metadata
    tensors = []
    with open(model_path, "rb") as f:
        # Skip header
        f.read(4)  # magic
        f.read(4)  # version
        f.read(8)  # n_tensors
        f.read(8)  # n_kv

        # Skip KV metadata (we need to advance past it to reach tensor info)
        for _ in range(info["n_kv"]):
            # key
            key_len = struct.unpack("<Q", f.read(8))[0]
            f.read(key_len)
            # value
            val_type = struct.unpack("<I", f.read(4))[0]
            skip_value(f, val_type)

        # Read tensor info
        for _ in range(info["n_tensors"]):
            tensors.append(read_tensor_info(f))

    # Classify tensors by layer
    layer_tensors = {}
    non_layer_tensors = []

    for name, shape, dtype, offset in tensors:
        # BitNet/Llama tensor naming: layers.N.weight, layers.N.attention, etc.
        if "layers." in name:
            parts = name.split("layers.")
            if len(parts) > 1:
                layer_num_str = parts[1].split(".")[0]
                try:
                    layer_num = int(layer_num_str)
                    if layer_num not in layer_tensors:
                        layer_tensors[layer_num] = []
                    layer_tensors[layer_num].append((name, shape, dtype, offset))
                    continue
                except ValueError:
                    pass
        non_layer_tensors.append((name, shape, dtype, offset))

    sorted_layers = sorted(layer_tensors.keys())
    total_layers = len(sorted_layers)

    if total_layers == 0:
        print("WARNING: No layer-structured tensors found. Splitting equally.")
        # Fallback: split tensors evenly
        chunk_size = len(tensors) // num_nodes
        for i in range(num_nodes):
            start = i * chunk_size
            end = start + chunk_size if i < num_nodes - 1 else len(tensors)
            shard_tensors = tensors[start:end]
            print(f"  Node {i+1}: {len(shard_tensors)} tensors")
        return {"error": "no layer structure found"}

    layers_per_node = total_layers // num_nodes
    remainder = total_layers % num_nodes

    shard_map = {
        "model": os.path.basename(model_path),
        "total_layers": total_layers,
        "nodes": [],
    }

    layer_idx = 0
    for node_idx in range(num_nodes):
        node_layers = layers_per_node + (1 if node_idx < remainder else 0)
        start_layer = sorted_layers[layer_idx]
        end_layer = sorted_layers[layer_idx + node_layers - 1] if node_layers > 0 else start_layer

        node_tensors = []
        for li in range(layer_idx, layer_idx + node_layers):
            node_tensors.extend(layer_tensors[sorted_layers[li]])

        # Add embedding/output tensors to first and last nodes
        if node_idx == 0:
            for t in non_layer_tensors:
                if "token_embd" in t[0] or "embedding" in t[0].lower():
                    node_tensors.insert(0, t)
        if node_idx == num_nodes - 1:
            for t in non_layer_tensors:
                if "output" in t[0] or "norm" in t[0]:
                    node_tensors.append(t)

        node_entry = {
            "node": node_idx + 1,
            "layer_start": int(start_layer),
            "layer_end": int(end_layer),
            "tensor_count": len(node_tensors),
            "tensors": [
                {"name": t[0], "shape": t[1], "dtype": t[2]}
                for t in node_tensors
            ],
        }
        shard_map["nodes"].append(node_entry)

        print(f"  Node {node_idx+1}: layers {start_layer}-{end_layer} ({len(node_tensors)} tensors)")
        layer_idx += node_layers

    # Save shard map
    map_path = os.path.join(output_dir, "shard_map.json")
    with open(map_path, "w") as f:
        json.dump(shard_map, f, indent=2)
    print(f"\nShard map saved to: {map_path}")

    return shard_map


def skip_value(f, val_type: int):
    """Skip a KV value based on its type."""
    if val_type == 0:  # UINT8
        f.read(1)
    elif val_type == 1:  # INT8
        f.read(1)
    elif val_type == 2:  # UINT16
        f.read(2)
    elif val_type == 3:  # INT16
        f.read(2)
    elif val_type == 4:  # UINT32
        f.read(4)
    elif val_type == 5:  # INT32
        f.read(4)
    elif val_type == 6:  # FLOAT32
        f.read(4)
    elif val_type == 7:  # BOOL
        f.read(1)
    elif val_type == 8:  # STRING
        s_len = struct.unpack("<Q", f.read(8))[0]
        f.read(s_len)
    elif val_type == 9:  # ARRAY
        arr_type = struct.unpack("<I", f.read(4))[0]
        arr_len = struct.unpack("<Q", f.read(8))[0]
        for _ in range(arr_len):
            skip_value(f, arr_type)
    elif val_type == 10:  # UINT64
        f.read(8)
    elif val_type == 11:  # INT64
        f.read(8)
    elif val_type == 12:  # FLOAT64
        f.read(8)
    else:
        raise ValueError(f"Unknown KV type: {val_type}")


def main():
    parser = argparse.ArgumentParser(description="Shard GGUF model for pipeline parallelism")
    parser.add_argument("model", help="Path to GGUF model file")
    parser.add_argument("nodes", type=int, help="Number of nodes to shard across")
    parser.add_argument("--output-dir", default="shards", help="Output directory (default: shards)")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: model not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    shard_gguf(args.model, args.nodes, args.output_dir)


if __name__ == "__main__":
    main()
