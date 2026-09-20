#!/usr/bin/env python3
"""Extract the Bonsai tokenizer from the GGUF into tokenizer.json.

Emits what cluster/src/infer/tokenizer.rs consumes: tokens (GPT-2
byte-level alphabet), token_type, merges (ranked), special ids, pre type.
Run: python3 tools/dump_tokenizer.py <model.gguf> shards_bonsai27_n1 [...]
"""
import json
import struct
import sys

from shard_model import read_kv, read_str, skip_kv_value


def extract(path):
    with open(path, "rb") as f:
        f.read(4)  # magic
        f.read(4)  # version
        f.read(8)  # n_tensors
        n_kv = struct.unpack("<Q", f.read(8))[0]
        kv = {}
        for _ in range(n_kv):
            key = read_str(f)
            vt = struct.unpack("<I", f.read(4))[0]
            if key.startswith("tokenizer.") or key in ("general.name",):
                kv[key] = read_kv(f, vt)
            else:
                skip_kv_value(f, vt)
    tok = kv["tokenizer.ggml.tokens"]
    types = kv["tokenizer.ggml.token_type"]
    merges = kv["tokenizer.ggml.merges"]
    assert len(tok) == len(types), "tokens/token_type arity"
    out = {
        "model": kv.get("tokenizer.ggml.model", "gpt2"),
        "pre": kv.get("tokenizer.ggml.pre", "qwen35"),
        "tokens": tok,
        "token_types": types,
        "merges": merges,
        "eos": kv.get("tokenizer.ggml.eos_token_id"),
        "bos": kv.get("tokenizer.ggml.bos_token_id"),
        "pad": kv.get("tokenizer.ggml.padding_token_id"),
        "add_bos": bool(kv.get("tokenizer.ggml.add_bos_token", 0)),
    }
    return out


def main():
    gguf = sys.argv[1]
    out_dirs = sys.argv[2:]
    tok = extract(gguf)
    n_special = sum(1 for t in tok["token_types"] if t != 1)
    print(f"model={tok['model']} pre={tok['pre']} tokens={len(tok['tokens'])} "
          f"merges={len(tok['merges'])} non-normal={n_special} "
          f"eos={tok['eos']} bos={tok['bos']} add_bos={tok['add_bos']}")
    for d in out_dirs:
        out = f"{d}/tokenizer.json"
        with open(out, "w") as f:
            json.dump(tok, f, ensure_ascii=False)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
