#!/usr/bin/env python3
"""
test_shard_model.py — Rung B1 gate for tools/shard_model.py (Track B,
docs/AIR_PATH.md): the MTP draft head must be kept, not filtered, and must
land on the brain node (draft.bmts, node 1), with the card accounting it.

Builds synthetic GGUFs (qwen35 with nextn, qwen35 no-nextn, bitnet) in a
temp dir, runs the sharder, asserts the draft record, the BMTS layout, and
the --no-draft regression path.
"""
import json
import os
import struct
import subprocess
import sys
import tempfile

TOOLS = os.path.dirname(os.path.abspath(__file__))
SHARDER = os.path.join(TOOLS, "shard_model.py")

GGUF_MAGIC = 0x46554747


def write_str(f, s):
    b = s.encode("utf-8")
    f.write(struct.pack("<Q", len(b)))
    f.write(b)


def write_kv(f, key, vtype, value):
    write_str(f, key)
    f.write(struct.pack("<I", vtype))
    if vtype == 4:
        f.write(struct.pack("<i", value))
    elif vtype == 6:
        f.write(struct.pack("<f", value))
    elif vtype == 8:
        write_str(f, value)
    elif vtype == 9:
        atype, items = value
        f.write(struct.pack("<IQ", atype, len(items)))
        for s in items:
            write_str(f, s)
    else:
        raise ValueError(f"unsupported test KV type {vtype}")


def make_gguf(path, arch, n_layer, nextn, extra_nextn_tensor=True):
    """Write a minimal valid GGUF. All tensors f32, 1-D shape [16]."""
    if arch == "qwen35":
        kvs = [
            ("general.architecture", 8, "qwen35"),
            ("qwen35.block_count", 4, n_layer),
            ("qwen35.embedding_length", 4, 16),
            ("qwen35.attention.head_count", 4, 4),
            ("qwen35.attention.head_count_kv", 4, 1),
            ("qwen35.attention.key_length", 4, 4),
            ("qwen35.attention.value_length", 4, 4),
            ("qwen35.attention.layer_norm_rms_epsilon", 6, 1e-5),
            ("qwen35.rope.freq_base", 6, 10000.0),
            ("qwen35.rope.dimension_count", 4, 8),
            ("qwen35.full_attention_interval", 4, 2),
            ("qwen35.nextn_predict_layers", 4, nextn),
            ("qwen35.ssm.conv_kernel", 4, 4),
            ("qwen35.ssm.state_size", 4, 4),
            ("qwen35.ssm.group_count", 4, 1),
            ("qwen35.ssm.time_step_rank", 4, 1),
            ("qwen35.ssm.inner_size", 4, 16),
            ("tokenizer.ggml.tokens", 9, (8, ["a", "b"])),
            ("general.alignment", 4, 32),
        ]
    else:  # bitnet
        kvs = [
            ("general.architecture", 8, "bitnet"),
            ("bitnet.block_count", 4, n_layer),
            ("bitnet.embedding_length", 4, 16),
            ("tokenizer.ggml.tokens", 9, (8, ["a", "b"])),
            ("general.alignment", 4, 32),
        ]

    names = [f"blk.{i}.attn_norm.weight" for i in range(n_layer)]
    names += ["token_embd.weight", "output_norm.weight"]
    if nextn and extra_nextn_tensor:
        names.append("nextn_lm_head.weight")

    with open(path, "wb") as f:
        f.write(struct.pack("<IIQQ", GGUF_MAGIC, 3, len(names), len(kvs)))
        for key, vt, val in kvs:
            write_kv(f, key, vt, val)
        # tensor headers (offsets are relative to aligned data start)
        off = 0
        for n in names:
            write_str(f, n)
            f.write(struct.pack("<I", 1))          # nd = 1
            f.write(struct.pack("<Q", 16))         # shape [16]
            f.write(struct.pack("<I", 0))          # dtype f32
            f.write(struct.pack("<Q", off))        # offset
            off += 64
        # pad data section to alignment
        while f.tell() % 32:
            f.write(b"\x00")
        for _ in names:
            f.write(b"\x00" * 64)


def read_bmts_header(path):
    with open(path, "rb") as f:
        magic, version, node, n_tensors, meta_len = struct.unpack("<IHHII", f.read(16))
    return magic, version, node, n_tensors, meta_len


def run_sharder(gguf, outdir, extra=None):
    cmd = [sys.executable, SHARDER, gguf, "2", "--output-dir", outdir]
    if extra:
        cmd += extra
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, f"sharder failed:\n{r.stdout}\n{r.stderr}"
    return r.stdout


def test_qwen35_draft_kept_on_brain():
    with tempfile.TemporaryDirectory() as d:
        gguf = os.path.join(d, "m.gguf")
        out = os.path.join(d, "shards")
        make_gguf(gguf, "qwen35", n_layer=4, nextn=1)
        out_text = run_sharder(gguf, out)

        smap = json.load(open(os.path.join(out, "shard_map.json")))
        card = json.load(open(os.path.join(out, "model.json")))

        # Trunk covers layers 0..2 only (keep_layers = 4 - 1 = 3)
        trunk = [l for n in smap["nodes"] for l in n["layers"]]
        assert trunk == [0, 1, 2], f"trunk layers wrong: {trunk}"

        # Draft record on the brain node
        assert "draft" in smap, "draft record missing from shard_map"
        d = smap["draft"]
        assert d["node"] == 1, d
        assert d["layers"] == [3, 3], d
        assert d["tensors"] == 2, d  # blk.3.* + nextn_lm_head.weight
        assert d["bytes"] > 0, d

        # Card accounts the draft head
        assert card["keep_layers"] == 3
        assert card["nextn"] == 1
        assert card["draft_layers"] == [3, 3]
        assert card["draft_node"] == 1
        assert card["draft_bytes"] > 0

        # draft.bmts is a valid BMTS shard owned by node 1
        magic, _, node, nt, _ = read_bmts_header(os.path.join(out, "draft.bmts"))
        assert magic == 0x4F55524F, "draft.bmts not BMTS"
        assert node == 1
        assert nt == 2

        assert "kept 2 MTP draft tensors on brain node 1" in out_text, out_text


def test_qwen35_no_nextn_has_no_draft():
    with tempfile.TemporaryDirectory() as d:
        gguf = os.path.join(d, "m.gguf")
        out = os.path.join(d, "shards")
        make_gguf(gguf, "qwen35", n_layer=4, nextn=0)
        out_text = run_sharder(gguf, out)
        smap = json.load(open(os.path.join(out, "shard_map.json")))
        card = json.load(open(os.path.join(out, "model.json")))
        assert "draft" not in smap, smap
        assert "draft_layers" not in card
        assert card["keep_layers"] == 4
        assert "filtered" not in out_text, out_text


def test_bitnet_untouched():
    with tempfile.TemporaryDirectory() as d:
        gguf = os.path.join(d, "m.gguf")
        out = os.path.join(d, "shards")
        make_gguf(gguf, "bitnet", n_layer=4, nextn=0)
        run_sharder(gguf, out)
        smap = json.load(open(os.path.join(out, "shard_map.json")))
        assert "draft" not in smap
        trunk = [l for n in smap["nodes"] for l in n["layers"]]
        assert trunk == [0, 1, 2, 3], trunk


def test_no_draft_flag_regression():
    with tempfile.TemporaryDirectory() as d:
        gguf = os.path.join(d, "m.gguf")
        out = os.path.join(d, "shards")
        make_gguf(gguf, "qwen35", n_layer=4, nextn=1)
        out_text = run_sharder(gguf, out, extra=["--no-draft"])
        smap = json.load(open(os.path.join(out, "shard_map.json")))
        card = json.load(open(os.path.join(out, "model.json")))
        assert "draft" not in smap, smap
        assert "draft_layers" not in card
        trunk = [l for n in smap["nodes"] for l in n["layers"]]
        assert trunk == [0, 1, 2], trunk  # blk.3 dropped, keep_layers still 3
        assert "filtered 2 non-text tensors (vision/nextn)" in out_text, out_text


if __name__ == "__main__":
    for fn in [test_qwen35_draft_kept_on_brain,
               test_qwen35_no_nextn_has_no_draft,
               test_bitnet_untouched,
               test_no_draft_flag_regression]:
        fn()
        print(f"  ok: {fn.__name__}")
    print("SHARDER GATES GREEN")