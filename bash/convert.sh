#!/usr/bin/env bash
set -euo pipefail

cd /data-share/chenxuanyi/internship/JuDGE_RL

FILES=(
    "outputs/qwen3_direct_raw.json:::outputs/qwen3_direct.jsonl"
    "outputs/qwen3_icl_raw.json:::outputs/qwen3_icl.jsonl"
    "outputs/qwen3_sft_raw.json:::outputs/qwen3_sft.jsonl"
    "outputs/qwen3_rl_full_raw.json:::outputs/qwen3_rl_full.jsonl"
    "outputs/qwen3_rl_lora_raw.json:::outputs/qwen3_rl_lora.jsonl"

)

python - <<PY
import json
from pathlib import Path

fd2id = {}
with open("data/test.json", "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        fd2id[obj["fd"]] = obj["text_id"]

# bash 的数组会被注入到这里（作为字符串列表）
FILES = """${FILES[@]}""".split()

def convert_one(src_path: str, dst_path: str):
    src = Path(src_path)
    dst = Path(dst_path)

    if not src.exists():
        print(f"[skip] {src} not found")
        return

    print(f"[convert] {src} → {dst}")

    # 支持 JSON / JSONL 自动识别
    if src.suffix == ".json":
        data = json.load(open(src, "r", encoding="utf-8"))
    else:
        data = [json.loads(line) for line in open(src, "r", encoding="utf-8")]

    with open(dst, "w", encoding="utf-8") as out:
        for item in data:
            fd = item.get("exp_ans") or item.get("output") or item.get("fd")
            gen = item.get("gen_ans") or item.get("document") or item.get("gen")
            cid = fd2id.get(fd)

            if cid and gen is not None:
                out.write(json.dumps({"id": cid, "document": gen}, ensure_ascii=False) + "\n")

    print(f"[done] wrote {dst}\n")

# 遍历每个 src::dst
for pair in FILES:
    src, dst = pair.split(":::")
    convert_one(src, dst)

PY
