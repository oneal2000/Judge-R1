#!/usr/bin/env bash
set -euo pipefail

cd /data-share/chenxuanyi/internship/JuDGE_RL

# 根据MRAG模式选择文件后缀
USE_MRAG=${USE_MRAG:-false}
if [[ "${USE_MRAG}" == "true" ]]; then
  SUFFIX="_mrag"
else
  SUFFIX=""
fi

FILES=(
    "outputs/qwen25_direct${SUFFIX}_raw.json:::outputs/qwen25_direct${SUFFIX}.jsonl"
    "outputs/qwen25_icl${SUFFIX}_raw.json:::outputs/qwen25_icl${SUFFIX}.jsonl"
    "outputs/qwen25_sft${SUFFIX}_raw.json:::outputs/qwen25_sft${SUFFIX}.jsonl"
    "outputs/qwen25_rl${SUFFIX}_raw.json:::outputs/qwen25_rl${SUFFIX}.jsonl"
    "outputs/qwen3_direct${SUFFIX}_raw.json:::outputs/qwen3_direct${SUFFIX}.jsonl"
    "outputs/qwen3_icl${SUFFIX}_raw.json:::outputs/qwen3_icl${SUFFIX}.jsonl"
    "outputs/qwen3_sft${SUFFIX}_raw.json:::outputs/qwen3_sft${SUFFIX}.jsonl"
    "outputs/qwen3_rl${SUFFIX}_raw.json:::outputs/qwen3_rl${SUFFIX}.jsonl"
)

echo "=========================================="
echo "  格式转换"
echo "  MRAG模式: ${USE_MRAG}"
echo "=========================================="

python - <<PY
import json
from pathlib import Path

fd2id = {}
with open("data/test.json", "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        fd2id[obj["fd"]] = obj["text_id"]

FILES = """${FILES[@]}""".split()

def convert_one(src_path: str, dst_path: str):
    src = Path(src_path)
    dst = Path(dst_path)

    if not src.exists():
        print(f"[skip] {src} not found")
        return

    print(f"[convert] {src} → {dst}")

    if src.suffix == ".json":
        data = json.load(open(src, "r", encoding="utf-8"))
    else:
        data = [json.loads(line) for line in open(src, "r", encoding="utf-8")]

    with open(dst, "w", encoding="utf-8") as out:
        for item in data:
            fd = item.get("exp_ans") or item.get("output") or item.get("fd")
            gen = item.get("gen_ans") or item.get("document") or item.get("gen")
            cid = item.get("id") or item.get("text_id") or fd2id.get(fd)

            if cid and gen is not None:
                out.write(json.dumps({"id": cid, "document": gen}, ensure_ascii=False) + "\n")

    print(f"[done] wrote {dst}\n")

for pair in FILES:
    src, dst = pair.split(":::")
    convert_one(src, dst)

PY

echo "✅ 格式转换完成"