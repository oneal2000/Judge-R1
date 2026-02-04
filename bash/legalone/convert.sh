#!/usr/bin/env bash
set -euo pipefail

cd /data-share/chenxuanyi/internship/JuDGE_RL

# ============================================================
# LegalOne-4B 输出格式转换脚本
# 
# 用法:
#   bash bash/legalone/convert.sh               # 标准模式
#   USE_MRAG=true bash bash/legalone/convert.sh # MRAG 模式
# ============================================================

USE_MRAG=${USE_MRAG:-false}
if [[ "${USE_MRAG}" == "true" ]]; then
    SUFFIX="_mrag"
else
    SUFFIX=""
fi

FILES=(
    "outputs/legalone_direct${SUFFIX}_raw.json:::outputs/legalone_direct${SUFFIX}.jsonl"
    "outputs/legalone_icl${SUFFIX}_raw.json:::outputs/legalone_icl${SUFFIX}.jsonl"
    "outputs/legalone_sft${SUFFIX}_raw.json:::outputs/legalone_sft${SUFFIX}.jsonl"
)

echo "=========================================="
echo "  LegalOne-4B 格式转换"
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

    converted_count = 0
    skipped_count = 0
    with open(dst, "w", encoding="utf-8") as out:
        for item in data:
            fd = item.get("exp_ans") or item.get("output") or item.get("fd")
            # 使用 None 检查而不是 or，避免空字符串被跳过
            gen = item.get("gen_ans")
            if gen is None:
                gen = item.get("document")
            if gen is None:
                gen = item.get("gen")
            cid = item.get("id") or item.get("text_id") or fd2id.get(fd)

            if cid is not None:
                # 即使 gen 是空字符串也保留记录
                if gen is None:
                    gen = ""  # 默认为空字符串
                out.write(json.dumps({"id": cid, "document": gen}, ensure_ascii=False) + "\n")
                converted_count += 1
            else:
                skipped_count += 1
                print(f"  [warn] skipped entry without id: {item.get('text_id', 'unknown')}")
    
    print(f"[done] wrote {dst} ({converted_count} entries, {skipped_count} skipped)")

    print(f"[done] wrote {dst}\n")

for pair in FILES:
    src, dst = pair.split(":::")
    convert_one(src, dst)

PY

echo "✅ LegalOne-4B 格式转换完成"
