#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/paths.sh"
cd "${PROJECT_ROOT}"

# 所有模型前缀和实验模式
PREFIXES="qwen25 qwen3"
ALL_MODES="direct icl sft mrag rl sft_mrag sft_rl mrag_rl sft_mrag_rl"

# 动态生成文件对列表
FILES=()
for prefix in $PREFIXES; do
    for mode in $ALL_MODES; do
        src="outputs/${prefix}_${mode}_raw.json"
        dst="outputs/${prefix}_${mode}.jsonl"
        if [ -f "${src}" ]; then
            FILES+=("${src}:::${dst}")
        fi
    done
done

if [ ${#FILES[@]} -eq 0 ]; then
    echo "⚠️  没有找到任何 _raw.json 文件，请先运行 bash bash/gen.sh"
    exit 0
fi

echo "=========================================="
echo "  格式转换"
echo "  找到 ${#FILES[@]} 个文件待转换"
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
