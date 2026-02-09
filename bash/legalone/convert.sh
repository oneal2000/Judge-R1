#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../paths.sh"
cd "${PROJECT_ROOT}"

# ============================================================
# LegalOne 输出格式转换脚本（4B + 1.7B）
# ============================================================

USE_MRAG=${USE_MRAG:-false}
LEGALONE_MODEL_SET=${LEGALONE_MODEL_SET:-4b}   # all | 4b | 1.7b

if [[ "${USE_MRAG}" == "true" ]]; then
    SUFFIX="_mrag"
else
    SUFFIX=""
fi

MODEL_PREFIXES=()
case "${LEGALONE_MODEL_SET}" in
    all)
        MODEL_PREFIXES=("legalone" "legalone17b")
        ;;
    4b)
        MODEL_PREFIXES=("legalone")
        ;;
    1.7b|1_7b|17b)
        MODEL_PREFIXES=("legalone17b")
        ;;
    *)
        echo "❌ 不支持的 LEGALONE_MODEL_SET=${LEGALONE_MODEL_SET} (可选: all|4b|1.7b)"
        exit 1
        ;;
esac

FILES=()
for prefix in "${MODEL_PREFIXES[@]}"; do
    FILES+=("outputs/${prefix}_direct${SUFFIX}_raw.json:::outputs/${prefix}_direct${SUFFIX}.jsonl")
    FILES+=("outputs/${prefix}_icl${SUFFIX}_raw.json:::outputs/${prefix}_icl${SUFFIX}.jsonl")
    FILES+=("outputs/${prefix}_sft${SUFFIX}_raw.json:::outputs/${prefix}_sft${SUFFIX}.jsonl")
done

echo "=========================================="
echo "  LegalOne 格式转换"
echo "  模型集合: ${LEGALONE_MODEL_SET}"
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

    print(f"[convert] {src} -> {dst}")

    if src.suffix == ".json":
        data = json.load(open(src, "r", encoding="utf-8"))
    else:
        data = [json.loads(line) for line in open(src, "r", encoding="utf-8")]

    converted_count = 0
    skipped_count = 0
    with open(dst, "w", encoding="utf-8") as out:
        for item in data:
            fd = item.get("exp_ans") or item.get("output") or item.get("fd")
            gen = item.get("gen_ans")
            if gen is None:
                gen = item.get("document")
            if gen is None:
                gen = item.get("gen")
            cid = item.get("id") or item.get("text_id") or fd2id.get(fd)

            if cid is not None:
                if gen is None:
                    gen = ""
                out.write(json.dumps({"id": cid, "document": gen}, ensure_ascii=False) + "\\n")
                converted_count += 1
            else:
                skipped_count += 1
                print(f"  [warn] skipped entry without id: {item.get('text_id', 'unknown')}")

    print(f"[done] wrote {dst} ({converted_count} entries, {skipped_count} skipped)")

for pair in FILES:
    src, dst = pair.split(":::")
    convert_one(src, dst)
PY

echo "✅ LegalOne 格式转换完成"
