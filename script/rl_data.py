import argparse
import json
import random
from pathlib import Path


SYSTEM_PROMPT = (
    "你是一个法律助理。根据给定的案件事实，撰写一份完整且具有法律效力的刑事判决书，"
    "结构需包含事实、法律分析、裁判理由和最终裁判结果，语言需正式、客观、清晰。"
)


def build_messages(fact: str) -> list[dict]:
    user_prompt = (
        f"任务背景: 根据以下提供的案件事实，生成一份完整的刑法判决书。"
        f"判决书需包括案件事实、法律分析、裁判理由以及最终裁判结论。\n"
        f"本案件事实：{fact}\n"
        f"本案件的完整判决书为："
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def convert(src: Path, dst_dir: Path, max_samples: int | None, seed: int) -> Path:
    rng = random.Random(seed)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "train.jsonl"

    with src.open("r", encoding="utf-8") as fin:
        data = [json.loads(line) for line in fin]

    if max_samples:
        rng.shuffle(data)
        data = data[:max_samples]

    with dst.open("w", encoding="utf-8") as fout:
        for item in data:
            fact = item.get("text") or item.get("input") or ""
            reference = item.get("fd") or item.get("output") or ""
            text_id = item.get("text_id") or ""
            if not fact or not reference:
                continue

            fout.write(
                json.dumps(
                    {
                        "id": text_id,
                        "messages": build_messages(fact),
                        "reference_document": reference,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    return dst


def main():
    parser = argparse.ArgumentParser(description="Build RL dataset jsonl for GRPO training.")
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("data/train.json"),
        help="Source json (train.json) with fields text/fd.",
    )
    parser.add_argument(
        "--dst_dir",
        type=Path,
        default=Path("data/rl_train"),
        help="Output directory to place train.jsonl (used by swift rlhf).",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional cap on samples for quick experiments.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dst = convert(args.src, args.dst_dir, args.max_samples, args.seed)
    print(f"RL dataset written to {dst}")


if __name__ == "__main__":
    main()
