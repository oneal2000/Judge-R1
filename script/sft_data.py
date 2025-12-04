import json, pathlib
src = pathlib.Path("data/train.json"); dst = pathlib.Path("data/train_sft.json")
with src.open() as fin, dst.open("w", encoding="utf-8") as fout:
    for line in fin:
        obj = json.loads(line)
        fout.write(json.dumps({"input": obj["text"], "output": obj["fd"]}, ensure_ascii=False) + "\n")
print("wrote", dst)