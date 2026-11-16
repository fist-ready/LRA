"""按 topic-v2.0 准备标注批次，校验短编号 JSONL 并解码为 CSV。"""
import argparse
import csv
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
CATALOG = BASE / "data/book_catalog"
DICT = json.loads((BASE / "book_topics_v2.json").read_text(encoding="utf8"))
TOPICS = {int(k):v for k,v in DICT["topics"].items()}
SPECIAL = {int(k):v for k,v in DICT["special"].items()}
SET_FORMAT = "topic-set-v1"


def prepare(start, size, path):
    if start < 1 or size < 1:
        raise ValueError("起始数据行和批量大小必须大于零")
    with (CATALOG/"索书号书目与主题标注.csv").open(encoding="utf-8-sig",newline="") as f:
        from itertools import islice
        books = [{k:r[k] for k in ("索书号","题名","责任者","中图分类段","冲突字段")}
                 for r in islice(csv.DictReader(f),start-1,start-1+size)]
    if not books:
        raise ValueError("起始行超出目录")
    conflicts = {b["索书号"]:set() for b in books if b["冲突字段"]}
    if conflicts:
        with (CATALOG/"来源书目明细.csv").open(encoding="utf-8-sig",newline="") as f:
            for r in csv.DictReader(f):
                if r["索书号"] in conflicts:
                    conflicts[r["索书号"]].add((r["题名"],r["责任者"]))
        for b in books:
            if b["索书号"] in conflicts:
                b["题名责任者变体"] = sorted(conflicts[b["索书号"]])
    with path.open("x",encoding="utf8") as f:
        json.dump({"version":DICT["version"],"format":SET_FORMAT,"books":books},f,ensure_ascii=False,indent=2)


def decode(batch, answers, output):
    source = json.loads(batch.read_text(encoding="utf-8-sig"))
    if source["version"] != DICT["version"]:
        raise ValueError("批次与词表版本不同")
    layout = source.get("format","primary-secondary")
    if layout not in (SET_FORMAT,"primary-secondary"):
        raise ValueError("未知的标注格式")
    codes = [r["索书号"] for r in source["books"]]
    expected = set(codes)
    if len(codes)!=len(expected):
        raise ValueError("输入索书号重复")
    rows = {}
    for num,line in enumerate(answers.read_text(encoding="utf-8-sig").splitlines(),1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row,list) or len(row)<2 or (layout=="primary-secondary" and len(row)>4):
            raise ValueError(f"第 {num} 行格式错误")
        code,*ids = row
        if not isinstance(code,str) or code not in expected or code in rows:
            raise ValueError(f"第 {num} 行索书号错误或重复")
        if any(type(i) is not int or i not in TOPICS and i not in SPECIAL for i in ids):
            raise ValueError(f"第 {num} 行有未定义编号")
        if len(set(ids))!=len(ids) or (len(ids)>1 and any(i in SPECIAL or i==99 for i in ids)):
            raise ValueError(f"第 {num} 行编号重复或特殊编号未单独使用")
        if layout==SET_FORMAT:
            ids.sort()
            rows[code] = [code,"|".join(map(str,ids)),"|".join(TOPICS[i] for i in ids if i in TOPICS),
                          SPECIAL.get(ids[0],"模型初标"),DICT["version"],layout]
            continue
        primary,*secondary = ids
        rows[code] = [code,primary,"|".join(map(str,secondary)),TOPICS.get(primary,""),
                      "|".join(TOPICS[i] for i in secondary),SPECIAL.get(primary,"模型初标"),DICT["version"]]
    if expected!=rows.keys():
        raise ValueError(f"缺少 {len(expected-rows.keys())} 个索书号")
    with output.open("x",encoding="utf-8-sig",newline="") as f:
        writer = csv.writer(f)
        if layout==SET_FORMAT:
            writer.writerow(["索书号","主题编号集合","主题名称集合","标注状态","词表版本","标注格式"])
        else:
            writer.writerow(["索书号","主主题编号","次主题编号","主主题","次主题","标注状态","词表版本"])
        writer.writerows(rows[code] for code in codes)


if __name__=="__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    subs = cli.add_subparsers(dest="action",required=True)
    prep = subs.add_parser("prepare")
    prep.add_argument("--start",type=int,default=1)
    prep.add_argument("--size",type=int,default=50)
    prep.add_argument("output",type=Path)
    dec = subs.add_parser("decode")
    for name in ["batch","answers","output"]:
        dec.add_argument(name,type=Path)
    args = cli.parse_args()
    if args.action=="prepare":
        prepare(args.start,args.size,args.output)
    else:
        decode(args.batch,args.answers,args.output)

