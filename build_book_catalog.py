"""构建按索书号去重的标注底表；派生 CSV 与比赛模型相互独立。"""
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent
PACKAGE = BASE.parent / "C题初赛验证集"
FIELDS = ["索书号","题名","责任者","出版社","出版日期","标准编码","版次"]
LABELS = ["主主题","次主题","体裁或用途","关键词","标注依据","置信度","标注状态"]
RANK = {"入藏目录":0,"历史借阅":1,"验证历史书单":2}
MISSING = {"","未知","不详","null","none","nan","<na>","n/a"}


def clean(value):
    text = (value or "").strip()
    return "" if text.casefold() in MISSING else text


def read(path, encoding):
    with path.open(encoding=encoding,newline="") as stream:
        yield from csv.DictReader(stream)


def observations():
    for source,path,encoding,date_field in [
        ("历史借阅",BASE/"data/processed/borrowing_cleaned.csv","utf-8-sig","处理时间"),
        ("入藏目录",PACKAGE/"中文入藏（新书）.csv","gb18030","处理日期"),
    ]:
        for row in read(path,encoding):
            record = [clean(row.get("出版者" if k=="出版社" else k)) for k in FIELDS]
            if record[0]:
                yield source,tuple(record),row[date_field].split(" ")[0]
    for row in read(PACKAGE/"验证集发放.csv","utf-8-sig"):
        codes = row["历史借阅索书号列表"].split("|")
        titles = row["历史借阅题名列表"].split("|")
        if len(codes)!=len(titles):
            raise ValueError("验证历史索书号与题名列表长度不等，须先核对来源")
        for code,title in zip(codes,titles):
            if clean(code):
                yield "验证历史书单",(clean(code),clean(title),"","","","",""),""


def write_rows(path,columns,rows):
    with path.open("w",encoding="utf-8-sig",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def build(out):
    out.mkdir(parents=True,exist_ok=True)
    main = out/"索书号书目与主题标注.csv"
    detail = out/"来源书目明细.csv"
    if main.exists() or detail.exists():
        raise FileExistsError("输出已经存在；请指定新的 --out，以保护已有标注和明细。")
    grouped = defaultdict(dict)
    for source,record,date in observations():
        summary = grouped[record[0]].setdefault((source,record),[0,""])
        summary[0] += 1
        if date:
            year,month,day = re.split(r"[-/]",date)
            iso = f"{int(year):04}-{int(month):02}-{int(day):02}"
            summary[1] = min(summary[1] or iso,iso)
    books,variants = [],[]
    for code,entries in sorted(grouped.items()):
        # 始终选一条完整的原始书目，避免跨版本拼接题名与作者。
        keys = sorted(entries,key=lambda k:(not k[1][1],RANK[k[0]],-sum(bool(v) for v in k[1]),k[1]))
        source,record = keys[0]
        row = dict(zip(FIELDS,record))
        row["中图分类段"] = code.split("/",1)[0]
        row["来源"] = "|".join(s for s in RANK if any(k[0]==s for k in keys))
        row["代表记录来源"] = source
        conflicts = [f for i,f in enumerate(FIELDS[1:],1) if len({r[i] for _,r in keys if r[i]})>1]
        row["冲突字段"] = "|".join(conflicts)
        row.update(dict.fromkeys(LABELS,""))
        row["标注状态"] = "待核对书目" if conflicts or not record[1] else "待标注"
        books.append(row)
        for i,key in enumerate(keys):
            s,r = key
            n,dt = entries[key]
            variants.append(dict(zip(FIELDS,r)) | {"来源":s,"来源行数":n,"该来源首次记录日期":dt,
                                                    "采用为代表记录":int(i==0)})
    write_rows(detail,[*FIELDS,"来源","来源行数","该来源首次记录日期","采用为代表记录"],variants)
    write_rows(main,[*FIELDS,"中图分类段","来源","代表记录来源","冲突字段",*LABELS],books)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=BASE/"data/book_catalog")
    build(parser.parse_args().out)

