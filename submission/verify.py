"""校验初赛 CSV 的读者、顺序、合法候选及个人历史排除。"""
import argparse
import hashlib
from pathlib import Path

import pandas as pd

ROOT=Path(__file__).resolve().parent

def read(path,encoding="utf-8-sig"):
    return pd.read_csv(path,dtype=str,keep_default_na=False,encoding=encoding)

def verify(path):
    rows=read(path)
    people=read(ROOT/"inputs/验证集发放.csv")
    history=read(ROOT/"work/processed/borrowing_cleaned.csv")
    books=read(ROOT/"inputs/中文入藏（新书）.csv","gb18030")
    past=history[pd.to_datetime(history["处理时间"],format="mixed")<pd.Timestamp("2025-09-01")]
    legal=(set(past["索书号"].str.strip())|set(books["索书号"].str.strip()))-{"","未知","不详"}
    assert list(rows.columns)==["读者ID","推荐索书号列表"]
    assert rows["读者ID"].is_unique and len(rows)==len(people)==956
    assert rows["读者ID"].tolist()==people["读者ID"].tolist()
    introduced=set(books["索书号"])-set(past["索书号"])
    new_slots=0
    for pred,old in zip(rows["推荐索书号列表"],people["历史借阅索书号列表"],strict=True):
        top=pred.split("|")
        seen={c.strip() for c in old.split("|") if c.strip()}
        assert len(top)==len(set(top))==10 and set(top)<=legal and not set(top)&seen
        new_slots+=len(set(top)&introduced)
    print(f"{len(rows)} 人，{10*len(rows)} 个合法位置。仅在入藏目录出现的书占 {new_slots} 个位置。")
    actual=hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"SHA-256 {actual}")
    return actual

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=ROOT/"验证集结果.csv")
    parser.add_argument("--check-reference",action="store_true")
    args=parser.parse_args()
    actual=verify(args.output)
    if args.check_reference:
        expected=(ROOT/"reference.sha256").read_text("utf8").split()[0]
        if actual!=expected:
            raise SystemExit("文件结构合格，但与发布参考文件字节不同")
