from pathlib import Path
import pandas as pd
from preprocess import clean_frame, read_source, parse_dates

USER, ITEM, TIME = "读者条码", "条形码", "处理时间"
MISSING = {"", "未知", "不详", "null", "none", "nan", "n/a", "<na>"}


def attribute_value(value):
    if pd.isna(value):
        return ""
    value = str(value).strip()
    return "" if value.casefold() in MISSING else value


ALIASES = {"经管学院":"经济与管理学院","数计学院":"数学与计算机学院","智造学院":"智能制造学院",
           "土建学院":"土木与建筑工程学院","电气学院":"电气信息工程学院","生化学院":"生物与化学工程学院",
           "数学与计算机学院（大数据学院）":"数学与计算机学院","生物与化学工程学院（农学院）":"生物与化学工程学院"}
def college(value):
    name = attribute_value(value)
    return ALIASES.get(name,name)



def read_history(path):
    if Path(path).suffix.lower() == ".xlsx":
        history = clean_frame(read_source(path))["cleaned"]
    else:
        history = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    history[TIME] = parse_dates(history[TIME])
    if history[TIME].isna().any():
        raise ValueError("训练数据存在无效借阅时间")
    return history.sort_values([TIME, USER, ITEM], kind="stable")
