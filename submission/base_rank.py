"""从随包数据重建冻结的索书号推荐器，输出初赛官方格式。"""
import argparse
from pathlib import Path

import pandas as pd

from data_io import USER, TIME, attribute_value, college, read_history
from newcomer_portrait import ProfileRecommender, portrait
from older_call_recall import CALL, WEIGHTS, OlderCallRecall, blend

ROOT = Path(__file__).resolve().parent


def base_lists(history, readers, cutoff):
    history = history.loc[history[TIME] < cutoff].copy()
    history[CALL] = history[CALL].map(attribute_value)
    history["单位"] = history["单位"].map(college)
    index = OlderCallRecall(history, cutoff)
    popular = sorted(index.readers, key=lambda code: (-len(index.readers[code]), code))
    if readers["读者ID"].eq("").any() or readers["读者ID"].duplicated().any():
        raise ValueError("读者 ID 为空或重复")
    evidence = []
    for row in readers.to_dict("records"):
        issued = pd.to_datetime(row.get("办证日期") or None)
        if issued is not None and issued < cutoff:
            seq = [s.strip() for s in row["历史借阅索书号列表"].split("|") if s.strip()]
            if seq:
                evidence.append((row["读者ID"],row.get("性别",""),seq))
    profiles = ProfileRecommender(history,cutoff,index,evidence)
    results = []
    for row in readers.to_dict("records"):
        seq = [v.strip() for v in row["历史借阅索书号列表"].split("|") if v.strip()]
        seen = set(seq)
        if row["用户类型"] not in {"老人", "新人"}:
            raise ValueError("无效的用户类型")
        if not seq and row["用户类型"] == "新人":
            attrs = portrait(row.get("学院",""),row.get("专业",""),row.get("年级",""),
                            row.get("读者类别",""),row.get("性别",""),"新人",cutoff)
            top = profiles.predict(attrs)
        elif not seq:
            top = index.popular[:10]
        else:
            channels = index.channels(seq, college(row["学院"]))
            if row["用户类型"] == "老人":
                top = blend(channels, WEIGHTS)
            else:
                channels["base"] = [i for i in popular if i not in seen][:100]
                top = blend(channels, {"base": 1, "cf": .75, "content": 1})
        results.append(top)
    return results
