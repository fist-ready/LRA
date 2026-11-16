"""从借阅流水重建开发窗答案，复算模型选型时存档的前十推荐。"""
from collections import defaultdict
from math import log2
from pathlib import Path

import numpy as np
import pandas as pd

from data_io import TIME, USER, attribute_value, read_history
from newcomer_portrait import GAP

ROOT = Path(__file__).resolve().parent
WINDOWS = ("2023-09-01", "2024-03-01", "2024-09-01")
EPOCHS = (8, 12, 16)
SEEDS = (2026, 2027)


def main():
    history = read_history(ROOT/"work/processed/borrowing_cleaned.csv")
    history["索书号"] = history["索书号"].map(attribute_value)
    predictions = pd.read_csv(ROOT/"evidence/dev_top10.csv.gz", dtype={"user":str})
    totals = []
    for date in WINDOWS:
        start = pd.Timestamp(date)
        past = history[history[TIME] < start]
        legal = set(past["索书号"]) - {""}
        seen = past.groupby(USER)["索书号"].agg(set).to_dict()
        older = set(past.loc[past[TIME] < start-GAP, USER])
        future = history[history[TIME].between(start, start+pd.DateOffset(months=2), inclusive="left")]
        truth = {user: set(rows["索书号"]) - seen.get(user, set()) - {""} for user,rows in future.groupby(USER)}
        truth = {user: books for user,books in truth.items() if books}
        for seed in SEEDS:
            for epoch in EPOCHS:
                part = predictions[(predictions.window==date)&(predictions.seed==seed)&(predictions.epoch==epoch)]
                if not part.user.is_unique or set(part.user)!=set(truth):
                    raise ValueError(f"{date}/{seed}/{epoch} 读者集合与评测答案不一致")
                groups = defaultdict(list)
                for user, top in part[["user","pred"]].itertuples(index=False,name=None):
                    ranked = top.split("|")
                    books = set(ranked)
                    if len(ranked)!=10 or len(books)!=10 or books-legal or books & seen.get(user,set()):
                        raise ValueError("开发预测的合法性校验失败")
                    answers = truth[user]
                    positions = [i for i,item in enumerate(ranked,2) if item in answers]
                    hit, recall = int(bool(positions)), len(positions)/len(answers)
                    if user in older:
                        ndcg = sum(1/log2(i) for i in positions)/sum(1/log2(i) for i in range(2,2+min(10,len(answers))))
                        groups["老人"].append(40*hit+30*recall+30*ndcg)
                    else:
                        groups["新人"].append(50*(hit+recall))
                new, old = np.mean(groups["新人"]), np.mean(groups["老人"])
                totals.append((date,seed,epoch,new,old,.4*new+.6*old))
    keys = ["window","seed","epoch"]
    frame = pd.DataFrame(totals,columns=keys+["新人","老人","total"]).set_index(keys).sort_index()
    reference = pd.read_csv(ROOT/"epoch_scores.csv").set_index(keys).sort_index()
    pd.testing.assert_frame_equal(frame,reference,rtol=0,atol=1e-9)
    print(frame.total.groupby(["epoch","window"]).mean().unstack().assign(平均分=lambda d:d.mean(axis=1)).round(3))
    print("三个窗口、两个种子、三个轮次均已从流水复算通过。")


if __name__=="__main__":
    main()
