"""独立核对索书号排序逐人输出，并按跨窗读者聚类估计分差区间。"""
from pathlib import Path

import numpy as np
import pandas as pd

from baseline import TIME, USER, attribute_value, load_interactions

BASE = Path(__file__).resolve().parent
OUT = BASE/"outputs/older_call_ranker"


def verify_scores(directory):
    data = load_interactions(BASE/"data/processed/borrowing_cleaned.csv")
    data["索书号"] = data["索书号"].map(attribute_value)
    result = pd.read_csv(directory/"per_user.csv",dtype={"user":str})
    assert not result.duplicated(["window","user","method"]).any()
    score_table = pd.read_csv(directory/"summary.csv").set_index(["window","method"])
    for window,rows in result.groupby("window"):
        start = pd.Timestamp(window)
        end = start+pd.DateOffset(months=2) if window!="2024-11-01" else data[TIME].max()+pd.Timedelta(1,"ns")
        history = data.loc[data[TIME]<start]
        old = set(history.loc[history[TIME]<start-(pd.Timestamp("2025-09-01")-pd.Timestamp("2025-01-17")),USER])
        seen = history.groupby(USER)["索书号"].agg(set)
        catalog = set(history["索书号"])-{""}
        future = data.loc[data[TIME].between(start,end,inclusive="left")].groupby(USER)["索书号"].agg(set)
        truth = {u:items-seen[u]-{""} for u,items in future.items() if u in old}
        eligible = {u for u,items in truth.items() if items}
        for method,group in rows.groupby("method"):
            assert set(group.user)==eligible
            scores = []
            for r in group.itertuples():
                pred = r.pred.split("|")
                assert len(set(pred))==len(pred)==10 and set(pred)<=catalog-seen[r.user]
                correct = [k for k,v in enumerate(pred,2) if v in truth[r.user]]
                n = len(truth[r.user])
                recall = len(correct)/n
                ndcg = sum(1/np.log2(k) for k in correct)/sum(1/np.log2(k) for k in range(2,2+min(10,n)))
                scores.append(40*bool(correct)+30*recall+30*ndcg)
            np.testing.assert_allclose(scores,group.score,atol=1e-12)
            np.testing.assert_allclose(np.mean(scores),score_table.loc[(window,method),"score"])
    return result


def main():
    result = verify_scores(OUT)
    rows = []
    for name,part in result.groupby(result.window.eq("2024-11-01").map({True:"复用诊断",False:"开发"})):
        pivot = part.pivot(index=["user","window"],columns="method",values="score")
        for reference in ("原流程","旧树全池","等权融合"):
            d = (pivot["新借排序"]-pivot[reference]).unstack()
            a = d.to_numpy()
            rng = np.random.default_rng(20261105)
            samples = [np.nanmean(a[rng.integers(len(a),size=len(a))],axis=0).mean() for _ in range(4000)]
            low,high = np.quantile(samples,[.025,.975])
            rows.append(dict(split=name,relative_to=reference,delta=d.mean().mean(),low=low,high=high))
    intervals = pd.DataFrame(rows)
    intervals.to_csv(OUT/"paired.csv",encoding="utf-8-sig",index=False)
    print("核验通过",len(result),"行"); print(intervals.round(4).to_string(index=False))


if __name__=="__main__":
    main()

