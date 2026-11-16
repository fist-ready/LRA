"""核验画像 CSV，并生成只对完全无历史新人启用画像的备选文件。"""
import numpy as np
import pandas as pd

from baseline import BASE, USER,TIME, attribute_value, load_interactions
from newcomer_replay import CALL,DIAG,DEV
from evaluate_newcomer_portrait import OUT, PACKAGE


def main():
    data = load_interactions(BASE/"data/processed/borrowing_cleaned.csv")
    data[CALL] = data[CALL].map(attribute_value)
    roster = pd.read_csv(PACKAGE/"验证集发放.csv",dtype=str,keep_default_na=False).set_index("读者ID")
    base = pd.read_csv(BASE/"outputs/cold_demand_v1/提交结果_C题初赛.csv",dtype=str).set_index("读者ID")
    new = pd.read_csv(OUT/"验证集结果_新人画像.csv",dtype=str).set_index("读者ID")
    assert base.index.equals(new.index) and set(new.index)==set(roster.index) and len(new)==new.index.nunique()==956
    catalog = set(data.loc[data[TIME]<pd.Timestamp("2025-09-01"),CALL])-{""}
    only_cold = base.copy()
    for u,s in new["推荐索书号列表"].items():
        r = roster.loc[u]
        top = s.split("|")
        assert len(set(top))==len(top)==10 and set(top)<=catalog-set(r["历史借阅索书号列表"].split("|"))
        if r["用户类型"]=="老人":
            assert s==base.at[u,"推荐索书号列表"]
        elif not r["历史借阅索书号列表"].strip():
            only_cold.at[u,"推荐索书号列表"] = s
    only_cold.to_csv(OUT/"验证集结果_仅无历史新人画像.csv",encoding="utf-8-sig")
    frames = [pd.read_csv(OUT/name,dtype={"user":str}) for name in ["per_user.csv","registration_per_user.csv"]]
    rows = pd.concat(frames)
    for date in [*DEV,DIAG]:
        start = pd.Timestamp(date)
        end = data[TIME].max()+pd.Timedelta(1,"ns") if date==DIAG else start+pd.DateOffset(months=2)
        seen = data.loc[data[TIME]<start].groupby(USER)[CALL].agg(set).to_dict()
        future = data.loc[data[TIME].between(start,end,inclusive="left")].groupby(USER)[CALL].agg(set)
        for r in rows.loc[rows.window.eq(date)].itertuples():
            y = future[r.user]-seen.get(r.user,set())-{""}
            hits = len(set(r.pred.split("|")) & y)
            assert np.isclose(r.score,50*(bool(hits)+hits/len(y)))
    print("已独立复算",len(rows),"份逐人预测得分并核验 956 位目标读者")
    

if __name__=="__main__":
    main()

