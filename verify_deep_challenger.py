"""用原始流水独立核验挑战版的四窗数据与官方格式，并精确复现旧对照。"""
import json

import numpy as np
import pandas as pd

from baseline import BASE,USER,TIME,attribute_value,load_interactions
from newcomer_replay import DEV,DIAG,CALL,GAP
from roster_replay import score

OUT = BASE/"outputs/deep_challenger_v1"


def main():
    old_dir = BASE/"outputs/catalog_v1"
    data = load_interactions(BASE/"data/processed/borrowing_cleaned.csv")
    data[CALL] = data[CALL].map(attribute_value)
    rows_checked = 0
    for date in [*DEV,DIAG]:
        rows = pd.read_csv(OUT/(date+".csv"),dtype={"user":str})
        old = pd.read_csv(old_dir/(date+".csv"),dtype={"user":str})
        for a,b in [("基础","基础"),("冻结BGE融合","画像书目与语义")]:
            def preds(f,m):
                return f.loc[f.method.eq(m)].set_index(["scenario","user"]).pred.sort_index()
            pd.testing.assert_series_equal(preds(rows,a),preds(old,b))
        start = pd.Timestamp(date)
        end = start+pd.DateOffset(months=2) if date!=DIAG else data[TIME].max()+pd.Timedelta(1,"ns")
        h = data.loc[data[TIME]<start]
        seen = h.groupby(USER)[CALL].agg(set).to_dict()
        catalog = set(h[CALL])-{""}
        elders = set(h.loc[h[TIME]<start-GAP,USER])
        answers = {u:set(g[CALL])-seen.get(u,set())-{""} for u,g in data.loc[data[TIME].between(start,end,inclusive="left")].groupby(USER)}
        for r in rows.itertuples():
            top = r.pred.split("|")
            assert len(set(top))==len(top)==10 and set(top)<=catalog-seen.get(r.user,set())
            _,_,_,s = score(top,answers[r.user],"老人" if r.user in elders else "新人")
            assert abs(s-r.score)<1e-8
        rows_checked += len(rows)
    sub = pd.read_csv(OUT/"验证集结果_双塔挑战.csv",dtype=str)
    d = BASE.parent/"C题初赛验证集"
    roster = pd.read_csv(d/"验证集发放.csv",dtype=str,keep_default_na=False).set_index("读者ID")
    catalog = pd.read_csv(d/"中文入藏（新书）.csv",dtype=str,encoding="gb18030",keep_default_na=False)
    history = set(data.loc[data[TIME]<pd.Timestamp("2025-09-01"),CALL])-{""}
    new = set(catalog[CALL].map(attribute_value))-{""}-history
    assert sub.columns.tolist()==["读者ID","推荐索书号列表"]
    assert len(sub)==sub["读者ID"].nunique()==len(roster)==956 and set(sub["读者ID"])==set(roster.index)
    new_count = 0
    for u,s in sub.itertuples(index=False):
        top = s.split("|")
        assert len(top)==len(set(top))==10 and set(top)<=history|new
        assert not set(top)&set(roster.at[u,"历史借阅索书号列表"].split("|"))
        new_count += len(set(top)&new)
    results = dict(checked=rows_checked,readers=len(sub),new_slots=new_count)
    (OUT/"verification.json").write_text(json.dumps(results,indent=2),"utf8")
    print(results)


if __name__=="__main__":
    main()

