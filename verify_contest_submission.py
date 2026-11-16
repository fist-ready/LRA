"""核验建议提交，并把最终方案三部分重新组合计分。"""
import json
import numpy as np
import pandas as pd

from baseline import BASE,USER,TIME,attribute_value,load_interactions
from newcomer_replay import DEV, DIAG, CALL
from cold_demand import OUT
from roster_replay import score


def main():
    data = load_interactions(BASE/"data/processed/borrowing_cleaned.csv")
    data[CALL] = data[CALL].map(attribute_value)
    package = BASE.parent/"C题初赛验证集"
    roster = pd.read_csv(package/"验证集发放.csv",dtype=str,keep_default_na=False).set_index("读者ID")
    books = pd.read_csv(package/"中文入藏（新书）.csv",dtype=str,encoding="gb18030",keep_default_na=False)
    train = set(data.loc[data[TIME]<pd.Timestamp("2025-09-01"),CALL])-{""}
    extra = set(books[CALL].map(attribute_value))-{""}-train
    sub = pd.read_csv(OUT/"提交结果_C题初赛.csv",dtype=str,keep_default_na=False)
    assert sub.columns.tolist()==["读者ID","推荐索书号列表"]
    assert len(sub)==sub["读者ID"].nunique()==len(roster)==956 and set(sub["读者ID"])==set(roster.index)
    new_slots = 0
    for u,s in sub.itertuples(index=False):
        top = s.split("|")
        assert len(set(top))==len(top)==10 and set(top)<=train|extra
        assert not set(top)&set(roster.at[u,"历史借阅索书号列表"].split("|"))
        new_slots += len(set(top)&extra)
    cold = pd.read_csv(OUT/"per_user.csv",dtype={"user":str})
    choice = json.loads((OUT/"locked_plan.json").read_text("utf8"))["selected"]
    series = BASE/"outputs/series_v1"
    rule = json.loads((series/"locked_plan.json").read_text("utf8"))["chosen"]
    portable = pd.read_csv(series/"per_user.csv",dtype={"user":str})
    lag = pd.read_csv(BASE/"outputs/roster_v1/per_user.csv",dtype={"user":str})
    prior = pd.read_csv(BASE/"outputs/newcomer_replay/summary.csv").query("method=='开发选定'").set_index("window")
    rows = []
    for date in [*DEV,DIAG]:
        for mode,src,ref in [("完整历史",portable.loc[portable.scenario.eq("历史回放")],rule),("发放间隔",lag,"基础")]:
            g = src.loc[src.window.eq(date)&src.method.eq(ref)].set_index("user")
            c = cold.loc[cold.scenario.eq(mode)&cold.window.eq(date)]
            baseline = c.loc[c.method.eq("近期热门")].set_index("user")
            match = baseline.index.intersection(g.index)
            assert len(match)==len(baseline) and g.loc[match,"group"].eq("新人").all()
            # 对齐两份回放分类时，以有无有效索书号作为无历史依据。
            np.testing.assert_allclose(g.loc[match,"score"],baseline.loc[match,"score"],atol=1e-9)
            g.loc[match,"score"] = c.loc[c.method.eq(choice)].set_index("user").loc[match,"score"]
            vals = g.groupby("group").score.mean()
            rows.append(dict(window=date,scenario=mode,older=vals["老人"],newcomer=vals["新人"],
                             total=.6*vals["老人"]+.4*vals["新人"],previous=prior.loc[date,"total"] if mode=="完整历史" else None))
    pd.DataFrame(rows).to_csv(OUT/"final_scores.csv",index=False,encoding="utf-8-sig")
    (OUT/"verification.json").write_text(json.dumps(dict(users=len(sub),slots=10*len(sub),new_slots=new_slots,
                                                       eligible_catalog=len(train|extra)),indent=2),"utf8")
    print(pd.DataFrame(rows).round(4));print("正式提交校验通过",new_slots)


if __name__=="__main__":
    main()

