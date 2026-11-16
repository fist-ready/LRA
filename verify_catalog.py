"""脱离推荐器，从读者清单和历史流水验证提交及历史回放。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

from baseline import BASE, USER, TIME, attribute_value, load_interactions
from newcomer_replay import DEV, DIAG, GAP, CALL
OUT = BASE/"outputs/catalog_v1"


def verify():
    data = load_interactions(BASE/"data/processed/borrowing_cleaned.csv")
    data[CALL] = data[CALL].map(attribute_value)
    directory = BASE.parent/"C题初赛验证集"
    roster = pd.read_csv(directory/"验证集发放.csv",dtype=str,keep_default_na=False).set_index("读者ID")
    books = pd.read_csv(directory/"中文入藏（新书）.csv",encoding="gb18030",dtype=str,keep_default_na=False)
    hist = set(data.loc[data[TIME]<pd.Timestamp("2025-09-01"),CALL])-{""}
    new = set(books[CALL].map(attribute_value))-{""}-hist
    export = {}
    for variant in ("选定","内容"):
        sub = pd.read_csv(OUT/("提交结果_"+variant+".csv"),dtype=str,keep_default_na=False)
        assert sub.columns.tolist()==["读者ID","推荐索书号列表"]
        assert len(sub)==sub["读者ID"].nunique()==len(roster)==956 and set(sub["读者ID"])==set(roster.index)
        count = readers = 0
        for u,s in sub.itertuples(index=False):
            top = s.split("|")
            assert len(set(top))==len(top)==10 and set(top)<=hist|new
            assert not set(top)&set(roster.at[u,"历史借阅索书号列表"].split("|"))
            hits = len(set(top)&new)
            count += hits
            readers += bool(hits)
        export[variant] = dict(new_slots=count,readers_with_new=readers)
    plan = json.loads((OUT/"locked_plan.json").read_text("utf8"))
    comparisons = []
    previous = pd.read_csv(BASE/"outputs/newcomer_replay/summary.csv").query("method=='开发选定'").set_index("window")
    for date in [*DEV,DIAG]:
        start = pd.Timestamp(date)
        end = start+pd.DateOffset(months=2) if date!=DIAG else data[TIME].max()+pd.Timedelta(1,"ns")
        h = data.loc[data[TIME]<start]
        seen = h.groupby(USER)[CALL].agg(set).to_dict()
        truth = {u:set(g[CALL])-seen.get(u,set())-{""} for u,g in data.loc[data[TIME].between(start,end,inclusive="left")].groupby(USER)}
        truth = {u:y for u,y in truth.items() if y}
        catalog = set(h[CALL])-{""}
        old = set(h.loc[h[TIME]<start-GAP,USER])
        rows = pd.read_csv(OUT/(date+".csv"),dtype={"user":str})
        for (scenario,method),g in rows.groupby(["scenario","method"]):
            assert len(g)==g.user.nunique()==len(truth) and set(g.user)==set(truth)
            for row in g.itertuples():
                u = row.user
                top = row.pred.split("|")
                assert len(set(top))==len(top)==10 and set(top)<=catalog-seen.get(u,set())
                y = truth[u]
                positions = [i for i,item in enumerate(top,2) if item in y]
                hit,rec = bool(positions),len(positions)/len(y)
                ndcg = sum(1/np.log2(i) for i in positions)/sum(1/np.log2(i) for i in range(2,2+min(10,len(y))))
                score = 40*hit+30*rec+30*ndcg if u in old else 50*(hit+rec)
                assert abs(score-row.score)<1e-9
        chosen = rows.loc[rows.method.eq(rows.segment.map(plan))].copy()
        chosen["method"] = "选定"
        whole = pd.concat([rows,chosen])
        for (scenario,method),g in whole.groupby(["scenario","method"]):
            scores = g.groupby("group").score.mean()
            total = .4*scores["新人"]+.6*scores["老人"]
            comparisons.append(dict(window=date,scenario=scenario,method=method,older=scores["老人"],newcomer=scores["新人"],
                                    total=total,previous=previous.loc[date,"total"] if scenario=="历史回放" else None))
    pd.DataFrame(comparisons).to_csv(OUT/"comparison.csv",index=False,encoding="utf-8-sig")
    (OUT/"verification.json").write_text(json.dumps(export,ensure_ascii=False,indent=2),"utf8")
    print(export)


if __name__=="__main__":
    verify()

