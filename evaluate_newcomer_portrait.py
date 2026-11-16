"""新人画像对照：常规回放、隐藏历史但保留画像的代理实验、官方格式输出。"""
import hashlib
import json
from collections import Counter

import numpy as np
import pandas as pd

from baseline import BASE,USER,TIME,attribute_value,load_interactions
from newcomer_replay import DEV,DIAG,GAP,CALL
from newcomer_portrait import ProfileRecommender,portrait,WEIGHTS

OUT = BASE/"outputs/newcomer_portrait_v1"
PACKAGE = BASE.parent/"C题初赛验证集"


def per_user(u,date,mode,segment,y,model,p,seq):
    method = "个人历史" if seq else "近期"
    base = model.base.predict(seq,p["学院"],[method])[method]
    altered,used = model.predict(p,seq)
    for name,top in [("主提交",base),("画像",altered)]:
        assert len(set(top))==len(top)==10 and set(top)<=model.ids.keys()-set(seq)
        hits = len(set(top)&y)
        yield dict(window=date,mode=mode,segment=segment,user=u,method=name,truth=len(y),
                   pred="|".join(top),hit=int(hits>0),recall=hits/len(y),score=50*(bool(hits)+hits/len(y)),
                   used="|".join(used) if name=="画像" else "")


def replay(data,date):
    start = pd.Timestamp(date)
    end = start+pd.DateOffset(months=2) if date!=DIAG else data[TIME].max()+pd.Timedelta(1,"ns")
    h = data.loc[data[TIME]<start]
    old = set(h.loc[h[TIME]<start-GAP,USER])
    model = ProfileRecommender(h,start)
    profiles = {}
    fields = ["单位","专业","年级","读者级别","性别"]
    for key in fields:
        profiles[key] = h.loc[h[key].map(attribute_value).ne("")].groupby(USER)[key].last().to_dict() if key in h else {}
    truth,attrs,rows = {},{},[]
    target = data.loc[data[TIME].between(start,end,inclusive="left")&~data[USER].isin(old)]
    for u,g in target.groupby(USER):
        seq = model.base.index.profile.get(u,[])
        y = set(g[CALL])-set(seq)-{""}
        if not y:
            continue
        p = portrait(*(profiles[k].get(u,"") for k in fields),"新人",start)
        truth[u],attrs[u] = y,p
        rows.extend(per_user(u,date,"常规","有历史" if seq else "无历史",y,model,p,seq))
    for fold in range(5):
        users = {u for u,p in attrs.items() if any(p[k] for k in fields_to_check)
                 and int(hashlib.sha256(u.encode()).hexdigest()[:8],16)%5==fold}
        if users:
            proxy = ProfileRecommender(h.loc[~h[USER].isin(users)],start)
            for u in sorted(users):
                rows.extend(per_user(u,date,"仅画像模拟","无历史",truth[u],proxy,attrs[u],[]))
    return rows


fields_to_check = ("学院","专业","年级","读者类别","性别")


def official(data):
    v = pd.read_csv(PACKAGE/"验证集发放.csv",dtype=str,keep_default_na=False)
    base = pd.read_csv(BASE/"outputs/cold_demand_v1/提交结果_C题初赛.csv",dtype=str).set_index("读者ID")
    assert base.index.is_unique and set(base.index)==set(v["读者ID"])
    start = pd.Timestamp("2025-09-01")
    history = lambda s:[x.strip() for x in s.split("|") if x.strip()]
    evidence = []
    for row in v.to_dict("records"):
        issued = pd.to_datetime(row["办证日期"] or None)
        if issued is not None and issued<start and history(row["历史借阅索书号列表"]):
            evidence.append((row["读者ID"],row["性别"],history(row["历史借阅索书号列表"])))
    model = ProfileRecommender(data,start,evidence)
    output = base.copy()
    detail = []
    for row in v.loc[v["用户类型"].eq("新人")].to_dict("records"):
        u = row["读者ID"]
        seq = history(row["历史借阅索书号列表"])
        p = portrait(*(row[k] for k in ("学院","专业","年级","读者类别","性别","用户类型")),start)
        top,used = model.predict(p,seq)
        assert len(set(top))==len(top)==10 and set(top)<=model.ids.keys()-set(seq)
        before = base.loc[u,"推荐索书号列表"].split("|")
        output.loc[u,"推荐索书号列表"] = "|".join(top)
        detail.append(dict(user=u,segment="有历史" if seq else "无历史",used="|".join(used),
                           changed=set(before)!=set(top),replacements=len(set(top)-set(before))))
    assert len(output)==956 and len(detail)==674
    output.to_csv(OUT/"验证集结果_新人画像.csv",encoding="utf-8-sig")
    pd.DataFrame(detail).to_csv(OUT/"official_audit.csv",index=False,encoding="utf-8-sig")


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    data = load_interactions(BASE/"data/processed/borrowing_cleaned.csv")
    data[CALL] = data[CALL].map(attribute_value)
    official(data)
    rows = []
    for date in [*DEV,DIAG]:
        print(date,flush=True)
        rows.extend(replay(data,date))
    f = pd.DataFrame(rows)
    f.to_csv(OUT/"per_user.csv",index=False,encoding="utf-8-sig")
    sums = f.groupby(["window","mode","method"])[["score","hit","recall"]].mean()
    older = pd.read_csv(BASE/"outputs/cold_demand_v1/final_scores.csv").query("scenario=='完整历史'").set_index("window")
    sums["users"] = f.groupby(["window","mode","method"]).size()
    sums["total"] = [.4*row.score+.6*older.at[w,"older"] if mode=="常规" else np.nan
                     for (w,mode,_),row in sums.iterrows()]
    np.testing.assert_allclose(sums.xs(("常规","主提交"),level=("mode","method")).total,older.sort_index().total)
    sums.to_csv(OUT/"summary.csv",encoding="utf-8-sig")
    print(sums.round(4).to_string())


if __name__=="__main__":
    main()

