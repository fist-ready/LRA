"""模拟训练末日与预测日间隔，用已发放的全体读者截止前书单补充基础模型。"""
from collections import Counter
from math import sqrt
import json

import numpy as np
import pandas as pd

from baseline import BASE, USER, TIME, attribute_value, load_interactions
from newcomer_replay import CALL, DEV, DIAG, GAP, college
from older_call_recall import OlderCallRecall, WEIGHTS, ranked, blend

OUT = BASE/"outputs/roster_v1"
SRC = BASE/"outputs/catalog_v1"
PACKAGE = BASE.parent/"C题初赛验证集"
CONFIG = {"基础":0,"轻融合":.5,"等权":1.,"同伴优先":2.}


def field(v):
    return attribute_value(v).removesuffix("本科")


def score(top,y,group):
    pos = [i for i,x in enumerate(top,2) if x in y]
    recall = len(pos)/len(y)
    ndcg = sum(1/np.log2(i) for i in pos)/sum(1/np.log2(i) for i in range(2,2+min(10,len(y))))
    hit = int(bool(pos))
    return hit,recall,ndcg,50*(hit+recall) if group=="新人" else 40*hit+30*recall+30*ndcg


def recommend(history,start,roster,legal):
    old = OlderCallRecall(history,start)
    pop = sorted(old.readers,key=lambda i:(-len(old.readers[i]),i))
    public_degree = Counter(i for r in roster for i in set(r["seq"]))
    public = []
    for r in roster:
        seq = list(dict.fromkeys(r["seq"]))
        # 用预测前最后借阅日衡量整份书单的新近程度，不把它当作每本书的日期。
        days = max(0,(start-r["last"]).days) if pd.notna(r["last"]) else 730
        freshness = .15+2**(-days/120)
        public.append((set(seq),{x.split("/")[0] for x in seq},freshness,
                       [(x,1/sqrt(1+i/8)) for i,x in enumerate(reversed(seq)) if x in legal]))
    for n,r in enumerate(roster):
        seq = r["seq"]
        seen,cats,_,_ = public[n]
        channel = old.channels(seq,r["dept"])
        if not seq:
            base = old.popular[:100]
        elif r["group"]=="老人":
            base = blend(channel,WEIGHTS)
        else:
            channel["base"] = [i for i in pop if i not in seen][:100]
            base = blend(channel,{"base":1,"cf":.75,"content":1})
        votes = Counter()
        for j,(books,themes,fresh,items) in enumerate(public):
            if n==j or not items:
                continue
            peer = roster[j]
            affinity = .3 + 2*bool(r["dept"] and r["dept"]==peer["dept"]) + 4*bool(r["major"] and r["major"]==peer["major"])
            affinity += sum(5/sqrt(public_degree[i]) for i in seen&books)+2*len(cats&themes)/sqrt(max(1,len(cats)))
            scale = affinity*fresh/sqrt(len(books))
            for i,recency in items:
                if i not in seen:
                    votes[i] += scale*recency
        orders = {"base":base,"public":ranked(votes)[:100]}
        predictions = {name:(blend(orders,{"base":1,"public":w}) if w else base)[:10] for name,w in CONFIG.items()}
        yield r,predictions


def staged(data,date):
    start = pd.Timestamp(date)
    boundary = start-GAP
    train = data.loc[data[TIME]<boundary]
    history = data.loc[data[TIME]<start]
    end = start+pd.DateOffset(months=2) if date!=DIAG else data[TIME].max()+pd.Timedelta(1,"ns")
    past = {u:g for u,g in history.groupby(USER)}
    old = set(train[USER])
    readers,truth = [],{}
    for u,g in data.loc[data[TIME].between(start,end,inclusive="left")].groupby(USER):
        h = past.get(u,history.iloc[:0])
        seq = h.loc[h[CALL].ne("")].drop_duplicates(CALL,keep="first")[CALL].tolist()
        y = set(g[CALL])-set(seq)-{""}
        if not y:
            continue
        profile = {}
        for key,fn in [("单位",college),("专业",field)]:
            vals = h[key].map(fn)
            profile[key] = next((v for v in reversed(vals.tolist()) if v),"")
        readers.append(dict(user=u,seq=seq,dept=profile["单位"],major=profile["专业"],last=h[TIME].max(),
                            group="老人" if u in old else "新人"))
        truth[u] = y
    legal = set(train[CALL])|{i for r in readers for i in r["seq"]}
    rows = []
    for r,plans in recommend(train,start,readers,legal):
        y = truth[r["user"]]
        for m,top in plans.items():
            assert len(set(top))==len(top)==10 and set(top)<=legal-set(r["seq"])
            hit,recall,ndcg,s = score(top,y,r["group"])
            rows.append(dict(window=date,user=r["user"],group=r["group"],method=m,pred="|".join(top),
                             hit=hit,recall=recall,ndcg=ndcg,score=s))
    return pd.DataFrame(rows)


def export(data,plan):
    roster = pd.read_csv(PACKAGE/"验证集发放.csv",dtype=str,keep_default_na=False)
    b = pd.read_csv(PACKAGE/"中文入藏（新书）.csv",dtype=str,keep_default_na=False,encoding="gb18030")
    start = pd.Timestamp("2025-09-01")
    train = data.loc[data[TIME]<start]
    readers = [dict(user=r["读者ID"],seq=[i.strip() for i in r["历史借阅索书号列表"].split("|") if i.strip()],
                    dept=college(r["学院"]),major=field(r["专业"]),last=pd.to_datetime(r["最近借阅日期"] or None),
                    group=r["用户类型"]) for r in roster.to_dict("records")]
    legal = (set(train[CALL])|set(b[CALL].map(attribute_value))|{i for r in readers for i in r["seq"]})-{""}
    result = []
    for r,orders in recommend(train,start,readers,legal):
        top = orders[plan[r["group"]]]
        assert len(set(top))==len(top)==10 and set(top)<=legal-set(r["seq"])
        result.append({"读者ID":r["user"],"推荐索书号列表":"|".join(top)})
    out = pd.DataFrame(result)
    assert len(out)==out["读者ID"].nunique()==956
    out.to_csv(OUT/"提交结果_读者书单融合.csv",index=False,encoding="utf-8-sig")


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    data = load_interactions(BASE/"data/processed/borrowing_cleaned.csv")
    data[CALL] = data[CALL].map(attribute_value)
    frames = []
    for d in [*DEV,DIAG]:
        if d==DIAG:
            dev = pd.concat(frames)
            scores = dev.groupby(["group","method","window"]).score.mean().groupby(["group","method"]).mean()
            plan = {group:max(CONFIG,key=scores.loc[group].get) for group in ["新人","老人"]}
            (OUT/"locked_plan.json").write_text(json.dumps(plan,ensure_ascii=False),"utf8")
        frames.append(staged(data,d))
    results = pd.concat(frames)
    results.to_csv(OUT/"per_user.csv",index=False,encoding="utf-8-sig")
    selected = results.loc[results.method.eq(results.group.map(plan))].copy()
    selected["method"] = "选定"
    both = pd.concat([results,selected])
    groups = both.groupby(["window","method","group"])[["score","hit","recall","ndcg"]].mean()
    groups.to_csv(OUT/"groups.csv",encoding="utf-8-sig")
    table = groups.score.unstack()
    table["total"] = .4*table["新人"]+.6*table["老人"]
    table.to_csv(OUT/"summary.csv",encoding="utf-8-sig")
    export(data,plan)
    print(plan);print(table.total.unstack().round(3))


if __name__=="__main__":
    main()

