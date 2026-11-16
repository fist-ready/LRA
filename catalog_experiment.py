"""原始回放、遮蔽图书/读者实验及官方格式导出。"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from baseline import BASE, USER, TIME, attribute_value, load_interactions
from catalog_recommend import Catalog, META, CALL, METHODS, Profile, bucket, clc
from newcomer_replay import DEV, DIAG, GAP
from ranking_models import TextEncoder
from call_number_replay import metrics

OUT = BASE/"outputs/catalog_v1"
PACKAGE = BASE.parent/"C题初赛验证集"


def write(name,data):
    pd.DataFrame(data).to_csv(OUT/name,index=False,encoding="utf-8-sig")


def measure(top,y,group):
    vals = metrics(top,y)
    h,r,n = vals["hit"],vals["recall"],vals["ndcg"]
    return 50*(h+r) if group=="新人" else 40*h+30*r+30*n


def experiments(data,date,encoder):
    start = pd.Timestamp(date)
    end = start+pd.DateOffset(months=2) if date!=DIAG else data[TIME].max()+pd.Timedelta(1,"ns")
    history = data.loc[data[TIME]<start]
    meta = history.loc[history[CALL].ne(""),META].drop_duplicates(CALL,keep="last")
    old = set(history.loc[history[TIME]<start-GAP,USER])
    seen = history.groupby(USER)[CALL].agg(set).to_dict()
    truth = {u:set(g[CALL])-seen.get(u,set())-{""} for u,g in
             data.loc[data[TIME].between(start,end,inclusive="left")].groupby(USER)}
    truth = {u:v for u,v in truth.items() if v}
    mask = {i for i in meta[CALL] if bucket(i,5)==0}
    rows = []
    for scenario in ("历史回放","冷书模拟"):
        h = history if scenario=="历史回放" else history.loc[~history[CALL].isin(mask)]
        fields = {key:h.loc[h[key].map(attribute_value).ne("")].groupby(USER)[key].last().to_dict()
                  for key in ("专业","单位")}
        model = Catalog(h,start,meta,encoder)
        for u,y in truth.items():
            group = "老人" if u in old else "新人"
            seq = model.index.profile.get(u,[])
            segment = "老人" if group=="老人" else ("短历史新人" if seq else "无历史新人")
            plans = model.recommend(seq,[],fields["专业"].get(u,""),fields["单位"].get(u,""),group)
            for name,order in plans.items():
                # 原有已借仍不计分；模拟中遮蔽部分不得作为兴趣种子。
                pool = [i for i in order if i not in seen.get(u,set())]
                top = pool[:10]
                assert len(set(top))==10
                hits = y.intersection(top)
                vals = metrics(top,y)
                rows.append(dict(window=date,scenario=scenario,user=u,group=group,segment=segment,method=name,
                                 score=measure(top,y,group),truth=len(y),hits=len(hits),pool_hits=len(y.intersection(pool)),
                                 masked_hits=len(hits&mask),masked_truth=len(y&mask),pred="|".join(top),**{k:vals[k] for k in ('hit','recall','ndcg')}))
    # 单独模拟只有画像的新人；建统计时完全移除同折读者。
    profile_rows = []
    sample = sorted(u for u in truth if u not in old and u in seen)
    for fold in range(5):
        users = [u for u in sample if bucket(u,5)==fold]
        if not users:
            continue
        train = history.loc[~history[USER].isin(users)&history[CALL].ne("")]
        profile = Profile(train)
        items = sorted(train[CALL].unique())
        pairs = train.drop_duplicates([USER,CALL],keep="last")
        weights = .2+.8*2**(-(start-pairs[TIME]).dt.days/120)
        hot = weights.groupby(pairs[CALL]).sum().reindex(items).to_numpy()
        for u in users:
            mh = history.loc[history[USER].eq(u)]
            values = [mh[key].map(attribute_value).loc[lambda s:s.ne("")] for key in ["专业","单位"]]
            m,d = [s.iloc[-1] if len(s) else "" for s in values]
            if not (m or d):
                continue
            lifts = profile.lifts(m,d)
            alt = (hot+.25)*np.array([lifts.get(clc(i),1) for i in items])
            def top_order(scores):
                return [items[i] for i in np.argsort(-scores,kind="stable")[:100]]
            base = top_order(hot)
            from older_call_recall import blend
            for name,top in [("基础",base[:10]),("画像",blend({"base":base,"profile":top_order(alt)},{"base":1,"profile":.7})[:10])]:
                profile_rows.append(dict(window=date,user=u,method=name,score=measure(top,truth[u],"新人")))
    return pd.DataFrame(rows),pd.DataFrame(profile_rows)


def choose(dev,proxy):
    live = dev.loc[dev.scenario.eq("历史回放")]
    plan = {}
    for segment,part in live.groupby("segment"):
        scores = part.groupby(["window","method"]).score.mean().groupby("method").mean()
        plan[segment] = max(METHODS,key=scores.get)
    p = proxy.groupby(["window","method"]).score.mean().groupby("method").mean()
    plan["无历史新人"] = max(["基础","画像"],key=p.get)
    return plan


def official(data,encoder,plan):
    start = pd.Timestamp("2025-09-01")
    history = data.loc[data[TIME]<start]
    hmeta = history.loc[history[CALL].ne(""),META]
    b = pd.read_csv(PACKAGE/"中文入藏（新书）.csv",encoding="gb18030",dtype=str,keep_default_na=False)
    b[CALL] = b[CALL].map(attribute_value)
    # 保留历史书目优先；官方明确发放整个目录为可用候选。
    meta = pd.concat([b.loc[b[CALL].ne(""),META],hmeta])
    model = Catalog(history,start,meta,encoder)
    roster = pd.read_csv(PACKAGE/"验证集发放.csv",dtype=str,keep_default_na=False)
    new = set(b[CALL])-set(hmeta[CALL])
    result,stats = [],Counter()
    for row in roster.to_dict("records"):
        seq = [s.strip() for s in row["历史借阅索书号列表"].split("|") if s.strip()]
        titles = row["历史借阅题名列表"].split("|")
        if len(titles)!=len(seq):
            titles = []  # 位置不可靠时仍保留索书号种子
        segment = "老人" if row["用户类型"]=="老人" else ("短历史新人" if seq else "无历史新人")
        outputs = model.recommend(seq,titles,row["专业"],row["学院"],row["用户类型"])
        for variant,method in [("选定",plan[segment]),("内容",METHODS[-1])]:
            top = outputs[method][:10]
            assert len(set(top))==len(top)==10 and not set(top)&set(seq) and set(top)<=model.ids.keys()
            stats[variant] += len(set(top)&new)
            result.append({"variant":variant,"读者ID":row["读者ID"],"推荐索书号列表":"|".join(top)})
    for variant,part in pd.DataFrame(result).groupby("variant"):
        assert len(part)==len(set(part["读者ID"]))==len(roster)==956
        write("提交结果_"+variant+".csv",part.drop(columns="variant"))
    return dict(stats)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    data = load_interactions(BASE/"data/processed/borrowing_cleaned.csv")
    data[CALL] = data[CALL].map(attribute_value)
    encoder = TextEncoder(BASE/".model_deps/bge-small-zh-v1.5",OUT/"embeddings")
    all_rows,profiles = [],[]
    for date in [*DEV,DIAG]:
        if date==DIAG:
            plan = choose(pd.concat(all_rows),pd.concat(profiles))
            (OUT/"locked_plan.json").write_text(json.dumps(plan,ensure_ascii=False,indent=2),"utf8")
        print("回放",date,flush=True)
        rows,pr = experiments(data,date,encoder)
        write(date+".csv",rows)
        all_rows.append(rows)
        profiles.append(pr)
    details = pd.concat(all_rows)
    write("profile_proxy.csv",pd.concat(profiles))
    selected = details.loc[details.method.eq(details.segment.map(plan))].copy()
    selected["method"] = "选定"
    summary = pd.concat([details,selected]).groupby(["scenario","window","method","group"]).agg(
        users=("user","size"),score=("score","mean"),hit=("hit","mean"),recall=("recall","mean"),ndcg=("ndcg","mean"),
        pool_hits=("pool_hits","sum"),truth=("truth","sum"),masked_hits=("masked_hits","sum"),masked_truth=("masked_truth","sum")).reset_index()
    write("summary.csv",summary)
    stats = official(data,encoder,plan)
    (OUT/"export_stats.json").write_text(json.dumps(stats,ensure_ascii=False,indent=2),"utf8")
    print(plan,stats,flush=True)


if __name__=="__main__":
    main()

