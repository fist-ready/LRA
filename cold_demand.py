"""按新人 HitRate 和 Recall 估计整张列表覆盖的需求，单独评价无历史读者。"""
from collections import Counter,defaultdict
import json
import math

import pandas as pd

from baseline import BASE,USER,TIME,attribute_value,load_interactions
from newcomer_replay import CALL,DEV,DIAG,GAP
from roster_replay import score

OUT = BASE/"outputs/cold_demand_v1"
METHODS = ["近期热门","宏平均需求","名单覆盖","初借名单覆盖"]


def predictions(history,start):
    pop = Counter()
    rows = defaultdict(list)
    time0 = history[TIME].min()
    recent = history.drop_duplicates([USER,CALL],keep="last")
    for u,i,t in recent[[USER,CALL,TIME]].itertuples(index=False):
        pop[i] += .2+.8*2**(-(start-t).days/120)
    for u,g in history.groupby(USER):
        first = g[TIME].min()
        for block,part in g.groupby(((g[TIME]-first).dt.days//60).to_numpy()):
            # 在初借之后按独立的60天书单估计需求，不让高频读者的每一借权重相同。
            books = sorted(part[CALL].unique())
            w = .2+.8*2**(-(start-part[TIME].max()).days/120)
            rows["all"].append((books,w))
            if block==0 and first>=time0+pd.Timedelta(days=60):
                rows["first"].append((books,w))
    def greedy(profiles,discount):
        counts = Counter()
        reverse = defaultdict(list)
        for j,(books,w) in enumerate(profiles):
            for i in books:
                counts[i] += w*(.5+.5/len(books))
                reverse[i].append(j)
        covered = set()
        selected = []
        for _ in range(10):
            best = min(counts,key=lambda i:(-counts[i],i))
            selected.append(best)
            if discount:
                for j in reverse[best]:
                    if j not in covered:
                        covered.add(j)
                        books,w = profiles[j]
                        for i in books:
                            if i in counts:
                                counts[i] -= .5*w
            del counts[best]
        return selected
    return dict(zip(METHODS,[sorted(pop,key=lambda i:(-pop[i],i))[:10],greedy(rows["all"],False),
                             greedy(rows["all"],True),greedy(rows["first"],True)]))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    data = load_interactions(BASE/"data/processed/borrowing_cleaned.csv")
    data[CALL] = data[CALL].map(attribute_value)
    data = data.loc[data[CALL].ne("")]
    rows = []
    for date in [*DEV,DIAG]:
        if date==DIAG:
            dev = pd.DataFrame(rows)
            s = dev.groupby(["scenario","method","window"]).score.mean().groupby(["scenario","method"]).mean().unstack()
            gain = s.sub(s["近期热门"],axis=0)
            selected = max(METHODS,key=lambda m:gain[m].min())
            (OUT/"locked_plan.json").write_text(json.dumps({"selected":selected,"criterion":"完整历史和发放间隔两种开发回放各自增益取较小者"},ensure_ascii=False),"utf8")
        start = pd.Timestamp(date)
        end = start+pd.DateOffset(months=2) if date!=DIAG else data[TIME].max()+pd.Timedelta(1,"ns")
        h = data.loc[data[TIME]<start]
        new = data.loc[data[TIME].between(start,end,inclusive="left")&~data[USER].isin(h[USER])]
        for regime,train in [("完整历史",h),("发放间隔",h.loc[h[TIME]<start-GAP])]:
            for method,top in predictions(train,start).items():
                for u,g in new.groupby(USER):
                    y = set(g[CALL])
                    hit,recall,_,s = score(top,y,"新人")
                    rows.append(dict(window=date,scenario=regime,method=method,user=u,hit=hit,recall=recall,score=s))
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT/"per_user.csv",index=False,encoding="utf-8-sig")
    summary = frame.groupby(["window","scenario","method"])[["score","hit","recall"]].mean()
    summary.to_csv(OUT/"summary.csv",encoding="utf-8-sig")
    directory = BASE.parent/"C题初赛验证集"
    roster = pd.read_csv(directory/"验证集发放.csv",dtype=str,keep_default_na=False).set_index("读者ID")
    src = pd.read_csv(BASE/"outputs/series_v1/提交结果_C题初赛.csv",dtype=str)
    start = pd.Timestamp("2025-09-01")
    hist = data.loc[data[TIME]<start]
    top = predictions(hist,start)[selected]
    mask = set(roster.index[roster["用户类型"].eq("新人")&roster["历史借阅索书号列表"].str.strip().eq("")])
    assert len(mask)==610 and len(set(top))==10 and set(top)<=set(hist[CALL])
    src.loc[src["读者ID"].isin(mask),"推荐索书号列表"] = "|".join(top)
    src.to_csv(OUT/"提交结果_C题初赛.csv",encoding="utf-8-sig",index=False)
    print(selected);print(summary.score.unstack().round(3))


if __name__=="__main__":
    main()

