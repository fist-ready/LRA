"""Paired historical test of calendar/cohort priors and one-position reranking."""
import hashlib
import json
import math
from collections import Counter, defaultdict
from functools import lru_cache

import numpy as np
import pandas as pd

from baseline import BASE, USER, TIME, attribute_value
from catalog_recommend import CALL, clc, major
from event_encoder_b import read_data
from newcomer_portrait import stage
from newcomer_replay import DEV, DIAG, college
from prior_swap_folds import OUT, fold_data, horizon_ends
from roster_replay import score

VARIANTS = ("原版", "时序先验", "时序群体先验", "随机换位", "证据换位", "多目标换位", "先验加多目标", "双证据择换")


def term(t):
    return "春夏" if 3 <= t.month <= 8 else "秋冬"


class DemandPrior:
    """Estimate smoothed fine-class demand using only unique reader/class evidence before cutoff."""
    def __init__(self, frame, cutoff):
        self.cutoff = cutoff
        self.counts = defaultdict(Counter)
        self.support = defaultdict(set)
        last = {}
        fields = [USER, CALL, TIME, "专业", "单位", "年级"]
        for u, book, t, m, d, year in frame[fields].itertuples(index=False, name=None):
            category = clc(book)
            keys = [("all","*"), ("term",term(t)), ("major",major(m)),
                    ("college",college(d)), ("stage",stage(year,t))]
            for key, value in keys:
                if value:
                    last[key,value,u,category] = t
        for (key,value,u,category), t in last.items():
            self.support[key,value].add(u)
            w = .2 + .8 * 2 ** (-(cutoff - t).days / 120)
            self.counts[key,value][category] += w
        self.mass = {k: sum(v.values()) for k,v in self.counts.items()}
        self.raw = Counter(k[-1] for k in last if k[0] == "all")
        self.raw_n = sum(self.raw.values())

    @lru_cache(maxsize=100000)
    def lift(self, category, m, d, y, group_features):
        # Log lift against the long-run all-reader category distribution.
        p = self.raw.get(category, 0) / self.raw_n
        if not p:
            return 0.
        keys = [("all","*",.35), ("term",term(self.cutoff),.65)]
        if group_features:
            keys += [("major",m,.4), ("college",d,.25), ("stage",y,.15)]
        total = 0.
        for key, value, weight in keys:
            idx = (key,value)
            if not value or len(self.support.get(idx, ())) < 5:
                continue
            posterior = (self.counts[idx][category] + 50*p) / (self.mass[idx]+50)
            total += weight * max(-1., min(1., math.log(posterior/p)))
        return total


def first_class(code):
    letter = code[:1].upper()
    return letter if "A" <= letter <= "Z" else ""


def prior_rerank(release, attrs, prior, group_features=True):
    order = release[:30]
    values = [math.exp(.35*prior.lift(clc(c), *attrs, group_features))/(20+i)
              for i,c in enumerate(order,1)]
    return [order[i] for i in sorted(range(len(order)),key=lambda i:-values[i])[:10]]


def predictions(r, prior, authors, tail):
    head = r["release"][:10]
    history = r["recent"]
    cats = {clc(c) for c in history}
    auth = {authors.get(c,"") for c in history} - {""}
    m, d, y = r["attrs"][:3]
    attrs = (major(m),college(d),stage(y,prior.cutoff))

    def swaps(initial):
        keep = initial[:9]
        classes = {first_class(c) for c in keep}
        candidates = [(rank,c) for rank,c in enumerate(r["event"],1) if c not in initial and
                      (clc(c) in cats or authors.get(c,"") in auth or
                       prior.lift(clc(c), *attrs, True) >= math.log(1.25))]
        if not candidates:
            return (initial, initial, initial)
        def utility(item):
            rank,c = item
            return (1/math.log2(rank+1)
                    + .35*int(bool(first_class(c)) and first_class(c) not in classes)
                    + .20*int(c in tail))
        random_pick = min((c for _,c in candidates),
                          key=lambda c: hashlib.sha256(f"2026|{r['user']}|{c}".encode()).digest())
        diverse = max(candidates,key=utility)[1]
        return keep+[random_pick], keep+[candidates[0][1]], keep+[diverse]

    # Exploratory follow-up: require two signals and permit keeping position ten.
    # On the same nine retained books the unchanged list is a feasible comparator.
    base_rel = {c:1/(20+i) for i,c in enumerate(r["base"],1)}
    event_rel = {c:.5/(20+i) for i,c in enumerate(r["event"],1)}
    def evidence_swap():
        initial = head[-1]
        kept = {first_class(c) for c in head[:9]}
        def utility(c):
            rel = (base_rel.get(c,0) + event_rel.get(c,0)) / (1.5/21)
            return (.50*rel + .30*int(bool(first_class(c)) and first_class(c) not in kept)
                    + .20*int(c in tail))
        choices = [initial]
        for c in r["event"]:
            signals = (clc(c) in cats, authors.get(c,"") in auth,
                       prior.lift(clc(c), *attrs, True) >= math.log(1.25))
            if c not in head and sum(signals) >= 2:
                choices.append(c)
        return head[:9]+[max(choices,key=utility)]

    group = prior_rerank(r["release"], attrs, prior)
    random, evidence, diverse = swaps(head)
    return dict(zip(VARIANTS, (head, prior_rerank(r["release"], attrs, prior, False), group, random, evidence, diverse,
                               swaps(group)[2], evidence_swap())))


def reference(date, seed):
    if date in DEV:
        f = BASE/"outputs/release_event_epochs_v1"/f"{date}_{seed}.csv"
        df = pd.read_csv(f, dtype={"user":str}).query("epoch == 8")
    else:
        f = BASE/"outputs/event_encoder_b_v1"/str(seed)/f"{date}_gru.csv"
        df = pd.read_csv(f, dtype={"user":str}).query("epoch == 8 and view == 'official'")
    return df


def assess_fold(data, date, masked, seed=2026):
    start = pd.Timestamp(date)
    scenario = "masked" if masked else "normal"
    full, visible, hidden, _ = fold_data(data,date,masked)
    prior = DemandPrior(visible,start)
    authors = {c:attribute_value(a) for c,a in full[[CALL,"责任者"]].itertuples(index=False)}
    legal = set(authors)
    frequency = visible.groupby(CALL)[USER].nunique().to_dict()
    popular = sorted(legal, key=lambda c:(-frequency.get(c,0),c))[:math.ceil(.2*len(legal))]
    tail = legal-set(popular)
    banned = full.groupby(USER)[CALL].agg(set).to_dict()
    answers = {}
    for label,end in horizon_ends(date).items():
        future = data.loc[(data[TIME]>=start) & (data[TIME]<end)]
        answers[label] = {u:books for u,g in future.groupby(USER)
                          if (books:=set(g[CALL])-banned.get(u,set())-{""})}
    base = reference(date, seed)
    if date in DEV:
        anchor = base.set_index("user").pred.to_dict()
    else:
        anchor = {(r.user,r.method):r.pred for r in base.itertuples()}
    if date in DEV:
        assert set(anchor) == set(answers["short"])
    else:
        assert set(base.user) == set(answers["short"])
    rows, funnel = [], []
    detected = set()
    for line in (OUT/f"{date}_{scenario}.jsonl").read_text("utf8").splitlines():
        r = json.loads(line)
        u = r["user"]
        assert u not in detected
        detected.add(u)
        warm = bool(r["visible_history"])
        if not masked and u in answers["short"]:
            expected = anchor[u] if date in DEV else anchor[u,"事件轻融合" if warm else "当前提交"]
            assert "|".join(r["release"][:10]) == expected, (date,u,"锚点不一致")
        variants = predictions(r, prior, authors, tail)
        for label, amap in answers.items():
            y = amap.get(u)
            if not y:
                continue
            cold = y & hidden
            event = set(r["event"])
            released = set(r["release"])
            # The intersection below tracks identical positives, even for overlapping channels.
            funnel.append(dict(date=date,horizon=label,scenario=scenario,user=u,group=r["group"],
                               history=int(warm),truth=len(y),legal=len(y&legal),cold=len(cold),
                               raw100=len(cold&event),used100=len(cold&event) if warm else 0,
                               merged100=len(cold&released),top10=len(cold&set(r["release"][:10]))))
            for method,ls in variants.items():
                assert len(ls)==len(set(ls))==10 and set(ls)<=legal and not set(ls)&banned.get(u,set())
                hit,rec,nd,s = score(ls,y,r["group"])
                n_cold = len(set(ls)&cold)
                rows.append(dict(date=date,horizon=label,scenario=scenario,user=u,group=r["group"],
                                 history=int(warm),method=method,hit=hit,recall=rec,ndcg=nd,points=s,
                                 cover=len({first_class(c) for c in ls}-{""}),tail=sum(c in tail for c in ls)/10,
                                 cold_hits=n_cold,cold_recall=n_cold/len(cold) if cold else np.nan,
                                 cold_slots=sum(c in hidden for c in ls),
                                 changed=set(ls)!=set(variants["原版"])))
    assert detected == set.union(*(set(v) for v in answers.values()))
    prior.lift.cache_clear()
    return rows,funnel


def save(df,name):
    df.to_csv(OUT/name,index=False,encoding="utf-8-sig",float_format="%.8f")


def summarize(df):
    keys = ["scenario","horizon","date","method"]
    measures = ["points","hit","recall","ndcg"]
    groups = df.groupby(keys+["group"])[measures].mean().reset_index()
    save(groups,"group_scores.csv")
    groups[measures] = groups[measures].mul(groups.group.map({"新人":.4,"老人":.6}),axis=0)
    total = groups.groupby(keys)[measures].sum()
    totals = df.groupby(keys).agg(readers=("user","size"),categories=("cover","mean"),
             tail_fraction=("tail","mean"),cold_recall=("cold_recall","mean"),
             cold_hits=("cold_hits","sum"),cold_slots=("cold_slots","sum"),changed=("changed","sum"))
    result = total.join(totals).reset_index()
    save(result,"fold_scores.csv")
    seg = df.groupby(keys+["group","history"]).agg(readers=("user","size"),points=("points","mean"),
             hit=("hit","mean"),recall=("recall","mean"),cold_recall=("cold_recall","mean"))
    save(seg.reset_index(),"segments.csv")
    return result


def paired(df, filename="paired.csv"):
    rng = np.random.default_rng(20261107)
    result=[]
    for (scenario,horizon), frame in df.groupby(["scenario","horizon"]):
        parts = {"开发":frame[frame.date.isin(DEV)]}
        if horizon=="short":
            parts["复用诊断"] = frame[frame.date==DIAG]
        for label, data in parts.items():
            ids=sorted(set(data.user))
            uid={u:i for i,u in enumerate(ids)}
            weights=rng.poisson(1,(2000,len(ids))).astype(float)
            keys=["date","group","user"]
            pts=data.pivot(index=keys,columns="method",values="points")
            diffs=pts.subtract(pts["原版"],axis=0)
            samples=np.zeros((len(weights),len(pts.columns)))
            means=np.zeros(len(pts.columns))
            for (date,g), group in diffs.groupby(level=["date","group"]):
                idx=[uid[u] for u in group.index.get_level_values("user")]
                w=weights[:,idx]
                scale=(.4 if g=="新人" else .6)/data.date.nunique()
                samples += scale*(w@group.to_numpy())/w.sum(axis=1)[:,None]
                means += scale*group.mean().to_numpy()
            for i,m in enumerate(pts.columns):
                lo,hi=np.quantile(samples[:,i],[.025,.975])
                result.append(dict(scenario=scenario,horizon=horizon,stage=label,method=m,
                                   delta=means[i],lo=lo,hi=hi))
    save(pd.DataFrame(result),filename)


def main():
    import argparse
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=[2026,2027], default=2026)
    args = parser.parse_args()
    if args.seed != 2026:
        OUT = OUT / str(args.seed)
    data = read_data()
    rows, funnel = [], []
    for date in [*DEV,DIAG]:
        for masked in ([False,True] if args.seed == 2026 else [False]):
            a,b=assess_fold(data,date,masked,args.seed)
            rows.extend(a)
            funnel.extend(b)
    df=pd.DataFrame(rows)
    assert not df.duplicated(["date","horizon","scenario","method","user"]).any()
    save(df,"reader_scores.csv")
    stages=pd.DataFrame(funnel)
    save(stages.groupby(["scenario","horizon","date","group","history"])[
        ["truth","legal","cold","raw100","used100","merged100","top10"]].sum().reset_index(),"funnel.csv")
    summarize(df)
    paired(df)
    print("完成逐人前十锚点校验、候选合法性校验与配对评测", len(df),flush=True)


if __name__ == "__main__":
    main()
