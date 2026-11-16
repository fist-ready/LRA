"""只用开发结果选择预留名额，再评估复用诊断。"""
import numpy as np
import pandas as pd
from catboost import CatBoostRanker

from baseline import USER
from call_number_replay import CALL, metrics
from older_call_ranker import FEATURES, OUT as PREVIOUS, fit
from prepare_older_windows import ALL, EXTRA, OUT
from verify_older_call_ranker import verify_scores
from ranking import DEVELOPMENT, DIAGNOSTIC, atomic_json

SLOTS = (4,8)
BASES = ("等权融合","扩窗学习")
CHOICES = [f"{base}+{q}" for base in BASES for q in range(3)]


def reserve(order, exclusive, count):
    """按新通道顺序提拔独有候选；已在目标位置之前则保留原位。"""
    result = list(order)
    for item,position in zip(exclusive[:count],SLOTS):
        i = result.index(item)
        if i >= position:
            result.insert(position-1,result.pop(i))
    return result[:10]


def audit(frame):
    exclusive = frame.old_rr.eq(0) & frame.new_rr.gt(0)
    hit = frame.label.eq(1)
    return dict(queries=frame["query"].nunique(),pairs=len(frame),positive=int(hit.sum()),
                exclusive=int(exclusive.sum()),exclusive_positive=int((hit & exclusive).sum()),
                exclusive_positive_queries=frame.loc[hit & exclusive,"query"].nunique())


def bootstrap(rows,choice,reference):
    wide = rows.pivot(index=["user","window"],columns="method",values="score")
    delta = (wide[choice]-wide[reference]).unstack().to_numpy()
    rng = np.random.default_rng(20261105)
    reps = [np.nanmean(delta[rng.integers(len(delta),size=len(delta))],axis=0).mean() for _ in range(4000)]
    lo,hi = np.quantile(reps,[.025,.975])
    return dict(delta=np.nanmean(delta,axis=0).mean(),low=lo,high=hi)


def main():
    rows,windows,training,past = [],[],{},[]
    prev = pd.read_csv(PREVIOUS/"per_user.csv",dtype={"user":str})
    prev_summary = pd.read_csv(PREVIOUS/"summary.csv")
    for date in ALL:
        if date==DIAGNOSTIC:
            dev = pd.DataFrame(rows)
            scores = dev.groupby(["window","method"]).score.mean().groupby("method").mean()
            winner = max(CHOICES,key=lambda name:scores[name])
            atomic_json(OUT/"locked_plan.json",dict(winner=winner,dev={k:scores[k] for k in CHOICES},
                        slot_positions=SLOTS,selection="三个开发窗口老人组得分等权；同分先等权、再少预留"))
        f,answers,end = pd.read_pickle(OUT/date/"calls.pkl")
        assert all(e<=pd.Timestamp(date) for _,e,_ in past)
        windows.append(dict(date=date,end=end.isoformat(),extra=date in EXTRA,**audit(f)))
        if date in [*DEVELOPMENT,DIAGNOSTIC]:
            train = pd.concat([g for _,_,g in past],ignore_index=True)
            model,stats = fit(train)
            training[date] = dict(dates=[d for d,_,_ in past],**stats,**audit(train))
            model.save_model(str(OUT/date/"expanded.cbm"))
            f = f.copy()
            f["扩窗学习"] = CatBoostRanker().load_model(str(OUT/date/"expanded.cbm")).predict(f[FEATURES])
            f["等权融合"] = f.old_rr+f.new_rr
            f["原窗学习"] = CatBoostRanker().load_model(str(PREVIOUS/date/"ranker.cbm")).predict(f[FEATURES])
            for u,g in f.groupby(USER):
                truth = answers[u]
                positive = set(g.loc[g.label.eq(1),CALL])
                assert positive == set(g[CALL]) & truth
                exclusive = g.loc[g.old_rr.eq(0)].sort_values(["new_rr",CALL],ascending=[False,True])[CALL].tolist()
                exclusives = set(exclusive)
                for base in (*BASES,"原窗学习","new_rr"):
                    order = g.sort_values([base,CALL],ascending=[False,True])[CALL].tolist()
                    for q in (range(3) if base in BASES else [0]):
                        method = f"{base}+{q}" if base in BASES else {"原窗学习":"原窗学习","new_rr":"新召回"}[base]
                        top = reserve(order,exclusive,q)
                        assert len(set(top))==len(top)==10
                        m = metrics(top,truth)
                        rows.append(dict(window=date,user=u,method=method,pred="|".join(top),truth=len(truth),
                                         pool_hits=len(positive),new_hits=len(set(top)&truth&exclusives),
                                         score=40*m["hit"]+30*m["recall"]+30*m["ndcg"],**m))
            print(date,stats,flush=True)
        past.append((date,end,f))
    details = pd.DataFrame(rows)
    for a,b in [("原窗学习","新借排序"),("等权融合+0","等权融合"),("新召回","新召回")]:
        def indexed(df,method):
            return df.loc[df.method.eq(method)].set_index(["window","user"]).pred.sort_index()
        pd.testing.assert_series_equal(indexed(details,a),indexed(prev,b))
    grouped = details.groupby(["window","method"])
    summary = grouped[["score","hit","recall","ndcg"]].mean()
    summary["pool_coverage"] = grouped.pool_hits.sum()/grouped.truth.sum()
    summary["new_hits"] = grouped.new_hits.sum()
    for (date,method),row in summary.iterrows():
        baseline = prev_summary.loc[prev_summary.window.eq(date)&prev_summary.method.eq("等权融合")].iloc[0]
        summary.loc[(date,method),"total"] = baseline.total+.6*(row.score-baseline.score)
    details.to_csv(OUT/"per_user.csv",index=False,encoding="utf-8-sig")
    summary.to_csv(OUT/"summary.csv",encoding="utf-8-sig")
    pd.DataFrame(windows).to_csv(OUT/"positive_counts.csv",index=False,encoding="utf-8-sig")
    atomic_json(OUT/"training.json",training)
    paired = []
    for name,part in details.groupby(details.window.eq(DIAGNOSTIC).map({True:"复用诊断",False:"开发"})):
        for choice,ref in [("扩窗学习+0","原窗学习"), ("扩窗学习+0","等权融合+0"),
                           ("等权融合+1","等权融合+0"), ("等权融合+2","等权融合+0")]:
            paired.append(dict(split=name,method=choice,reference=ref,**bootstrap(part,choice,ref)))
    pd.DataFrame(paired).to_csv(OUT/"paired.csv",index=False,encoding="utf-8-sig")
    verify_scores(OUT)
    print(summary.score.unstack().round(3));print(winner)


if __name__=="__main__":
    main()

