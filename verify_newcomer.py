"""独立复核得分并单列学院画像代理实验，不改变已冻结策略。"""
import hashlib
import json
import numpy as np
import pandas as pd

from newcomer_replay import BASE, OUT, DEV, DIAG, GAP, CALL, USER, TIME, Newcomer, college
from baseline import attribute_value, load_interactions


def main():
    d = load_interactions(BASE/"data/processed/borrowing_cleaned.csv")
    d[CALL] = d[CALL].map(attribute_value)
    f = pd.read_csv(OUT/"per_user.csv",dtype={"user":str})
    plans = json.loads((OUT/"locked_plan.json").read_text("utf8"))
    summaries = pd.read_csv(OUT/"summary.csv").set_index(["window","method"])
    proxy,changes = [],[]
    for date in [*DEV,DIAG]:
        start = pd.Timestamp(date)
        end = d[TIME].max()+pd.Timedelta(1,"ns") if date==DIAG else start+pd.DateOffset(months=2)
        h = d.loc[d[TIME]<start]
        seen = h.groupby(USER)[CALL].agg(set).to_dict()
        catalog = set(h[CALL])-{""}
        old = set(h.loc[h[TIME]<start-GAP,USER])
        targets = d.loc[d[TIME].between(start,end,inclusive="left")].groupby(USER)[CALL].agg(set)
        truth = {u:y-seen.get(u,set())-{""} for u,y in targets.items() if u not in old}
        eligible = {u for u,y in truth.items() if y}
        detail = f.loc[f.window.eq(date)]
        selected = detail.loc[detail.method.eq(detail.segment.map(plans))]
        baseline = detail.loc[detail.method.eq("全局")]
        for method,rows in [("对照",baseline),("开发选定",selected)]:
            assert rows.user.is_unique and set(rows.user)==eligible
            for r in rows.itertuples():
                top = r.pred.split("|")
                assert len(set(top))==len(top)==10 and set(top)<=catalog-seen.get(r.user,set())
                hits = len(truth[r.user].intersection(top))
                assert r.hit==bool(hits) and np.isclose(r.recall,hits/len(truth[r.user]))
            s = summaries.loc[(date,method)]
            assert np.isclose(s.total,.6*s.older+20*(rows.hit.mean()+rows.recall.mean()))
        diff = selected.set_index("user").score-baseline.set_index("user").score
        changes.append(diff.rename(date))
        dept = h.loc[h["单位"].map(attribute_value).ne("")].groupby(USER)["单位"].last()
        # 只模拟有独立画像的用户。遮蔽组所有人的流水均从索引剔除。
        sample = sorted(eligible.intersection(dept.index))
        folds = {u:int(hashlib.sha256(u.encode()).hexdigest()[:8],16)%5 for u in sample}
        for fold in range(5):
            users = [u for u in sample if folds[u]==fold]
            if not users:
                continue
            model = Newcomer(h.loc[~h[USER].isin(users)],start)
            for u in users:
                for name,top in model.predict([],college(dept[u]),["全局","近期","学院平滑"]).items():
                    hits = len(truth[u].intersection(top))
                    proxy.append(dict(window=date,user=u,method=name,hit=bool(hits),
                                      recall=hits/len(truth[u]),score=50*(bool(hits)+hits/len(truth[u]))))
    p = pd.DataFrame(proxy)
    p.groupby(["window","method"]).agg(users=("user","size"),hit=("hit","mean"),
                                      recall=("recall","mean"),score=("score","mean")).to_csv(OUT/"profile_proxy.csv",encoding="utf-8-sig")
    paired = []
    for name,series in [("开发",changes[:3]),("复用诊断",changes[3:])]:
        a = pd.concat(series,axis=1).to_numpy()
        rng = np.random.default_rng(20261105)
        draws = [.4*np.nanmean(a[rng.integers(len(a),size=len(a))],axis=0).mean() for _ in range(4000)]
        lo,hi = np.quantile(draws,[.025,.975])
        paired.append(dict(split=name,total_delta=.4*np.nanmean(a,axis=0).mean(),low=lo,high=hi))
    print(pd.DataFrame(paired))
    pd.DataFrame(paired).to_csv(OUT/"paired.csv",encoding="utf-8-sig",index=False)


if __name__=="__main__":
    main()

