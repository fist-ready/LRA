import numpy as np
import pandas as pd
from report_event_epochs import D, load
from newcomer_replay import DEV


def main():
    eight=load("users_epoch8.csv")
    four=load("paired_rerun_epoch4.csv")
    target="事件轻融合"
    key=["window","group","user"]
    ref=eight.pivot(index=key,columns="method",values="score")
    paired=four.loc[four.method.eq(target)].set_index(key).score
    d=pd.DataFrame({"八轮减四轮":ref[target]-paired,"八轮减提交":ref[target]-ref["当前提交"],
                    "八轮减旧双塔融合":ref[target]-ref["前轮双塔轻融合"]}).reset_index()
    assert not d.isna().any().any()
    # 同一读者在不同窗共享重采样权重；窗口、分组权重保持固定。
    users,ix=np.unique(d.user,return_inverse=True)
    rng=np.random.default_rng(2026)
    w=rng.poisson(1.,size=(4000,len(users)))
    diffs=d.iloc[:,3:].to_numpy()
    est=np.zeros((len(w),diffs.shape[1]))
    point=np.zeros(diffs.shape[1])
    for (window,group),grp in d.groupby(["window","group"]):
        js=grp.index.to_numpy()
        weight=(.4 if group=="新人" else .6)/len(DEV)
        a=w[:,ix[js]]
        est+=weight*(a@diffs[js]/a.sum(axis=1)[:,None])
        point+=weight*diffs[js].mean(axis=0)
    out=pd.DataFrame([point,*np.quantile(est,[.025,.975],axis=0)],index=["差值","近似下界","近似上界"],columns=d.columns[3:])
    out.to_csv(D/"epoch_study_uncertainty.csv",encoding="utf-8-sig")
    print(out.round(4))
    for df in (four,eight):
        g=df[df.method.eq(target)].groupby(["window","group"])[["hit","recall","ndcg","score"]].mean().groupby("group").mean()
        print(g.round(4))


if __name__=="__main__":
    main()
