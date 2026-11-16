"""GRU / 单层 SASRec × 8 / 32 步。用正式混合策略及同窗同读者作配对。"""
import argparse
import json
import math
import numpy as np
import pandas as pd
import torch
from event_encoder_b import read_data, SEEDS
from event_objective_c import BASE, DEV, DIAG, RUN, FUSION, save
from evaluate_event_suite import load_model, prepare, evaluate, train_checkpoints
from roster_replay import score
from verify_event_objective_c import compare

OUT=BASE/"outputs/sasrec_8_32_v1"
ARMS=("gru8","sasrec8","gru32","sasrec32")
METRICS=["score","hit","recall","ndcg"]

def fit(date, seed):
    directory=OUT/str(seed)
    directory.mkdir(parents=True,exist_ok=True)
    data=read_data()
    ck,_=load_model(RUN/date/"model.pt")
    m=ck["manifest"]
    for depth in (8,32):
        m.update(depth=depth,seed=seed)
        records,h=prepare(data,date,ck)
        info={q.user:(len(q.seen),targets,group) for q,group,targets,*_ in records}
        seen={q.user:{ck["codes"][i] for i in q.seen} for q,*_ in records}
        legal=set(ck["legal"])
        for encoder in ("gru","sasrec"):
            arm=f"{encoder}{depth}"
            path=directory/f"{date}_{arm}.csv"
            if path.exists():
                continue
            if arm=="gru8":
                previous=BASE/"outputs/event_encoder_b_v1"/str(seed)/f"{date}_gru.csv"
                raw=pd.read_csv(previous,dtype={"user":str}).query("epoch==8 and view=='official'")
            else:
                m["encoder"]=encoder
                model,=train_checkpoints(ck,h,date,"joint",path.with_suffix(".json"),aligned_init=True,checkpoints=(8,))
                raw=evaluate(ck,model,records,date,8)
            sizes=raw.user.map(lambda u:info[u][0])
            routed=raw[(sizes.gt(0)&raw.method.eq(FUSION)) | (sizes.eq(0)&raw.method.eq("当前提交"))].copy()
            assert routed.user.is_unique and set(routed.user)==set(info)
            for row in routed.itertuples():
                pred=row.pred.split("|")
                n,targets,group=info[row.user]
                assert len(set(pred))==len(pred)==10 and set(pred)<=legal and not seen[row.user].intersection(pred)
                assert group==row.group and np.allclose(score(pred,targets,group),[row.hit,row.recall,row.ndcg,row.score])
            routed["n_history"]=routed.user.map(lambda u:info[u][0])
            routed["arm"]=arm
            routed["seed"]=seed
            routed["updates"]=8*math.ceil(sum(m["queries"].values())/m["batch_size"])
            raw_direct=raw[raw.method.eq("事件直排")].set_index("user").score
            routed["direct_score"]=routed.user.map(raw_direct)
            save(routed,directory/f"{date}_{arm}.csv")
            print("完成",date,seed,arm,flush=True)

def segment_intervals(rows):
    """分层描述沿用窗内等读者、窗间等权；同一读者跨窗/种子绑定。"""
    results=[]
    for label, minimum in [(">=2",2),(">8",9)]:
        scores=(rows[rows.n_history.ge(minimum)]
                .pivot_table(index=["window","user"],columns="arm",values="score",aggfunc="mean"))
        assert scores.notna().all().all()
        reader, indices=np.unique(scores.index.get_level_values("user"),return_inverse=True)
        weights=np.random.default_rng(2026).poisson(1.,size=(4000,len(reader)))
        for left,right in [("sasrec8","gru8"),("sasrec32","gru32"),("sasrec32","gru8"),("gru32","gru8")]:
            diff=scores[left]-scores[right]
            draws=[]
            for ix in diff.groupby(level="window").indices.values():
                w=weights[:,indices[ix]]
                draws.append(w@diff.iloc[ix].to_numpy()/w.sum(axis=1))
            lo,hi=np.quantile(np.mean(draws,axis=0),[.025,.975])
            results.append(dict(segment=label,left=left,right=right,
                                delta=diff.groupby(level="window").mean().mean(),lo=lo,hi=hi,
                                reader_windows=len(scores),unique_readers=len(reader)))
    return pd.DataFrame(results)


def summarize(stage):
    dates=DEV if stage=="dev" else [DIAG]
    rows=pd.concat([pd.read_csv(OUT/str(seed)/f"{date}_{arm}.csv",dtype={"user":str})
                    for date in dates for arm in ARMS for seed in SEEDS],ignore_index=True)
    grp=rows.groupby(["window","user"])
    assert grp.size().eq(8).all() and grp.n_history.nunique().eq(1).all()
    assert grp.truth.nunique().eq(1).all() and grp.group.nunique().eq(1).all()
    assert rows[rows.n_history.eq(0)].groupby(["window","user"]).pred.nunique().eq(1).all()
    keys=["window","seed","arm"]
    metrics=METRICS+["direct_score"]
    groups=rows.groupby(keys+["group"])[metrics].mean().reset_index()
    weighted=groups[metrics].mul(groups.group.map({"新人":.4,"老人":.6}),axis=0)
    scores=pd.concat([groups[keys],weighted],axis=1).groupby(keys).sum().reset_index()
    save(scores,OUT/f"{stage}_scores.csv")
    save(segment_intervals(rows),OUT/f"{stage}_segment_paired.csv")
    segments=[]
    for label,mask in [(">=2",rows.n_history.ge(2)),(">8",rows.n_history.gt(8))]:
        sub=rows[mask].groupby(keys).agg(n=("user","size"),score=("score","mean"),hit=("hit","mean"),recall=("recall","mean")).reset_index()
        segments.append(sub.assign(segment=label))
    save(pd.concat(segments),OUT/f"{stage}_segments.csv")
    averaged=rows.groupby(["window","user","group","arm"])[["score","hit"]].mean().reset_index()
    pairs=[("sasrec8","gru8"),("sasrec32","gru32"),("gru32","gru8"),("sasrec32","sasrec8"),("sasrec32","gru8")]
    save(compare(averaged,pairs,variant_column="arm").drop(columns=["gained","lost"]),OUT/f"{stage}_paired.csv")
    # 探索性分流：>8 已在实验前指定为子组，使用该组最优配置是看过开发结果后的选择。
    route=rows[((rows.n_history>8)&(rows.arm=="sasrec32"))|((rows.n_history<=8)&(rows.arm=="gru8"))].copy()
    route["arm"]="long_sasrec32"
    baseline=rows[rows.arm.eq("gru8")]
    split=pd.concat([baseline,route],ignore_index=True)
    by_group=split.groupby(keys+["group"])[METRICS].mean().reset_index()
    by_group[METRICS]=by_group[METRICS].mul(by_group.group.map({"新人":.4,"老人":.6}),axis=0)
    save(by_group.groupby(keys)[METRICS].sum().reset_index(),OUT/f"{stage}_routing.csv")
    route_avg=split.groupby(["window","user","group","arm"])[["score","hit"]].mean().reset_index()
    route_delta=compare(route_avg,[("long_sasrec32","gru8")],variant_column="arm")
    save(route_delta.drop(columns=["gained","lost"]),OUT/f"{stage}_routing_paired.csv")
    rank=scores.groupby("arm").score.mean().sort_values(ascending=False)
    print(stage,rank.to_dict(),flush=True)
    if stage=="dev":
        (OUT/"selection.json").write_text(json.dumps({"ranking":rank.to_dict(),"winner":rank.index[0],"epoch":8},indent=2),"utf8")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    action=p.add_mutually_exclusive_group(required=True)
    action.add_argument("--fold",choices=[*DEV,DIAG])
    action.add_argument("--summarize",choices=["dev","diagnostic"])
    p.add_argument("--seed",type=int,choices=SEEDS)
    args=p.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.fold:
        for seed in ((args.seed,) if args.seed is not None else SEEDS):
            fit(args.fold,seed)
    else:
        summarize(args.summarize)
