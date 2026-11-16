"""复核实验 C 的逐人命中和下一事件指标，按读者聚类估算八轮目标差值。"""
import argparse
from itertools import combinations
import json

import numpy as np
import pandas as pd

from event_objective_c import OUT, OBJECTIVES, FUSION, DEV, DIAG, save
from evaluate_event_suite import BASE, load_interactions, CALL, TIME, USER, attribute_value
from roster_replay import score


def verify_fold(data, date, objectives, out_dir, *, variant_column="objective", return_all=False):
    start = pd.Timestamp(date)
    end = data[TIME].max()+pd.Timedelta(1,"ns") if date==DIAG else start+pd.DateOffset(months=2)
    history = data.loc[data[TIME]<start]
    seen = history.groupby(USER)[CALL].agg(set).to_dict()
    legal = set(history[CALL])-{""}
    future = data.loc[data[TIME].between(start,end,inclusive="left") & data[CALL].ne("")]
    truth, first = {}, {}
    for user, rows in future.groupby(USER):
        known = seen.get(user,set())
        answers = set(rows[CALL])-known
        if not answers:
            continue
        truth[user] = answers
        rows = rows.sort_values(TIME,kind="stable")
        blocks = rows[TIME].diff().gt(pd.Timedelta(minutes=30)).cumsum()
        for _, event in rows.groupby(blocks):
            novel = set(event[CALL])-known
            if novel:
                first[user] = novel
                break
    rows = pd.concat([pd.read_csv(out_dir/f"{date}_{o}.csv",dtype={"user":str}) for o in objectives])
    assert rows.groupby(["epoch",variant_column,"method"]).user.nunique().eq(len(truth)).all()
    for r in rows.itertuples():
        top=r.pred.split("|")
        assert len(top)==len(set(top))==10 and set(top)<=legal and not set(top)&seen.get(r.user,set())
        assert r.truth==len(truth[r.user])
        assert np.allclose(score(top,truth[r.user],r.group),[r.hit,r.recall,r.ndcg,r.score])
        assert np.isclose(len(set(top)&first[r.user])/len(first[r.user]),r.next_recall)
    if return_all:
        return rows
    return rows.loc[(rows.epoch==8) & (rows.method==FUSION)].copy()


def compare(frame, pairs, *, variant_column="objective"):
    key=["window","group","user"]
    scores=frame.pivot(index=key,columns=variant_column,values="score")
    hits=frame.pivot(index=key,columns=variant_column,values="hit")
    users=scores.index.get_level_values("user")
    uniques,idx=np.unique(users,return_inverse=True)
    weights=np.random.default_rng(2026).poisson(1.,size=(4000,len(uniques)))
    result=[]
    for left,right in pairs:
        diff=scores[left]-scores[right]
        hit_delta=hits[left]-hits[right]
        simulations=np.zeros(len(weights))
        observed=0.
        for (_,group),positions in diff.groupby(level=["window","group"]).indices.items():
            weight=(.4 if group=="新人" else .6)/frame.window.nunique()
            w=weights[:,idx[positions]]
            values=diff.iloc[positions].to_numpy()
            observed+=weight*values.mean()
            simulations+=weight*((w@values)/w.sum(axis=1))
        low,high=np.quantile(simulations,[.025,.975])
        result.append(dict(left=left,right=right,delta=observed,low=low,high=high,
                           gained=int(hit_delta.gt(0).sum()),lost=int(hit_delta.lt(0).sum())))
    return pd.DataFrame(result)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--replicate-seed",type=int)
    args=parser.parse_args()
    out_dir=OUT if args.replicate_seed is None else OUT/f"seed_{args.replicate_seed}"
    decision=json.loads((OUT/"decision_before_diagnostic.json").read_text("utf8"))
    diag_pair=[decision["winner"],decision["diagnostic_comparator"]]
    data=load_interactions(BASE/"data/processed/borrowing_cleaned.csv")
    data[CALL]=data[CALL].map(attribute_value)
    dev=pd.concat([verify_fold(data,date,OBJECTIVES,out_dir) for date in DEV])
    splits=[("dev",dev,combinations(OBJECTIVES,2))]
    if args.replicate_seed is None:
        splits.append(("diagnostic",verify_fold(data,DIAG,diag_pair,out_dir),[diag_pair]))
    for name,df,pairs in splits:
        result=compare(df,pairs)
        save(result,out_dir/f"{name}_paired.csv")
        print(name,result.round(4).to_string(index=False))
    print("已从借阅流水重新计分，所有结果一致。")


if __name__=="__main__":
    main()
