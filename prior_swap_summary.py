"""Aggregate two paired seeds; keep the exploratory newcomer-only route explicit."""
import numpy as np
import pandas as pd

from newcomer_replay import DEV, DIAG
from prior_swap_study import OUT, paired, save


def main():
    frames = []
    for seed, folder in [(2026,OUT),(2027,OUT/"2027")]:
        frame = pd.read_csv(folder/"reader_scores.csv",dtype={"user":str})
        frame = frame.loc[frame.scenario.eq("normal")].assign(seed=seed)
        # Exploratory route, computed after looking at seed 2026 subgroup scores.
        target = frame.group.eq("新人") & frame.history.eq(1)
        routed = frame[(target & frame.method.eq("时序群体先验")) |
                       (~target & frame.method.eq("原版"))].copy()
        frames.append(pd.concat([frame,routed.assign(method="仅有史新人用先验")]))
    data = pd.concat(frames,ignore_index=True)
    keys = ["scenario","horizon","date","group","history","user","method"]
    assert data.groupby(keys).size().eq(2).all()
    measures = ["points","hit","recall","ndcg","cover","tail","changed"]
    avg = data.groupby(keys,as_index=False)[measures].mean()
    paired(avg,"two_seed_paired.csv")
    reports = []
    for (horizon,tag), block in data.groupby(["horizon",data.date.map(lambda d:"诊断" if d==DIAG else "开发")]):
        for method, part in block.groupby("method"):
            groups = part.groupby(["seed","date","group"])[["points","hit","recall","ndcg"]].mean().reset_index()
            cols=["points","hit","recall","ndcg"]
            groups[cols]=groups[cols].mul(groups.group.map({"新人":.4,"老人":.6}),axis=0)
            by_seed=groups.groupby(["seed","date"])[cols].sum().groupby(level=0).mean()
            row=dict(horizon=horizon,stage=tag,method=method,seed_2026=by_seed.at[2026,"points"],
                     seed_2027=by_seed.at[2027,"points"],**by_seed.mean().to_dict())
            row.update(part.groupby(["seed","date"])[["cover","tail","changed"]].mean().mean().to_dict())
            reports.append(row)
    save(pd.DataFrame(reports),"two_seed_summary.csv")
    first = avg[avg.date.eq(DEV[0])].copy()
    early = set(first.loc[first.horizon.eq("short"),"user"])
    first["cohort"] = np.where(first.user.isin(early),"首两月可评","仅长窗可评")
    cohort = first.groupby(["cohort","horizon","method","group"]).agg(
        readers=("user","size"),points=("points","mean")).reset_index()
    cohort["weighted"] = cohort.points * cohort.group.map({"新人":.4,"老人":.6})
    save(cohort.groupby(["cohort","horizon","method"]).agg(
        readers=("readers","sum"),points=("weighted","sum")).reset_index(),"horizon_cohorts.csv")
    seg=avg.groupby(["date","horizon","group","history","method"]).agg(
        readers=("user","size"),points=("points","mean"),hit=("hit","mean"),recall=("recall","mean"))
    save(seg.reset_index(),"two_seed_segments.csv")


if __name__=="__main__":
    main()
