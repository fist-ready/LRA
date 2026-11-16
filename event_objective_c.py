"""实验 C。开发窗按固定八轮预算和事件轻融合综合分选目标，再核验复用诊断。"""
import argparse
import json
import math

import pandas as pd
import torch

from evaluate_event_suite import RUN, BASE, load_interactions, CALL, TIME, USER, DEV, DIAG
from evaluate_event_suite import attribute_value, load_model, prepare, evaluate, train_checkpoints

OUT = BASE / "outputs" / "event_objective_c_v1"
OBJECTIVES = ("next", "window", "joint")
FUSION = "事件轻融合"
METRICS = ["hit", "recall", "ndcg", "score", "next_recall"]


def save(frame, path):
    tmp = path.with_suffix(".tmp")
    frame.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)


def first_targets(data, date, records):
    start = pd.Timestamp(date)
    end = data[TIME].max() + pd.Timedelta(1, "ns") if date == DIAG else start + pd.DateOffset(months=2)
    truth = {q.user: answers for q, _, answers, *_ in records}
    future = data.loc[data[TIME].between(start, end, inclusive="left") & data[CALL].ne("")]
    result = {}
    for user, rows in future.groupby(USER):
        if user not in truth:
            continue
        rows = rows.sort_values(TIME, kind="stable")
        groups = rows[TIME].diff().gt(pd.Timedelta(minutes=30)).cumsum()
        for _, event in rows.groupby(groups):
            books = set(event[CALL]) & truth[user]
            if books:
                result[user] = books
                break
    assert result.keys() == truth.keys()
    return result


def run_fold(data, date, objectives, out_dir=OUT, seed=2026):
    ck, _ = load_model(RUN / date / "model.pt")
    ck["manifest"] = dict(ck["manifest"], seed=seed)
    cfg = ck["manifest"]
    assert (cfg["encoder"], cfg["events"], cfg["topics"]) == ("gru", "gap30", True)
    records, history = prepare(data, date, ck)
    targets = first_targets(data, date, records)
    steps = math.ceil(sum(cfg["queries"].values()) / cfg["batch_size"])
    out = []
    for objective in objectives:
        path = out_dir / f"{date}_{objective}.csv"
        if not path.exists():
            four, eight = train_checkpoints(ck, history, date, objective, out_dir / f"{date}_{objective}_loss.json")
            frames = []
            for epoch, model in [(4, four), (8, eight)]:
                part = evaluate(ck, model, records, date, epoch)
                part["objective"] = objective
                part["seed"] = seed
                part["updates"] = steps * epoch
                part["next_recall"] = [len(set(pred.split("|")) & targets[u]) / len(targets[u])
                                       for u, pred in zip(part.user, part.pred)]
                frames.append(part)
            save(pd.concat(frames), path)
        frame = pd.read_csv(path, dtype={"user":str})
        assert frame.objective.eq(objective).all() and frame.window.eq(date).all()
        assert len(frame) == len(records)*8 and not frame.duplicated(["epoch", "method", "user"]).any()
        assert (frame.updates == frame.epoch*steps).all() and set(frame.user)==set(targets)
        out.append(frame)
        print("已核验", date, objective, steps*8, "次更新", flush=True)
    return pd.concat(out, ignore_index=True)


def summarize(rows, prefix, out_dir=OUT):
    keys = ["window","epoch","objective","method"]
    groups = rows.groupby(keys+["group"])[METRICS].mean().reset_index()
    save(groups, out_dir / f"{prefix}_groups.csv")
    for metric in METRICS:
        groups[metric] *= groups.group.map({"新人":.4,"老人":.6})
    scores = groups.groupby(keys)[METRICS].sum().reset_index()
    save(scores, out_dir / f"{prefix}_scores.csv")
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicate-seed", type=int, help="仅在三个开发窗复核另一个随机种子")
    args = parser.parse_args()
    seed = 2026 if args.replicate_seed is None else args.replicate_seed
    if args.replicate_seed == 2026:
        parser.error("默认主实验已经使用 2026，请指定另一个种子")
    out_dir = OUT if args.replicate_seed is None else OUT / f"seed_{seed}"
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_interactions(BASE/"data/processed/borrowing_cleaned.csv")
    data[CALL] = data[CALL].map(attribute_value)
    if "性别" not in data:
        data["性别"] = ""
    dev = pd.concat([run_fold(data,date,OBJECTIVES,out_dir,seed) for date in DEV], ignore_index=True)
    scores = summarize(dev,"dev",out_dir)
    if args.replicate_seed is not None:
        return
    eligible = scores.loc[(scores.epoch==8) & (scores.method==FUSION)]
    ranking = eligible.groupby("objective").score.mean().sort_values(ascending=False,kind="stable")
    best = ranking.index[0]
    reference = "joint" if best != "joint" else ranking.index[1]
    decision = {"selection":"3 个开发窗八轮预算的事件轻融合综合分等权均值",
                "development":ranking.to_dict(), "winner":best, "diagnostic_comparator":reference,
                "seed":2026, "torch_threads":1}
    (OUT/"decision_before_diagnostic.json").write_text(json.dumps(decision,ensure_ascii=False,indent=2),"utf8")
    print("开发选择",decision,flush=True)
    diag=run_fold(data,DIAG,[best,reference])
    summarize(diag,"diagnostic")
    for frame in [dev, diag]:
        baselines = frame[frame.method.isin(["当前提交","前轮双塔轻融合"])]
        assert baselines.groupby(["window","user","method"]).pred.nunique().eq(1).all()
        assert frame.groupby(["window","user"]).truth.nunique().eq(1).all()


if __name__ == "__main__":
    main()
