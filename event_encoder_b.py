"""实验 B：固定联合目标，在相同借阅历史上比较三种读者编码。"""
import argparse
import json
import math

import pandas as pd
import torch

from event_objective_c import BASE, DEV, DIAG, FUSION, METRICS, RUN, first_targets, save
from event_sequences import DEPTH, events
from baseline import TIME, USER
from evaluate_event_suite import CALL, attribute_value, evaluate, load_interactions, load_model, prepare, train_checkpoints
from verify_event_objective_c import compare, verify_fold

OUT = BASE / "outputs" / "event_encoder_b_v1"
ENCODERS = ("book_mean", "mean", "gru")
SEEDS = (2026, 2027)
VIEWS = ("official", "history")


def fit_fold(date, seed, data):
    directory = OUT / str(seed)
    directory.mkdir(parents=True, exist_ok=True)
    ck, _ = load_model(RUN/date/"model.pt")
    cfg = ck["manifest"]
    assert (cfg["events"], cfg["objective"], cfg["topics"]) == ("gap30", "joint", True)
    records, history = prepare(data, date, ck)
    first = first_targets(data, date, records)
    budget = math.ceil(sum(cfg["queries"].values()) / cfg["batch_size"])
    for encoder in ENCODERS:
        path = directory / f"{date}_{encoder}.csv"
        if not path.exists():
            ck["manifest"] = dict(cfg, seed=seed, encoder=encoder)
            snapshots = train_checkpoints(ck, history, date, "joint", directory/f"{date}_{encoder}_loss.json",
                                          aligned_init=True)
            results = []
            for epoch, model in zip((4,8),snapshots):
                for view in (VIEWS if epoch==8 else ("official",)):
                    rows = evaluate(ck, model, records, date, epoch, official=view=="official")
                    rows["encoder"] = encoder
                    rows["seed"] = seed
                    rows["view"] = view
                    rows["updates"] = budget*epoch
                    rows["next_recall"] = [len(first[u].intersection(pred.split("|")))/len(first[u])
                                           for u,pred in zip(rows.user,rows.pred)]
                    results.append(rows)
            save(pd.concat(results),path)
        rows = pd.read_csv(path,dtype={"user":str})
        assert len(rows)==len(records)*12 and rows.encoder.eq(encoder).all() and rows.seed.eq(seed).all()
        assert set(rows.user)==set(first) and rows.window.eq(date).all() and (rows.updates==rows.epoch*budget).all()
        assert not rows.duplicated(["epoch","view","user","method"]).any()
        print("完成",date,seed,encoder,budget*8,flush=True)


def summarize(dates, stage):
    data = read_data()
    coverage = []
    for date in dates:
        past = data.loc[(data[TIME] < pd.Timestamp(date)) & data[CALL].ne("")]
        seqs = dict(events(past, {book:book for book in past[CALL].unique()}, "gap30"))
        roster = pd.read_csv(OUT/str(SEEDS[0])/f"{date}_gru.csv", usecols=["user"],dtype={"user":str})
        for user in roster.user.unique():
            recent = seqs.get(user, [])[-DEPTH:]
            has_basket = any(len(event.items) > 1 for event in recent)
            bucket = "含多本事件" if has_basket else ("仅单本事件" if recent else "无历史")
            coverage.append({"window":date,"user":user,"prefix_type":bucket})
    prefixes = pd.DataFrame(coverage)
    save(prefixes.groupby(["window","prefix_type"]).size().rename("readers").reset_index(),
         OUT/f"{stage}_history_counts.csv")
    results, pairs, fusions = [], [], []
    for seed in SEEDS:
        directory = OUT/str(seed)
        per_seed = []
        for date in dates:
            frame = verify_fold(data,date,ENCODERS,directory,variant_column="encoder",return_all=True)
            assert frame.groupby(["window","user"]).truth.nunique().eq(1).all()
            fixed = frame.loc[frame.method.isin(["当前提交","前轮双塔轻融合"])]
            assert fixed.groupby(["user","method"]).pred.nunique().eq(1).all()
            # GRU 为 C 的联合监督锚点。若上轮有同窗结果则逐行核对正式书单预测。
            c_dir = BASE/"outputs"/"event_objective_c_v1"
            if seed != 2026:
                c_dir = c_dir/f"seed_{seed}"
            previous = c_dir/f"{date}_joint.csv"
            if previous.exists():
                keys = ["epoch","user","method"]
                prior = pd.read_csv(previous,dtype={"user":str}).set_index(keys).pred.sort_index()
                current = frame[(frame.encoder=="gru") & (frame.view=="official")].set_index(keys).pred.sort_index()
                if not current.equals(prior):
                    raise ValueError(f"{date} {seed} GRU 与实验 C 的逐人锚点不一致")
            per_seed.append(frame)
        rows = pd.concat(per_seed,ignore_index=True)
        keys=["seed","window","epoch","view","encoder","method","group"]
        results.append(rows.groupby(keys)[METRICS].mean().reset_index())
        fusion=rows[(rows.epoch==8)&(rows.method==FUSION)]
        fusions.append(fusion)
        for view, subset in fusion.groupby("view"):
            pair=compare(subset,[("mean","book_mean"),("gru","mean"),("gru","book_mean")],variant_column="encoder")
            pairs.append(pair.assign(seed=seed,view=view))
    all_fused = pd.concat(fusions).merge(prefixes,on=["window","user"],validate="many_to_one")
    segments = all_fused.groupby(["window","seed","encoder","view","prefix_type"])[METRICS].mean().reset_index()
    save(segments,OUT/f"{stage}_history_segments.csv")
    # 每位读者先平均两个种子，同一读者跨窗保持相同的抽样权重。
    averaged = all_fused.groupby(["window","group","user","encoder","view"])[["score","hit"]].mean().reset_index()
    for view, subset in averaged.groupby("view"):
        pair = compare(subset,[("mean","book_mean"),("gru","mean"),("gru","book_mean")],variant_column="encoder")
        pairs.append(pair.drop(columns=["gained","lost"]).assign(seed="average",view=view))
    groups=pd.concat(results)
    save(groups,OUT/f"{stage}_groups.csv")
    groups[METRICS]=groups[METRICS].mul(groups.group.map({"新人":.4,"老人":.6}),axis=0)
    scores=groups.groupby(keys[:-1])[METRICS].sum().reset_index()
    save(scores,OUT/f"{stage}_scores.csv")
    save(pd.concat(pairs),OUT/f"{stage}_paired.csv")
    primary=scores[(scores.epoch==8)&(scores.method==FUSION)&(scores.view=="official")]
    ranking=primary.groupby("encoder").score.mean().sort_values(ascending=False)
    print(stage,ranking.to_dict(),flush=True)
    if stage=="dev":
        decision={"winner":ranking.index[0],"scores":ranking.to_dict(),"seeds":SEEDS,
                  "criterion":"三个开发窗和两个种子等权，八轮官方书单视图轻融合综合分",
                  "diagnostic":"固定三组与两个种子，同时复核两个预设差值"}
        (OUT/"decision_before_diagnostic.json").write_text(json.dumps(decision,ensure_ascii=False,indent=2),"utf8")


def read_data():
    data=load_interactions(BASE/"data/processed/borrowing_cleaned.csv")
    data[CALL]=data[CALL].map(attribute_value)
    if "性别" not in data:
        data["性别"]=""
    return data


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--fold",choices=[*DEV,DIAG])
    parser.add_argument("--seed",type=int,choices=SEEDS)
    parser.add_argument("--summarize",choices=["dev","diagnostic"])
    args=parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.summarize:
        summarize(DEV if args.summarize=="dev" else [DIAG],args.summarize)
        return
    if not args.fold:
        parser.error("请提供 --fold 或 --summarize")
    if args.fold==DIAG and not (OUT/"decision_before_diagnostic.json").is_file():
        parser.error("请先按开发集结果运行 --summarize dev")
    data=read_data()
    for seed in (SEEDS if args.seed is None else [args.seed]):
        fit_fold(args.fold,seed,data)


if __name__=="__main__":
    main()
