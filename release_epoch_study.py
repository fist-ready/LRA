"""为正式 GRU-8 架构比较用户指定的训练轮数，输出三个开发窗的综合分。"""
import argparse

import pandas as pd
import torch

from event_encoder_b import read_data
from event_objective_c import BASE, RUN, DEV, FUSION, save
from evaluate_event_suite import load_model, prepare, evaluate, iter_train_checkpoints

OUT = BASE / "outputs" / "release_event_epochs_v1"
EPOCHS = (8, 12, 16)
SEEDS = (2026, 2027)


def parse_epochs(values):
    try:
        epochs = sorted({int(part) for value in values for part in value.split(",")})
    except ValueError as exc:
        raise ValueError("轮数应为正整数，例如 --epochs 4 8 12 或 --epochs 4,8,12") from exc
    if not epochs or epochs[0] < 1:
        raise ValueError("轮数必须大于零")
    return tuple(epochs)


def fold(date, epochs=EPOCHS, out=OUT, fresh=False):
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    ck, _ = load_model(RUN/date/"model.pt")
    records, history = prepare(read_data(), date, ck)
    warm = {q.user for q, *_ in records if q.seen}
    for seed in SEEDS:
        dest = out / f"{date}_{seed}.csv"
        if dest.exists() and not fresh:
            print(f"复用已完成回放 {date}，种子 {seed}", flush=True)
            continue
        print(f"训练 {date}，种子 {seed}，共 {max(epochs)} 轮", flush=True)
        ck["manifest"]["seed"] = seed
        rows = []
        for epoch, model in iter_train_checkpoints(ck, history, date, "joint", dest.with_suffix(".json"), checkpoints=epochs):
            part = evaluate(ck, model, records, date, epoch)
            mask = part.user.isin(warm)
            routed = part.loc[(mask & part.method.eq(FUSION)) | (~mask & part.method.eq("当前提交"))].copy()
            assert routed.user.is_unique and len(routed) == len(records)
            routed["seed"] = seed
            rows.append(routed)
        save(pd.concat(rows), dest)


def summarize(epochs=EPOCHS, out=OUT):
    parts = []
    for date in DEV:
        for seed in SEEDS:
            frame = pd.read_csv(out/f"{date}_{seed}.csv", dtype={"user":str})
            if (not frame.window.eq(date).all() or not frame.seed.eq(seed).all()
                    or set(frame.epoch) != set(epochs) or frame.duplicated(["epoch","user"]).any()):
                raise ValueError(f"结果与请求的轮数或窗口不符：{date}，种子 {seed}")
            parts.append(frame)
    df = pd.concat(parts)
    scores = df.groupby(["window","seed","epoch","group"]).score.mean().unstack()
    scores["total"] = .4*scores["新人"] + .6*scores["老人"]
    save(scores.reset_index(), out/"scores.csv")
    table = scores.total.groupby(["epoch","window"]).mean().unstack().loc[list(epochs),list(DEV)]
    if table.isna().any().any():
        raise ValueError("三窗分组得分不完整")
    table.columns = [d[:7]+" 开发窗" for d in DEV]
    table["平均综合分"] = table.mean(axis=1)
    table.index.name = "训练轮数"
    best = int(table["平均综合分"].idxmax())
    lines = ["| 训练轮数 | " + " | ".join(table.columns) + " |", "|---:|---:|---:|---:|---:|"]
    for epoch, row in table.iterrows():
        lines.append(f"| {epoch} 轮 | " + " | ".join(f"{score:.3f}" for score in row) + " |")
    markdown = "\n".join(lines)
    (out/"epoch_table.md").write_text(markdown+"\n","utf8")
    table.reset_index().to_csv(out/"epoch_table.csv",index=False,encoding="utf-8-sig",float_format="%.3f")
    (out/"epochs.txt").write_text(str(best),"utf8")
    console = table.reset_index()
    console["训练轮数"] = console["训练轮数"].map(lambda n: f"{n} 轮")
    with pd.option_context("display.unicode.east_asian_width", True):
        print("\n"+console.to_string(index=False,float_format=lambda value:f"{value:.3f}"))
    print(f"开发均分最高：{best} 轮（{table.loc[best,'平均综合分']:.3f}）。结果目录：{out}",flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs",nargs="+",metavar="N",help="一个或多个正整数，例如 4 8 12；自动训练三个开发窗并制表")
    parser.add_argument("--fold",choices=DEV,help="仅运行指定窗口（用于分窗执行）")
    parser.add_argument("--fresh",action="store_true",help="重新训练指定轮数组合，即使已有该组合的缓存")
    args = parser.parse_args()
    if args.fresh and not args.epochs:
        parser.error("--fresh 请配合 --epochs 指定本次实验的轮数")
    if not args.epochs:
        if args.fold:
            fold(args.fold)
        else:
            summarize()
        return
    try:
        epochs = parse_epochs(args.epochs)
    except ValueError as exc:
        parser.error(str(exc))
    out = BASE/"outputs"/"custom_epoch_study"/("_".join(map(str,epochs)))
    for date in ([args.fold] if args.fold else DEV):
        fold(date, epochs, out, args.fresh)
    if not args.fold:
        summarize(epochs, out)


if __name__=="__main__":
    main()
