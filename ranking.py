"""第三轮：滚动历史训练学习排序，冻结方案后比较复用诊断集；支持缓存续跑。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from advanced import FusionSpace, fit_channels, paired_bootstrap, quick_metrics
from advanced_models import AdvancedConfig, HistoryContext, set_seed
from baseline import (BASE, ITEM, TIME, USER, evaluate_rankings, ground_truth, load_interactions,
                      load_readers, recommendations_frame, save_csv, source_info, timestamp, write_json)
from optimized import previous_space, subset_diagnostics
from optimized_models import OptimizationConfig, RepeatExploreSpace, RepeatGate
from ranking_models import (CandidateFeatures, TextEncoder, candidate_coverage, fit_ranker,
                            labeled, rank_candidates, training_data)

SNAPSHOTS = ["2022-09-01", "2023-03-01", "2023-05-01", "2023-09-01",
             "2024-03-01", "2024-05-01", "2024-09-01"]
DEVELOPMENT = ["2023-09-01", "2024-03-01", "2024-09-01"]
DIAGNOSTIC = "2024-11-01"
MODEL_REPO = "BAAI/bge-small-zh-v1.5"
MODEL_REVISION = "7999e1d3359715c523056ef9478215996d62a620"
SCHEMA = 1


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_json(temporary, value)
    temporary.replace(path)


def prepare_model(model_dir):
    """从原作者官方仓库下载；固定 revision，拒绝执行远程代码。"""
    import requests
    from concurrent.futures import ThreadPoolExecutor
    from time import sleep
    model_dir = Path(model_dir)
    provenance = model_dir / "provenance.json"
    if provenance.is_file():
        stored = json.loads(provenance.read_text(encoding="utf-8"))
        for name, expected in stored["files_sha256"].items():
            if hashlib.sha256((model_dir / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"模型文件 {name} 校验失败。")
        return stored
    manifest = model_dir / "official_manifest.json"
    if manifest.is_file():
        info = json.loads(manifest.read_text(encoding="utf-8"))
    else:
        for attempt in range(4):
            try:
                response = requests.get(f"https://huggingface.co/api/models/{MODEL_REPO}/revision/{MODEL_REVISION}?blobs=true", timeout=30)
                response.raise_for_status()
                info = response.json()
                atomic_json(manifest, info)
                break
            except requests.RequestException:
                if attempt == 3:
                    raise
                sleep(1)
    revision = info["sha"]
    available = {row["rfilename"] for row in info["siblings"]}
    weights = "model.safetensors" if "model.safetensors" in available else "pytorch_model.bin"
    files = [name for name in ("config.json", "tokenizer.json", "tokenizer_config.json",
                               "special_tokens_map.json", "vocab.txt", weights) if name in available]
    model_dir.mkdir(parents=True, exist_ok=True)
    for name in files:
        target = model_dir / name
        metadata = next(row for row in info["siblings"] if row["rfilename"] == name)
        def verified(path):
            if not path.is_file():
                return False
            value = path.read_bytes()
            if "lfs" in metadata:
                return hashlib.sha256(value).hexdigest() == metadata["lfs"]["sha256"]
            return hashlib.sha1(b"blob " + str(len(value)).encode() + b"\0" + value).hexdigest() == metadata["blobId"]
        if verified(target):
            continue
        url = f"https://huggingface.co/{MODEL_REPO}/resolve/{revision}/{name}"
        size = metadata["size"]
        step = 8 * 1024 * 1024
        count = (size + step - 1) // step
        def fetch(index):
            begin, end = index * step, min(size, (index + 1) * step) - 1
            part = model_dir / f"{revision}.{name}.part{index}"
            if part.is_file() and part.stat().st_size == end - begin + 1:
                return part
            for attempt in range(4):
                try:
                    with requests.get(url, headers={"Range": f"bytes={begin}-{end}"}, stream=True, timeout=(20, 60)) as remote:
                        remote.raise_for_status()
                        if remote.status_code == 206:
                            if remote.headers.get("Content-Range") != f"bytes {begin}-{end}/{size}":
                                raise ValueError("下载分段范围与请求不符。")
                        elif not (begin == 0 and end == size - 1):
                            raise ValueError("下载服务器不支持分段。")
                        with part.open("wb") as stream:
                            for chunk in remote.iter_content(256 * 1024):
                                stream.write(chunk)
                    if part.stat().st_size != end - begin + 1:
                        raise ValueError("下载分段不完整。")
                    print(f"模型下载 {name}: 分段 {index + 1}/{count} 完成", flush=True)
                    return part
                except (requests.RequestException, ValueError):
                    if attempt == 3:
                        raise
                    sleep(1)
        with ThreadPoolExecutor(max_workers=8) as executor:
            parts = list(executor.map(fetch, range(count)))
        partial = target.with_suffix(target.suffix + ".complete")
        with partial.open("wb") as stream:
            for part in parts:
                stream.write(part.read_bytes())
        if not verified(partial):
            raise ValueError(f"{name} 与官方文件哈希不一致。")
        partial.replace(target)
    meta = {"repo": MODEL_REPO, "revision": revision,
            "files_sha256": {name: hashlib.sha256((model_dir / name).read_bytes()).hexdigest() for name in files},
            "pooling": "CLS, L2 normalized", "max_tokens": 128, "fine_tuned_on_competition": False}
    atomic_json(provenance, meta)
    return meta


def legacy_channels(context, users, plan, directory):
    previous = plan["previous_plan"]
    weights = previous["selection"]["weights"]
    raw, summary = fit_channels(context, users, context.config, Path(directory) / "previous",
                                frozen_epochs=previous["selection"]["epochs"],
                                needed=(*weights, "recent"), k=10)
    raw["previous"] = previous_space(raw, users, context.config.rrf_constant).predict(weights, 200)
    gate = RepeatGate(context, OptimizationConfig(**plan["config"]))
    recipe = plan["recommendation"]
    if recipe["strategy"] != "repeat_explore":
        raise ValueError("本轮参照接口要求已冻结的复借探索方案。")
    raw["optimized"] = RepeatExploreSpace(context, raw, users, gate).predict(
        recipe["weights"], 200, recipe["gate_scale"])
    return raw, {"previous": summary, "gate": gate.summary}


def context_for(data, start, plan):
    return HistoryContext(data.loc[data[TIME].lt(timestamp(start))], start,
                          AdvancedConfig(**plan["previous_plan"]["config"]))


def build_snapshot(data, start, end, plan, encoder, output, identity):
    directory = Path(output) / "snapshots" / str(start)[:10]
    manifest_path = directory / "manifest.json"
    expected = {"identity": identity, "start": timestamp(start).isoformat(),
                "end": timestamp(end).isoformat()}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if all(manifest.get(key) == value for key, value in expected.items()):
            print(f"复用完整快照 {str(start)[:10]}", flush=True)
            return manifest
        raise ValueError(f"{directory} 的缓存来源不匹配；请使用新的 --output-dir。")
    context = context_for(data, start, plan)
    future = data.loc[data[TIME].ge(timestamp(start)) & data[TIME].lt(timestamp(end))]
    truth = ground_truth(future)
    if not truth:
        raise ValueError("预测窗口没有可评价的借阅。")
    users = sorted(truth)
    print(f"生成快照 {str(start)[:10]}：历史 {len(context.history)} 条，目标 {len(users)} 位读者", flush=True)
    raw, summary = legacy_channels(context, users, plan, directory)
    features, new = CandidateFeatures(context, encoder).build(users, raw)
    raw.update(new)
    features = labeled(features, truth, start, end)
    directory.mkdir(parents=True, exist_ok=True)
    feature_path = directory / "features.pkl"
    features.to_pickle(feature_path.with_suffix(".tmp"))
    feature_path.with_suffix(".tmp").replace(feature_path)
    atomic_json(directory / "channels.json", raw)
    atomic_json(directory / "training_summary.json", summary)
    manifest = {**expected, "path": str(feature_path), "directory": str(directory),
                "history_max": context.history[TIME].max().isoformat(),
                "users": len(users), "rows": len(features),
                "coverage": candidate_coverage(features, truth, context)}
    atomic_json(manifest_path, manifest)
    return manifest


def plans():
    result = [{"name": "optimized", "kind": "reference"}]
    # 预先声明的固定内容融合；不为诊断集寻找权重。
    result.append({"name": "expanded_fixed", "kind": "fixed"})
    for expanded in (False, True):
        prefix = "expanded" if expanded else "existing"
        for trees in (100, 250):
            for weight in (.25, .5, 1.0):
                result.append({"name": f"{prefix}_{trees}_blend{weight:g}", "kind": "ranker",
                               "expanded": expanded, "trees": trees, "weight": weight})
            result.append({"name": f"{prefix}_{trees}_ranker", "kind": "ranker",
                           "expanded": expanded, "trees": trees, "weight": None})
    return result


def strategy_predictions(raw, ranked, recipe, k=10):
    users = list(raw["optimized"])
    if recipe["kind"] == "reference":
        return {u: raw["optimized"][u][:k] for u in users}
    if recipe["kind"] == "fixed":
        return FusionSpace({n: raw[n] for n in ("optimized", "semantic", "category")}, users).predict(
            {"optimized": 1., "semantic": .25, "category": .25}, k)
    rank = ranked[(recipe["expanded"], recipe["trees"])]
    if recipe["weight"] is None:
        return {u: rank[u][:k] for u in users}
    return FusionSpace({"optimized": raw["optimized"], "ranker": rank}, users).predict(
        {"optimized": 1., "ranker": recipe["weight"]}, k)


def evaluate_window(data, snap, snapshots, recipes, plan, output, diagnostic=False):
    directory = Path(snap["directory"])
    evaluation_path = directory / "evaluation.json"
    cache_key = {"recipes": recipes, "training_windows": [[s["start"], s["end"]] for s in snapshots
                                                        if timestamp(s["end"]) <= timestamp(snap["start"])],
                 "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if evaluation_path.is_file():
        cached = json.loads(evaluation_path.read_text(encoding="utf-8"))
        if cached.get("cache_key") == cache_key:
            detail = pd.read_csv(directory / "per_user.csv", dtype={USER: str})
            rows = {name: group.drop(columns="method").reset_index(drop=True)
                    for name, group in detail.groupby("method")}
            print(f"复用完成的排序比较 {str(snap['start'])[:10]}", flush=True)
            return cached, rows
    context = context_for(data, snap["start"], plan)
    target = data.loc[data[TIME].ge(timestamp(snap["start"])) & data[TIME].lt(timestamp(snap["end"]))]
    truth = ground_truth(target)
    raw = json.loads((directory / "channels.json").read_text(encoding="utf-8"))
    features = pd.read_pickle(snap["path"])
    ranked, training = {}, {}
    for expanded in sorted({r["expanded"] for r in recipes if r["kind"] == "ranker"}):
        frame, used = training_data(snapshots, snap["start"], expanded)
        model_path = directory / ("expanded.cbm" if expanded else "existing.cbm")
        print(f"排序训练 {str(snap['start'])[:10]} expanded={expanded}，"
              f"{frame['query'].nunique()} 组，{len(frame)} 候选，{frame['label'].sum()} 正例", flush=True)
        model, columns = fit_ranker(frame, expanded, model_path)
        training[str(expanded)] = {"snapshots": used, "rows": len(frame), "queries": int(frame["query"].nunique()),
                                  "max_label_end": max(x["end"] for x in used), "model": str(model_path)}
        for trees in sorted({r["trees"] for r in recipes if r["kind"] == "ranker" and r["expanded"] == expanded}):
            ranked[(expanded, trees)] = rank_candidates(model, columns, features, raw["optimized"],
                                                        expanded, trees, k=200)
        del frame, model
    predictions = {r["name"]: strategy_predictions(raw, ranked, r) for r in recipes}
    for name in ("baseline", "previous", "recent"):
        predictions[name] = {u: items[:10] for u, items in raw[name].items()}
    metrics, per_user, subsets = {}, {}, {}
    for name, prediction in predictions.items():
        if any(len(x) != 10 or len(set(x)) != 10 or not set(x) <= context.baseline.items for x in prediction.values()):
            raise ValueError("推荐必须为 10 个不重复的历史馆藏条码。")
        metrics[name], per_user[name] = evaluate_rankings(prediction, truth, context.baseline, 10)
        subsets[name] = subset_diagnostics(prediction, truth, context, 10)
        save_csv(recommendations_frame(prediction, 10), directory / (name + ".csv"))
    save_csv(pd.concat([v.assign(method=n) for n, v in per_user.items()], ignore_index=True), directory / "per_user.csv")
    report = {"start": snap["start"], "end": snap["end"], "methods": metrics, "cache_key": cache_key,
              "subsets": subsets, "training": training, "coverage": snap["coverage"]}
    atomic_json(directory / "evaluation.json", report)
    print("窗口成绩 " + str(snap["start"])[:10] + ": " +
          ", ".join(n + "=" + format(v["all"]["score"], ".6f") for n, v in metrics.items()), flush=True)
    return report, per_user


def evaluate(args):
    started = perf_counter()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    plan = json.loads(Path(args.previous_plan).read_text(encoding="utf-8"))
    original = source_info(args.input)
    if original["sha256"] != plan["input"]["sha256"]:
        raise ValueError("本轮比较要求与上一轮相同的输入文件。")
    set_seed(2026, 4)
    model_meta = prepare_model(args.model_dir)
    identity_payload = {"source": original["sha256"], "schema": SCHEMA, "previous": plan,
                        "encoder": model_meta,
                        "feature_code": hashlib.sha256((BASE / "ranking_models.py").read_bytes()).hexdigest()}
    identity = hashlib.sha256(json.dumps(identity_payload, sort_keys=True).encode()).hexdigest()
    protocol = {**identity_payload, "identity": identity, "snapshots": SNAPSHOTS, "development": DEVELOPMENT,
                "diagnostic_start": DIAGNOSTIC, "diagnostic_reused_not_blind": True,
                "recipes": plans(), "selection": "equal mean composite over three development windows",
                "ranker_training": "only complete earlier snapshot labels; no injected positives; warm groups with candidate positives",
                "cold_predictions": "ranker falls back to frozen optimized lists; no independent profiles in evaluation"}
    existing = output / "protocol.json"
    if existing.is_file() and json.loads(existing.read_text(encoding="utf-8"))["identity"] != identity:
        raise ValueError("输出目录对应不同特征版本，请另选 --output-dir。")
    atomic_json(existing, protocol)
    data = load_interactions(args.input)
    encoder = TextEncoder(args.model_dir, output / "embeddings")
    snapshots, development = [], []
    for start in SNAPSHOTS:
        atomic_json(output / "progress.json", {"stage": "development", "current": start,
                                               "completed": [s["start"] for s in snapshots]})
        end = timestamp(start) + pd.DateOffset(months=2)
        snap = build_snapshot(data, start, end, plan, encoder, output, identity)
        if start in DEVELOPMENT:
            report, _ = evaluate_window(data, snap, snapshots, plans(), plan, output)
            development.append(report)
        snapshots.append(snap)
    trials = []
    for recipe in plans():
        scores = [fold["methods"][recipe["name"]]["all"] for fold in development]
        trials.append({"recipe": recipe, "per_window": scores,
                       "mean_score": float(np.mean([row["score"] for row in scores]))})
    chosen = max(trials, key=lambda x: (x["mean_score"], x["recipe"]["kind"] == "reference"))
    # 先写锁定计划；随后才构造诊断候选或读取诊断成绩。
    locked = {"schema": SCHEMA, "identity": identity, "input": original, "previous_plan": plan,
              "recipe": chosen["recipe"], "selection": chosen,
              "model_dir": str(Path(args.model_dir).resolve()), "encoder": model_meta,
              "training_snapshots": snapshots}
    atomic_json(output / "trials.json", trials)
    atomic_json(output / "locked_plan.json", locked)
    print("已冻结：" + json.dumps(chosen, ensure_ascii=False), flush=True)
    atomic_json(output / "progress.json", {"stage": "diagnostic", "selected": chosen["recipe"]})
    end = data[TIME].max() + pd.Timedelta(nanoseconds=1)
    diagnostic = build_snapshot(data, DIAGNOSTIC, end, plan, encoder, output, identity)
    # 两个排序消融以开发集各自最好的设置比较；诊断分不参与选型。
    comparisons = [plans()[0], plans()[1]]
    for expanded in (False, True):
        comparisons.append(max((t for t in trials if t["recipe"].get("expanded") is expanded),
                               key=lambda t: t["mean_score"])["recipe"])
    comparisons.append(chosen["recipe"])
    comparisons = list({r["name"]: r for r in comparisons}.values())
    report, rows = evaluate_window(data, diagnostic, snapshots, comparisons, plan, output, True)
    reference = json.loads((BASE / "outputs/optimized/experiment_report.json").read_text(encoding="utf-8"))
    deltas = {}
    for new, old in (("baseline", "baseline"), ("previous", "previous"), ("recent", "recent"), ("optimized", "selected")):
        expected = reference["diagnostic"]["methods"][old]["all"]
        current = report["methods"][new]["all"]
        deltas[new] = {k: current[k] - expected[k] for k in ("recall@10", "ndcg@10", "score")}
    if any(abs(x) > 1e-12 for values in deltas.values() for x in values.values()):
        raise ValueError("旧模型分数没有精确复现，停止解释新模型成绩。")
    selected = chosen["recipe"]["name"]
    bootstrap = {name: paired_bootstrap(rows[name], rows[selected], samples=5000)
                 for name in ("baseline", "previous", "recent", "optimized")}
    result = {"input": original, "protocol": protocol, "development": development, "trials": trials,
              "selection": chosen, "diagnostic": report, "paired_comparisons": bootstrap,
              "reference_deltas": deltas, "elapsed_seconds": perf_counter() - started}
    atomic_json(output / "experiment_report.json", result)
    save_csv(pd.DataFrame([{"method": name, **scores["all"]}
                          for name, scores in report["methods"].items()]), output / "comparison.csv")
    atomic_json(output / "progress.json", {"stage": "evaluation_complete", "selected": selected,
                                          "score": report["methods"][selected]["all"]["score"]})
    print("实验完成：" + str(output / "experiment_report.json"), flush=True)


def predict(args):
    from catboost import CatBoostRanker
    locked = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    if source_info(args.input)["sha256"] != locked["input"]["sha256"]:
        raise ValueError("预测输入必须匹配冻结方案的数据。")
    set_seed(2026, 4)
    data = load_interactions(args.input)
    cutoff = timestamp(args.as_of) if args.as_of else data[TIME].max() + pd.Timedelta(nanoseconds=1)
    external = None
    if args.readers:
        supplied = load_readers(args.readers)
        readers = supplied[USER].tolist()
        external = supplied.set_index(USER).to_dict("index")
    else:
        readers = sorted(data.loc[data[TIME].lt(cutoff), USER].unique())
    if args.readers:
        supplied = pd.read_csv(args.readers, dtype=str, keep_default_na=False)
        if "画像时间" in supplied:
            times = pd.to_datetime(supplied["画像时间"], errors="raise")
            if times.isna().any() or times.ge(cutoff).any():
                raise ValueError("独立画像时间必须非空且早于预测时点。")
    context = context_for(data, cutoff, locked["previous_plan"])
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    raw, _ = legacy_channels(context, readers, locked["previous_plan"], output)
    recipe = locked["recipe"]
    ranked = {}
    if recipe["kind"] != "reference":
        encoder = TextEncoder(locked["model_dir"], Path(args.plan).parent / "embeddings")
        features, new = CandidateFeatures(context, encoder).build(readers, raw, external=external)
        raw.update(new)
        if recipe["kind"] == "ranker":
            # 加入诊断窗口的完整历史标签用于最终训练；不影响已冻结选型。
            snapshots = list(locked["training_snapshots"])
            diag_path = Path(args.plan).parent / "snapshots" / DIAGNOSTIC / "manifest.json"
            if diag_path.is_file():
                diagnostic = json.loads(diag_path.read_text(encoding="utf-8"))
                if diagnostic["identity"] != locked["identity"]:
                    raise ValueError("诊断缓存来源与冻结计划不一致。")
                snapshots.append(diagnostic)
            frame, used = training_data(snapshots, cutoff, recipe["expanded"])
            model, columns = fit_ranker(frame, recipe["expanded"], output / "ranker.cbm")
            ranked[(recipe["expanded"], recipe["trees"])] = rank_candidates(
                model, columns, features, raw["optimized"], recipe["expanded"], recipe["trees"], 200)
            atomic_json(output / "ranker_training.json", used)
    prediction = strategy_predictions(raw, ranked, recipe)
    # 真正新读者若有独立画像，沿用已有的明确画像先验回退，不推断未来画像。
    if external:
        from optimized_models import AttributePrior
        cold = [u for u in readers if u not in context.user_index and external.get(u)]
        if cold:
            prediction.update(AttributePrior(context, OptimizationConfig()).rank(cold, 10, external))
    if any(len(v) != 10 or len(set(v)) != 10 or not set(v) <= context.baseline.items for v in prediction.values()):
        raise ValueError("预测列表不合法。")
    save_csv(recommendations_frame(prediction, 10), output / "result.csv")
    atomic_json(output / "prediction_summary.json", {"readers": len(readers), "cutoff": cutoff.isoformat(),
                                                    "recipe": recipe, "input": source_info(args.input)})
    print(f"已生成 {len(readers)} 位读者的 Top-10。", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare-model", help="下载官方固定版本语义模型")
    prep.add_argument("--model-dir", default=str(BASE / ".model_deps/bge-small-zh-v1.5"))
    evaluation = commands.add_parser("evaluate", help="自动复用完整快照，运行开发与诊断比较")
    evaluation.add_argument("--previous-plan", default=str(BASE / "outputs/optimized/locked_plan.json"))
    evaluation.add_argument("--model-dir", default=str(BASE / ".model_deps/bge-small-zh-v1.5"))
    evaluation.add_argument("--output-dir", default=str(BASE / "outputs/ranking"))
    prediction = commands.add_parser("predict")
    prediction.add_argument("--plan", default=str(BASE / "outputs/ranking/locked_plan.json"))
    prediction.add_argument("--readers")
    prediction.add_argument("--as-of")
    prediction.add_argument("--output-dir", default=str(BASE / "outputs/ranking_prediction"))
    for sub in (evaluation, prediction):
        sub.add_argument("--input", default=str(BASE / "data/2022-2025处理后借阅信息.xlsx"))
    args = parser.parse_args()
    if args.command == "prepare-model":
        print(json.dumps(prepare_model(args.model_dir), ensure_ascii=False))
    elif args.command == "evaluate":
        evaluate(args)
    else:
        predict(args)


if __name__ == "__main__":
    main()
