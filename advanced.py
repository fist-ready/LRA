"""多路推荐实验：BPR / 属性 BPR / Item2Vec / GRU / 内容，以及验证集选择的融合。"""

from __future__ import annotations

import argparse
import json
import math
import platform
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from baseline import (
    BASE, ITEM, TIME, USER, evaluate_rankings, frame_summary, ground_truth,
    load_interactions, load_readers, recommendations_frame, save_csv, source_info,
    target_diagnostics, temporal_split, timestamp, write_json,
)
from advanced_models import (
    AdvancedConfig, ContentModel, HistoryContext, fit_neural, neural_rankings,
    top_indices, torch,
)

CHANNELS = ("baseline", "recent", "content", "bpr_mf", "attribute_bpr", "item2vec", "gru")
NEURAL_CHANNELS = ("bpr_mf", "attribute_bpr", "item2vec", "gru")
FULL_WEIGHTS = {"baseline": 1.0, "attribute_bpr": 0.4, "item2vec": 0.4, "gru": 0.4, "content": 0.4}


class FusionSpace:
    """在各路历史候选的并集上做加权倒数排名融合，不读取真值构建候选。"""

    def __init__(self, rankings, users, constant=20.0):
        self.users, self.channels = list(users), list(rankings)
        self.rows = []
        if not math.isfinite(constant) or constant <= 0:
            raise ValueError("融合常数必须为有限正数。")
        for user in self.users:
            pool = sorted({item for channel in rankings.values() for item in channel.get(user, [])})
            index = {item: position for position, item in enumerate(pool)}
            matrix = np.zeros((len(pool), len(self.channels)), dtype=np.float64)
            for column, name in enumerate(self.channels):
                items = rankings[name].get(user, [])
                if len(items) != len(set(items)):
                    raise ValueError(f"{name} 的推荐包含重复条码。")
                for rank, item in enumerate(items, 1):
                    matrix[index[item], column] = 1.0 / (constant + rank)
            self.rows.append((pool, matrix))

    def predict(self, weights, k=10):
        if k < 1 or any(name not in self.channels for name in weights):
            raise ValueError("k 或融合通道无效。")
        vector = np.asarray([weights.get(name, 0.0) for name in self.channels], dtype=np.float64)
        if not np.isfinite(vector).all() or (vector < 0).any() or vector.sum() <= 0:
            raise ValueError("融合权重必须非负、有限，且至少一个为正。")
        predictions = {}
        for user, (pool, matrix) in zip(self.users, self.rows):
            scores = matrix @ vector
            scores[scores <= 0] = -np.inf
            predictions[user] = [pool[index] for index in top_indices(scores, k)]
        return predictions


def quick_metrics(predictions, truth, k):
    from baseline import ranking_metrics
    values = [ranking_metrics(predictions.get(user, []), relevant, k) for user, relevant in truth.items()]
    return {key: float(np.mean([row[key] for row in values])) for key in (f"recall@{k}", f"ndcg@{k}", "score")}


def select_fusion(space, truth, k=10):
    """预定义全融合、单路对照及至多两轮前向选择，只接收验证真值。"""
    trials, cache = [], {}

    def evaluate(weights, label):
        weights = {name: float(value) for name, value in weights.items() if value > 0}
        key = tuple(sorted(weights.items()))
        if key not in cache:
            metrics = quick_metrics(space.predict(weights, k), truth, k)
            row = {"name": label, "weights": weights, **metrics}
            cache[key] = row
            trials.append(row)
        return cache[key]

    for name in space.channels:
        evaluate({name: 1.0}, name)
    evaluate(FULL_WEIGHTS, "full_fusion")
    current = evaluate({"baseline": 1.0}, "baseline")
    for round_index in range(2):
        candidates = [current]
        for name in space.channels:
            if name in current["weights"]:
                continue
            for weight in (0.1, 0.25, 0.5, 1.0):
                candidates.append(evaluate({**current["weights"], name: weight}, f"forward_{round_index + 1}_{name}_{weight}"))
        best = max(candidates, key=lambda row: (row["score"], row[f"ndcg@{k}"], -len(row["weights"])))
        if best["score"] <= current["score"] + 1e-12:
            break
        current = best
    selected = max(trials, key=lambda row: (row["score"], row[f"ndcg@{k}"], -len(row["weights"])))
    return selected, trials


def paired_bootstrap(baseline_rows, alternative_rows, k=10, samples=5000, seed=2026):
    """以读者为重采样单位，保留同一读者两套推荐的配对关系。"""
    if samples < 1:
        raise ValueError("bootstrap 次数必须为正。")
    left = baseline_rows.set_index(USER).sort_index()
    right = alternative_rows.set_index(USER).sort_index()
    if left.empty or not left.index.equals(right.index) or not left.index.is_unique:
        raise ValueError("bootstrap 必须比较同一组非空且唯一的读者。")
    columns = [f"recall@{k}", f"ndcg@{k}", "score"]
    differences = right[columns].to_numpy() - left[columns].to_numpy()
    rng = np.random.default_rng(seed)
    simulated = []
    for start in range(0, samples, 250):
        indices = rng.integers(0, len(left), size=(min(250, samples - start), len(left)))
        simulated.append(differences[indices].mean(axis=1))
    simulated = np.concatenate(simulated)
    result = {"users": len(left), "bootstrap_samples": samples, "unit": "paired reader", "metrics": {}}
    for column, name in enumerate(columns):
        delta = float(differences[:, column].mean())
        base = float(left[name].mean())
        low, high = np.percentile(simulated[:, column], [2.5, 97.5])
        result["metrics"][name] = {
            "baseline": base, "alternative": float(right[name].mean()), "absolute_delta": delta,
            "relative_delta": delta / base if base else None,
            "ci95_low": float(low), "ci95_high": float(high),
            "ci95_excludes_zero_positive": bool(low > 0),
            "readers_improved": int((differences[:, column] > 1e-12).sum()),
            "readers_worsened": int((differences[:, column] < -1e-12).sum()),
            "readers_tied": int((np.abs(differences[:, column]) <= 1e-12).sum()),
        }
    return result


def diagnostic_comparisons(per_user, k=10, samples=5000, seed=2026):
    """解释收益来源的附加比较；不参与模型选择，也不替换预设的主要比较。"""
    result = {}
    if "recent" in per_user:
        result["recent_vs_baseline"] = paired_bootstrap(per_user["baseline"], per_user["recent"], k, samples, seed)
        result["selected_vs_recent"] = paired_bootstrap(per_user["recent"], per_user["selected"], k, samples, seed)
    for name in per_user:
        if name.startswith("without_"):
            result["selected_vs_" + name] = paired_bootstrap(per_user[name], per_user["selected"], k, samples, seed)
    return result


def fit_channels(context, users, config, output_dir, truth=None, frozen_epochs=None, k=10, needed=None):
    needed = set(needed or CHANNELS)
    rankings, training = {}, {}
    candidate_k = max(k, config.candidate_k)
    if "baseline" in needed:
        rankings["baseline"] = context.baseline_rankings(users, candidate_k)
    if "recent" in needed:
        rankings["recent"] = context.recent_rankings(users, candidate_k)
    if "content" in needed:
        print("构建历史题名、作者和分类的内容分支。", flush=True)
        rankings["content"] = ContentModel(context).rank(users, candidate_k)
    # GRU 使用同一训练期 Item2Vec 初始化，因此需要先训练该表示，即使其融合权重为零。
    train_names = set(needed) & set(NEURAL_CHANNELS)
    if "gru" in train_names:
        train_names.add("item2vec")
    pretrained = None
    for name in NEURAL_CHANNELS:
        if name not in train_names:
            continue
        epoch = frozen_epochs[name] if frozen_epochs is not None else None
        model, summary = fit_neural(
            name, context, config, users if truth is not None else None, truth, k,
            epochs=epoch, pretrained=pretrained if name == "gru" else None,
            output_dir=Path(output_dir) / "checkpoints",
        )
        training[name] = summary
        if name == "item2vec":
            pretrained = model.input_embedding.weight.detach().cpu().numpy().copy()
        if name in needed:
            rankings[name] = neural_rankings(name, model, context, users, candidate_k)
        del model
    return {name: rankings[name] for name in CHANNELS if name in rankings}, training


def variants(space, selection, k):
    result = {name: space.predict({name: 1.0}, k) for name in space.channels}
    result["full_fusion"] = space.predict(FULL_WEIGHTS, k)
    result["selected"] = space.predict(selection, k)
    if len(selection) > 1:
        for name in selection:
            reduced = {channel: weight for channel, weight in selection.items() if channel != name}
            result["without_" + name] = space.predict(reduced, k)
    return result


def save_metrics(stage, predictions, truth, context, k, output_dir):
    all_rows, methods, per_user_tables = [], {}, {}
    for name, ranking in predictions.items():
        metrics, rows = evaluate_rankings(ranking, truth, context.baseline, k)
        methods[name], per_user_tables[name] = metrics, rows
        detail = rows.copy()
        detail.insert(0, "method", name)
        all_rows.append(detail)
        save_csv(recommendations_frame(ranking, k), Path(output_dir) / f"{stage}_{name}.csv")
        values = metrics["all"]
        print(f"{stage:10s} {name:20s} Recall@{k}={values[f'recall@{k}']:.6f} "
              f"NDCG@{k}={values[f'ndcg@{k}']:.6f} score={values['score']:.6f}", flush=True)
    save_csv(pd.concat(all_rows, ignore_index=True), Path(output_dir) / f"{stage}_per_user.csv")
    return methods, per_user_tables


def verify_reference(report, reference_path):
    reference_path = Path(reference_path)
    if not reference_path.is_file():
        return {"available": False}
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    comparable = reference["input"]["sha256"] == report["input"]["sha256"] and reference["boundaries"] == report["boundaries"] and reference["k"] == report["k"]
    result = {"available": True, "same_input_boundaries_and_k": comparable}
    if comparable:
        deltas = {}
        for stage in ("validation", "test"):
            old = reference["stages"][stage]["methods"]["hybrid"]["all"]["score"]
            new = report["stages"][stage]["methods"]["baseline"]["all"]["score"]
            deltas[stage] = new - old
        result["baseline_score_deltas"] = deltas
        result["baseline_reproduced"] = all(abs(delta) < 1e-12 for delta in deltas.values())
    return result


def run_experiment(input_path, output_dir, config=None, valid_start="2024-09-01",
                   test_start="2024-11-01", test_end=None, k=10, sheet=None, bootstrap_samples=5000):
    config = config or AdvancedConfig()
    if k < 1 or config.candidate_k < k:
        raise ValueError("必须满足 1 <= k <= candidate_k。")
    started = perf_counter()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_interactions(input_path, sheet)
    train, valid, test = temporal_split(data, valid_start, test_start, test_end)
    report = {
        "schema_version": 1, "input": source_info(input_path), "k": k, "config": asdict(config),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "torch": torch.__version__, "device": "cpu"},
        "boundaries": {"valid_start": timestamp(valid_start).isoformat(), "test_start": timestamp(test_start).isoformat(),
                       "test_end_exclusive": timestamp(test_end).isoformat() if test_end else None},
        "data": {"train": frame_summary(train), "validation": frame_summary(valid), "test": frame_summary(test)},
        "protocol": "same temporal splits and historical barcode catalog as baseline; all future-active readers; repeats retained; validation-only epoch and fusion selection",
        "stages": {},
    }
    write_json(output_dir / "experiment_config.json", {key: value for key, value in report.items() if key != "stages"})
    print(f"沿用时间切分：train={len(train)}, validation={len(valid)}, test={len(test)}。", flush=True)
    print("验证阶段：训练与画像只使用 valid_start 之前的数据。", flush=True)
    context = HistoryContext(train, valid_start, config)
    truth = ground_truth(valid)
    rankings, training = fit_channels(context, sorted(truth), config, output_dir / "validation", truth=truth, k=k)
    space = FusionSpace(rankings, sorted(truth), config.rrf_constant)
    selected, trials = select_fusion(space, truth, k)
    epochs = {name: row["selected_epoch"] for name, row in training.items()}
    selection = {"weights": selected["weights"], "epochs": epochs, "based_on": "validation only",
                 "validation_score": selected["score"], "validation_trial_count": len(trials)}
    report["selection"] = selection
    # 在开始测试期模型拟合和测试指标计算前，把方法、权重、轮数写成冻结计划。
    locked_plan = {"schema_version": 1, "input": report["input"], "config": asdict(config), "k": k,
                   "boundaries": report["boundaries"], "selection": selection, "full_fusion_weights": FULL_WEIGHTS}
    write_json(output_dir / "locked_plan.json", locked_plan)
    write_json(output_dir / "validation_trials.json", trials)
    save_csv(pd.DataFrame([{**row, "weights": json.dumps(row["weights"], ensure_ascii=False, sort_keys=True)} for row in trials]), output_dir / "validation_trials.csv")
    val_predictions = variants(space, selected["weights"], k)
    metrics, _ = save_metrics("validation", val_predictions, truth, context, k, output_dir)
    report["stages"]["validation"] = {"training": training, "history": frame_summary(train),
                                      "diagnostics": target_diagnostics(train, valid, k, config.exclude_seen), "methods": metrics}
    print(f"已冻结融合权重 {selected['weights']}，训练轮数 {epochs}。", flush=True)
    del context, space, rankings

    print("测试阶段：只合并 train + validation，按冻结方案重新训练。", flush=True)
    history = pd.concat([train, valid], ignore_index=True)
    context = HistoryContext(history, test_start, config)
    truth = ground_truth(test)
    rankings, training = fit_channels(context, sorted(truth), config, output_dir / "test", frozen_epochs=epochs, k=k)
    space = FusionSpace(rankings, sorted(truth), config.rrf_constant)
    test_predictions = variants(space, selected["weights"], k)
    metrics, per_user = save_metrics("test", test_predictions, truth, context, k, output_dir)
    report["stages"]["test"] = {"training": training, "history": frame_summary(history),
                                "diagnostics": target_diagnostics(history, test, k, config.exclude_seen), "methods": metrics}
    report["paired_comparison"] = {
        name: paired_bootstrap(per_user["baseline"], per_user[name], k, bootstrap_samples, config.seed)
        for name in ("selected", "full_fusion")
    }
    report["diagnostic_comparisons"] = diagnostic_comparisons(per_user, k, bootstrap_samples, config.seed)
    report["reference_baseline"] = verify_reference(report, BASE / "outputs/baseline/evaluation_report.json")
    report["elapsed_seconds"] = round(perf_counter() - started, 3)
    rows = [{"split": stage, "method": name, "segment": segment, **values}
            for stage, details in report["stages"].items() for name, segments in details["methods"].items()
            for segment, values in segments.items()]
    save_csv(pd.DataFrame(rows), output_dir / "metrics.csv")
    write_json(output_dir / "experiment_report.json", report)
    comparison = report["paired_comparison"]["selected"]["metrics"]["score"]
    print(f"完成：综合分变化 {comparison['absolute_delta']:+.6f}，配对 bootstrap 95% 区间 "
          f"[{comparison['ci95_low']:+.6f}, {comparison['ci95_high']:+.6f}]。", flush=True)
    return report


def run_prediction(input_path, output_dir, plan_path, readers_path=None, as_of=None, sheet=None):
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    config, k = AdvancedConfig(**plan["config"]), plan["k"]
    selection = plan["selection"]
    data = load_interactions(input_path, sheet)
    cutoff = timestamp(as_of) if as_of else data[TIME].max() + pd.Timedelta(1, "ns")
    data = data.loc[data[TIME].lt(cutoff)].copy()
    context = HistoryContext(data, cutoff, config)
    readers = load_readers(readers_path) if readers_path else pd.DataFrame({USER: context.users})
    users = readers[USER].tolist()
    weights = selection["weights"]
    rankings, training = fit_channels(context, users, config, output_dir,
                                     frozen_epochs=selection["epochs"], k=k, needed=weights)
    predictions = FusionSpace(rankings, users, config.rrf_constant).predict(weights, k)
    if any(len(items) != k for items in predictions.values()):
        raise ValueError("存在可推荐图书不足 k 的读者，停止导出。")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "result.csv").resolve() == Path(input_path).resolve():
        raise ValueError("输出不能覆盖输入文件。")
    save_csv(recommendations_frame(predictions, k), output_dir / "result.csv")
    write_json(output_dir / "prediction_summary.json", {
        "input": source_info(input_path), "history": frame_summary(data), "cutoff": cutoff.isoformat(),
        "selection": selection, "training": training, "readers": len(users), "k": k,
        "format": "local wide CSV; adapt to official submission template when available",
        "external_reader_attributes_used": False,
    })
    print(f"全历史推荐完成：{len(users)} 位读者，每人 {k} 册。", flush=True)
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("evaluate", "predict"):
        command = commands.add_parser(name)
        command.add_argument("--input", type=Path, default=BASE / "data/2022-2025处理后借阅信息.xlsx")
        command.add_argument("--output-dir", type=Path, default=BASE / ("outputs/advanced" if name == "evaluate" else "outputs/advanced_prediction"))
        command.add_argument("--sheet", default=None)
        if name == "evaluate":
            command.add_argument("--valid-start", default="2024-09-01")
            command.add_argument("--test-start", default="2024-11-01")
            command.add_argument("--test-end", default=None)
            command.add_argument("--k", type=int, default=10)
            command.add_argument("--bootstrap-samples", type=int, default=5000)
            for field, value in asdict(AdvancedConfig()).items():
                option = "--" + field.replace("_", "-")
                if isinstance(value, bool):
                    command.add_argument(option, action="store_true", default=value)
                else:
                    command.add_argument(option, type=type(value), default=value)
        else:
            command.add_argument("--plan", type=Path, default=BASE / "outputs/advanced/locked_plan.json")
            command.add_argument("--readers", type=Path, default=None, help="读者 CSV；当前仅使用读者条码")
            command.add_argument("--as-of", default=None)
    args = parser.parse_args()
    try:
        if args.command == "evaluate":
            config = AdvancedConfig(**{field: getattr(args, field) for field in asdict(AdvancedConfig())})
            run_experiment(args.input, args.output_dir, config, args.valid_start, args.test_start,
                           args.test_end, args.k, args.sheet, args.bootstrap_samples)
        else:
            run_prediction(args.input, args.output_dir, args.plan, args.readers, args.as_of, args.sheet)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f"错误：{error}\n")


if __name__ == "__main__":
    main()
