"""多时间窗口选择 EASE / RP3 / 书目聚合 / 属性先验 / 复借探索，再比较旧模型。"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from advanced import CHANNELS as PREVIOUS_CHANNELS, FusionSpace, fit_channels, paired_bootstrap, quick_metrics
from advanced_models import AdvancedConfig, HistoryContext, set_seed, torch
from baseline import (
    BASE, ITEM, TIME, USER, evaluate_rankings, frame_summary, ground_truth,
    load_interactions, load_readers, recommendations_frame, save_csv, source_info,
    target_diagnostics, timestamp, write_json,
)
from optimized_models import (
    AttributePrior, EASEModel, EntityMatrix, OptimizationConfig, RepeatExploreSpace,
    RepeatGate, RP3Model, no_attribute_baseline,
)


DEVELOPMENT_WINDOWS = [
    ("2023-09-01", "2023-11-01"),
    ("2024-03-01", "2024-05-01"),
    ("2024-09-01", "2024-11-01"),
]
EASE_GRID = (10.0, 100.0, 1000.0)
BETA_GRID = (0.0, 0.5, 1.0)
FAMILIES = ("rp3", "title_rp3", "ease", "title_ease")
CORE_CHANNELS = ("baseline", "previous", "recent", "attributes")
SOURCES = [
    {"title": "Embarrassingly Shallow Autoencoders for Sparse Data", "url": "https://arxiv.org/abs/1905.03375"},
    {"title": "Updatable, Accurate, Diverse, and Scalable Recommendations for Interactive Applications", "url": "https://doi.org/10.1145/2955101"},
    {"title": "RepeatNet: A Repeat Aware Neural Recommendation Machine", "url": "https://arxiv.org/abs/1812.02646"},
    {"title": "Revisiting Alternative Experimental Settings for Evaluating Top-N Item Recommendation Algorithms", "url": "https://arxiv.org/abs/2010.04484"},
]


def parameter_name(family, value):
    return family + "_" + format(float(value), "g").replace(".", "p")


def check_windows(windows, diagnostic_start):
    parsed = [(timestamp(start), timestamp(end)) for start, end in windows]
    if not parsed or any(start >= end for start, end in parsed):
        raise ValueError("开发窗口必须非空，且起点早于排他终点。")
    if any(parsed[index][1] > parsed[index + 1][0] for index in range(len(parsed) - 1)):
        raise ValueError("开发窗口须按时间排序且互不重叠。")
    if any(end > timestamp(diagnostic_start) for _, end in parsed):
        raise ValueError("开发窗口不能进入诊断区间。")
    return parsed


def collapse_channels(rankings, chosen):
    result = {name: rankings[name] for name in CORE_CHANNELS}
    for family in FAMILIES:
        result[family] = rankings[parameter_name(family, chosen[family])]
    return result


def previous_space(rankings, users, constant):
    """保留上一轮评估的 7 列顺序；零权重列无需训练，但不能改变浮点求和布局。"""
    ordered = {name: rankings.get(name, {user: [] for user in users}) for name in PREVIOUS_CHANNELS}
    return FusionSpace(ordered, users, constant)


def build_channels(context, users, prior_plan, config, output_dir, chosen=None,
                   needed=None, external=None, ablations=True):
    """只有开发阶段会训练网格；诊断/预测阶段仅接受已经冻结的参数。"""
    needed = set(needed or (*CORE_CHANNELS, *FAMILIES))
    output_dir = Path(output_dir)
    k = max(prior_plan["k"], context.config.candidate_k)
    channels, training = {}, {}
    old_weights = prior_plan["selection"]["weights"]
    if "previous" in needed or ablations:
        old, old_training = fit_channels(context, users, context.config, output_dir / "previous",
                                         frozen_epochs=prior_plan["selection"]["epochs"],
                                         k=prior_plan["k"], needed=old_weights)
        channels["previous"] = previous_space(old, users, context.config.rrf_constant).predict(old_weights, k)
        training["previous"] = old_training
    else:
        old = {}
    if "baseline" in needed or ablations:
        channels["baseline"] = old.get("baseline") or context.baseline_rankings(users, k)
    if "recent" in needed:
        channels["recent"] = context.recent_rankings(users, k)
    # 预测时为独立提供了画像的新读者预备属性回退；离线不读取未来画像。
    prior = AttributePrior(context, config)
    if "attributes" in needed or external or ablations:
        channels["attributes"] = prior.rank(users, k, external)
        training["attributes"] = prior.summary
    if ablations:
        channels["baseline_no_attributes"] = no_attribute_baseline(context, users, k)
        altered = dict(old)
        if "baseline" in altered:
            altered["baseline"] = channels["baseline_no_attributes"]
        channels["previous_no_attributes"] = previous_space(altered, users, context.config.rrf_constant).predict(old_weights, k)
        channels["attributes_no_stage"] = AttributePrior(context, config, use_stage=False).rank(users, k, external)
    prior.distribution.cache_clear()

    for title_level in (False, True):
        graph_family, ease_family = (("title_rp3", "title_ease") if title_level else ("rp3", "ease"))
        if not ({graph_family, ease_family} & needed):
            continue
        matrix = EntityMatrix(context, title_level=title_level)
        if graph_family in needed:
            values = (chosen[graph_family],) if chosen is not None else BETA_GRID
            for beta in values:
                label = parameter_name(graph_family, beta)
                print(f"构建 {label}，矩阵 {matrix.x.shape}。", flush=True)
                model = RP3Model(matrix, beta=beta, topk=config.graph_topk)
                channels[label] = model.rank(users, k)
                training[label] = model.summary
                del model
        if ease_family in needed:
            values = (chosen[ease_family],) if chosen is not None else EASE_GRID
            for regularization in values:
                label = parameter_name(ease_family, regularization)
                print(f"构建 {label}，使用用户维度精确求解。", flush=True)
                model = EASEModel(matrix, regularization, config.max_dual_users)
                channels[label] = model.rank(users, k)
                training[label] = model.summary
                del model
                gc.collect()
        del matrix
    gate = RepeatGate(context, config)
    training["repeat_gate"] = gate.summary
    write_json(output_dir / "training_summary.json", training)
    return channels, training, gate, prior


def aggregate_metrics(rows, k=10):
    return {name: float(np.mean([row[name] for row in rows]))
            for name in (f"recall@{k}", f"ndcg@{k}", "score")}


def select_parameters(folds, k=10):
    """同一参数在所有开发窗口共同选择，不为每个窗口挑不同参数。"""
    chosen, trials = {}, []
    for family in FAMILIES:
        values = EASE_GRID if "ease" in family else BETA_GRID
        candidates = []
        for value in values:
            label = parameter_name(family, value)
            metrics = [quick_metrics({user: items[:k] for user, items in fold["raw"][label].items()},
                                     fold["truth"], k) for fold in folds]
            row = {"family": family, "value": value, "per_window": metrics, **aggregate_metrics(metrics, k)}
            candidates.append(row)
            trials.append(row)
        best = max(candidates, key=lambda row: (row["score"], row[f"ndcg@{k}"], -row["value"]))
        chosen[family] = best["value"]
    return chosen, trials


def predict_plan(fold, plan, k, no_attributes=False):
    if no_attributes:
        if "no_attribute_rankings" not in fold:
            altered = dict(fold["rankings"])
            altered["baseline"] = fold["raw"]["baseline_no_attributes"]
            altered["previous"] = fold["raw"]["previous_no_attributes"]
            altered["attributes"] = {user: fold["context"].fallback(user, fold["context"].config.candidate_k)
                                     for user in fold["users"]}
            fold["no_attribute_rankings"] = altered
        rankings = fold["no_attribute_rankings"]
    else:
        rankings = fold["rankings"]
    if plan["strategy"] == "rrf":
        name = "no_attribute_rrf" if no_attributes else "rrf"
        if name not in fold:
            ordered = {channel: rankings.get(channel, {user: [] for user in fold["users"]})
                       for channel in (*CORE_CHANNELS, *FAMILIES)}
            fold[name] = FusionSpace(ordered, fold["users"], fold["context"].config.rrf_constant)
        return fold[name].predict(plan["weights"], k)
    if plan["strategy"] != "repeat_explore":
        raise ValueError("未知的推荐策略。")
    name = "no_attribute_repeat_explore" if no_attributes else "repeat_explore"
    if name not in fold:
        gate = (RepeatGate(fold["context"], fold["optimization_config"], use_attributes=False)
                if no_attributes else fold["gate"])
        fold[name] = RepeatExploreSpace(fold["context"], rankings, fold["users"], gate,
                                        fold["context"].config.rrf_constant)
    return fold[name].predict(plan["weights"], k, plan["gate_scale"])


def select_strategy(folds, k=10, frozen_plan=None):
    trials, cache = [], {}

    def evaluate(plan, label):
        key = json.dumps(plan, sort_keys=True)
        if key not in cache:
            per_window = [quick_metrics(predict_plan(fold, plan, k), fold["truth"], k) for fold in folds]
            row = {"name": label, "plan": plan, "per_window": per_window, **aggregate_metrics(per_window, k)}
            cache[key] = row
            trials.append(row)
        return cache[key]

    current = max([evaluate({"strategy": "rrf", "weights": {name: 1.0}}, name)
                   for name in (*CORE_CHANNELS, *FAMILIES)], key=lambda row: row["score"])
    for round_index in range(2):
        candidates = [current]
        for name in (*CORE_CHANNELS, *FAMILIES):
            if name in current["plan"]["weights"]:
                continue
            for weight in (0.25, 0.5, 1.0):
                plan = {"strategy": "rrf", "weights": {**current["plan"]["weights"], name: weight}}
                candidates.append(evaluate(plan, f"forward_{round_index + 1}_{name}_{weight}"))
        best = max(candidates, key=lambda row: (row["score"], row[f"ndcg@{k}"], -len(row["plan"]["weights"])))
        if best["score"] <= current["score"] + 1e-12:
            break
        current = best
    exploration_weights = [
        {"previous": 1.0}, {"title_rp3": 1.0}, {"title_ease": 1.0}, {"attributes": 1.0},
        {"previous": 0.5, "title_rp3": 1.0, "title_ease": 1.0, "attributes": 0.5},
        current["plan"]["weights"],
    ]
    for index, weights in enumerate(exploration_weights):
        for scale in (0.5, 1.0, 2.0):
            evaluate({"strategy": "repeat_explore", "weights": weights, "gate_scale": scale},
                     f"repeat_explore_{index}_{scale}")
    selected = (evaluate(frozen_plan, "reused_frozen_selection") if frozen_plan is not None else
                max(trials, key=lambda row: (row["score"], row[f"ndcg@{k}"],
                                             -len(row["plan"]["weights"]), row["plan"]["strategy"] == "rrf")))
    return selected, trials


def subset_diagnostics(predictions, truth, context, k):
    """按真值成因分解命中，使用所有方法相同的推荐列表，不重排、不补候选。"""
    parts = {name: [] for name in ("repeat", "known_new_to_user", "unknown_catalog")}
    for user, relevant in truth.items():
        index = context.user_index.get(user)
        seen = {context.items[item] for item in context.seen[index]} if index is not None else set()
        subsets = {"repeat": relevant & seen,
                   "known_new_to_user": (relevant & context.baseline.items) - seen,
                   "unknown_catalog": relevant - context.baseline.items}
        for name, subset in subsets.items():
            if subset:
                parts[name].append((len(subset), len(set(predictions[user][:k]) & subset)))
    return {name: {"users_with_truth": len(rows), "truth_pairs": sum(row[0] for row in rows),
                   "hits": sum(row[1] for row in rows),
                   "pair_recall": sum(row[1] for row in rows) / sum(row[0] for row in rows) if rows else None,
                   "user_macro_recall": float(np.mean([hits / size for size, hits in rows])) if rows else None}
            for name, rows in parts.items()}


def save_fold(fold, selected_plan, output_dir, k):
    output_dir = Path(output_dir)
    predictions = {name: {user: items[:k] for user, items in ranking.items()}
                   for name, ranking in fold["rankings"].items()}
    predictions["selected"] = predict_plan(fold, selected_plan, k)
    predictions["baseline_no_attributes"] = {user: items[:k] for user, items in fold["raw"]["baseline_no_attributes"].items()}
    predictions["previous_no_attributes"] = {user: items[:k] for user, items in fold["raw"]["previous_no_attributes"].items()}
    predictions["attributes_no_stage"] = {user: items[:k] for user, items in fold["raw"]["attributes_no_stage"].items()}
    predictions["selected_no_user_attributes"] = predict_plan(fold, selected_plan, k, no_attributes=True)
    if len(selected_plan["weights"]) > 1:
        for name in selected_plan["weights"]:
            reduced = {**selected_plan, "weights": {key: value for key, value in selected_plan["weights"].items() if key != name}}
            predictions["without_" + name] = predict_plan(fold, reduced, k)
    metrics, per_user, detail, subsets = {}, {}, [], {}
    for name, ranking in predictions.items():
        if any(len(items) != k or len(set(items)) != k or not set(items) <= fold["context"].baseline.items
               for items in ranking.values()):
            raise ValueError(f"{name} 的推荐列表长度、去重或历史候选范围不合法。")
        scores, rows = evaluate_rankings(ranking, fold["truth"], fold["context"].baseline, k)
        metrics[name], per_user[name] = scores, rows
        subsets[name] = subset_diagnostics(ranking, fold["truth"], fold["context"], k)
        frame = rows.copy()
        frame.insert(0, "method", name)
        detail.append(frame)
        save_csv(recommendations_frame(ranking, k), output_dir / f"{name}.csv")
        if name in ("baseline", "previous", "recent", "selected"):
            all_scores = scores["all"]
            print(f"{fold['name']:16s} {name:10s} Recall={all_scores[f'recall@{k}']:.6f} "
                  f"NDCG={all_scores[f'ndcg@{k}']:.6f} score={all_scores['score']:.6f}", flush=True)
    save_csv(pd.concat(detail, ignore_index=True), output_dir / "per_user.csv")
    return metrics, per_user, subsets


def check_previous(report, old_report_path):
    old_report_path = Path(old_report_path)
    if not old_report_path.is_file():
        return {"available": False}
    old = json.loads(old_report_path.read_text(encoding="utf-8"))
    if old["input"]["sha256"] != report["input"]["sha256"] or old["k"] != report["k"]:
        return {"available": True, "same_input_and_k": False}
    deltas = {}
    for new_name, old_name in (("baseline", "baseline"), ("previous", "selected"), ("recent", "recent")):
        previous = old["stages"]["test"]["methods"][old_name]["all"]
        current = report["diagnostic"]["methods"][new_name]["all"]
        deltas[new_name] = {metric: current[metric] - previous[metric]
                            for metric in (f"recall@{report['k']}", f"ndcg@{report['k']}", "score")}
    same = report["diagnostic"]["start"] == old["boundaries"]["test_start"] and report["diagnostic"]["end_exclusive"] == old["boundaries"]["test_end_exclusive"]
    reproduced = all(abs(delta) < 1e-12 for row in deltas.values() for delta in row.values())
    if same and not reproduced:
        raise ValueError("历史模型未能精确复现，请先修复可比性再解释结果。")
    return {"available": True, "same_input_and_k": True, "same_boundaries": same,
            "metric_deltas": deltas, "exactly_reproduced": reproduced if same else None}


def run_evaluation(input_path, output_dir, prior_plan_path, windows=None, diagnostic_start="2024-11-01",
                   diagnostic_end=None, config=None, sheet=None, bootstrap_samples=5000, reuse_selection=None):
    started = perf_counter()
    config = config or OptimizationConfig()
    windows = check_windows(windows or DEVELOPMENT_WINDOWS, diagnostic_start)
    prior_plan = json.loads(Path(prior_plan_path).read_text(encoding="utf-8"))
    advanced_config = AdvancedConfig(**prior_plan["config"])
    k = prior_plan["k"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(advanced_config.seed, advanced_config.threads)
    data = load_interactions(input_path, sheet)
    frozen = json.loads(Path(reuse_selection).read_text(encoding="utf-8")) if reuse_selection else None
    if frozen and (frozen["input"]["sha256"] != source_info(input_path)["sha256"]
                   or frozen["config"] != asdict(config) or frozen["previous_plan"] != prior_plan):
        raise ValueError("复用的冻结计划必须具有相同输入、配置和上一轮参照。")
    report = {
        "schema_version": 1, "input": source_info(input_path), "k": k, "config": asdict(config),
        "previous_config": asdict(advanced_config), "previous_selection": prior_plan["selection"],
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "torch": torch.__version__,
                        "logical_cpu_count": os.cpu_count(),
                        "thread_environment": {key: os.environ.get(key) for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}},
        "sources": SOURCES,
        "protocol": {"development_windows": [[start.isoformat(), end.isoformat()] for start, end in windows],
                     "selection_objective": "equal-weight mean of per-window user-macro composite score",
                     "ease_grid": EASE_GRID, "rp3_beta_grid": BETA_GRID, "rrf_forward_rounds": 2,
                     "rrf_added_weights": [0.25, 0.5, 1.0], "repeat_gate_scales": [0.5, 1.0, 2.0],
                     "diagnostic_start": timestamp(diagnostic_start).isoformat(),
                     "diagnostic_end_exclusive": timestamp(diagnostic_end).isoformat() if diagnostic_end else None,
                     "diagnostic_is_reused_previous_test": True,
                     "caution": "The old test has already informed research direction. It is a reused diagnostic, not a new blind holdout. Rolling development results are also selection results.",
                     "truth": "historical full barcode catalog; all future-active readers; repeats and unknown-catalog truth retained",
                     "external_reader_profiles_used": False},
    }
    report["protocol"]["reused_frozen_selection"] = bool(frozen)
    # 在任何开发标签打分前记录来源、搜索空间与协议，防止事后隐藏失败路线。
    write_json(output_dir / "experiment_protocol.json", report)
    folds = []
    for start, end in windows:
        history = data.loc[data[TIME].lt(start)].copy()
        target = data.loc[data[TIME].ge(start) & data[TIME].lt(end)].copy()
        if history.empty or target.empty:
            raise ValueError("开发窗口没有历史或真值，请检查日期。")
        name = start.strftime("%Y-%m-%d")
        print(f"开发窗口 {name} -> {end.date()}：历史 {len(history)}，目标 {len(target)}。", flush=True)
        context = HistoryContext(history, start, advanced_config)
        truth = ground_truth(target)
        users = sorted(truth)
        raw, training, gate, _ = build_channels(context, users, prior_plan, config, output_dir / name)
        folds.append({"name": name, "context": context, "truth": truth, "users": users, "raw": raw,
                      "gate": gate, "training": training, "optimization_config": config,
                      "history_summary": frame_summary(history), "target_summary": frame_summary(target),
                      "diagnostics": target_diagnostics(history, target, k)})
    chosen, parameter_trials = select_parameters(folds, k)
    if frozen:
        chosen = frozen["parameters"]
    for fold in folds:
        fold["rankings"] = collapse_channels(fold["raw"], chosen)
    selected, trials = select_strategy(folds, k, frozen["recommendation"] if frozen else None)
    plan = {"schema_version": 1, "input": report["input"], "k": k, "config": asdict(config),
            "previous_plan": prior_plan, "parameters": chosen, "recommendation": selected["plan"],
            "selection": {"based_on": f"{len(windows)} rolling development windows only", "score": selected["score"],
                          "per_window": selected["per_window"], "strategy_trials": len(trials)},
            "reader_profile_fallback": "new readers with independently provided known profiles -> smoothed attribute prior"}
    # 诊断期重训、推断与标签打分前冻结。之后不能用诊断表现改变分支或参数。
    write_json(output_dir / "locked_plan.json", plan)
    write_json(output_dir / "parameter_trials.json", parameter_trials)
    write_json(output_dir / "strategy_trials.json", trials)
    save_csv(pd.DataFrame([{**{key: value for key, value in row.items() if key not in ("plan", "per_window")},
                           "plan": json.dumps(row["plan"], ensure_ascii=False, sort_keys=True)} for row in trials]),
             output_dir / "strategy_trials.csv")
    print(f"多窗口选择已冻结：{selected['plan']}；参数 {chosen}。", flush=True)
    report["parameters"], report["selection"] = chosen, plan["selection"]
    report["recommendation"] = selected["plan"]
    report["development"] = {}
    for fold in folds:
        metrics, _, subsets = save_fold(fold, selected["plan"], output_dir / fold["name"], k)
        report["development"][fold["name"]] = {"history": fold["history_summary"], "target": fold["target_summary"],
                                                "diagnostics": fold["diagnostics"], "methods": metrics, "subsets": subsets}
    methods = list(next(iter(report["development"].values()))["methods"])
    report["development_mean"] = {name: aggregate_metrics([details["methods"][name]["all"]
                                                          for details in report["development"].values()], k) for name in methods}
    del folds
    gc.collect()
    cutoff = timestamp(diagnostic_start)
    history = data.loc[data[TIME].lt(cutoff)].copy()
    target = data.loc[data[TIME].ge(cutoff)].copy()
    if diagnostic_end:
        target = target.loc[target[TIME].lt(timestamp(diagnostic_end))].copy()
    if target.empty:
        raise ValueError("诊断区间没有真值。")
    context = HistoryContext(history, cutoff, advanced_config)
    truth = ground_truth(target)
    users = sorted(truth)
    raw, training, gate, _ = build_channels(context, users, prior_plan, config, output_dir / "diagnostic", chosen=chosen)
    diagnostic = {"name": "diagnostic", "context": context, "truth": truth, "users": users, "raw": raw,
                  "rankings": collapse_channels(raw, chosen), "gate": gate, "optimization_config": config}
    metrics, per_user, subsets = save_fold(diagnostic, selected["plan"], output_dir / "diagnostic", k)
    report["diagnostic"] = {"start": cutoff.isoformat(),
                            "end_exclusive": timestamp(diagnostic_end).isoformat() if diagnostic_end else None,
                            "history": frame_summary(history), "target": frame_summary(target),
                            "training": training, "diagnostics": target_diagnostics(history, target, k),
                            "methods": metrics, "subsets": subsets}
    comparisons = {"selected_vs_" + name: paired_bootstrap(per_user[name], per_user["selected"], k,
                                                           bootstrap_samples, advanced_config.seed)
                   for name in ("baseline", "previous", "recent", "selected_no_user_attributes")}
    comparisons["baseline_attributes_vs_none"] = paired_bootstrap(per_user["baseline_no_attributes"],
                                                                  per_user["baseline"], k, bootstrap_samples, advanced_config.seed)
    comparisons["attribute_stage_vs_no_stage"] = paired_bootstrap(per_user["attributes_no_stage"],
                                                                 per_user["attributes"], k, bootstrap_samples, advanced_config.seed)
    for name in metrics:
        if name.startswith("without_"):
            comparisons["selected_vs_" + name] = paired_bootstrap(per_user[name], per_user["selected"], k,
                                                                  bootstrap_samples, advanced_config.seed)
    report["paired_comparisons"] = comparisons
    report["previous_reproduction"] = check_previous(report, BASE / "outputs/advanced/experiment_report.json")
    rows = []
    for stage, details in [*report["development"].items(), ("diagnostic", report["diagnostic"])]:
        rows.extend({"window": stage, "method": method, "segment": segment, **values}
                    for method, segments in details["methods"].items() for segment, values in segments.items())
    save_csv(pd.DataFrame(rows), output_dir / "metrics.csv")
    report["elapsed_seconds"] = round(perf_counter() - started, 3)
    write_json(output_dir / "experiment_report.json", report)
    print(f"完成，耗时 {report['elapsed_seconds']} 秒；诊断区间仅比较，不重新选择。", flush=True)
    return report


def run_prediction(input_path, output_dir, plan_path, readers_path=None, as_of=None, sheet=None):
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    prior_plan = plan["previous_plan"]
    advanced_config = AdvancedConfig(**prior_plan["config"])
    config = OptimizationConfig(**plan["config"])
    set_seed(advanced_config.seed, advanced_config.threads)
    data = load_interactions(input_path, sheet)
    cutoff = timestamp(as_of) if as_of else data[TIME].max() + pd.Timedelta(1, "ns")
    history = data.loc[data[TIME].lt(cutoff)].copy()
    context = HistoryContext(history, cutoff, advanced_config)
    readers = load_readers(readers_path) if readers_path else pd.DataFrame({USER: context.users})
    users = readers[USER].tolist()
    external = {row[USER]: row for row in readers.to_dict("records")} if readers_path else None
    if readers_path and "画像时间" in readers:
        known_at = pd.to_datetime(readers["画像时间"], errors="raise")
        if known_at.isna().any() or known_at.ge(cutoff).any():
            raise ValueError("独立画像时间必须非空且严格早于预测时点。")
    needed = set(plan["recommendation"]["weights"])
    raw, training, gate, prior = build_channels(context, users, prior_plan, config, output_dir,
                                                chosen=plan["parameters"], needed=needed, external=external,
                                                ablations=False)
    rankings = {name: raw[name] if name in CORE_CHANNELS else raw[parameter_name(name, plan["parameters"][name])]
                for name in needed}
    fold = {"context": context, "users": users, "rankings": rankings, "gate": gate, "optimization_config": config}
    predictions = predict_plan(fold, plan["recommendation"], plan["k"])
    profile_readers = []
    if external:
        for user in users:
            if user not in context.user_index and prior.keys(prior.profile(user, external), cutoff):
                predictions[user] = prior.rank([user], plan["k"], external)[user]
                profile_readers.append(user)
    if any(len(items) != plan["k"] or len(set(items)) != plan["k"] or not set(items) <= context.baseline.items
           for items in predictions.values()):
        raise ValueError("推荐数量、去重或历史候选范围不合法。")
    output_dir = Path(output_dir)
    if (output_dir / "result.csv").resolve() == Path(input_path).resolve():
        raise ValueError("输出不能覆盖输入。")
    save_csv(recommendations_frame(predictions, plan["k"]), output_dir / "result.csv")
    write_json(output_dir / "prediction_summary.json", {
        "input": source_info(input_path), "history": frame_summary(history), "cutoff": cutoff.isoformat(),
        "readers": len(users), "k": plan["k"], "parameters": plan["parameters"],
        "recommendation": plan["recommendation"], "training": training,
        "external_profiles_provided": bool(external), "cold_readers_using_profile_fallback": len(profile_readers),
        "note": "Profiles must be independently known before prediction. No future interactions or catalog used.",
    })
    prior.distribution.cache_clear()
    print(f"全历史推荐完成：{len(users)} 位读者，每人 {plan['k']} 册。", flush=True)
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("evaluate", "predict"):
        command = commands.add_parser(name)
        command.add_argument("--input", type=Path, default=BASE / "data/2022-2025处理后借阅信息.xlsx")
        command.add_argument("--output-dir", type=Path, default=BASE / ("outputs/optimized" if name == "evaluate" else "outputs/optimized_prediction"))
        command.add_argument("--sheet", default=None)
        if name == "evaluate":
            command.add_argument("--previous-plan", type=Path, default=BASE / "outputs/advanced/locked_plan.json")
            command.add_argument("--dev-window", nargs=2, action="append", help="可重复给出 START END，均为排他终点")
            command.add_argument("--diagnostic-start", default="2024-11-01")
            command.add_argument("--diagnostic-end", default=None)
            command.add_argument("--bootstrap-samples", type=int, default=5000)
            command.add_argument("--reuse-selection", type=Path, default=None, help="重算已冻结方案，不根据重算结果更换参数或组合")
        else:
            command.add_argument("--plan", type=Path, default=BASE / "outputs/optimized/locked_plan.json")
            command.add_argument("--readers", type=Path, default=None, help="读者条码，可附专业/系别/年级；可选画像时间必须早于预测")
            command.add_argument("--as-of", default=None)
    args = parser.parse_args()
    try:
        if args.command == "evaluate":
            run_evaluation(args.input, args.output_dir, args.previous_plan, args.dev_window,
                           args.diagnostic_start, args.diagnostic_end, sheet=args.sheet,
                           bootstrap_samples=args.bootstrap_samples, reuse_selection=args.reuse_selection)
        else:
            run_prediction(args.input, args.output_dir, args.plan, args.readers, args.as_of, args.sheet)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f"错误：{error}\n")


if __name__ == "__main__":
    main()
