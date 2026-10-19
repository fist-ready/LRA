"""按时间验证的图书 Top-N 基线：热门度、ItemCF 和属性融合。"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import platform
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from time import perf_counter

import pandas as pd

from preprocess import clean_frame, parse_dates, read_source, save_csv

BASE = Path(__file__).resolve().parent
USER, ITEM, TIME = "读者条码", "条形码", "处理时间"
ATTRIBUTES = {"专业": 0.6, "系别": 0.3, "年级": 0.1}
METHODS = ("popular", "itemcf", "hybrid")
MISSING = {"", "未知", "不详", "null", "none", "nan", "n/a", "<na>"}


def attribute_value(value):
    if pd.isna(value):
        return ""
    value = str(value).strip()
    return "" if value.casefold() in MISSING else value


def timestamp(value):
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tzinfo is not None:
        raise ValueError("时间必须是有效的、不带时区的本地时间。")
    return result


def prepare_events(data):
    """校验建模输入，不悄悄删除坏行，也不将条码转成数值。"""
    missing = {USER, ITEM, TIME} - set(data.columns)
    if missing:
        raise ValueError(f"缺少建模字段：{sorted(missing)}")
    data = data.copy()
    for column in (USER, ITEM):
        data[column] = data[column].astype("string").str.strip()
        if data[column].isna().any() or data[column].eq("").any():
            raise ValueError(f"{column}存在空值，请先完成数据清洗。")
    data[TIME] = parse_dates(data[TIME])
    if data[TIME].isna().any():
        raise ValueError("处理时间存在空值或解析失败，请先完成数据清洗。")
    if not pd.api.types.is_datetime64_dtype(data[TIME]):
        raise ValueError("处理时间必须是不带时区的本地时间。")
    if "操作类型" in data and not data["操作类型"].eq("借书").all():
        raise ValueError("输入包含非借书操作，请先使用 preprocess.py 清洗。")
    return data.sort_values([TIME, USER, ITEM], kind="stable").reset_index(drop=True)


def load_interactions(path, sheet=None):
    """原始 Excel 复用现有清洗模块；清洗 CSV 的所有字段先按文本读取。"""
    path = Path(path)
    if path.suffix.lower() == ".xlsx":
        data = clean_frame(read_source(path, sheet))["cleaned"]
    elif path.suffix.lower() == ".csv":
        if sheet is not None:
            raise ValueError("CSV 输入不支持 --sheet。")
        data = pd.read_csv(path, dtype="string", keep_default_na=False, encoding="utf-8-sig")
    else:
        raise ValueError("输入文件必须是 .xlsx 或清洗后的 .csv。")
    return prepare_events(data)


def temporal_split(data, valid_start="2024-09-01", test_start="2024-11-01", test_end=None):
    """所有读者共用时间边界，区间左闭右开。"""
    valid_start, test_start = timestamp(valid_start), timestamp(test_start)
    test_end = timestamp(test_end) if test_end is not None else None
    if valid_start >= test_start or (test_end is not None and test_start >= test_end):
        raise ValueError("必须满足 valid_start < test_start < test_end（若指定）。")
    train = data.loc[data[TIME] < valid_start].copy()
    valid = data.loc[data[TIME].ge(valid_start) & data[TIME].lt(test_start)].copy()
    mask = data[TIME].ge(test_start)
    if test_end is not None:
        mask &= data[TIME].lt(test_end)
    test = data.loc[mask].copy()
    for name, frame in (("train", train), ("validation", valid), ("test", test)):
        if frame.empty:
            raise ValueError(f"{name}为空，请根据实际时间范围调整切分边界。")
    return train, valid, test


@dataclass(frozen=True)
class ModelConfig:
    neighbors: int = 50
    max_history: int = 100
    half_life_days: float = 180.0
    shrinkage: float = 10.0
    group_min_users: int = 5
    exclude_seen: bool = False

    def __post_init__(self):
        if self.neighbors < 1 or self.max_history < 1 or self.group_min_users < 1:
            raise ValueError("neighbors、max_history 和 group_min_users 必须为正整数。")
        if not math.isfinite(self.half_life_days) or self.half_life_days <= 0:
            raise ValueError("half_life_days 必须为有限正数。")
        if not math.isfinite(self.shrinkage) or self.shrinkage < 0:
            raise ValueError("shrinkage 必须为有限非负数。")


def unit_scale(scores):
    maximum = max(scores.values(), default=0.0)
    return {item: value / maximum for item, value in scores.items()} if maximum > 0 else {}


class ItemCFBaseline:
    """只从传入的历史构建候选库、画像、热门度及物品共现图，无额外依赖。"""

    def __init__(self, config=None):
        self.config = config or ModelConfig()
        self.fitted = False

    def fit(self, train, as_of=None):
        train = prepare_events(train)
        if train.empty:
            raise ValueError("不能用空训练集拟合模型。")
        self.as_of = timestamp(as_of) if as_of is not None else train[TIME].max() + pd.Timedelta(1, "ns")
        if train[TIME].ge(self.as_of).any():
            raise ValueError("训练记录必须严格早于 as_of，禁止把预测区间的记录传入 fit。")
        self.fitted = False
        self.items = set(train[ITEM])
        self.profiles = defaultdict(dict)
        for field in ATTRIBUTES:
            if field not in train:
                continue
            values = train[field].map(attribute_value)
            known = train.loc[values.ne(""), [USER]].assign(value=values[values.ne("")])
            for user, value in known.drop_duplicates(USER, keep="last").itertuples(index=False, name=None):
                self.profiles[user][field] = value

        # 同一读者对同一册书仅贡献一次，保留最近一次借阅时间。
        pairs = train.drop_duplicates([USER, ITEM], keep="last").copy()
        pairs["_weight"] = 2.0 ** (-(self.as_of - pairs[TIME]).dt.total_seconds() / (86400 * self.config.half_life_days))
        self.popularity = unit_scale(pairs.groupby(ITEM, sort=True)["_weight"].sum().to_dict())
        # 包括极端时间衰减至 0 的条码，使候选库与训练图书集合保持一致。
        self.popular_items = sorted(self.items, key=lambda item: (-self.popularity.get(item, 0.0), item))
        self.seen, self.history = {}, {}
        self.group_scores = defaultdict(Counter)
        group_users = defaultdict(set)
        frequencies = Counter()
        for user, rows in pairs.groupby(USER, sort=True):
            self.seen[user] = set(rows[ITEM])
            recent = rows.sort_values([TIME, ITEM], ascending=[False, True], kind="stable").head(self.config.max_history)
            self.history[user] = dict(zip(recent[ITEM], zip(recent[TIME], recent["_weight"])))
            frequencies.update(self.history[user].keys())
            for field, value in self.profiles[user].items():
                key = (field, value)
                group_users[key].add(user)
                for item, weight in rows[[ITEM, "_weight"]].itertuples(index=False, name=None):
                    self.group_scores[key][item] += weight
        self.group_scores = {
            key: unit_scale(scores) for key, scores in self.group_scores.items()
            if len(group_users[key]) >= self.config.group_min_users
        }

        # 无须构造 20847 × 20847 的稠密矩阵。每个无序图书对存权重和共同读者数。
        cooccurrence = {}
        for history in self.history.values():
            penalty = 1.0 / math.log1p(len(history))
            for left, right in combinations(sorted(history), 2):
                gap_days = abs((history[left][0] - history[right][0]).total_seconds()) / 86400
                contribution = penalty * 2.0 ** (-gap_days / self.config.half_life_days)
                pair = cooccurrence.setdefault((left, right), [0.0, 0])
                pair[0] += contribution
                pair[1] += 1
        graph = defaultdict(dict)
        for (left, right), (weight, support) in cooccurrence.items():
            similarity = weight / math.sqrt(frequencies[left] * frequencies[right])
            similarity *= support / (support + self.config.shrinkage)
            graph[left][right] = similarity
            graph[right][left] = similarity
        self.similarities = {
            item: heapq.nsmallest(self.config.neighbors, neighbors.items(), key=lambda pair: (-pair[1], pair[0]))
            for item, neighbors in graph.items()
        }
        self.training_summary = {
            "events": len(train), "users": len(self.seen), "items": len(self.items),
            "unique_user_item_pairs": len(pairs), "cooccurrence_pairs": len(cooccurrence),
            "as_of": self.as_of.isoformat(), "last_training_event": train[TIME].max().isoformat(),
        }
        self.fitted = True
        return self

    def recommend(self, user, k=10, method="hybrid", profile=None):
        """返回不重复的条码列表。profile 仅接收预测时独立提供的读者属性。"""
        if not self.fitted:
            raise RuntimeError("请先 fit。")
        if k < 1:
            raise ValueError("k 必须为正整数。")
        if method not in METHODS:
            raise ValueError(f"未知基线：{method}")
        user = str(user)
        blocked = self.seen.get(user, set()) if self.config.exclude_seen else set()
        history = self.history.get(user, {})
        scores = defaultdict(float)
        if method != "popular":
            cf = defaultdict(float)
            for item, (_, weight) in history.items():
                for neighbor, similarity in self.similarities.get(item, []):
                    cf[neighbor] += weight * similarity
            cf_weight = 0.95 if method == "itemcf" else 0.60
            for item, value in unit_scale(cf).items():
                scores[item] += cf_weight * value
        if method == "hybrid":
            reader_profile = dict(self.profiles.get(user, {}))
            if profile is not None:
                reader_profile.update({field: attribute_value(value) for field, value in profile.items() if field in ATTRIBUTES})
            group = defaultdict(float)
            available_weight = 0.0
            for field, field_weight in ATTRIBUTES.items():
                group_vector = self.group_scores.get((field, reader_profile.get(field, "")), {})
                if group_vector:
                    available_weight += field_weight
                    for item, value in group_vector.items():
                        group[item] += field_weight * value
            if available_weight:
                for item, value in group.items():
                    scores[item] += 0.20 * value / available_weight
            repeat = unit_scale({item: values[1] for item, values in history.items()})
            for item, value in repeat.items():
                scores[item] += 0.15 * value

        # 非个性化候选只有热门度分。加入前 k 个未屏蔽的热门项，足以精确求融合 Top-k。
        fallback = []
        for item in self.popular_items:
            if item not in blocked:
                fallback.append(item)
                if len(fallback) == k:
                    break
        candidates = (set(scores) | set(fallback)) - blocked
        pop_weight = 1.0 if method == "popular" else 0.05
        return heapq.nsmallest(
            k, candidates,
            key=lambda item: (-(scores.get(item, 0.0) + pop_weight * self.popularity.get(item, 0.0)), item),
        )


def ground_truth(events):
    return events.groupby(USER, sort=True)[ITEM].agg(set).to_dict()


def ranking_metrics(recommendations, relevant, k=10):
    """二值相关性；Recall 分母为全部真值册数；IDCG 截断到 min(k, 真值册数)。"""
    if k < 1:
        raise ValueError("k 必须为正整数。")
    relevant = set(relevant)
    if not relevant:
        raise ValueError("无真值用户不参与离线指标平均。")
    top = list(recommendations)[:k]
    if len(set(top)) != len(top):
        raise ValueError("推荐列表含重复条码。")
    hits = sum(item in relevant for item in top)
    dcg = sum(1.0 / math.log2(rank + 2) for rank, item in enumerate(top) if item in relevant)
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(min(k, len(relevant))))
    recall, ndcg = hits / len(relevant), dcg / idcg
    return {"hits": hits, f"recall@{k}": recall, f"ndcg@{k}": ndcg, "score": 0.6 * recall + 0.4 * ndcg}


def evaluate_rankings(recommendations, truth, model, k=10):
    rows = []
    for user in sorted(truth):
        rows.append({
            USER: user, "segment": "warm" if user in model.seen else "cold",
            "truth_items": len(truth[user]), "known_truth_items": len(truth[user] & model.items),
            **ranking_metrics(recommendations.get(user, []), truth[user], k),
        })
    per_user = pd.DataFrame(rows)
    if per_user.empty:
        raise ValueError("评估区间没有有效真值。")
    metrics = {}
    for segment in ("all", "warm", "cold"):
        part = per_user if segment == "all" else per_user.loc[per_user["segment"].eq(segment)]
        metrics[segment] = {
            "users": len(part),
            **{name: float(part[name].mean()) if len(part) else None for name in (f"recall@{k}", f"ndcg@{k}", "score")},
        }
    return metrics, per_user


def frame_summary(data):
    return {
        "events": len(data), "users": data[USER].nunique(), "items": data[ITEM].nunique(),
        "unique_user_item_pairs": len(data.drop_duplicates([USER, ITEM])),
        "first_event": data[TIME].min().isoformat() if len(data) else None,
        "last_event": data[TIME].max().isoformat() if len(data) else None,
    }


def target_diagnostics(history, target, k, exclude_seen=False):
    """未来新书仍留在真值中；额外报告当前候选库能达到的理想上限。"""
    truth = ground_truth(target)
    catalog = set(history[ITEM])
    seen = ground_truth(history)
    pairs = target[[USER, ITEM]].drop_duplicates()
    known_pairs = pairs[ITEM].isin(catalog)
    repeated = sum(item in seen.get(user, set()) for user, item in pairs.itertuples(index=False, name=None))
    oracle = []
    for user, relevant in truth.items():
        reachable = relevant & catalog
        if exclude_seen:
            reachable -= seen.get(user, set())
        oracle.append(ranking_metrics(sorted(reachable)[:k], relevant, k))
    cold_users = sum(user not in seen for user in truth)
    return {
        "target_users": len(truth), "warm_users": len(truth) - cold_users, "cold_users": cold_users,
        "truth_pairs": len(pairs), "known_item_truth_pairs": int(known_pairs.sum()),
        "new_item_truth_pairs": int((~known_pairs).sum()), "repeated_truth_pairs": repeated,
        "known_item_pair_rate": float(known_pairs.mean()),
        f"oracle_recall@{k}": sum(row[f"recall@{k}"] for row in oracle) / len(oracle),
        f"oracle_ndcg@{k}": sum(row[f"ndcg@{k}"] for row in oracle) / len(oracle),
    }


def recommendations_frame(recommendations, k=10):
    columns = [USER] + [f"推荐{i}" for i in range(1, k + 1)]
    return pd.DataFrame(
        [[user] + list(items)[:k] + [""] * max(0, k - len(items)) for user, items in recommendations.items()],
        columns=columns,
    )


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def source_info(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"file": path.name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def run_evaluation(input_path, output_dir, valid_start="2024-09-01", test_start="2024-11-01",
                   test_end=None, config=None, k=10, sheet=None):
    started = perf_counter()
    if k < 1:
        raise ValueError("k 必须为正整数。")
    config = config or ModelConfig()
    data = load_interactions(input_path, sheet)
    train, valid, test = temporal_split(data, valid_start, test_start, test_end)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in (("train", train), ("validation", valid), ("test", test)):
        save_csv(frame, output_dir / "splits" / f"{name}.csv")
    print(f"时间切分：train={len(train)}，validation={len(valid)}，test={len(test)}", flush=True)
    report = {
        "schema_version": 1, "input": source_info(input_path), "k": k, "config": asdict(config),
        "environment": {"python": platform.python_version(), "pandas": pd.__version__},
        "boundaries": {"valid_start": timestamp(valid_start).isoformat(), "test_start": timestamp(test_start).isoformat(),
                       "test_end_exclusive": timestamp(test_end).isoformat() if test_end else None},
        "data": {"all": frame_summary(data), "train": frame_summary(train), "validation": frame_summary(valid), "test": frame_summary(test),
                 "unused_after_test_end_events": int(data[TIME].ge(timestamp(test_end)).sum()) if test_end else 0},
        "metric_protocol": "macro over all future-active users; binary barcode relevance; full historical catalog; repeated targets retained",
        "stages": {},
    }
    metric_rows = []
    selected = None
    for stage, history, target, cutoff in (
        ("validation", train, valid, valid_start),
        ("test", pd.concat([train, valid], ignore_index=True), test, test_start),
    ):
        # test 阶段只重新拟合历史，参数及方法选择已在 validation 阶段冻结。
        print(f"拟合 {stage} 模型，历史记录 {len(history)} 条……", flush=True)
        model = ItemCFBaseline(config).fit(history, as_of=cutoff)
        truth = ground_truth(target)
        result = {
            "training": model.training_summary,
            "diagnostics": target_diagnostics(history, target, k, config.exclude_seen), "methods": {},
        }
        per_user_frames = []
        for method in METHODS:
            predictions = {user: model.recommend(user, k=k, method=method) for user in truth}
            metrics, per_user = evaluate_rankings(predictions, truth, model, k)
            result["methods"][method] = metrics
            per_user.insert(0, "method", method)
            per_user_frames.append(per_user)
            save_csv(recommendations_frame(predictions, k), output_dir / f"{stage}_{method}.csv")
            for segment, values in metrics.items():
                metric_rows.append({"split": stage, "method": method, "segment": segment, **values})
            values = metrics["all"]
            print(f"{stage:10s} {method:7s} Recall@{k}={values[f'recall@{k}']:.6f} "
                  f"NDCG@{k}={values[f'ndcg@{k}']:.6f} score={values['score']:.6f}", flush=True)
        save_csv(pd.concat(per_user_frames, ignore_index=True), output_dir / f"{stage}_per_user.csv")
        report["stages"][stage] = result
        if stage == "validation":
            selected = max(METHODS, key=lambda method: (
                result["methods"][method]["all"]["score"], result["methods"][method]["all"][f"ndcg@{k}"],
                -METHODS.index(method),
            ))
            report["selection"] = {"method": selected, "based_on": "validation only", "config": asdict(config)}
            print(f"验证集选定方法：{selected}。随后仅在本地测试集评估。", flush=True)
    report["elapsed_seconds"] = round(perf_counter() - started, 3)
    save_csv(pd.DataFrame(metric_rows), output_dir / "metrics.csv")
    write_json(output_dir / "evaluation_report.json", report)
    print(f"评估完成。选定方法：{selected}；输出目录：{output_dir.resolve()}", flush=True)
    return report


def load_readers(path):
    readers = pd.read_csv(path, dtype="string", keep_default_na=False, encoding="utf-8-sig")
    if USER not in readers:
        raise ValueError(f"读者清单必须包含“{USER}”列。")
    readers[USER] = readers[USER].str.strip()
    if readers.empty or readers[USER].eq("").any() or readers[USER].duplicated().any():
        raise ValueError("读者清单不能为空，读者条码必须非空且唯一。")
    return readers


def run_prediction(input_path, output_dir, method="auto", selection_report=None, readers_path=None,
                   config=None, k=10, sheet=None, as_of=None):
    if k < 1:
        raise ValueError("k 必须为正整数。")
    if method == "auto":
        selection_report = Path(selection_report) if selection_report else BASE / "outputs/baseline/evaluation_report.json"
        with selection_report.open(encoding="utf-8") as stream:
            selection = json.load(stream)
        if selection["k"] != k:
            raise ValueError("k 与验证报告不一致，请重新评估或显式选择 --method。")
        selected_config = ModelConfig(**selection["selection"]["config"])
        if config is not None and config != selected_config:
            raise ValueError("auto 模式使用验证报告参数，不能覆盖；请重新评估或显式选择 --method。")
        config, method = selected_config, selection["selection"]["method"]
    config = config or ModelConfig()
    if method not in METHODS:
        raise ValueError(f"未知基线：{method}")
    data = load_interactions(input_path, sheet)
    if as_of is not None:
        data = data.loc[data[TIME].lt(timestamp(as_of))].copy()
    model = ItemCFBaseline(config).fit(data, as_of=as_of)
    readers = load_readers(readers_path) if readers_path else pd.DataFrame({USER: sorted(model.seen)})
    predictions = {}
    for row in readers.to_dict("records"):
        user = row[USER]
        predictions[user] = model.recommend(user, k=k, method=method, profile=row)
        if len(predictions[user]) != k:
            raise ValueError(f"读者 {user} 的可推荐图书不足 {k} 册，请检查候选库或 --exclude-seen。")
    output_dir = Path(output_dir)
    if (output_dir / "result.csv").resolve() == Path(input_path).resolve():
        raise ValueError("输出 result.csv 不能覆盖输入文件。")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_csv(recommendations_frame(predictions, k), output_dir / "result.csv")
    summary = {
        "input": source_info(input_path), "method": method, "k": k, "config": asdict(config),
        "selection_report": Path(selection_report).name if selection_report else None,
        "training": model.training_summary, "readers": len(readers),
        "cold_readers": sum(user not in model.seen for user in predictions),
        "format": "local wide CSV: 读者条码,推荐1,...,推荐k; adapt to the official template when released",
    }
    write_json(output_dir / "prediction_summary.json", summary)
    print(f"预测完成：{len(readers)} 位读者，每人 {k} 册；方法 {method}；{output_dir / 'result.csv'}", flush=True)
    return predictions


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("evaluate", "predict"):
        command = commands.add_parser(name, help="本地时间验证" if name == "evaluate" else "全历史训练并生成推荐")
        command.add_argument("--input", type=Path, default=BASE / "data/2022-2025处理后借阅信息.xlsx")
        command.add_argument("--output-dir", type=Path, default=BASE / ("outputs/baseline" if name == "evaluate" else "outputs/prediction"))
        command.add_argument("--sheet", default=None)
        command.add_argument("--k", type=int, default=10)
        command.add_argument("--neighbors", type=int, default=None)
        command.add_argument("--max-history", type=int, default=None)
        command.add_argument("--half-life-days", type=float, default=None)
        command.add_argument("--shrinkage", type=float, default=None)
        command.add_argument("--group-min-users", type=int, default=None)
        command.add_argument("--exclude-seen", action="store_true", default=None)
        if name == "evaluate":
            command.add_argument("--valid-start", default="2024-09-01")
            command.add_argument("--test-start", default="2024-11-01")
            command.add_argument("--test-end", default=None, help="本地测试的排他上界；默认到输入数据末尾")
        else:
            command.add_argument("--method", choices=("auto", *METHODS), default="auto")
            command.add_argument("--selection-report", type=Path, default=None)
            command.add_argument("--readers", type=Path, default=None, help="读者 CSV：读者条码，可附专业/系别/年级")
            command.add_argument("--as-of", default=None, help="仅使用严格早于此时刻的记录")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    overrides = {name: getattr(args, name) for name in asdict(ModelConfig()) if getattr(args, name) is not None}
    try:
        config = ModelConfig(**overrides) if overrides else None
        if args.command == "evaluate":
            run_evaluation(args.input, args.output_dir, args.valid_start, args.test_start,
                           args.test_end, config, args.k, args.sheet)
        else:
            run_prediction(args.input, args.output_dir, args.method, args.selection_report,
                           args.readers, config, args.k, args.sheet, args.as_of)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f"错误：{error}\n")


if __name__ == "__main__":
    main()
