"""时间隔离的语义召回、类别先验、候选特征和 CatBoost 排序。

预训练编码器不在比赛流水上微调；只编码当前历史可见元数据。
排序标签独立生成，绝不把未召回的正例强行加入训练候选。
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from advanced_models import ContentModel, clc_prefix, top_indices, torch
from baseline import ITEM, TIME, USER, attribute_value, timestamp, write_json
from optimized_models import study_stage

OLD_CHANNELS = ("baseline", "content", "item2vec", "recent", "previous", "optimized")
NEW_CHANNELS = ("semantic", "category")
CAT_FEATURES = ["major", "department", "stage", "clc"]


class TextEncoder:
    """官方 BGE 的 CLS + L2 归一化。分批保存内容寻址向量，可中断恢复。"""

    def __init__(self, model_dir, cache_dir, batch_size=64):
        from transformers import AutoModel, AutoTokenizer
        self.model_dir = Path(model_dir)
        provenance = json.loads((self.model_dir / "provenance.json").read_text(encoding="utf-8"))
        self.identity = provenance["repo"] + "@" + provenance["revision"] + ":CLS:128:v1"
        self.cache_dir = Path(cache_dir) / hashlib.sha256(self.identity.encode()).hexdigest()[:16]
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        self.model = AutoModel.from_pretrained(str(model_dir), local_files_only=True).eval()
        self.batch_size = batch_size
        self.dimension = self.model.config.hidden_size

    def encode(self, texts):
        keys = [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts]
        unique = dict(zip(keys, texts))
        missing = [(key, text) for key, text in unique.items()
                   if not (self.cache_dir / (key + ".npy")).is_file()]
        for begin in range(0, len(missing), self.batch_size):
            batch = missing[begin:begin + self.batch_size]
            tokens = self.tokenizer([text for _, text in batch], padding=True, truncation=True,
                                    max_length=128, return_tensors="pt")
            with torch.inference_mode():
                vector = self.model(**tokens).last_hidden_state[:, 0]
                vector = torch.nn.functional.normalize(vector, p=2, dim=1).cpu().numpy()
            for (key, _), row in zip(batch, vector):
                path = self.cache_dir / (key + ".npy")
                with path.with_suffix(".tmp").open("wb") as stream:
                    np.save(stream, row.astype(np.float32), allow_pickle=False)
                path.with_suffix(".tmp").replace(path)
            if begin % (self.batch_size * 20) == 0:
                print(f"语义编码：{min(begin + len(batch), len(missing))}/{len(missing)} 个新增文本", flush=True)
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        result = np.stack([np.load(self.cache_dir / (key + ".npy"), allow_pickle=False) for key in keys])
        if not np.isfinite(result).all():
            raise ValueError("语义向量包含无效值。")
        return result


def book_text(meta):
    # 卷次、数字、版本不删除，图书相似并不等于同一册。
    return "；".join(field + "：" + attribute_value(meta.get(field, ""))
                    for field in ("题名", "责任者", "索书号") if attribute_value(meta.get(field, ""))) or "未知图书"


class CandidateFeatures:
    """只依赖 HistoryContext；既生成新召回，又为现有/扩展候选构造同一套特征。"""

    def __init__(self, context, encoder):
        self.context = context
        self.vectors = encoder.encode([book_text(context.metadata.get(item, {})) for item in context.items])
        self.prefix = np.asarray([clc_prefix(context.metadata.get(item, {}).get("索书号", ""))
                                  for item in context.items])
        self.authors = np.asarray([attribute_value(context.metadata.get(item, {}).get("责任者", ""))
                                  for item in context.items])
        h = context.history
        self.item_degree = np.bincount(context.pair_items, minlength=len(context.items)).astype(float)
        self.popularity = np.asarray([context.baseline.popularity.get(i, 0) for i in context.items])
        self.recent30 = np.zeros(len(context.items))
        self.recent180 = np.zeros(len(context.items))
        for days, array in ((30, self.recent30), (180, self.recent180)):
            counts = h.loc[h[TIME].ge(context.cutoff - pd.Timedelta(days=days))].groupby(ITEM)[USER].nunique()
            for item, count in counts.items():
                array[context.item_index[item]] = count
        self.last_days = {}
        self.pair_counts = h.groupby([USER, ITEM]).size().to_dict()
        self.user_counts = h.groupby(USER).size().to_dict()
        self.pair_last = {(u, i): max(0., (context.cutoff - time).total_seconds() / 86400)
                          for u, i, time in context.pairs[[USER, ITEM, TIME]].itertuples(index=False, name=None)}
        self.user_last = h.groupby(USER)[TIME].max().map(
            lambda t: max(0., (context.cutoff - t).total_seconds() / 86400)).to_dict()
        # 基于事件发生时专业/阶段计数：类别统计比单册统计共享更多样本。
        self.category_counts = defaultdict(Counter)
        self.group_item_counts = defaultdict(Counter)
        unique = h.drop_duplicates([USER, ITEM], keep="last")
        for row in unique.to_dict("records"):
            idx = context.item_index[row[ITEM]]
            major = attribute_value(row.get("专业", ""))
            stage = study_stage(row, row[TIME])
            keys = [("global",)]
            if major:
                keys.append(("major", major))
                if stage:
                    keys.append(("major_stage", major, stage))
            for key in keys:
                if self.prefix[idx]:
                    self.category_counts[key][self.prefix[idx]] += 1
                self.group_item_counts[key][idx] += 1
        self.global_category = self.category_counts[("global",)]
        self.global_category_total = max(1, sum(self.global_category.values()))
        self.category_degree = Counter()
        for index, prefix in enumerate(self.prefix):
            self.category_degree[prefix] += self.item_degree[index]
        self.copy_count = Counter(book_text(context.metadata.get(item, {})) for item in context.items)
        self.copies = np.asarray([self.copy_count[book_text(context.metadata.get(item, {}))] for item in context.items])

    def user_scores(self, user, profile=None):
        c = self.context
        profile = profile or c.profiles.get(user, {})
        major = attribute_value(profile.get("专业", ""))
        stage = study_stage(profile, c.cutoff)
        keys = [("major_stage", major, stage), ("major", major), ("global",)]
        key = next((k for k in keys if sum(self.category_counts.get(k, {}).values()) >= 10), ("global",))
        counter = self.category_counts[key]
        total = sum(counter.values())
        ids, weights = c.recent.get(user, (np.array([], dtype=int), np.array([])))
        ids, weights = ids[:5], weights[:5]
        semantic_mean = np.zeros(len(c.items), dtype=np.float32)
        semantic_max = semantic_mean.copy()
        affinity = Counter()
        author_affinity = Counter()
        if len(ids):
            # 用 Torch 进行明确线程数的矩阵乘法，不改变旧 Item2Vec 的 NumPy BLAS 环境。
            similarities = (torch.from_numpy(self.vectors) @ torch.from_numpy(self.vectors[ids]).T).numpy()
            normalized = weights / max(float(weights.sum()), 1e-12)
            semantic_mean = similarities @ normalized
            semantic_max = (similarities * np.sqrt(weights / max(float(weights.max()), 1e-12))).max(axis=1)
            for item, weight in zip(ids, normalized):
                if self.prefix[item]:
                    affinity[self.prefix[item]] += float(weight)
                if self.authors[item]:
                    author_affinity[self.authors[item]] += float(weight)
        categories = set(self.prefix)
        probability = {prefix: (counter[prefix] + 20 * self.global_category[prefix] / self.global_category_total)
                       / max(total + 20, 1) for prefix in categories}
        group_cat = np.asarray([probability[p] for p in self.prefix])
        personal_cat = np.asarray([affinity[p] for p in self.prefix])
        personal_author = np.asarray([author_affinity[a] for a in self.authors])
        group_items = self.group_item_counts.get(key, {})
        group_item = np.asarray([group_items.get(i, 0) for i in range(len(c.items))])
        category_score = ((0.65 * personal_cat + 0.35 * group_cat)
                          * np.log1p(self.recent180 + 0.1 * self.item_degree)
                          / np.asarray([max(1, self.category_degree[p]) ** 0.25 for p in self.prefix]))
        return {"semantic_mean": semantic_mean, "semantic_max": semantic_max,
                "category_personal": personal_cat, "author_personal": personal_author,
                "category_group": group_cat, "group_item_count": group_item,
                "semantic_score": .7 * semantic_max + .3 * semantic_mean,
                "category_score": category_score}

    def build(self, users, channels, per_channel=200, new_k=100, external=None):
        """返回候选表和扩展通道。没有标签参数，因此标签不能影响候选构建。"""
        c = self.context
        expanded = {name: {} for name in NEW_CHANNELS}
        frames = []
        for user in users:
            profile = dict(c.profiles.get(user, {}))
            if external and user in external:
                profile.update({k: v for k, v in external[user].items() if attribute_value(v)})
            scores = self.user_scores(user, profile)
            blocked = set(c.blocked(user))
            # 新通道面向未借图书；已借图书保留在原有通道中。
            seen_ids = c.seen[c.user_index[user]] if user in c.user_index else set()
            for name in NEW_CHANNELS:
                values = scores[name + "_score"]
                indices = top_indices(values + self.popularity * 1e-12, new_k, seen_ids | blocked)
                expanded[name][user] = [c.items[i] for i in indices]
            existing = set().union(*(set(channels[n][user][:per_channel]) for n in OLD_CHANNELS))
            codes = sorted((existing | set(expanded["semantic"][user]) | set(expanded["category"][user]))
                           - {c.items[i] for i in blocked})
            ids = np.asarray([c.item_index[item] for item in codes])
            data = {USER: [user] * len(codes), ITEM: codes, "existing": [int(i in existing) for i in codes],
                    "warm": [int(user in c.user_index)] * len(codes)}
            for name in (*OLD_CHANNELS, *NEW_CHANNELS):
                ranking = channels[name][user][:per_channel] if name in OLD_CHANNELS else expanded[name][user]
                positions = {item: rank for rank, item in enumerate(ranking, 1)}
                data["rr_" + name] = [1 / (20 + positions[item]) if item in positions else 0 for item in codes]
            data["channel_count"] = np.sum([np.asarray(data["rr_" + n]) > 0 for n in (*OLD_CHANNELS, *NEW_CHANNELS)], axis=0)
            for name in ("semantic_mean", "semantic_max", "category_personal", "author_personal",
                         "category_group", "group_item_count"):
                data[name] = scores[name][ids]
            for name, array in (("degree", self.item_degree), ("popular", self.popularity),
                                ("recent30", self.recent30), ("recent180", self.recent180), ("copies", self.copies)):
                data[name] = np.log1p(array[ids])
            data["trend"] = (self.recent30[ids] + 1) / (self.recent180[ids] + 6)
            data["seen"] = [int(i in seen_ids) for i in ids]
            data["borrow_count"] = [math.log1p(self.pair_counts.get((user, item), 0)) for item in codes]
            data["last_gap"] = [math.log1p(self.pair_last.get((user, item), 3650)) for item in codes]
            data["user_events"] = math.log1p(self.user_counts.get(user, 0))
            data["user_unique"] = math.log1p(len(seen_ids))
            data["user_last"] = math.log1p(self.user_last.get(user, 3650))
            data["month"] = c.cutoff.month
            data["major"] = attribute_value(profile.get("专业", "")) or "未知"
            data["department"] = attribute_value(profile.get("系别", "")) or "未知"
            data["stage"] = study_stage(profile, c.cutoff) or "未知"
            data["clc"] = [self.prefix[i] or "未知" for i in ids]
            frame = pd.DataFrame(data)
            for col in frame:
                if col not in (USER, ITEM, *CAT_FEATURES):
                    frame[col] = frame[col].astype(np.float32)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True), expanded


def feature_columns(frame, expanded):
    excluded = {USER, ITEM, "label", "query", "existing", "warm"}
    # 现有召回+排序也可用相同内容/属性特征；唯一差异是候选及新通道名次。
    if not expanded:
        excluded |= {"rr_semantic", "rr_category", "channel_count"}
    return [name for name in frame if name not in excluded]


def labeled(frame, truth, start, end):
    if timestamp(start) >= timestamp(end):
        raise ValueError("标签窗口必须向前。")
    result = frame.copy()
    result["label"] = [int(item in truth.get(user, set()))
                       for user, item in result[[USER, ITEM]].itertuples(index=False, name=None)]
    result["query"] = str(start) + "|" + result[USER]
    return result


def training_data(snapshots, cutoff, expanded):
    available = []
    used = []
    for snap in snapshots:
        if timestamp(snap["end"]) > timestamp(cutoff):
            continue
        frame = pd.read_pickle(snap["path"])
        if not expanded:
            frame = frame.loc[frame["existing"].eq(1)]
        # 未来没有命中候选的组无法提供排序对；不注入其遗漏正例。
        frame = frame.loc[frame["warm"].eq(1)]
        useful = frame.groupby("query")["label"].transform("sum").gt(0)
        frame = frame.loc[useful]
        if len(frame):
            available.append(frame)
            used.append({"start": snap["start"], "end": snap["end"],
                         "queries": int(frame["query"].nunique()), "positive_candidates": int(frame["label"].sum())})
    if not available:
        raise ValueError("没有标签完整且候选包含正例的更早训练快照。")
    result = pd.concat(available, ignore_index=True).sort_values(["query", ITEM], kind="stable").reset_index(drop=True)
    return result, used


def fit_ranker(frame, expanded, path, iterations=250):
    from catboost import CatBoostRanker, Pool
    columns = feature_columns(frame, expanded)
    model = CatBoostRanker(iterations=iterations, depth=4, learning_rate=.05, l2_leaf_reg=10,
                          loss_function="YetiRank:mode=NDCG;top=10;dcg_denominator=LogPosition",
                          random_seed=2026, thread_count=4, verbose=False, allow_writing_files=False)
    pool = Pool(frame[columns], label=frame["label"], group_id=frame["query"],
                cat_features=CAT_FEATURES)
    model.fit(pool)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))
    write_json(Path(path).with_suffix(".features.json"),
               {"columns": columns, "importance": dict(zip(columns, map(float, model.get_feature_importance(pool))))})
    return model, columns


def rank_candidates(model, columns, frame, fallback, expanded, trees=250, k=10):
    eligible = frame if expanded else frame.loc[frame["existing"].eq(1)]
    warm = eligible.loc[eligible["warm"].eq(1)].copy()
    warm["prediction"] = model.predict(warm[columns], ntree_end=trees) if len(warm) else []
    result = {user: items[:k] for user, items in fallback.items()}
    for user, group in warm.groupby(USER, sort=False):
        result[user] = group.sort_values(["prediction", ITEM], ascending=[False, True], kind="stable")[ITEM].head(k).tolist()
    return result


def candidate_coverage(frame, truth, context=None):
    result = {}
    for name, subset in (("existing", frame.loc[frame["existing"].eq(1)]), ("expanded", frame)):
        pools = subset.groupby(USER)[ITEM].agg(set).to_dict()
        hits = sum(len(pools.get(u, set()) & y) for u, y in truth.items())
        result[name] = {"users": len(truth), "candidate_hit_pairs": hits,
                        "truth_pairs": sum(map(len, truth.values())),
                        "user_macro_recall": float(np.mean([len(pools.get(u, set()) & y) / len(y)
                                                          for u, y in truth.items()])),
                        "mean_candidates": float(np.mean([len(pools.get(u, set())) for u in truth]))}
        if context is not None:
            counts = {key: {"truth_pairs": 0, "candidate_hit_pairs": 0}
                      for key in ("repeat", "known_new_to_user", "unknown_catalog")}
            for user, relevant in truth.items():
                index = context.user_index.get(user)
                seen = {context.items[i] for i in context.seen[index]} if index is not None else set()
                subsets = {"repeat": relevant & seen,
                           "known_new_to_user": (relevant & context.baseline.items) - seen,
                           "unknown_catalog": relevant - context.baseline.items}
                for key, values in subsets.items():
                    counts[key]["truth_pairs"] += len(values)
                    counts[key]["candidate_hit_pairs"] += len(values & pools.get(user, set()))
            result[name]["subsets"] = counts
    return result
