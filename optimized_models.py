"""稀疏线性/图推荐、书目聚合、分层属性先验与因果复借门控。

算法按论文公式独立实现；复借门控是面向借阅周期的轻量改造，不是 RepeatNet 复现。
"""

from __future__ import annotations

import copy
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from time import perf_counter

import numpy as np
import pandas as pd

# 同时兼容标准安装和本项目 .model_deps 隔离环境。
from advanced_models import HistoryContext, latest_attributes, top_indices
from baseline import ATTRIBUTES, ITEM, TIME, USER, attribute_value, ground_truth, timestamp

try:
    from scipy import linalg, sparse
    from scipy.special import expit, logit
except ModuleNotFoundError as error:
    raise ModuleNotFoundError("请先安装 requirements-optimized.txt。") from error


@dataclass(frozen=True)
class OptimizationConfig:
    graph_topk: int = 100
    profile_strength: float = 20.0
    profile_min_users: int = 3
    gate_months: int = 2
    gate_windows: int = 3
    gate_l2: float = 2.0
    max_dual_users: int = 8000

    def __post_init__(self):
        if min(self.graph_topk, self.profile_min_users, self.gate_months,
               self.gate_windows, self.max_dual_users) < 1:
            raise ValueError("数量参数必须为正整数。")
        if any(not math.isfinite(x) or x <= 0 for x in (self.profile_strength, self.gate_l2)):
            raise ValueError("平滑和正则参数必须为有限正数。")


def normalized_text(value):
    text = unicodedata.normalize("NFKC", attribute_value(value)).casefold()
    return re.sub(r"\s+", "", text)


def study_stage(profile, at):
    """9 月换学年；入学年份不等同于学习阶段。不能合理推算时返回空值。"""
    grade = attribute_value(profile.get("年级", ""))
    match = re.fullmatch(r"((?:19|20)\d{2})(?:\.0|级)?", grade)
    if not match:
        return ""
    at = timestamp(at)
    academic_year = at.year - int(at.month < 9)
    stage = academic_year - int(match.group(1)) + 1
    return str(stage) if 1 <= stage <= 8 else ""


def no_attribute_baseline(context, users, k):
    """只关闭属性项，ItemCF、复借、热门与所有系数保持相同；不修改原对象。"""
    model = copy.copy(context.baseline)
    model.group_scores = {}
    return {user: model.recommend(user, k, method="hybrid") for user in users}


class EntityMatrix:
    """训练二值矩阵 + 查询衰减矩阵。书目键保留题名中的版本/卷次信息。"""

    def __init__(self, context: HistoryContext, title_level=False):
        self.context, self.title_level = context, title_level
        keys = []
        for item in context.items:
            meta = context.metadata.get(item, {})
            title = normalized_text(meta.get("题名", ""))
            author = normalized_text(meta.get("责任者", ""))
            keys.append(("title", title, author) if title_level and title else ("barcode", item, ""))
        self.keys = sorted(set(keys))
        key_index = {key: index for index, key in enumerate(self.keys)}
        self.item_groups = np.asarray([key_index[key] for key in keys], dtype=np.int64)
        pair_groups = self.item_groups[context.pair_items]
        self.x = sparse.csr_matrix((np.ones(len(pair_groups)), (context.pair_users, pair_groups)),
                                   shape=(len(context.users), len(self.keys)), dtype=np.float64)
        self.x.data[:] = 1.0
        self.x.sort_indices()
        self.query = sparse.lil_matrix(self.x.shape, dtype=np.float64)
        for user, (items, weights) in context.recent.items():
            values = {}
            for item, weight in zip(items, weights):
                group = int(self.item_groups[item])
                values[group] = max(values.get(group, 0.0), float(weight))
            row = context.user_index[user]
            self.query.rows[row] = sorted(values)
            self.query.data[row] = [values[index] for index in sorted(values)]
        self.query = self.query.tocsr()
        popularity = np.asarray([context.baseline.popularity.get(item, 0.0) for item in context.items])
        self.tie_break = popularity * 1e-12
        counts = np.bincount(self.item_groups, minlength=len(self.keys))
        totals = np.bincount(self.item_groups, weights=popularity, minlength=len(self.keys))
        self.copy_prior = (0.5 / counts[self.item_groups]
                           + 0.5 * np.divide(popularity, totals[self.item_groups],
                                             out=1.0 / counts[self.item_groups],
                                             where=totals[self.item_groups] > 0))

    def expand(self, scores):
        scores = np.asarray(scores, dtype=np.float64)
        if self.title_level:
            # 先预测书目，再按历史复本分布分配到条码。负关系不作为书目召回证据。
            scores = np.maximum(scores[self.item_groups], 0.0) * self.copy_prior
        return scores + self.tie_break

    def rankings(self, score_function, users, k):
        context, result = self.context, {}
        known = [user for user in users if user in context.user_index]
        for start in range(0, len(known), 64):
            batch = known[start:start + 64]
            query = self.query[[context.user_index[user] for user in batch]]
            values = score_function(query)
            for user, row in zip(batch, values):
                scores = self.expand(row)
                result[user] = [context.items[index] for index in top_indices(scores, k, context.blocked(user))]
        for user in users:
            if user not in result:
                result[user] = context.fallback(user, k)
        return {user: result[user] for user in users}

    def summary(self):
        return {"users": self.x.shape[0], "entities": self.x.shape[1], "nonzeros": self.x.nnz,
                "barcodes": len(self.context.items), "title_level": self.title_level}


class RP3Model:
    """alpha=1 的三步随机游走，目的图书度数惩罚 beta，随后保留行 Top-k。"""

    def __init__(self, matrix, beta=0.5, topk=100):
        if not math.isfinite(beta) or beta < 0 or topk < 1:
            raise ValueError("beta 必须非负且有限，topk 必须为正。")
        started = perf_counter()
        self.matrix = matrix
        x = matrix.x
        user_degree = np.asarray(x.sum(axis=1)).ravel()
        item_degree = np.asarray(x.sum(axis=0)).ravel()
        transition = sparse.diags(1.0 / item_degree) @ x.T @ sparse.diags(1.0 / user_degree) @ x
        transition = (transition @ sparse.diags(item_degree ** (-beta))).tocsr()
        transition.setdiag(0)
        transition.eliminate_zeros()
        data, indices, indptr = [], [], [0]
        for row in range(transition.shape[0]):
            begin, end = transition.indptr[row:row + 2]
            ids, values = transition.indices[begin:end], transition.data[begin:end]
            order = np.lexsort((ids, -values))[:topk]
            indices.extend(ids[order])
            data.extend(values[order])
            indptr.append(len(indices))
        self.similarity = sparse.csr_matrix((data, indices, indptr), shape=transition.shape)
        self.summary = {**matrix.summary(), "beta": beta, "alpha": 1.0, "topk": topk,
                        "similarity_nonzeros": self.similarity.nnz,
                        "elapsed_seconds": round(perf_counter() - started, 3)}

    def scores(self, query):
        return (query @ self.similarity).toarray()

    def rank(self, users, k):
        return self.matrix.rankings(self.scores, users, k)


class EASEModel:
    """精确 EASE 的用户维度 Woodbury 解，避免构造 2 万册平方的矩阵。"""

    def __init__(self, matrix, regularization=100.0, max_users=8000):
        if not math.isfinite(regularization) or regularization <= 0:
            raise ValueError("EASE 正则必须为有限正数。")
        if matrix.x.shape[0] > max_users:
            raise ValueError(f"EASE 用户维度 {matrix.x.shape[0]} 超过保护上限 {max_users}。")
        started = perf_counter()
        self.matrix = matrix
        x = matrix.x
        gram = (x @ x.T).toarray()
        gram.flat[::len(gram) + 1] += regularization
        factor = linalg.cho_factor(gram, lower=True, overwrite_a=True, check_finite=False)
        self.inverse = linalg.cho_solve(factor, np.eye(len(gram)), overwrite_b=True, check_finite=False)
        columns = x.tocsc()
        leverage = np.empty(x.shape[1], dtype=np.float64)
        for column in range(x.shape[1]):
            ids = columns.indices[columns.indptr[column]:columns.indptr[column + 1]]
            leverage[column] = self.inverse[np.ix_(ids, ids)].sum()
        self.leverage = leverage
        self.denominator = 1.0 - leverage
        if not np.isfinite(self.inverse).all() or (self.denominator <= 0).any():
            raise ValueError("EASE 数值求解失败，请增加正则。")
        self.summary = {**matrix.summary(), "regularization": regularization,
                        "solver": "exact user-space Woodbury / Cholesky", "zero_diagonal": True,
                        "inverse_megabytes": self.inverse.nbytes / 1024 ** 2,
                        "elapsed_seconds": round(perf_counter() - started, 3)}

    def scores(self, query):
        x = self.matrix.x
        coefficients = (query @ x.T).dot(self.inverse)
        prediction = x.T.dot(coefficients.T).T
        return (prediction - query.toarray() * self.leverage) / self.denominator

    def rank(self, users, k):
        return self.matrix.rankings(self.scores, users, k)


class AttributePrior:
    """专业×学习阶段 -> 专业 -> 全局的平滑先验，另融合系别/学习阶段。"""

    WEIGHTS = {"major_stage": 0.5, "major": 0.25, "department": 0.15, "stage": 0.1}

    def __init__(self, context, config, use_stage=True):
        self.context, self.config, self.use_stage = context, config, use_stage
        raw = defaultdict(dict)
        support = defaultdict(set)
        fields = [USER, ITEM, TIME] + [field for field in ATTRIBUTES if field in context.history]
        for row in context.history[fields].to_dict("records"):
            weight = 2.0 ** (-(context.cutoff - row[TIME]).total_seconds()
                              / (86400 * context.config.half_life_days))
            for key in self.keys(row, row[TIME]):
                pair = (row[USER], context.item_index[row[ITEM]])
                raw[key][pair] = max(raw[key].get(pair, 0.0), weight)
                support[key].add(row[USER])
        self.counts = {}
        for key, pairs in raw.items():
            if len(support[key]) >= config.profile_min_users:
                counter = Counter()
                for (_, item), weight in pairs.items():
                    counter[item] += weight
                self.counts[key] = counter
        values = np.asarray([context.baseline.popularity.get(item, 0.0) for item in context.items])
        self.global_probability = values / values.sum()
        self.summary = {"groups": len(self.counts), "min_users": config.profile_min_users,
                        "prior_strength": config.profile_strength, "stage": use_stage,
                        "history_profiles_only": True}

    def keys(self, profile, at):
        major = attribute_value(profile.get("专业", ""))
        department = attribute_value(profile.get("系别", ""))
        stage = study_stage(profile, at) if self.use_stage else ""
        keys = []
        if major and stage:
            keys.append(("major_stage", major, stage))
        if major:
            keys.append(("major", major))
        if department:
            keys.append(("department", department))
        if stage:
            keys.append(("stage", stage))
        return keys

    @lru_cache(maxsize=128)
    def distribution(self, key):
        parent = self.distribution(("major", key[1])) if key[0] == "major_stage" else self.global_probability
        counts = self.counts.get(key)
        if not counts:
            return parent
        strength = self.config.profile_strength
        result = parent * strength
        ids = np.fromiter(counts.keys(), dtype=np.int64)
        values = np.fromiter(counts.values(), dtype=np.float64)
        result[ids] += values
        return result / (values.sum() + strength)

    def profile(self, user, external=None):
        result = dict(self.context.profiles.get(user, {}))
        if external and user in external:
            result.update({field: value for field, raw in external[user].items()
                           if field in ATTRIBUTES and (value := attribute_value(raw))})
        return result

    def scores(self, user, external=None):
        keys = self.keys(self.profile(user, external), self.context.cutoff)
        available = [key for key in keys if key in self.counts]
        if not available:
            return self.global_probability.copy()
        total = sum(self.WEIGHTS[key[0]] for key in available)
        return sum(self.WEIGHTS[key[0]] / total * self.distribution(key) for key in available)

    def rank(self, users, k, external=None):
        context, result = self.context, {}
        for user in users:
            scores = self.scores(user, external)
            result[user] = [context.items[index] for index in top_indices(scores, k, context.blocked(user))]
        return result


def behavioral_features(rows, cutoff, use_attributes=True):
    events, unique = len(rows), rows[ITEM].nunique()
    gap = max(0.0, (cutoff - rows[TIME].max()).total_seconds() / 86400)
    active = int(rows[TIME].ge(cutoff - pd.Timedelta(days=60)).sum())
    values = [math.log1p(events), math.log1p(unique), (events - unique) / events,
              math.log1p(gap), math.log1p(active)]
    if use_attributes:
        profile = {}
        if "年级" in rows:
            grades = rows["年级"].map(attribute_value)
            grades = grades[grades.ne("")]
            if len(grades):
                profile["年级"] = grades.iloc[-1]
        stage = study_stage(profile, cutoff)
        values += [float(stage) if stage else 0.0, float(bool(stage))]
    return np.asarray(values, dtype=np.float64)


class RepeatGate:
    """只用训练期内的历史快照，预测未来两个月真值中复借所占比例。"""

    def __init__(self, context, config, use_attributes=True):
        self.context, self.config, self.use_attributes = context, config, use_attributes
        features, labels, anchors = [], [], []
        for offset in range(1, config.gate_windows + 1):
            anchor = context.cutoff - pd.DateOffset(months=config.gate_months * offset)
            end = anchor + pd.DateOffset(months=config.gate_months)
            before = context.history.loc[context.history[TIME].lt(anchor)]
            after = context.history.loc[context.history[TIME].ge(anchor) & context.history[TIME].lt(end)]
            seen = ground_truth(before)
            history = {user: rows for user, rows in before.groupby(USER, sort=True)}
            count = 0
            for user, relevant in ground_truth(after).items():
                if user not in seen:
                    continue
                features.append(behavioral_features(history[user], anchor, use_attributes))
                labels.append(len(relevant & seen[user]) / len(relevant))
                count += 1
            anchors.append({"start": anchor.isoformat(), "end_exclusive": end.isoformat(), "samples": count})
        dimension = 7 if use_attributes else 5
        self.mean, self.std = np.zeros(dimension), np.ones(dimension)
        self.coefficients = np.zeros(dimension + 1)
        self.coefficients[0] = float(logit(0.1))
        if features:
            raw = np.asarray(features)
            y = np.asarray(labels)
            self.mean, self.std = raw.mean(axis=0), np.maximum(raw.std(axis=0), 1e-6)
            design = np.column_stack([np.ones(len(raw)), (raw - self.mean) / self.std])
            self.coefficients[0] = float(logit(np.clip(y.mean(), 1e-4, 1 - 1e-4)))
            penalty = np.full(dimension + 1, config.gate_l2)
            penalty[0] = 1e-6
            for _ in range(40):
                probability = expit(design @ self.coefficients)
                gradient = design.T @ (probability - y) + penalty * self.coefficients
                weights = np.maximum(probability * (1 - probability), 1e-6)
                hessian = design.T @ (design * weights[:, None]) + np.diag(penalty)
                change = np.linalg.solve(hessian, gradient)
                # 截断过大 Newton 步长，避免近乎可分数据造成数值发散。
                change *= min(1.0, 5.0 / max(np.linalg.norm(change), 1e-12))
                self.coefficients -= change
                if np.linalg.norm(change) < 1e-7:
                    break
        self.probabilities = {}
        for user, rows in context.history.groupby(USER, sort=True):
            feature = behavioral_features(rows, context.cutoff, use_attributes)
            design = np.r_[1.0, (feature - self.mean) / self.std]
            self.probabilities[user] = float(expit(design @ self.coefficients))
        self.summary = {"training_snapshots": anchors, "training_samples": len(labels),
                        "mean_repeat_target": float(np.mean(labels)) if labels else None,
                        "uses_study_stage": use_attributes, "coefficients": self.coefficients.tolist(),
                        "feature_mean": self.mean.tolist(), "feature_std": self.std.tolist(),
                        "all_labels_strictly_before": context.cutoff.isoformat()}


class RepeatExploreSpace:
    """分别归一化复借和探索概率，再按训练历史估计的门控合并。"""

    def __init__(self, context, rankings, users, gate, rrf_constant=20.0):
        self.context, self.rankings, self.users = context, rankings, list(users)
        self.gate, self.constant = gate, rrf_constant
        counts = context.history.groupby([USER, ITEM], sort=True).size().to_dict()
        self.repeat = {}
        for user in users:
            values = {}
            if not context.config.exclude_seen and user in context.recent:
                ids, weights = context.recent[user]
                values = {context.items[index]: float(weight) * math.log1p(counts[user, context.items[index]])
                          for index, weight in zip(ids, weights)}
            total = sum(values.values())
            self.repeat[user] = {item: value / total for item, value in values.items()} if total else {}

    def predict(self, weights, k=10, gate_scale=1.0):
        if (k < 1 or not math.isfinite(gate_scale) or gate_scale <= 0
                or not weights or any(name not in self.rankings or not math.isfinite(weight) or weight <= 0
                                      for name, weight in weights.items())):
            raise ValueError("复借探索参数无效。")
        predictions = {}
        for user in self.users:
            index = self.context.user_index.get(user)
            seen = self.context.seen[index] if index is not None else set()
            seen_codes = {self.context.items[item] for item in seen}
            explore = defaultdict(float)
            for name, weight in weights.items():
                novel = [item for item in self.rankings[name][user] if item not in seen_codes]
                for rank, item in enumerate(novel, 1):
                    explore[item] += weight / (self.constant + rank)
            total = sum(explore.values())
            probability = self.gate.probabilities.get(user, 0.0)
            if probability > 0:
                probability = float(expit(logit(np.clip(probability, 1e-8, 1 - 1e-8)) + math.log(gate_scale)))
            repeat = self.repeat[user]
            if not repeat:
                probability = 0.0
            elif not total:
                probability = 1.0
            combined = {item: (1 - probability) * value / total for item, value in explore.items()} if total else {}
            combined.update({item: probability * value for item, value in repeat.items()})
            ranked = sorted(combined, key=lambda item: (-combined[item], item))[:k]
            if len(ranked) < k:
                ranked += [item for item in self.context.fallback(user, k) if item not in ranked][:k - len(ranked)]
            predictions[user] = ranked
        return predictions
