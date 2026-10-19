"""训练期隔离的 BPR、属性 BPR、Item2Vec、GRU 和内容推荐组件。"""

from __future__ import annotations

import copy
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from baseline import (
    ATTRIBUTES, BASE, ITEM, TIME, USER, ItemCFBaseline, ModelConfig,
    attribute_value, prepare_events, ranking_metrics, timestamp,
)

try:
    import torch
except ModuleNotFoundError:
    # Codex 本次使用项目内的隔离依赖；正常环境直接安装 requirements-advanced.txt。
    local_dependencies = BASE / ".model_deps"
    if local_dependencies.is_dir():
        sys.path.insert(0, str(local_dependencies))
    try:
        import torch
    except ModuleNotFoundError as error:
        raise ImportError("请先安装 requirements-advanced.txt 中的 PyTorch。") from error

from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class AdvancedConfig:
    dimensions: int = 32
    mf_epochs: int = 30
    item2vec_epochs: int = 12
    gru_epochs: int = 24
    batch_size: int = 1024
    negatives: int = 5
    learning_rate: float = 0.01
    weight_decay: float = 0.0001
    context_window: int = 3
    max_sequence: int = 30
    max_history: int = 100
    half_life_days: float = 180.0
    candidate_k: int = 200
    rrf_constant: float = 20.0
    seed: int = 2026
    threads: int = 4
    exclude_seen: bool = False

    def __post_init__(self):
        integers = (self.dimensions, self.mf_epochs, self.item2vec_epochs, self.gru_epochs,
                    self.batch_size, self.negatives, self.context_window, self.max_sequence,
                    self.max_history, self.candidate_k, self.threads)
        if min(integers) < 1:
            raise ValueError("维度、轮数、批大小、负例数、窗口、候选数和线程数必须为正整数。")
        for value in (self.learning_rate, self.half_life_days, self.rrf_constant):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("学习率、半衰期和 RRF 常数必须为有限正数。")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("正则强度必须为有限非负数。")


def set_seed(seed, threads=4):
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)


def top_indices(scores, k, blocked=()):
    """精确处理第 k 位同分，按已排序的条码索引破平局。"""
    scores = np.asarray(scores, dtype=np.float64).copy()
    if np.isnan(scores).any() or np.isposinf(scores).any():
        raise ValueError("模型输出含 NaN 或正无穷。")
    if len(blocked):
        scores[np.asarray(list(blocked), dtype=np.int64)] = -np.inf
    available = np.flatnonzero(np.isfinite(scores))
    count = min(k, len(available))
    if count == 0:
        return np.empty(0, dtype=np.int64)
    if count < len(available):
        threshold = np.partition(scores[available], len(available) - count)[len(available) - count]
        available = available[scores[available] >= threshold]
    order = np.lexsort((available, -scores[available]))
    return available[order[:count]]


def latest_attributes(data, key, fields):
    result = defaultdict(dict)
    for field in fields:
        if field not in data:
            continue
        values = data[field].map(attribute_value)
        known = data.loc[values.ne(""), [key]].assign(value=values[values.ne("")])
        for entity, value in known.drop_duplicates(key, keep="last").itertuples(index=False, name=None):
            result[entity][field] = value
    return dict(result)


def title_terms(title):
    """无需外部分词或语料：中文二元片段及英文词，全部由历史元数据产生。"""
    text = re.sub(r"\s+", "", attribute_value(title).lower())
    if not text:
        return []
    chinese = re.findall(r"[\u4e00-\u9fff]+", text)
    terms = ["字:" + word[i:i + 2] for word in chinese for i in range(len(word) - 1)]
    terms += ["词:" + word for word in re.findall(r"[a-z]+|\d+", text)]
    return sorted(set(terms))


def clc_prefix(value):
    match = re.match(r"([A-Za-z]+\d{0,3})", attribute_value(value))
    return match.group(1).upper() if match else ""


class HistoryContext:
    """所有词表、元数据、画像及负采样分布只由 cutoff 前记录建立。"""

    def __init__(self, history, cutoff, config):
        self.config = config
        self.history = prepare_events(history)
        self.cutoff = timestamp(cutoff)
        if self.history.empty or self.history[TIME].ge(self.cutoff).any():
            raise ValueError("历史必须非空且严格早于预测截止时点。")
        self.users = sorted(self.history[USER].unique())
        self.items = sorted(self.history[ITEM].unique())
        self.user_index = {user: index for index, user in enumerate(self.users)}
        self.item_index = {item: index for index, item in enumerate(self.items)}
        self.profiles = latest_attributes(self.history, USER, ATTRIBUTES)
        self.metadata = latest_attributes(self.history, ITEM, ("题名", "责任者", "索书号", "中图法一级类目"))
        self.baseline = ItemCFBaseline(ModelConfig(
            max_history=config.max_history, half_life_days=config.half_life_days,
            exclude_seen=config.exclude_seen,
        )).fit(self.history, self.cutoff)
        self.seen = [set() for _ in self.users]
        self.sequences = [[] for _ in self.users]
        self.recent = {}
        self.pairs = self.history.drop_duplicates([USER, ITEM], keep="last")
        self.pair_users = self.pairs[USER].map(self.user_index).to_numpy(dtype=np.int64)
        self.pair_items = self.pairs[ITEM].map(self.item_index).to_numpy(dtype=np.int64)
        self.pair_weights = np.power(2.0, -(self.cutoff - self.pairs[TIME]).dt.total_seconds().to_numpy()
                                     / (86400 * config.half_life_days)).astype(np.float32)
        for user, rows in self.history.groupby(USER, sort=True):
            index = self.user_index[user]
            self.sequences[index] = [(self.item_index[item], time) for item, time in rows[[ITEM, TIME]].itertuples(index=False, name=None)]
            self.seen[index] = {item for item, _ in self.sequences[index]}
            recent = self.pairs.loc[self.pairs[USER].eq(user)].sort_values([TIME, ITEM], ascending=[False, True], kind="stable").head(config.max_history)
            ids = recent[ITEM].map(self.item_index).to_numpy(dtype=np.int64)
            weights = np.power(2.0, -(self.cutoff - recent[TIME]).dt.total_seconds().to_numpy()
                               / (86400 * config.half_life_days)).astype(np.float32)
            self.recent[user] = (ids, weights)
        frequencies = np.bincount(self.pair_items, minlength=len(self.items)).astype(np.float64)
        self.negative_probability = np.power(frequencies, 0.75)
        self.negative_probability /= self.negative_probability.sum()
        self.negative_cdf = np.cumsum(self.negative_probability)
        self.negative_cdf[-1] = 1.0

    def sample_negatives(self, users, count, rng):
        users = np.asarray(users, dtype=np.int64)
        if any(len(self.seen[int(user)]) >= len(self.items) for user in users):
            raise ValueError("读者已借遍候选库，无法为该读者采样未观测图书。")
        result = np.searchsorted(self.negative_cdf, rng.random((len(users), count))).astype(np.int64)
        for row, user in enumerate(users):
            seen = self.seen[int(user)]
            for column in range(count):
                attempts = 0
                while int(result[row, column]) in seen:
                    result[row, column] = rng.integers(len(self.items))
                    attempts += 1
                    if attempts >= 100:
                        eligible = np.asarray(sorted(set(range(len(self.items))) - seen), dtype=np.int64)
                        result[row, column] = rng.choice(eligible)
                        break
        return result

    def blocked(self, user):
        index = self.user_index.get(user)
        return self.seen[index] if self.config.exclude_seen and index is not None else ()

    def fallback(self, user, k):
        return self.baseline.recommend(user, k, method="popular")

    def baseline_rankings(self, users, k):
        return {user: self.baseline.recommend(user, k, method="hybrid") for user in users}

    def recent_rankings(self, users, k):
        result = {}
        for user in users:
            if user not in self.recent:
                result[user] = self.fallback(user, k)
                continue
            scores = np.zeros(len(self.items), dtype=np.float32)
            ids, weights = self.recent[user]
            if len(weights):
                scores[ids] = weights / weights.max()
            scores += np.asarray([self.baseline.popularity.get(item, 0.0) for item in self.items], dtype=np.float32) * 0.001
            result[user] = [self.items[index] for index in top_indices(scores, k, self.blocked(user))]
        return result

    def feature_matrices(self, use_attributes):
        """ID 与共享属性处于同一次训练得到的向量空间，0 专用于 padding。"""
        user_vocabulary, item_vocabulary = {}, {}

        def encode(features, vocabulary):
            encoded = []
            for key, weight in features:
                if key not in vocabulary:
                    vocabulary[key] = len(vocabulary) + 1
                encoded.append((vocabulary[key], weight))
            return encoded

        user_rows, item_rows = [], []
        for user in self.users:
            features = [("读者ID:" + user, 1.0)]
            if use_attributes:
                features += [(field + ":" + value, 0.6) for field, value in self.profiles.get(user, {}).items()]
            user_rows.append(encode(features, user_vocabulary))
        for item in self.items:
            features = [("图书ID:" + item, 1.0)]
            meta = self.metadata.get(item, {})
            if use_attributes:
                for field, weight in (("题名", 0.8), ("责任者", 0.4)):
                    if meta.get(field):
                        features.append((field + ":" + meta[field], weight))
                prefix = clc_prefix(meta.get("索书号", ""))
                if prefix:
                    features.append(("分类:" + prefix, 0.4))
                terms = title_terms(meta.get("题名", ""))[:12]
                features += [(term, 0.25) for term in terms]
            item_rows.append(encode(features, item_vocabulary))

        def pad(rows):
            width = max(map(len, rows))
            ids = np.zeros((len(rows), width), dtype=np.int64)
            weights = np.zeros((len(rows), width), dtype=np.float32)
            for index, row in enumerate(rows):
                for column, (feature, weight) in enumerate(row):
                    ids[index, column], weights[index, column] = feature, weight
            weights /= np.sqrt((weights ** 2).sum(axis=1, keepdims=True))
            return torch.from_numpy(ids), torch.from_numpy(weights)

        return pad(user_rows), pad(item_rows), len(user_vocabulary) + 1, len(item_vocabulary) + 1


class BPRModel(nn.Module):
    def __init__(self, context, dimensions, use_attributes=False):
        super().__init__()
        user, item, user_count, item_count = context.feature_matrices(use_attributes)
        self.register_buffer("user_features", user[0])
        self.register_buffer("user_weights", user[1])
        self.register_buffer("item_features", item[0])
        self.register_buffer("item_weights", item[1])
        self.user_embedding = nn.Embedding(user_count, dimensions, padding_idx=0)
        self.item_embedding = nn.Embedding(item_count, dimensions, padding_idx=0)
        nn.init.normal_(self.user_embedding.weight, std=0.05)
        nn.init.normal_(self.item_embedding.weight, std=0.05)
        with torch.no_grad():
            self.user_embedding.weight[0].zero_()
            self.item_embedding.weight[0].zero_()

    def encode_users(self, users):
        return (self.user_embedding(self.user_features[users]) * self.user_weights[users].unsqueeze(-1)).sum(-2)

    def encode_items(self, items):
        return (self.item_embedding(self.item_features[items]) * self.item_weights[items].unsqueeze(-1)).sum(-2)


class Item2VecModel(nn.Module):
    def __init__(self, item_count, dimensions):
        super().__init__()
        self.input_embedding = nn.Embedding(item_count, dimensions)
        self.output_embedding = nn.Embedding(item_count, dimensions)
        nn.init.normal_(self.input_embedding.weight, std=0.1)
        nn.init.zeros_(self.output_embedding.weight)


class GRUSequenceModel(nn.Module):
    """按读者历史训练的 GRU 适配模型，不冒称原始会话 GRU4Rec 的复现。"""

    def __init__(self, item_count, dimensions, pretrained=None):
        super().__init__()
        self.embedding = nn.Embedding(item_count + 1, dimensions, padding_idx=0)
        nn.init.normal_(self.embedding.weight, std=0.1)
        with torch.no_grad():
            self.embedding.weight[0].zero_()
            if pretrained is not None:
                vectors = F.normalize(torch.as_tensor(pretrained), dim=-1)
                self.embedding.weight[1:].copy_(vectors * 0.3)
        self.gru = nn.GRU(dimensions, dimensions, batch_first=True)
        self.projection = nn.Linear(dimensions, dimensions)
        self.dropout = nn.Dropout(0.1)

    def forward(self, sequence):
        states, _ = self.gru(self.dropout(self.embedding(sequence)))
        return self.projection(states)


def checkpoint_epochs(maximum):
    return sorted({max(1, maximum // 3), max(1, 2 * maximum // 3), maximum})


def validation_score(rankings, truth, k=10):
    return float(np.mean([ranking_metrics(rankings.get(user, []), relevant, k)["score"] for user, relevant in truth.items()]))


@torch.no_grad()
def neural_rankings(name, model, context, users, k):
    model.eval()
    result = {user: context.fallback(user, k) for user in users if user not in context.user_index}
    known = [user for user in users if user in context.user_index]
    if name in ("bpr_mf", "attribute_bpr"):
        item_vectors = model.encode_items(torch.arange(len(context.items)))
        for start in range(0, len(known), 128):
            batch = known[start:start + 128]
            user_vectors = model.encode_users(torch.tensor([context.user_index[user] for user in batch]))
            scores = (user_vectors @ item_vectors.T).numpy()
            for user, row in zip(batch, scores):
                result[user] = [context.items[index] for index in top_indices(row, k, context.blocked(user))]
    elif name == "item2vec":
        item_vectors = F.normalize(model.input_embedding.weight, dim=-1).numpy()
        vectors = []
        for user in known:
            ids, weights = context.recent[user]
            query = (item_vectors[ids] * weights[:, None]).sum(axis=0)
            vectors.append(query / max(np.linalg.norm(query), 1e-12))
        for start in range(0, len(known), 128):
            scores = np.asarray(vectors[start:start + 128]) @ item_vectors.T
            for user, row in zip(known[start:start + 128], scores):
                result[user] = [context.items[index] for index in top_indices(row, k, context.blocked(user))]
    elif name == "gru":
        item_vectors = model.embedding.weight[1:]
        for start in range(0, len(known), 128):
            batch = known[start:start + 128]
            sequences = [[item + 1 for item, _ in context.sequences[context.user_index[user]][-context.config.max_sequence:]] for user in batch]
            lengths = torch.tensor([len(sequence) for sequence in sequences])
            padded = torch.zeros((len(batch), int(lengths.max())), dtype=torch.long)
            for index, sequence in enumerate(sequences):
                padded[index, :len(sequence)] = torch.tensor(sequence)
            states = model(padded)[torch.arange(len(batch)), lengths - 1]
            scores = (states @ item_vectors.T).numpy()
            for user, row in zip(batch, scores):
                result[user] = [context.items[index] for index in top_indices(row, k, context.blocked(user))]
    else:
        raise ValueError(f"未知神经模型：{name}")
    return {user: result[user] for user in users}


def item2vec_pairs(context):
    centers, neighbors = [], []
    for sequence in context.sequences:
        for position, (item, time) in enumerate(sequence):
            left = max(0, position - context.config.context_window)
            right = min(len(sequence), position + context.config.context_window + 1)
            for other in range(left, right):
                other_item, other_time = sequence[other]
                if other != position and other_item != item and abs((time - other_time).total_seconds()) <= 180 * 86400:
                    centers.append(item)
                    neighbors.append(other_item)
    return np.asarray(centers, dtype=np.int64), np.asarray(neighbors, dtype=np.int64)


def sequence_batches(context):
    inputs, targets, user_ids = [], [], []
    for user_id, full_sequence in enumerate(context.sequences):
        sequence = full_sequence[-context.config.max_sequence - 1:]
        if len(sequence) < 2 or len(context.seen[user_id]) >= len(context.items):
            continue
        row_input = np.zeros(context.config.max_sequence, dtype=np.int64)
        row_target = np.zeros(context.config.max_sequence, dtype=np.int64)
        ids = [item + 1 for item, _ in sequence]
        row_input[:len(ids) - 1] = ids[:-1]
        row_target[:len(ids) - 1] = ids[1:]
        inputs.append(row_input)
        targets.append(row_target)
        user_ids.append(user_id)
    return np.asarray(inputs), np.asarray(targets), np.asarray(user_ids)


def fit_neural(name, context, config, validation_users=None, truth=None, k=10,
               epochs=None, pretrained=None, output_dir=None):
    offsets = {"bpr_mf": 11, "attribute_bpr": 23, "item2vec": 37, "gru": 53}
    set_seed(config.seed + offsets[name], config.threads)
    rng = np.random.default_rng(config.seed + offsets[name])
    started = perf_counter()
    if name in ("bpr_mf", "attribute_bpr"):
        model = BPRModel(context, config.dimensions, use_attributes=name == "attribute_bpr")
        maximum = epochs or config.mf_epochs
        eligible = np.asarray([index for index, user in enumerate(context.pair_users)
                               if len(context.seen[int(user)]) < len(context.items)], dtype=np.int64)
        if not len(eligible):
            raise ValueError("没有可用于 BPR 训练的正负样本。")
    elif name == "item2vec":
        model = Item2VecModel(len(context.items), config.dimensions)
        maximum = epochs or config.item2vec_epochs
        centers, neighbors = item2vec_pairs(context)
        if not len(centers):
            raise ValueError("没有可用于 Item2Vec 的历史共现对。")
        eligible = np.arange(len(centers))
    elif name == "gru":
        model = GRUSequenceModel(len(context.items), config.dimensions, pretrained)
        maximum = epochs or config.gru_epochs
        inputs, targets, sequence_users = sequence_batches(context)
        if not len(inputs):
            raise ValueError("没有长度至少为 2 的有效 GRU 训练序列。")
        eligible = np.arange(len(inputs))
    else:
        raise ValueError(f"未知模型：{name}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate if name != "gru" else config.learning_rate / 3,
                                 weight_decay=config.weight_decay)
    checkpoints = checkpoint_epochs(maximum) if truth is not None else [maximum]
    best_state, best_score, best_epoch, trace = None, -math.inf, maximum, []
    print(f"训练 {name}：{len(eligible)} 个训练样本，最多 {maximum} 轮。", flush=True)
    for epoch in range(1, maximum + 1):
        model.train()
        total_loss, batches = 0.0, 0
        order = rng.permutation(eligible)
        batch_size = min(config.batch_size, 64) if name == "gru" else config.batch_size
        for start in range(0, len(order), batch_size):
            selected = order[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            if name in ("bpr_mf", "attribute_bpr"):
                users = context.pair_users[selected]
                positives = context.pair_items[selected]
                negatives = context.sample_negatives(users, config.negatives, rng)
                user_vectors = model.encode_users(torch.from_numpy(users))
                positive_vectors = model.encode_items(torch.from_numpy(positives))
                negative_vectors = model.encode_items(torch.from_numpy(negatives))
                positive_scores = (user_vectors * positive_vectors).sum(-1)
                negative_scores = (user_vectors[:, None, :] * negative_vectors).sum(-1)
                weights = torch.from_numpy(context.pair_weights[selected])
                loss = (F.softplus(negative_scores - positive_scores[:, None]).mean(-1) * weights).sum() / weights.sum().clamp_min(1e-12)
            elif name == "item2vec":
                center = torch.from_numpy(centers[selected])
                neighbor = torch.from_numpy(neighbors[selected])
                negatives = torch.from_numpy(np.searchsorted(context.negative_cdf, rng.random((len(selected), config.negatives))).astype(np.int64))
                center_vectors = model.input_embedding(center)
                positive_scores = (center_vectors * model.output_embedding(neighbor)).sum(-1)
                negative_scores = (center_vectors[:, None, :] * model.output_embedding(negatives)).sum(-1)
                loss = -F.logsigmoid(positive_scores).mean() - F.logsigmoid(-negative_scores).sum(-1).mean()
            else:
                batch_input = torch.from_numpy(inputs[selected])
                batch_target = torch.from_numpy(targets[selected])
                mask = batch_target.ne(0)
                states = model(batch_input)[mask]
                positive_ids = batch_target[mask]
                user_ids = np.broadcast_to(sequence_users[selected, None], batch_target.shape)[mask.numpy()]
                negative_ids = torch.from_numpy(context.sample_negatives(user_ids, config.negatives, rng) + 1)
                positive_scores = (states * model.embedding(positive_ids)).sum(-1)
                negative_scores = (states[:, None, :] * model.embedding(negative_ids)).sum(-1)
                loss = F.softplus(negative_scores - positive_scores[:, None]).mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"{name} 第 {epoch} 轮损失非有限。")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach())
            batches += 1
        row = {"epoch": epoch, "loss": total_loss / max(batches, 1)}
        if epoch in checkpoints:
            if truth is not None:
                rankings = neural_rankings(name, model, context, validation_users, config.candidate_k)
                score = validation_score(rankings, truth, k)
                row["validation_score"] = score
                if score > best_score + 1e-12:
                    best_score, best_epoch, best_state = score, epoch, copy.deepcopy(model.state_dict())
            else:
                best_state = copy.deepcopy(model.state_dict())
            print(f"{name} epoch={epoch:2d} loss={row['loss']:.5f} "
                  + (f"validation_score={row['validation_score']:.6f}" if truth is not None else ""), flush=True)
        elif epoch % 5 == 0:
            print(f"{name} epoch={epoch:2d} loss={row['loss']:.5f}", flush=True)
        trace.append(row)
    model.load_state_dict(best_state)
    model.eval()
    summary = {"selected_epoch": best_epoch, "selected_by": "validation only" if truth is not None else "frozen plan",
               "training_samples": len(eligible), "trace": trace, "elapsed_seconds": round(perf_counter() - started, 3)}
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "name": name, "epoch": best_epoch,
                    "config": asdict(config), "items": context.items, "users": context.users}, output_dir / f"{name}.pt")
    return model, summary


class ContentModel:
    """历史题名片段、作者和分类的 TF-IDF 内容分支，没有外部预训练语料。"""

    def __init__(self, context):
        self.context = context
        raw = []
        frequencies = Counter()
        for item in context.items:
            meta = context.metadata.get(item, {})
            features = {term: 1.0 for term in title_terms(meta.get("题名", ""))}
            if meta.get("题名"):
                features["题名:" + meta["题名"]] = 2.0
            if meta.get("责任者"):
                features["作者:" + meta["责任者"]] = 0.7
            prefix = clc_prefix(meta.get("索书号", ""))
            if prefix:
                features["分类:" + prefix] = 0.7
            frequencies.update(features.keys())
            raw.append(features)
        self.vectors, inverted = [], defaultdict(list)
        for index, features in enumerate(raw):
            vector = {term: value * (math.log((1 + len(raw)) / (1 + frequencies[term])) + 1)
                      for term, value in features.items()}
            norm = math.sqrt(sum(value * value for value in vector.values())) or 1.0
            vector = {term: value / norm for term, value in vector.items()}
            self.vectors.append(vector)
            for term, value in vector.items():
                inverted[term].append((index, value))
        self.inverted = {term: (np.asarray([item for item, _ in rows]), np.asarray([value for _, value in rows]))
                         for term, rows in inverted.items()}

    def rank(self, users, k):
        context, result = self.context, {}
        for user in users:
            if user not in context.recent:
                result[user] = context.fallback(user, k)
                continue
            ids, weights = context.recent[user]
            query = defaultdict(float)
            for item, weight in zip(ids, weights):
                for term, value in self.vectors[item].items():
                    query[term] += float(weight) * value
            scores = np.zeros(len(context.items), dtype=np.float64)
            for term, value in query.items():
                target_ids, target_values = self.inverted[term]
                scores[target_ids] += value * target_values
            # 零内容关联时用很小的历史热门度破平局，不引入未来书目。
            scores += np.asarray([context.baseline.popularity.get(item, 0.0) for item in context.items]) * 1e-8
            result[user] = [context.items[index] for index in top_indices(scores, k, context.blocked(user))]
        return result
