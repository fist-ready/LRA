"""按索书号重放冻结开发快照。仅使用各窗口之前的候选和画像。"""
import json
import math
from collections import Counter
from pathlib import Path

import pandas as pd
from catboost import CatBoostRanker

from baseline import ITEM, TIME, USER, attribute_value, load_interactions
from ranking import DEVELOPMENT, rank_candidates

BASE = Path(__file__).resolve().parent
CALL = "索书号"
GROUP_GAP = pd.Timestamp("2025-09-01") - pd.Timestamp("2025-01-17")


def unique_new(items, seen):
    return list(dict.fromkeys(item for item in items if item and item not in seen))


def popularity(frame):
    counts = frame.drop_duplicates([USER, CALL])[CALL].value_counts().to_dict()
    return sorted(counts, key=lambda item: (-counts[item], item))


def fuse(lists):
    scores = Counter()
    for items in lists:
        for rank, item in enumerate(items, 1):
            scores[item] += 1 / (20 + rank)
    return sorted(scores, key=lambda item: (-scores[item], item))


def metrics(pred, truth):
    hits = [i for i, item in enumerate(pred, 1) if item in truth]
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(10, len(truth)) + 1))
    return dict(hit=int(bool(hits)), recall=len(hits) / len(truth),
                ndcg=sum(1 / math.log2(i + 1) for i in hits) / ideal,
                precision=len(hits) / 10)


def frozen_rankings(directory):
    def read(name):
        return json.loads((directory / name).read_text(encoding="utf8"))

    raw = read("channels.json")
    model = CatBoostRanker()
    model.load_model(str(directory / "expanded.cbm"))
    features = pd.read_pickle(directory / "features.pkl")
    ranks = rank_candidates(model, read("expanded.features.json")["columns"], features,
                            raw["optimized"], True, trees=100, k=200)
    return raw, ranks


def replay(data, date):
    directory = BASE / "outputs/ranking/snapshots" / date
    def read(name):
        return json.loads((directory / name).read_text(encoding="utf8"))

    start = pd.Timestamp(date)
    manifest = read("manifest.json")
    assert pd.Timestamp(manifest["start"]) == start and pd.Timestamp(manifest["history_max"]) < start
    history = data.loc[data[TIME] < start]
    valid = history.loc[history[CALL].ne("")]
    target = data.loc[data[TIME].between(start, pd.Timestamp(manifest["end"]), inclusive="left")]
    seen = valid.groupby(USER)[CALL].agg(set).to_dict()
    future = target.loc[target[CALL].ne("")].groupby(USER)[CALL].agg(set).to_dict()
    old = set(history.loc[history[TIME] < start - GROUP_GAP, USER])
    known_users = set(history[USER])
    department = history.loc[history["单位"].ne("")].groupby(USER)["单位"].last().to_dict()
    by_department = {d: popularity(g) for d, g in valid.groupby("单位") if d}
    popular = popularity(valid)
    catalog = set(popular)
    mapping = valid.drop_duplicates(ITEM, keep="last").set_index(ITEM)[CALL].to_dict()

    raw, ranks = frozen_rankings(directory)
    rows = []
    for user, answers in future.items():
        borrowed = seen.get(user, set())
        truth = answers - borrowed
        if not truth:
            continue
        group = "老人" if user in old else "新人"
        fallback = unique_new(popular, borrowed)[:10]
        college = by_department.get(department.get(user), [])
        def calls(channel):
            return unique_new((mapping.get(item) for item in channel[user]), borrowed)
        channels = [calls(raw["optimized"]), calls(ranks)]
        strategies = {"全局热门": fallback, "同学院热门": college,
                      "冻结模型适配": fuse(channels),
                      "适配加语义": fuse([*channels, calls(raw["semantic"])])}
        for name, pool in strategies.items():
            top = unique_new(pool, borrowed)[:10]
            top = unique_new(top + fallback, borrowed)[:10]
            assert len(top) == len(set(top)) == 10 and set(top) <= catalog - borrowed
            rows.append(dict(window=date, method=name, user=user, group=group, history=user in known_users,
                             truth=len(truth), catalog_hits=len(truth & catalog),
                             pool_hits=len(truth.intersection(pool) - borrowed), pred="|".join(top),
                             answers="|".join(sorted(truth)), **metrics(top, truth)))
    audit = dict(window=date, end=manifest["end"], group_boundary=str((start-GROUP_GAP).date()),
                 future_users=target[USER].nunique(), eligible=len(rows)//4,
                 only_missing=target[USER].nunique()-len(future),
                 only_repeat=sum(not (a-seen.get(u,set())) for u,a in future.items()),
                 historical_catalog=len(catalog))
    return rows, audit


def main():
    data = load_interactions(BASE / "data/processed/borrowing_cleaned.csv")
    for col in [CALL, "单位"]:
        data[col] = data[col].map(attribute_value)
    directory = BASE / "outputs/call_number_replay"
    directory.mkdir(parents=True, exist_ok=True)
    all_rows, audit = [], []
    for date in DEVELOPMENT:
        rows, counts = replay(data, date)
        all_rows.extend(rows)
        audit.append(counts)
    frame = pd.DataFrame(all_rows)
    frame.to_csv(directory / "per_user.csv", index=False, encoding="utf-8-sig")
    keys = ["window","method","group"]
    grouped = frame.groupby(keys)
    summary = grouped[["hit","recall","ndcg","precision"]].mean()
    summary["users"] = grouped.size()
    summary["candidate_pair_recall"] = grouped.pool_hits.sum() / grouped.truth.sum()
    summary["catalog_pair_recall"] = grouped.catalog_hits.sum() / grouped.truth.sum()
    summary.to_csv(directory / "groups.csv", encoding="utf-8-sig")
    new, old = (summary.xs(g, level="group") for g in ("新人","老人"))
    scores = 40*(0.5*new.hit + 0.5*new.recall) + 60*(0.4*old.hit + 0.3*old.recall + 0.3*old.ndcg)
    scores.rename("score_100").to_csv(directory / "scores.csv", encoding="utf-8-sig")
    pd.DataFrame(audit).to_csv(directory / "windows.csv", index=False, encoding="utf-8-sig")
    print(scores.unstack().round(4))


if __name__ == "__main__":
    main()

