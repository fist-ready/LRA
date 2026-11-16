"""在已有索书号回放的相同读者和答案上比较固定 100 个候选。"""
import argparse

import numpy as np
import pandas as pd

from baseline import ITEM, TIME, USER, attribute_value, load_interactions
from call_number_replay import BASE, CALL, DEVELOPMENT, frozen_rankings, fuse, metrics, replay, unique_new
from older_call_recall import OlderCallRecall, WEIGHTS, blend

LIMIT = 100


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic", action="store_true", help="仅运行复用诊断窗口")
    diagnostic = parser.parse_args().diagnostic
    data = load_interactions(BASE / "data/processed/borrowing_cleaned.csv")
    for key in (CALL, "单位"):
        data[key] = data[key].map(attribute_value)
    windows = ["2024-11-01"] if diagnostic else DEVELOPMENT
    if diagnostic:
        source = pd.DataFrame(replay(data, windows[0])[0])
    else:
        source = pd.read_csv(BASE / "outputs/call_number_replay/per_user.csv", dtype={"user": str})
    ref = source.loc[source.method.eq("冻结模型适配")]
    rows = []
    for date in windows:
        cutoff = pd.Timestamp(date)
        history = data.loc[data[TIME] < cutoff]
        index = OlderCallRecall(history, cutoff)
        depart = history.loc[history["单位"].ne("")].groupby(USER)["单位"].last().to_dict()
        mapping = history.loc[history[CALL].ne("")].drop_duplicates(ITEM, keep="last").set_index(ITEM)[CALL].to_dict()
        raw, ranks = frozen_rankings(BASE / "outputs/ranking/snapshots" / date)
        global_pop = sorted(index.readers, key=lambda x: (-len(index.readers[x]), x))
        for row in ref.loc[ref.window.eq(date) & ref.group.eq("老人")].itertuples():
            user = row.user
            seq = index.profile.get(user, [])
            seen = set(seq)
            channels = index.channels(seq, depart.get(user, ""))
            previous = fuse([unique_new((mapping.get(i) for i in source_list[user]), seen)
                             for source_list in (raw["optimized"], ranks)])
            fallback = unique_new(global_pop, seen)[:LIMIT]
            channels["old"] = unique_new(previous + fallback, seen)[:LIMIT]
            assert channels["old"][:10] == row.pred.split("|")
            strategies = {"冻结模型": channels["old"], "全局热门": fallback,
                          "近期热门与学院": blend(channels, {"hot":1, "college":.5}),
                          "增加共现": blend(channels, {"hot":1, "college":.5, "cf":.75}),
                          "增加共现和内容": blend(channels, WEIGHTS),
                          "再融合冻结模型": blend(channels, dict(WEIGHTS, old=1.))}
            truth = set(row.answers.split("|"))
            for method, pool in strategies.items():
                assert len(set(pool)) == len(pool) == LIMIT and not seen.intersection(pool)
                assert all(i in index.readers for i in pool)
                values = metrics(pool[:10], truth)
                rows.append(dict(window=date, method=method, user=user, truth=len(truth),
                                 pool_hits=len(truth.intersection(pool)), pred="|".join(pool[:10]),
                                 score=100*(.4*values["hit"]+.3*values["recall"]+.3*values["ndcg"]), **values))
    out = BASE / "outputs/older_call_recall"
    if diagnostic:
        out = out / "reused_diagnostic"
    out.mkdir(exist_ok=True, parents=True)
    details = pd.DataFrame(rows)
    details.to_csv(out/"per_user.csv", index=False, encoding="utf-8-sig")
    groups = details.groupby(["window","method"])
    summary = groups[["hit","recall","ndcg","score"]].mean()
    summary["pair_recall_100"] = groups.pool_hits.sum()/groups.truth.sum()
    summary["users"] = groups.size()
    cold = ref.loc[ref.group.eq("新人")].groupby("window")[["hit","recall"]].mean()
    summary["total_100"] = .6*summary.score + summary.index.get_level_values("window").map(20*(cold.hit+cold.recall))
    summary.to_csv(out/"summary.csv", encoding="utf-8-sig")
    print(summary.round(4).to_string())
    # 按读者成簇重采样，同一读者跨窗口共享一个权重；窗口仍然等权。
    pivot = details.pivot(index=["window","user"],columns="method",values="score")
    delta = (pivot["增加共现和内容"]-pivot["冻结模型"]).unstack("window")
    rng = np.random.default_rng(20261105)
    draws = [delta.iloc[rng.integers(len(delta),size=len(delta))].mean().mean() for _ in range(2000)]
    print("完整方案减冻结模型，老人分差及95%区间",delta.mean().mean(),np.quantile(draws,[.025,.975]))


if __name__ == "__main__":
    main()

