"""从导出的推荐文件独立重算第三轮指标，并核验历史候选与训练时间边界。"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from baseline import BASE, ITEM, TIME, USER, ground_truth, load_interactions, ranking_metrics, source_info, write_json


def verify(directory, prediction_directory=None):
    directory = Path(directory)
    report = json.loads((directory / "experiment_report.json").read_text(encoding="utf-8"))
    plan = json.loads((directory / "locked_plan.json").read_text(encoding="utf-8"))
    source = BASE / "data/2022-2025处理后借阅信息.xlsx"
    assert source_info(source)["sha256"] == report["input"]["sha256"] == plan["input"]["sha256"]
    assert max(report["trials"], key=lambda row: row["mean_score"])["mean_score"] == plan["selection"]["mean_score"]
    assert report["selection"]["recipe"] == plan["recipe"] == plan["selection"]["recipe"]
    data = load_interactions(source)
    verified_snapshots = 0
    manifests = {s["start"]: s for s in plan["training_snapshots"]}
    diagnostic_path = directory / "snapshots" / report["diagnostic"]["start"][:10] / "manifest.json"
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    manifests[diagnostic["start"]] = diagnostic
    for manifest in manifests.values():
        start, end = pd.Timestamp(manifest["start"]), pd.Timestamp(manifest["end"])
        assert manifest["identity"] == plan["identity"] and start < end
        assert pd.Timestamp(manifest["history_max"]) < start
        catalog = set(data.loc[data[TIME].lt(start), ITEM])
        truth = ground_truth(data.loc[data[TIME].ge(start) & data[TIME].lt(end)])
        features = pd.read_pickle(manifest["path"])
        assert len(features) == manifest["rows"] and not features.duplicated([USER, ITEM]).any()
        assert set(features[ITEM]) <= catalog and set(features[USER]) == set(truth)
        expected_labels = np.asarray([int(item in truth.get(user, set()))
                                      for user, item in features[[USER, ITEM]].itertuples(index=False, name=None)])
        assert np.array_equal(features["label"].to_numpy(), expected_labels)
        verified_snapshots += 1
    verified_files = 0
    for fold in [*report["development"], report["diagnostic"]]:
        start, end = pd.Timestamp(fold["start"]), pd.Timestamp(fold["end"])
        history = data.loc[data[TIME].lt(start)]
        catalog = set(history[ITEM])
        truth = ground_truth(data.loc[data[TIME].ge(start) & data[TIME].lt(end)])
        folder = directory / "snapshots" / str(start)[:10]
        for training in fold["training"].values():
            assert all(pd.Timestamp(s["end"]) <= start and pd.Timestamp(s["start"]) < start
                       for s in training["snapshots"])
        for method, metrics in fold["methods"].items():
            rec = pd.read_csv(folder / (method + ".csv"), dtype=str, keep_default_na=False)
            assert len(rec.columns) == 11 and rec[USER].is_unique and set(rec[USER]) == set(truth)
            scores = []
            for row in rec.itertuples(index=False, name=None):
                user, items = row[0], list(row[1:])
                assert len(items) == len(set(items)) == 10 and set(items) <= catalog and "" not in items
                scores.append(ranking_metrics(items, truth[user], 10))
            for key in ("recall@10", "ndcg@10", "score"):
                assert abs(np.mean([s[key] for s in scores]) - metrics["all"][key]) < 1e-12
            verified_files += 1
    reference_exact = all(abs(v) < 1e-12 for row in report["reference_deltas"].values() for v in row.values())
    assert reference_exact
    summary = {"source_unchanged": True, "files_recomputed": verified_files,
               "snapshots_verified": verified_snapshots,
               "historical_candidates_only": True, "training_labels_end_before_prediction": True,
               "reference_exact": reference_exact}
    if prediction_directory:
        prediction_directory = Path(prediction_directory)
        meta = json.loads((prediction_directory / "prediction_summary.json").read_text(encoding="utf-8"))
        assert meta["input"]["sha256"] == report["input"]["sha256"]
        assert meta["recipe"] == plan["recipe"]
        history = data.loc[data[TIME].lt(pd.Timestamp(meta["cutoff"]))]
        catalog = set(history[ITEM])
        rec = pd.read_csv(prediction_directory / "result.csv", dtype=str, keep_default_na=False)
        assert rec[USER].is_unique and len(rec) == meta["readers"]
        for row in rec.itertuples(index=False, name=None):
            assert len(row) == 11 and len(set(row[1:])) == 10 and set(row[1:]) <= catalog
        if meta["recipe"]["kind"] == "ranker":
            training = json.loads((prediction_directory / "ranker_training.json").read_text(encoding="utf-8"))
            assert training and all(pd.Timestamp(s["end"]) <= pd.Timestamp(meta["cutoff"]) for s in training)
            assert all(s["start"] in manifests and s["end"] == manifests[s["start"]]["end"] for s in training)
            summary["prediction_training_windows"] = len(training)
        summary["prediction_readers"] = len(rec)
    write_json(directory / "verification_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(BASE / "outputs/ranking"))
    parser.add_argument("--prediction-dir")
    args = parser.parse_args()
    verify(args.output_dir, args.prediction_dir)
