"""验证时间边界、信息隔离、排序指标与预测接口，不依赖额外测试框架。"""

import json
import math
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from baseline import (
    ITEM, TIME, USER, ItemCFBaseline, ModelConfig, evaluate_rankings, ground_truth,
    load_interactions, prepare_events, ranking_metrics, run_evaluation, run_prediction,
    target_diagnostics, temporal_split,
)


def events(rows):
    return prepare_events(pd.DataFrame(rows, columns=[USER, ITEM, TIME, "专业"]))


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.history = events([
            ("001", "A", "2024-01-01", "计算机"), ("001", "B", "2024-01-02", "计算机"),
            ("002", "A", "2024-01-03", "计算机"), ("002", "C", "2024-01-04", "计算机"),
            ("003", "D", "2024-01-05", "数学"), ("003", "E", "2024-01-06", "数学"),
        ])
        self.model = ItemCFBaseline(ModelConfig(group_min_users=1)).fit(self.history, "2024-02-01")

    def test_temporal_boundaries_and_no_userwise_shuffle(self):
        data = events([
            ("u", "a", "2024-08-31 23:59:59", "x"),
            ("u", "b", "2024-09-01 00:00:00", "x"),
            ("u", "c", "2024-11-01 00:00:00", "x"),
            ("u", "d", "2025-01-01 00:00:00", "x"),
        ])
        train, valid, test = temporal_split(data, test_end="2025-01-01")
        self.assertEqual([list(frame[ITEM]) for frame in (train, valid, test)], [["a"], ["b"], ["c"]])

    def test_invalid_or_empty_split_fails(self):
        with self.assertRaises(ValueError):
            temporal_split(self.history, "2024-02-01", "2024-01-01")
        with self.assertRaises(ValueError):
            temporal_split(self.history)

    def test_ndcg_recall_and_weighted_score_by_hand(self):
        result = ranking_metrics(["miss", "a", "b"], {"a", "b", "c"}, k=2)
        expected_ndcg = (1 / math.log2(3)) / (1 + 1 / math.log2(3))
        self.assertAlmostEqual(result["recall@2"], 1 / 3)
        self.assertAlmostEqual(result["ndcg@2"], expected_ndcg)
        self.assertAlmostEqual(result["score"], 0.6 / 3 + 0.4 * expected_ndcg)

    def test_recall_denominator_is_not_capped_at_k(self):
        result = ranking_metrics([str(i) for i in range(10)], set(map(str, range(20))))
        self.assertEqual(result["recall@10"], 0.5)
        self.assertEqual(result["ndcg@10"], 1.0)

    def test_duplicate_recommendations_and_empty_truth_fail(self):
        with self.assertRaises(ValueError):
            ranking_metrics(["a", "a"], {"a"})
        with self.assertRaises(ValueError):
            ranking_metrics(["a"], set())

    def test_truth_deduplicates_repeat_events(self):
        repeated = pd.concat([self.history, self.history.iloc[[0]]], ignore_index=True)
        self.assertEqual(ground_truth(repeated)["001"], {"A", "B"})

    def test_future_records_are_rejected_by_fit(self):
        with self.assertRaises(ValueError):
            ItemCFBaseline().fit(self.history, as_of="2024-01-04")

    def test_future_items_and_attributes_do_not_enter_model(self):
        original = self.model.recommend("001", k=3)
        future = events([("001", "FUTURE", "2024-02-03", "未来专业")])
        diagnostics = target_diagnostics(self.history, future, 3)
        self.assertEqual(diagnostics["new_item_truth_pairs"], 1)
        self.assertEqual(diagnostics["oracle_recall@3"], 0)
        self.assertNotIn("FUTURE", self.model.items)
        self.assertEqual(self.model.profiles["001"]["专业"], "计算机")
        self.assertEqual(original, self.model.recommend("001", k=3))

    def test_popularity_counts_unique_readers_not_repeat_frequency(self):
        repeated = pd.concat([self.history, self.history.iloc[[0]]], ignore_index=True)
        other = ItemCFBaseline(self.model.config).fit(repeated, "2024-02-01")
        self.assertEqual(self.model.popularity, other.popularity)
        self.assertEqual(self.model.similarities, other.similarities)

    def test_itemcf_cosine_iuf_shrinkage_and_no_diagonal(self):
        weight = (1 / math.log(3)) * 2 ** (-1 / 180)
        expected = weight / math.sqrt(2 * 1) * (1 / 11)
        self.assertAlmostEqual(dict(self.model.similarities["A"])["B"], expected)
        for item, neighbors in self.model.similarities.items():
            self.assertNotIn(item, dict(neighbors))

    def test_cold_fallback_and_reproducible_unique_catalog_output(self):
        popular = self.model.recommend("new", k=5, method="popular")
        for method in ("itemcf", "hybrid"):
            result = self.model.recommend("new", k=5, method=method)
            self.assertEqual(result, popular)
            self.assertEqual(len(result), len(set(result)))
            self.assertTrue(set(result) <= self.model.items)

    def test_seen_exclusion_does_not_pad_with_seen_or_unknown_items(self):
        model = ItemCFBaseline(ModelConfig(exclude_seen=True)).fit(self.history, "2024-02-01")
        for method in ("popular", "itemcf", "hybrid"):
            result = model.recommend("001", k=10, method=method)
            self.assertEqual(set(result), {"C", "D", "E"})

    def test_macro_metrics_include_cold_users_and_unseen_items(self):
        truth = {"001": {"A"}, "cold": {"unseen"}}
        metrics, _ = evaluate_rankings({"001": ["A"]}, truth, self.model, k=1)
        self.assertEqual(metrics["all"]["users"], 2)
        self.assertEqual(metrics["all"]["recall@1"], 0.5)
        self.assertEqual(metrics["cold"]["recall@1"], 0)

    def test_refit_resets_old_catalog_and_profiles(self):
        self.model.fit(events([("new", "only", "2024-03-01", "新专业")]), "2024-04-01")
        self.assertEqual(self.model.items, {"only"})
        self.assertNotIn("001", self.model.seen)
        self.assertNotIn("001", self.model.profiles)

    def test_max_history_only_caps_graph_not_exclusion_or_catalog(self):
        model = ItemCFBaseline(ModelConfig(max_history=1, exclude_seen=True)).fit(self.history, "2024-02-01")
        self.assertEqual(set(model.history["001"]), {"B"})
        self.assertEqual(model.seen["001"], {"A", "B"})
        self.assertEqual(model.items, {"A", "B", "C", "D", "E"})

    def test_csv_preserves_leading_zero_identifiers_and_rejects_bad_rows(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            path = Path(directory) / "events.csv"
            self.history.to_csv(path, index=False, encoding="utf-8-sig")
            self.assertIn("001", set(load_interactions(path)[USER]))
        invalid = self.history.copy()
        invalid.loc[0, TIME] = pd.NaT
        with self.assertRaises(ValueError):
            prepare_events(invalid)

    def test_validation_selection_refit_and_auto_prediction_end_to_end(self):
        data = pd.concat([
            self.history,
            events([("001", "C", "2024-09-01", "计算机"), ("new", "Z", "2024-09-02", "未来专业")]),
            events([("001", "Z", "2024-11-01", "计算机"), ("cold", "FUTURE", "2024-11-02", "未来专业")]),
        ], ignore_index=True)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            base = Path(directory)
            path = base / "input.csv"
            data.to_csv(path, index=False, encoding="utf-8-sig")
            report = run_evaluation(path, base / "evaluation", k=3)
            methods = report["stages"]["validation"]["methods"]
            selected = report["selection"]["method"]
            self.assertEqual(methods[selected]["all"]["score"], max(row["all"]["score"] for row in methods.values()))
            self.assertEqual(report["stages"]["test"]["training"]["events"], 8)
            self.assertEqual(report["stages"]["test"]["training"]["items"], 6)
            predictions = run_prediction(path, base / "prediction", selection_report=base / "evaluation/evaluation_report.json", k=3)
            self.assertEqual(len(predictions), 5)
            saved = pd.read_csv(base / "prediction/result.csv", dtype=str)
            self.assertEqual(saved.shape, (5, 4))
            metadata = json.loads((base / "prediction/prediction_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["training"]["events"], 10)
            self.assertEqual(metadata["method"], selected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
