"""第三轮优化：候选/标签隔离、历史快照训练、排序导出与预训练缓存测试。"""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from advanced_models import AdvancedConfig, HistoryContext, set_seed
from baseline import ITEM, TIME, USER, ground_truth
from ranking import atomic_json, plans, strategy_predictions
from ranking_models import (CAT_FEATURES, OLD_CHANNELS, CandidateFeatures, book_text, candidate_coverage,
                            feature_columns, fit_ranker, labeled, rank_candidates, training_data)


class FakeEncoder:
    """仅测试用确定性向量；正式流程必须加载官方权重，无静默降级。"""
    def encode(self, texts):
        values = np.asarray([list(hashlib.sha256(t.encode()).digest()[:16]) for t in texts], dtype=np.float32)
        return values / np.linalg.norm(values, axis=1, keepdims=True)


def fixture():
    rows = []
    for user in range(12):
        for j in range(3):
            item = (user * 2 + j) % 24
            rows.append({USER: f"u{user:02}", ITEM: f"00{item:02}", TIME: pd.Timestamp("2023-01-01") + pd.Timedelta(days=user+j),
                         "题名": f"计算机图书{item}", "责任者": f"作者{item%3}", "索书号": f"TP{item%3}",
                         "专业": "计算机", "系别": "计算机学院", "年级": "2022"})
    return pd.DataFrame(rows)


def build(exclude_seen=False):
    c = HistoryContext(fixture(), "2023-03-01", AdvancedConfig(threads=1, exclude_seen=exclude_seen))
    users = c.users + ["new"]
    channels = {n: c.baseline_rankings(users, 15) for n in OLD_CHANNELS}
    features, new = CandidateFeatures(c, FakeEncoder()).build(users, channels, per_channel=15, new_k=5)
    return c, channels, features, new


class RankingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_seed(2026, 1)

    def test_metadata_and_candidates_never_introduce_future_barcode(self):
        c, _, f, new = build()
        self.assertTrue(set(f[ITEM]) <= set(c.items))
        self.assertFalse(f.duplicated([USER, ITEM]).any())
        for route in new.values():
            for u, items in route.items():
                seen = c.seen[c.user_index[u]] if u in c.user_index else set()
                self.assertFalse(set(items) & {c.items[i] for i in seen})
        truth = {"new": {"FUTURE_ID"}}
        tagged = labeled(f, truth, "2023-03-01", "2023-05-01")
        pd.testing.assert_frame_equal(tagged[f.columns], f)
        self.assertEqual(tagged["label"].sum(), 0)

    def test_history_rejects_cutoff_event(self):
        data = fixture()
        data.loc[0, TIME] = pd.Timestamp("2023-03-01")
        with self.assertRaises(ValueError):
            HistoryContext(data, "2023-03-01", AdvancedConfig())

    def test_exclude_seen_applies_to_all_candidates(self):
        c, _, f, _ = build(True)
        for u, rows in f.groupby(USER):
            seen = c.seen[c.user_index[u]] if u in c.user_index else set()
            self.assertFalse(set(rows[ITEM]) & {c.items[i] for i in seen})

    def test_features_have_no_labels_or_identifiers(self):
        _, _, f, _ = build()
        for expanded in (False, True):
            cols = feature_columns(labeled(f, {}, "2023-03-01", "2023-05-01"), expanded)
            self.assertFalse(set(cols) & {USER, ITEM, "query", "label", "existing", "warm"})
            self.assertTrue(set(CAT_FEATURES) <= set(cols))
        self.assertNotIn("rr_semantic", feature_columns(f, False))
        self.assertIn("rr_semantic", feature_columns(f, True))
        nums = f.select_dtypes(include="number")
        self.assertTrue(np.isfinite(nums.to_numpy()).all())

    def test_training_uses_only_completed_prior_labels(self):
        _, _, f, _ = build()
        truth = {u: {rows.iloc[0][ITEM]} for u, rows in f.groupby(USER)}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "features.pkl"
            labeled(f, truth, "2023-03-01", "2023-05-01").to_pickle(path)
            snaps = [{"start": "2023-03-01", "end": "2023-05-01", "path": str(path)}]
            with self.assertRaises(ValueError):
                training_data(snaps, "2023-04-30", True)
            train, used = training_data(snaps, "2023-05-01", True)
            self.assertTrue(train["warm"].eq(1).all())
            self.assertEqual(used[0]["end"], "2023-05-01")
            self.assertTrue(train["query"].is_monotonic_increasing)

    def test_missing_candidate_positives_are_not_injected(self):
        _, _, f, _ = build()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "features.pkl"
            labeled(f, {u: {"not_a_candidate"} for u in f[USER].unique()},
                    "2023-03-01", "2023-05-01").to_pickle(path)
            with self.assertRaises(ValueError):
                training_data([{"start": "2023-03-01", "end": "2023-05-01", "path": str(path)}], "2023-06-01", True)

    def test_candidate_coverage_keeps_unreachable_truth_in_denominator(self):
        _, _, f, _ = build()
        u = f.iloc[0][USER]
        item = f.loc[f[USER].eq(u)].iloc[0][ITEM]
        coverage = candidate_coverage(f, {u: {item, "unreachable"}})
        self.assertEqual(coverage["expanded"]["user_macro_recall"], .5)
        self.assertGreaterEqual(coverage["expanded"]["candidate_hit_pairs"],
                                coverage["existing"]["candidate_hit_pairs"])

    def test_cold_profile_does_not_invent_personal_history(self):
        _, _, f, _ = build()
        cold = f.loc[f[USER].eq("new")]
        self.assertTrue(cold["warm"].eq(0).all())
        self.assertTrue(cold["seen"].eq(0).all())
        self.assertTrue(cold["semantic_max"].eq(0).all())
        self.assertTrue(cold["major"].eq("未知").all())

    def test_book_text_preserves_volume_and_author(self):
        self.assertNotEqual(book_text({"题名": "算法 第1卷"}), book_text({"题名": "算法 第2卷"}))
        self.assertNotEqual(book_text({"题名": "算法", "责任者": "甲"}), book_text({"题名": "算法", "责任者": "乙"}))

    def test_ranker_roundtrip_and_cold_fallback(self):
        from catboost import CatBoostRanker
        _, raw, f, new = build()
        truth = {u: {rows.loc[rows["seen"].eq(1)].iloc[0][ITEM]} for u, rows in f.groupby(USER) if u != "new"}
        train = labeled(f.loc[f["warm"].eq(1)], truth, "2023-03-01", "2023-05-01")
        train = train.sort_values(["query", ITEM])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.cbm"
            model, cols = fit_ranker(train, True, path, iterations=10)
            output = rank_candidates(model, cols, f, raw["optimized"], True, trees=10)
            restored = CatBoostRanker()
            restored.load_model(str(path))
            self.assertEqual(output, rank_candidates(restored, cols, f, raw["optimized"], True, trees=10))
            self.assertEqual(output["new"], raw["optimized"]["new"][:10])
            self.assertTrue(all(len(v) == len(set(v)) == 10 for v in output.values()))
            raw.update(new)
            recipe = {"kind": "ranker", "expanded": True, "trees": 10, "weight": .5}
            blended = strategy_predictions(raw, {(True, 10): output}, recipe)
            self.assertEqual(blended["new"], raw["optimized"]["new"][:10])

    def test_atomic_progress_and_predeclared_trials(self):
        recipes = plans()
        self.assertEqual(len({r["name"] for r in recipes}), len(recipes))
        self.assertEqual(len(recipes), 18)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "progress.json"
            atomic_json(path, {"stage": "finished"})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"stage": "finished"})
            self.assertFalse(path.with_suffix(".json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
