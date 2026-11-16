"""验证精确线性求解、三步游走、画像时间、复借探索和冻结后预测。"""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from advanced import CHANNELS, FusionSpace
from advanced_models import AdvancedConfig, HistoryContext
from baseline import ITEM, TIME, USER, save_csv, write_json
from optimized import CORE_CHANNELS, FAMILIES, check_windows, predict_plan, previous_space, run_evaluation, run_prediction
from optimized_models import (
    AttributePrior, EASEModel, EntityMatrix, OptimizationConfig, RepeatExploreSpace,
    RepeatGate, RP3Model, no_attribute_baseline, study_stage,
)


def fixture():
    rows = []
    for user in range(6):
        for offset in range(3):
            book = (user + offset) % 10
            rows.append({USER: f"00{user}", ITEM: f"000{book}", TIME: f"2024-01-{2 + user * 3 + offset:02d}",
                         "题名": f"图书{book}", "责任者": f"作者{book}", "索书号": f"TP3{book}",
                         "专业": "计算机" if user < 5 else "数学", "系别": "计算机学院", "年级": "2023"})
    rows.extend([{USER: "000", ITEM: "0008", TIME: "2024-02-01", "题名": "复本书 第一卷", "责任者": "甲",
                  "专业": "计算机", "系别": "计算机学院", "年级": "2023"},
                 {USER: "001", ITEM: "0009", TIME: "2024-02-02", "题名": "复本书 第一卷", "责任者": "甲",
                  "专业": "计算机", "系别": "计算机学院", "年级": "2023"}])
    data = pd.DataFrame(rows)
    data[TIME] = pd.to_datetime(data[TIME])
    return data


def context(data=None, cutoff="2024-03-01", exclude_seen=False):
    return HistoryContext(fixture() if data is None else data, cutoff,
                          AdvancedConfig(dimensions=4, mf_epochs=1, item2vec_epochs=1, gru_epochs=1,
                                         negatives=1, candidate_k=6, threads=1, exclude_seen=exclude_seen))


class OptimizedTests(unittest.TestCase):
    def test_woodbury_matches_dense_ease_including_zero_diagonal(self):
        matrix = EntityMatrix(context())
        model = EASEModel(matrix, regularization=10)
        x = matrix.x.toarray()
        precision = np.linalg.inv(x.T @ x + 10 * np.eye(x.shape[1]))
        coefficients = -precision / np.diag(precision)
        np.fill_diagonal(coefficients, 0)
        query = matrix.query[[0, 1, 2]]
        np.testing.assert_allclose(model.scores(query), query.toarray() @ coefficients, rtol=1e-11, atol=1e-11)

    def test_ease_no_self_reconstruction_with_disjoint_books(self):
        data = pd.DataFrame([{USER: f"0{i}", ITEM: f"00{i}", TIME: pd.Timestamp("2024-01-01")}
                             for i in range(4)])
        matrix = EntityMatrix(context(data))
        model = EASEModel(matrix, regularization=3)
        np.testing.assert_allclose(model.scores(matrix.query), 0, atol=1e-12)

    def test_ease_user_memory_guard_and_parameter_validation(self):
        matrix = EntityMatrix(context())
        with self.assertRaises(ValueError):
            EASEModel(matrix, -1)
        with self.assertRaises(ValueError):
            EASEModel(matrix, 10, max_users=1)

    def test_graph_matches_three_step_transition_and_degree_penalty(self):
        matrix = EntityMatrix(context())
        x = matrix.x.toarray()
        expected = (x.T / x.sum(axis=0)[:, None]) @ (x / x.sum(axis=1)[:, None])
        expected /= x.sum(axis=0)[None, :] ** 0.5
        np.fill_diagonal(expected, 0)
        model = RP3Model(matrix, beta=0.5, topk=100)
        np.testing.assert_allclose(model.similarity.toarray(), expected, rtol=1e-12, atol=1e-12)
        self.assertTrue((np.diff(RP3Model(matrix, topk=2).similarity.indptr) <= 2).all())

    def test_title_merge_keeps_author_and_volume_distinct(self):
        data = fixture()
        data.loc[data[ITEM].eq("0008"), "题名"] = "复本书 第一卷"
        data.loc[data[ITEM].eq("0009"), "题名"] = " 复本书　第一卷 "
        data.loc[data[ITEM].isin(["0008", "0009"]), "责任者"] = "甲"
        ctx = context(data)
        matrix = EntityMatrix(ctx, title_level=True)
        left, right = [ctx.item_index[item] for item in ("0008", "0009")]
        self.assertEqual(matrix.item_groups[left], matrix.item_groups[right])
        data.loc[data[ITEM].eq("0009"), "责任者"] = "乙"
        changed = EntityMatrix(context(data), title_level=True)
        self.assertNotEqual(changed.item_groups[left], changed.item_groups[right])
        data.loc[data[ITEM].eq("0009"), "责任者"] = "甲"
        data.loc[data[ITEM].eq("0009"), "题名"] = "复本书 第二卷"
        changed = EntityMatrix(context(data), title_level=True)
        self.assertNotEqual(changed.item_groups[left], changed.item_groups[right])

    def test_title_copy_probability_sums_to_one(self):
        matrix = EntityMatrix(context(), title_level=True)
        totals = np.bincount(matrix.item_groups, weights=matrix.copy_prior, minlength=len(matrix.keys))
        np.testing.assert_allclose(totals, 1)

    def test_study_stage_changes_at_academic_year_boundary(self):
        self.assertEqual(study_stage({"年级": "2022"}, "2024-08-31"), "2")
        self.assertEqual(study_stage({"年级": "2022级"}, "2024-09-01"), "3")
        self.assertEqual(study_stage({"年级": "2025"}, "2024-09-01"), "")
        self.assertEqual(study_stage({"年级": "未知"}, "2024-09-01"), "")

    def test_profile_backoff_and_independent_cold_reader(self):
        ctx = context()
        prior = AttributePrior(ctx, OptimizationConfig())
        np.testing.assert_allclose(prior.scores("new"), prior.global_probability)
        external = {"new": {"专业": "计算机", "系别": "计算机学院", "年级": "2023"}}
        np.testing.assert_allclose(prior.scores("new", external), prior.scores("000"))
        self.assertAlmostEqual(float(prior.scores("new", external).sum()), 1)
        self.assertTrue(np.isfinite(prior.scores("new", {"new": {"专业": "从未出现的专业"}})).all())
        prior.distribution.cache_clear()

    def test_missing_titles_do_not_merge_unrelated_barcodes(self):
        data = fixture().drop(columns=["题名", "责任者"])
        matrix = EntityMatrix(context(data), title_level=True)
        self.assertEqual(len(matrix.keys), data[ITEM].nunique())

    def test_attribute_ablation_does_not_mutate_baseline(self):
        ctx = context()
        original = ctx.baseline_rankings(ctx.users, 3)
        groups = set(ctx.baseline.group_scores)
        self.assertTrue(groups)
        no_attribute_baseline(ctx, ctx.users, 3)
        self.assertEqual(set(ctx.baseline.group_scores), groups)
        self.assertEqual(ctx.baseline_rankings(ctx.users, 3), original)

    def test_repeat_gate_uses_only_completed_training_windows(self):
        gate = RepeatGate(context(), OptimizationConfig(gate_months=1, gate_windows=2))
        for row in gate.summary["training_snapshots"]:
            self.assertLessEqual(pd.Timestamp(row["end_exclusive"]), pd.Timestamp("2024-03-01"))
        self.assertTrue(all(0 <= probability <= 1 for probability in gate.probabilities.values()))

    def test_repeat_explore_partition_and_exclude_seen(self):
        ctx = context()
        user = "000"
        ranking = {"all": {user: ctx.items}}
        gate = type("Gate", (), {"probabilities": {user: 0.0}})()
        predictions = RepeatExploreSpace(ctx, ranking, [user], gate).predict({"all": 1}, k=3)
        seen = {ctx.items[index] for index in ctx.seen[ctx.user_index[user]]}
        self.assertFalse(set(predictions[user]) & seen)
        gate.probabilities[user] = 1 - 1e-8
        predictions = RepeatExploreSpace(ctx, ranking, [user], gate).predict({"all": 1}, k=3)
        self.assertTrue(set(predictions[user]) <= seen)
        blocked = context(exclude_seen=True)
        predictions = RepeatExploreSpace(blocked, ranking, [user], gate).predict({"all": 1}, k=3)
        self.assertFalse(set(predictions[user]) & seen)

    def test_all_new_models_respect_candidate_catalog_and_cold_fallback(self):
        ctx = context(exclude_seen=True)
        matrix = EntityMatrix(ctx)
        models = [EASEModel(matrix, 10), RP3Model(matrix), AttributePrior(ctx, OptimizationConfig())]
        for model in models:
            rankings = model.rank(["000", "new"], 3)
            for user, items in rankings.items():
                self.assertEqual(len(items), 3)
                self.assertEqual(len(set(items)), 3)
                self.assertTrue(set(items) <= ctx.baseline.items)
            seen = {ctx.items[index] for index in ctx.seen[ctx.user_index["000"]]}
            self.assertFalse(set(rankings["000"]) & seen)
            self.assertEqual(rankings["new"], ctx.fallback("new", 3))

    def test_windows_reject_overlap_and_diagnostic_leakage(self):
        with self.assertRaises(ValueError):
            check_windows([("2024-01-01", "2024-06-01"), ("2024-05-01", "2024-07-01")], "2024-11-01")
        with self.assertRaises(ValueError):
            check_windows([("2024-09-01", "2024-12-01")], "2024-11-01")

    def test_previous_fusion_preserves_original_zero_weight_column_layout(self):
        users = ["u"]
        rankings = {"baseline": {"u": ["a", "b", "c"]}, "content": {"u": ["b", "a", "c"]},
                    "item2vec": {"u": ["c", "b", "a"]}}
        full = {name: rankings.get(name, {"u": ["z", "y", "x"]}) for name in CHANNELS}
        weights = {"baseline": 1, "content": 1, "item2vec": 0.5}
        expected = FusionSpace(full, users, 20).predict(weights, 3)
        actual = previous_space(rankings, users, 20)
        self.assertEqual(actual.channels, list(CHANNELS))
        self.assertEqual(actual.predict(weights, 3), expected)

    def test_current_fusion_same_when_prediction_omits_unselected_models(self):
        ctx = context()
        ranks = {"previous": {"u": ["a", "b", "c"]}, "recent": {"u": ["b", "a", "c"]}}
        full = {name: ranks.get(name, {"u": ["x", "y", "z"]}) for name in (*CORE_CHANNELS, *FAMILIES)}
        plan = {"strategy": "rrf", "weights": {"previous": 1, "recent": 0.5}}
        one = predict_plan({"context": ctx, "users": ["u"], "rankings": ranks}, plan, 3)
        two = predict_plan({"context": ctx, "users": ["u"], "rankings": full}, plan, 3)
        self.assertEqual(one, two)

    def test_small_end_to_end_frozen_selection_prediction_and_profile_timestamp(self):
        data = fixture()
        later = []
        for month in (3, 7, 11):
            for user, item in (("000", "0000"), ("001", "0008"), (f"new{month}", "0004")):
                later.append({USER: user, ITEM: item, TIME: pd.Timestamp(2024, month, 5),
                              "题名": "历史书", "责任者": "甲", "专业": "计算机", "系别": "计算机学院", "年级": "2023"})
        data = pd.concat([data, pd.DataFrame(later)], ignore_index=True)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            input_path, prior_path = root / "events.csv", root / "prior.json"
            save_csv(data, input_path)
            prior_plan = {"k": 3, "config": asdict(context().config),
                          "selection": {"weights": {"baseline": 1.0, "content": 1.0, "item2vec": 0.5},
                                        "epochs": {"item2vec": 1}}}
            write_json(prior_path, prior_plan)
            with contextlib.redirect_stdout(io.StringIO()):
                report = run_evaluation(input_path, root / "evaluation", prior_path,
                                         windows=[("2024-03-01", "2024-05-01"), ("2024-07-01", "2024-09-01")],
                                         diagnostic_start="2024-11-01", bootstrap_samples=50)
                locked = json.loads((root / "evaluation/locked_plan.json").read_text(encoding="utf-8"))
                self.assertEqual(locked["recommendation"], report["recommendation"])
                self.assertEqual(locked["parameters"], report["parameters"])
                repeated = run_evaluation(input_path, root / "repeated", prior_path,
                                           windows=[("2024-03-01", "2024-05-01"), ("2024-07-01", "2024-09-01")],
                                           diagnostic_start="2024-11-01", bootstrap_samples=50,
                                           reuse_selection=root / "evaluation/locked_plan.json")
                self.assertEqual(repeated["recommendation"], locked["recommendation"])
                self.assertEqual(repeated["parameters"], locked["parameters"])
                self.assertTrue(repeated["protocol"]["reused_frozen_selection"])
                readers = pd.DataFrame([{USER: "new-independent", "专业": "计算机", "系别": "计算机学院", "年级": "2023",
                                         "画像时间": "2024-12-01"}])
                save_csv(readers, root / "readers.csv")
                predictions = run_prediction(input_path, root / "prediction", root / "evaluation/locked_plan.json",
                                               root / "readers.csv", as_of="2025-01-01")
                self.assertEqual(len(predictions["new-independent"]), 3)
                summary = json.loads((root / "prediction/prediction_summary.json").read_text(encoding="utf-8"))
                self.assertEqual(summary["cold_readers_using_profile_fallback"], 1)
                readers["画像时间"] = "2025-02-01"
                save_csv(readers, root / "bad_readers.csv")
                with self.assertRaises(ValueError):
                    run_prediction(input_path, root / "invalid", root / "evaluation/locked_plan.json",
                                     root / "bad_readers.csv", as_of="2025-01-01")


if __name__ == "__main__":
    unittest.main(verbosity=2)
