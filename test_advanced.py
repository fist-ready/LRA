"""多路推荐的时间隔离、训练语义、融合、统计比较及完整流程验证。"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from baseline import ITEM, TIME, USER, prepare_events
from advanced import FusionSpace, paired_bootstrap, run_experiment, run_prediction
from advanced_models import (
    AdvancedConfig, ContentModel, GRUSequenceModel, HistoryContext, fit_neural,
    item2vec_pairs, neural_rankings, sequence_batches, top_indices, torch,
)


def events(rows):
    return prepare_events(pd.DataFrame(rows, columns=[USER, ITEM, TIME, "专业", "题名", "责任者", "索书号"]))


def fixture():
    return events([
        ("001", "A", "2024-01-01", "计算机", "Python数据分析", "作者甲", "TP311/1"),
        ("001", "B", "2024-01-02", "计算机", "机器学习", "作者乙", "TP181/2"),
        ("002", "A", "2024-01-03", "计算机", "Python数据分析", "作者甲", "TP311/1"),
        ("002", "C", "2024-01-04", "计算机", "Python数据分析", "作者甲", "TP311/1"),
        ("003", "D", "2024-01-05", "文学", "唐诗选读", "作者丙", "I222/3"),
        ("003", "E", "2024-01-06", "文学", "古代文学", "作者丁", "I206/4"),
    ])


class AdvancedTests(unittest.TestCase):
    def setUp(self):
        self.config = AdvancedConfig(dimensions=8, mf_epochs=2, item2vec_epochs=2, gru_epochs=2,
                                     batch_size=16, negatives=2, max_sequence=4, candidate_k=5, threads=1)
        self.history = fixture()
        self.context = HistoryContext(self.history, "2024-02-01", self.config)

    def test_context_rejects_future_records(self):
        with self.assertRaises(ValueError):
            HistoryContext(self.history, "2024-01-03", self.config)

    def test_vocabulary_profiles_and_metadata_only_from_history(self):
        self.assertEqual(set(self.context.items), {"A", "B", "C", "D", "E"})
        self.assertNotIn("new", self.context.user_index)
        self.assertEqual(self.context.profiles["001"]["专业"], "计算机")
        self.assertEqual(self.context.metadata["C"]["题名"], "Python数据分析")

    def test_negative_sampling_never_uses_known_positive(self):
        users = np.asarray([0, 1, 2] * 20)
        negatives = self.context.sample_negatives(users, 10, np.random.default_rng(1))
        for user, row in zip(users, negatives):
            self.assertFalse(set(row) & self.context.seen[user])

    def test_top_k_boundary_ties_and_blocked_items(self):
        self.assertEqual(top_indices([1, 1, 1, 1], 2).tolist(), [0, 1])
        self.assertEqual(top_indices([5, 4, 3], 10, {0}).tolist(), [1, 2])
        with self.assertRaises(ValueError):
            top_indices([np.nan], 1)

    def test_item2vec_context_stays_within_reader_and_time_window(self):
        centers, neighbors = item2vec_pairs(self.context)
        pairs = set(zip(centers.tolist(), neighbors.tolist()))
        ids = self.context.item_index
        self.assertIn((ids["A"], ids["B"]), pairs)
        self.assertNotIn((ids["B"], ids["D"]), pairs)
        self.assertNotIn((ids["A"], ids["A"]), pairs)

    def test_sequence_inputs_and_targets_are_shifted_without_future_input(self):
        inputs, targets, users = sequence_batches(self.context)
        row = int(np.flatnonzero(users == self.context.user_index["001"])[0])
        self.assertEqual(inputs[row].tolist(), [self.context.item_index["A"] + 1, 0, 0, 0])
        self.assertEqual(targets[row].tolist(), [self.context.item_index["B"] + 1, 0, 0, 0])

    def test_gru_prefix_outputs_do_not_change_when_future_is_appended(self):
        model = GRUSequenceModel(5, 8).eval()
        with torch.no_grad():
            prefix = model(torch.tensor([[1, 2]]))
            extended = model(torch.tensor([[1, 2, 3]]))
        self.assertTrue(torch.allclose(prefix, extended[:, :2], atol=1e-6))

    def test_content_retrieves_known_copy_with_identical_metadata(self):
        config = AdvancedConfig(exclude_seen=True)
        context = HistoryContext(self.history, "2024-02-01", config)
        ranking = ContentModel(context).rank(["001"], 3)["001"]
        self.assertEqual(ranking[0], "C")
        self.assertFalse(set(ranking) & {"A", "B"})

    def test_each_neural_model_trains_and_respects_catalog_and_cold_fallback(self):
        pretrained = None
        for name in ("bpr_mf", "attribute_bpr", "item2vec", "gru"):
            model, trace = fit_neural(name, self.context, self.config, epochs=2, pretrained=pretrained)
            self.assertTrue(all(np.isfinite(row["loss"]) for row in trace["trace"]))
            result = neural_rankings(name, model, self.context, ["001", "new"], 3)
            self.assertTrue(set(result["001"]) <= set(self.context.items))
            self.assertEqual(len(result["001"]), len(set(result["001"])))
            self.assertEqual(result["new"], self.context.fallback("new", 3))
            if name == "item2vec":
                pretrained = model.input_embedding.weight.detach().numpy().copy()

    def test_neural_training_is_reproducible_for_fixed_seed(self):
        one, _ = fit_neural("bpr_mf", self.context, self.config, epochs=2)
        two, _ = fit_neural("bpr_mf", self.context, self.config, epochs=2)
        for name, parameter in one.state_dict().items():
            self.assertTrue(torch.equal(parameter, two.state_dict()[name]))

    def test_single_channel_fusion_preserves_original_order(self):
        space = FusionSpace({"baseline": {"u": ["B", "A", "C"]}, "other": {"u": ["D"]}}, ["u"])
        self.assertEqual(space.predict({"baseline": 1}, 3)["u"], ["B", "A", "C"])

    def test_rank_fusion_and_validation_of_weights_and_duplicates(self):
        space = FusionSpace({"a": {"u": ["A", "B"]}, "b": {"u": ["B", "C"]}}, ["u"])
        self.assertEqual(space.predict({"a": 1, "b": 1}, 3)["u"], ["B", "A", "C"])
        with self.assertRaises(ValueError):
            space.predict({"a": 0})
        with self.assertRaises(ValueError):
            FusionSpace({"a": {"u": ["A", "A"]}}, ["u"])

    def test_paired_bootstrap_constant_improvement_and_no_change(self):
        left = pd.DataFrame({USER: ["u1", "u2"], "recall@10": [0.0, 0.0], "ndcg@10": [0.0, 0.0], "score": [0.0, 0.0]})
        right = left.copy()
        right[["recall@10", "ndcg@10", "score"]] = 1.0
        changed = paired_bootstrap(left, right, samples=100)["metrics"]["score"]
        same = paired_bootstrap(left, left, samples=100)["metrics"]["score"]
        self.assertEqual((changed["absolute_delta"], changed["ci95_low"], changed["ci95_high"]), (1, 1, 1))
        self.assertEqual((same["absolute_delta"], same["ci95_low"], same["ci95_high"]), (0, 0, 0))
        with self.assertRaises(ValueError):
            paired_bootstrap(left, right.iloc[:1], samples=10)

    def test_complete_evaluation_freezes_plan_and_predicts_from_full_history(self):
        data = pd.concat([
            self.history,
            events([("001", "C", "2024-09-01", "计算机", "Python数据分析", "作者甲", "TP311/1"),
                    ("new", "Z", "2024-09-02", "新专业", "未来题名", "新作者", "O1/1")]),
            events([("001", "Z", "2024-11-01", "计算机", "未来题名", "新作者", "O1/1"),
                    ("cold", "FUTURE", "2024-11-02", "新专业", "测试专有题名", "测试作者", "O2/1")]),
        ], ignore_index=True)
        config = AdvancedConfig(dimensions=4, mf_epochs=1, item2vec_epochs=1, gru_epochs=1,
                                batch_size=16, negatives=1, max_sequence=3, candidate_k=5, threads=1)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            base = Path(directory)
            source = base / "events.csv"
            data.to_csv(source, index=False, encoding="utf-8-sig")
            report = run_experiment(source, base / "evaluation", config, k=3, bootstrap_samples=20)
            self.assertIn("selected_vs_recent", report["diagnostic_comparisons"])
            plan_path = base / "evaluation/locked_plan.json"
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            self.assertEqual(plan["selection"], report["selection"])
            self.assertEqual(report["stages"]["test"]["history"]["events"], 8)
            for name in ("bpr_mf", "attribute_bpr", "item2vec", "gru"):
                self.assertEqual(report["stages"]["test"]["training"][name]["selected_epoch"], plan["selection"]["epochs"][name])
            test_predictions = pd.read_csv(base / "evaluation/test_selected.csv", dtype=str)
            self.assertNotIn("FUTURE", set(test_predictions.iloc[:, 1:].to_numpy().ravel()))
            result = run_prediction(source, base / "prediction", plan_path)
            self.assertEqual(len(result), 5)
            self.assertTrue(all(len(items) == 3 for items in result.values()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
