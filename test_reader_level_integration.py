import json
import tempfile
import unittest
from argparse import Namespace
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from advanced_models import AdvancedConfig, HistoryContext, set_seed
from baseline import BASE, USER, ITEM, TIME, source_info, ground_truth
from optimized_models import AttributePrior, OptimizationConfig
from ranking_models import CandidateFeatures, OLD_CHANNELS, labeled, fit_ranker, rank_candidates
from ranking import predict
from reader_level import ReaderLevelConfig
from reader_level_experiment import extend_features
from test_ranking import fixture, FakeEncoder


class LevelIntegration(unittest.TestCase):
    def setUp(self):
        set_seed(2026,1)
        self.data = fixture()
        self.data['读者级别'] = self.data[USER].map(lambda u: '学生' if int(u[1:])<6 else '教师')
        self.config = ReaderLevelConfig(True,2,2,.5)
        self.context = HistoryContext(self.data,'2023-03-01',AdvancedConfig(threads=1),reader_level=True)
        self.users = self.context.users+['new']
        self.raw = {n:self.context.baseline_rankings(self.users,20) for n in OLD_CHANNELS}

    def test_context_latest_nonmissing_and_cutoff(self):
        changed = self.data.copy()
        user = changed.iloc[0][USER]
        rows = changed[changed[USER].eq(user)].index
        changed.loc[rows,'读者级别'] = ['学生','教师','未知']
        context = HistoryContext(changed,'2023-03-01',AdvancedConfig(),reader_level=True)
        self.assertEqual(context.profiles[user]['读者级别'],'教师')
        off = HistoryContext(changed,'2023-03-01',AdvancedConfig())
        self.assertNotIn('读者级别',off.profiles[user])
        changed.loc[rows[-1],TIME] = pd.Timestamp('2023-03-01')
        with self.assertRaises(ValueError):
            HistoryContext(changed,'2023-03-01',AdvancedConfig(),reader_level=True)

    def test_features_off_compatibility_and_external_profile(self):
        plain,_ = CandidateFeatures(self.context,FakeEncoder()).build(self.users,self.raw)
        off,_ = CandidateFeatures(self.context,FakeEncoder(),ReaderLevelConfig()).build(self.users,self.raw)
        pd.testing.assert_frame_equal(plain,off)
        external = {'new':{'读者级别':'教师'}}
        enabled,_ = CandidateFeatures(self.context,FakeEncoder(),self.config).build(self.users,self.raw,external=external)
        reconstructed = extend_features(plain,self.context,self.config,external)
        for c in ['level_item_probability','level_category_probability','level_group_available']:
            np.testing.assert_array_equal(enabled[c],reconstructed[c])
        self.assertTrue(enabled.loc[enabled[USER].eq('new'),'level_group_available'].eq(1).all())
        self.assertTrue(np.isfinite(enabled.select_dtypes('number')).all().all())

    def test_prior_level_only_and_missing_fallback(self):
        old = AttributePrior(self.context,OptimizationConfig())
        off = AttributePrior(self.context,OptimizationConfig(),reader_level_config=ReaderLevelConfig())
        new = AttributePrior(self.context,OptimizationConfig(),reader_level_config=self.config)
        np.testing.assert_array_equal(old.scores('new'),off.scores('new'))
        np.testing.assert_array_equal(old.scores('new'),new.scores('new'))
        external={'new':{'读者级别':'教师'}}
        self.assertFalse(np.array_equal(old.scores('new',external),new.scores('new',external)))
        self.assertEqual(len(new.rank(['new'],10,external)['new']),10)

    def test_ranker_save_load_and_prediction_interface(self):
        from catboost import CatBoostRanker
        features,_ = CandidateFeatures(self.context,FakeEncoder(),self.config).build(self.users,self.raw)
        future=self.data.drop_duplicates(USER).copy()
        future[TIME]=pd.Timestamp('2023-03-15')
        train=labeled(features.loc[features['warm'].eq(1)],ground_truth(future),'2023-03-01','2023-05-01').sort_values(['query',ITEM])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            model,columns=fit_ranker(train,True,root/'ranker.cbm',iterations=10)
            restored=CatBoostRanker().load_model(str(root/'ranker.cbm'))
            np.testing.assert_array_equal(model.predict(train[columns]),restored.predict(train[columns]))
            self.assertIn('level_item_probability',json.loads((root/'ranker.features.json').read_text())['columns'])
            train.to_pickle(root/'features.pkl')
            data=pd.concat([self.data,future],ignore_index=True)
            data.to_csv(root/'input.csv',index=False,encoding='utf-8-sig')
            pd.DataFrame([{USER:'new','读者级别':'教师','画像时间':'2023-04-01'},{USER:'u00','读者级别':'学生','画像时间':'2023-04-01'}]).to_csv(root/'readers.csv',index=False,encoding='utf-8-sig')
            old=json.loads((BASE/'outputs/ranking/locked_plan.json').read_text(encoding='utf-8'))
            plan={**old,'input':source_info(root/'input.csv'),'reader_level':asdict(self.config),
                  'fit_iterations':10,'recipe':{**old['recipe'],'trees':10},
                  'training_snapshots':[{'start':'2023-03-01','end':'2023-05-01','path':str(root/'features.pkl')}]}
            (root/'plan.json').write_text(json.dumps(plan),encoding='utf-8')
            args=Namespace(plan=str(root/'plan.json'),input=str(root/'input.csv'),readers=str(root/'readers.csv'),as_of='2023-06-01',output_dir=str(root/'prediction'))
            with patch('ranking.TextEncoder',return_value=FakeEncoder()):
                predict(args)
            result=pd.read_csv(root/'prediction/result.csv',dtype=str)
            self.assertEqual(result.shape,(2,11))
            for row in result.itertuples(index=False,name=None):
                self.assertEqual(len(set(row[1:])),10)
                self.assertTrue(set(row[1:]) <= set(data[ITEM]))
            args.as_of='2023-04-01'
            with self.assertRaises(ValueError):
                predict(args)


if __name__=='__main__':
    unittest.main()
