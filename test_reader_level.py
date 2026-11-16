import unittest
from types import SimpleNamespace
import numpy as np
import pandas as pd
from reader_level import ReaderLevelConfig, ReaderLevelPreference
from baseline import USER, ITEM, TIME


class ReaderLevelTests(unittest.TestCase):
    def build(self, rows=None, enabled=True, minimum=2):
        rows = rows or [('a','x','2024-01-01','学生'), ('b','x','2024-01-01','学生'),
                        ('c','y','2024-01-01','教师')]
        frame = pd.DataFrame(rows, columns=[USER, ITEM, TIME, '读者级别'])
        frame[TIME] = pd.to_datetime(frame[TIME])
        context = SimpleNamespace(history=frame, cutoff=pd.Timestamp('2024-02-01'),
            items=['x','y'], item_index={'x':0,'y':1}, config=SimpleNamespace(half_life_days=180))
        return ReaderLevelPreference(context, ReaderLevelConfig(enabled, minimum, 2), ['A','B'])

    def test_distinct_readers_and_fallback(self):
        model = self.build()
        self.assertEqual(model.support['学生'], 2)
        self.assertEqual(model.probabilities('c')[2], 0)
        np.testing.assert_array_equal(model.probabilities('unknown')[0], model.global_probability)
        self.assertGreater(model.probabilities('new', {'读者级别':'学生'})[0][0], model.global_probability[0])

    def test_repeat_does_not_multiply_support_or_contribution(self):
        rows = [('a','x','2024-01-01','学生')]*10 + [('b','y','2024-01-01','教师')]
        one = self.build([rows[0],rows[-1]], minimum=1)
        many = self.build(rows, minimum=1)
        self.assertEqual(many.support['学生'], 1)
        np.testing.assert_array_equal(one.probabilities('a')[0], many.probabilities('a')[0])

    def test_identity_change_and_latest_nonmissing(self):
        model = self.build([('a','x','2024-01-01','学生'), ('a','y','2024-01-02','教师'),
                            ('a','x','2024-01-03','未知')], minimum=1)
        self.assertEqual(model.profiles['a'], '教师')
        self.assertEqual(model.counts['学生'][1], 0)
        self.assertEqual(model.counts['教师'][0], 0)

    def test_disabled_and_labels_not_merged(self):
        model = self.build(enabled=False)
        np.testing.assert_array_equal(model.probabilities('a')[0], model.global_probability)
        self.assertEqual(self.build().probabilities('new', {'读者级别':'本科学生'})[2], 0)

    def test_future_rejected(self):
        with self.assertRaises(ValueError):
            self.build([('a','x','2024-02-01','学生')])


if __name__ == '__main__':
    unittest.main()
