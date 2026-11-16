import unittest
import pandas as pd

from evaluate_older_windows import reserve, audit


class ReservedPositionsTests(unittest.TestCase):
    def test_two_distinct_positions_and_stable_order(self):
        order = list("ABCDEFGHIJKLM")
        self.assertEqual(reserve(order,["M","K"],2),list("ABCMDEFKGH"))
        self.assertEqual(order,list("ABCDEFGHIJKLM"))

    def test_existing_early_candidates_are_not_demoted(self):
        order = list("ABCDEFGHIJKL")
        self.assertEqual(reserve(order,["A","C"],2),order[:10])
        self.assertEqual(reserve(order,[],2),order[:10])
        self.assertEqual(reserve(order,["K"],0),order[:10])
        self.assertEqual(reserve(order,["K"],2),list("ABCKDEFGHI"))

    def test_exclusive_positives(self):
        features = pd.DataFrame({"query":["u","u","v","v"],"label":[1,1,1,0],
                                 "old_rr":[1,0,1,0],"new_rr":[0,1,1,1]})
        counts = audit(features)
        self.assertEqual(counts["positive"],3)
        self.assertEqual(counts["exclusive_positive"],1)
        self.assertEqual(counts["exclusive"],2)
        self.assertEqual(counts["exclusive_positive_queries"],1)


if __name__=="__main__":
    unittest.main()

