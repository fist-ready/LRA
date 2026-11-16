import unittest

import pandas as pd

from older_call_recall import CALL, TIME, USER, OlderCallRecall, WEIGHTS, blend


class OlderRecallTest(unittest.TestCase):
    def test_cutoff_copies_and_anonymous_profile(self):
        cutoff = pd.Timestamp("2024-03-01")
        rows = [("peer","I1/A","2024-01-01","甲"),("peer","I1/A","2024-02-01","甲"),
                ("peer","I1/B","2024-02-02","乙"),("other","I2/C","2024-02-01","甲"),
                ("other","I3/D","2024-03-01","甲"),("other","未知","2024-02-01","甲")]
        frame = pd.DataFrame(rows, columns=[USER,CALL,TIME,"责任者"])
        frame[TIME] = pd.to_datetime(frame[TIME])
        frame["单位"] = "学院"
        index = OlderCallRecall(frame, cutoff)
        self.assertEqual(set(index.popular), {"I1/A","I1/B","I2/C"})
        self.assertLess(index.hot["I1/A"], 1.)
        channels = index.channels(["I1/A"], "学院")
        self.assertEqual(channels["cf"], ["I1/B"])
        self.assertEqual(set(channels["content"]), {"I1/B","I2/C"})
        self.assertEqual(set(blend(channels, WEIGHTS)), {"I1/B","I2/C"})
        empty = index.channels([], "未知")
        self.assertEqual(blend(empty, WEIGHTS), index.popular)


if __name__ == "__main__":
    unittest.main()

