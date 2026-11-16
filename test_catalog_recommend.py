import unittest

import numpy as np
import pandas as pd

from catalog_recommend import CALL, META, TIME, USER, Catalog, Profile, major


class Encoder:
    def encode(self,texts):
        return np.array([[1,0] if "算法" in t else [0,1] for t in texts],dtype=np.float32)


class CatalogTests(unittest.TestCase):
    def setUp(self):
        rows = []
        for i in range(24):
            rows.append({USER:"u"+str(i),CALL:"TP1/"+str(i),"题名":"图书"+str(i),"责任者":"甲",
                         "专业":"计算机本科","单位":"数计学院",TIME:pd.Timestamp("2024-01-01")})
        self.h = pd.DataFrame(rows)
        self.meta = pd.concat([self.h[META],pd.DataFrame([["J2/Z","算法入门","乙"]],columns=META)])

    def test_external_title_can_recall_unborrowed_catalog_item(self):
        model = Catalog(self.h,pd.Timestamp("2024-02-01"),self.meta,Encoder())
        r = model.recommend(["X2/Q"],["算法"],"未知","","老人")
        self.assertIn("J2/Z",r["画像书目与语义"])
        self.assertNotIn("J2/Z",r["基础"])
        for order in r.values():
            self.assertEqual(len(order),len(set(order)))
            self.assertNotIn("X2/Q",order)

    def test_reader_support_and_fallback(self):
        many = pd.concat([self.h.iloc[:1]]*18)
        profile = Profile(many)
        self.assertEqual(profile.support["major","计算机"],1)
        self.assertEqual(major("计算机本科"),"计算机")
        self.assertEqual(profile.lifts("未知","未知"),{"TP1":1.0})

    def test_no_future_borrow_counts_and_filter_all_seen(self):
        h = self.h.copy()
        h.loc[0,TIME] = pd.Timestamp("2024-02-01")
        model = Catalog(h,pd.Timestamp("2024-02-01"),self.meta,Encoder())
        self.assertEqual(model.count[model.ids["TP1/0"]],0)
        for order in model.recommend(["TP1/1","TP1/0"],[],"计算机","数计学院","新人").values():
            self.assertFalse(set(order)&{"TP1/0","TP1/1"})


if __name__=="__main__":
    unittest.main()

