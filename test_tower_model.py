import unittest

import pandas as pd
import torch

from tower_model import CALL, TIME, USER, Towers, samples


class TowerTests(unittest.TestCase):
    def test_temporal_negatives_and_tied_positives(self):
        rows = [(f"p{i}",f"I/{i}",1) for i in range(30)]
        rows += [("target","I/30",2),("target","I/31",3),("target","I/32",3),
                 ("target","I/31",4),("future","I/33",5)]
        h = pd.DataFrame(rows,columns=[USER,CALL,TIME])
        h[TIME] = pd.to_datetime(h[TIME],unit="D")
        h["专业"],h["单位"],h["读者级别"] = "计算机","数计学院","学生"
        ids = {f"I/{i}":i+1 for i in range(34)}
        hist,fields,items = samples(h,ids)
        self.assertEqual(set(items[:,0].tolist()),{32,33})
        self.assertEqual(len(hist),2)
        self.assertTrue((hist[:,-1]==31).all())
        self.assertTrue((items[:,1:]<=30).all())
        self.assertTrue((fields!=0).all())

    def test_masked_history_cannot_affect_profile_only_prediction(self):
        m = Towers(12)
        profiles = torch.tensor([[1,2,3]])
        mask = torch.zeros(1,8,dtype=torch.bool)
        a = m.users(torch.randn(1,8,48),mask,profiles)
        b = m.users(torch.randn(1,8,48),mask,profiles)
        torch.testing.assert_close(a,b)
        self.assertTrue(torch.isfinite(a).all())


if __name__=="__main__":
    unittest.main()

