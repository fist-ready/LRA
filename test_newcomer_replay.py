import unittest
import pandas as pd

from newcomer_replay import CALL, TIME, USER, Newcomer


class NewcomerTests(unittest.TestCase):
    def test_cutoff_distinct_pairs_and_profile_fallback(self):
        t = pd.Timestamp("2024-05-01")
        data = pd.DataFrame([(u,book,t+pd.Timedelta(days=gap),dept,"甲") for u,book,gap,dept in [
            ("u","A",-3,"经管学院"),("u","A",-1,"经管学院"),("v","B",-1,"经济与管理学院"),
            ("x","C",-1,"另院"),("w","D",0,"经管学院")]],columns=[USER,CALL,TIME,"单位","责任者"])
        model = Newcomer(data,t)
        self.assertEqual(dict(model.pop),{"A":1,"B":1,"C":1})
        self.assertEqual(model.dept_order("未知"),model.global_order)
        self.assertEqual(model.dept_order("经管学院"),model.dept_order("经济与管理学院"))
        result = model.predict(["A"],"未知",["个人历史"])["个人历史"]
        self.assertEqual(set(result),{"B","C"})
        self.assertEqual(len(result),len(set(result)))


if __name__=="__main__":
    unittest.main()

