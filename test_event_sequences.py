import unittest

import numpy as np
import pandas as pd
import torch

from event_sequences import CALL, TIME, USER, NegativePool, events, queries
from event_network import EventNet,collate_queries,query_vectors


def fixture():
    times = ["2023-01-10 09:00","2023-01-10 09:30","2023-01-10 09:55","2023-01-10 13:00",
             "2023-03-01 09:00","2023-03-01 09:02","2023-03-01 09:04","2023-04-10 11:00"]
    h = pd.DataFrame({USER:"u",CALL:list("AABCADEF"),TIME:pd.to_datetime(times)})
    for key in ["专业","单位","年级","读者级别","性别"]:
        h[key] = ""
    return h,{v:i+1 for i,v in enumerate("ABCDEF")}


class EventsTests(unittest.TestCase):
    def test_sasrec_cannot_read_future_or_right_padding(self):
        model = EventNet(12,"sasrec").eval()
        inputs = torch.randn(2,32,64)
        for training in [True,False]:
            model.train(training)
            changed = inputs.clone()
            changed[:,4:] = torch.randn_like(changed[:,4:])
            with torch.set_grad_enabled(training):
                state = model.attentive_states(inputs)[:,:4]
                torch.testing.assert_close(state, model.attentive_states(changed)[:,:4])
                torch.testing.assert_close(state, model.attentive_states(inputs[:,:4]), atol=1e-5,rtol=1e-5)

    def test_depth_changes_only_visible_history(self):
        from event_sequences import Event, make_query
        records = [Event(i,i,(i,),("",)*5,(i,)) for i in range(1,41)]
        short,long = [make_query("u",50,records,[1,49],"next",d) for d in (8,32)]
        self.assertEqual((len(short.prefix),len(long.prefix)),(8,32))
        self.assertEqual(short.official,long.official[-8:])
        self.assertEqual(len(long.official),32)
        self.assertEqual(short.seen,long.seen)
        self.assertEqual(short.positives,long.positives)
        self.assertEqual(short.positives,(49,))

    def test_chain_inclusive_gap_dedup_and_controls(self):
        h,ids = fixture()
        seq = events(h,ids,"gap30")[0][1]
        self.assertEqual([e.items for e in seq],[(1,2),(3,),(1,4,5),(6,)])
        self.assertEqual(len(events(h,ids,"single")[0][1]),8)
        self.assertEqual(len(events(h,ids,"day")[0][1]),3)

    def test_window_labels_and_incomplete_horizon(self):
        h,ids = fixture()
        seq = events(h,ids,"gap30")
        first,end = pd.Timestamp("2023-01-01"),pd.Timestamp("2023-05-01")
        qs = queries(seq,first,end,"joint")
        mar = next(q for q in qs if q.task=="window" and q.time==pd.Timestamp("2023-03-01").value)
        self.assertEqual(mar.positives,(4,5,6))
        self.assertEqual(mar.seen,{1,2,3})
        self.assertTrue(all(e.end<mar.time for e in mar.prefix))
        self.assertNotIn(mar,queries(seq,first,pd.Timestamp("2023-04-30"),"window"))
        targets = [q for q in qs if q.task=="next" and 4 in q.positives]
        self.assertEqual(targets[0].positives,(4,5))
        self.assertEqual(NegativePool(h,ids).draw(mar,"D",np.random.default_rng(1)),[])

    def test_book_and_event_encoders_see_the_same_prefix(self):
        frame, ids = fixture()
        seq = events(frame, ids, "gap30")
        q = next(q for q in queries(seq, pd.Timestamp("2023-01-01"), pd.Timestamp("2023-05-01"), "next")
                 if len(q.prefix) == 3)
        def batch(official, hidden, per_book):
            return collate_queries([q], [official], [hidden], per_book=per_book)
        pooled, flat = batch(False, False, False), batch(False, False, True)
        self.assertEqual(sorted(pooled[0].tolist()), sorted(flat[0].tolist()))
        self.assertEqual(flat[0].tolist(), [i for event in q.prefix for i in event.ordered])
        self.assertEqual(pooled[2].item(), 3)
        self.assertGreater(flat[2].item(), pooled[2].item())
        for official, hidden in [(True,False),(False,True)]:
            for left,right in zip(batch(official,hidden,True), batch(official,hidden,False)):
                torch.testing.assert_close(left,right)
        common = []
        for encoder in ("gru", "mean", "book_mean"):
            torch.manual_seed(2026)
            model = EventNet(12, encoder, aligned_init=True)
            common.append({k:v for k,v in model.state_dict().items() if not k.startswith("gru.")})
        for key, params in common[0].items():
            torch.testing.assert_close(params, common[1][key], atol=0, rtol=0)
            torch.testing.assert_close(params, common[2][key], atol=0, rtol=0)

    def test_gru_and_mean_ignore_within_event_permutation(self):
        from dataclasses import replace
        h,ids = fixture()
        q = queries(events(h,ids,"gap30"),pd.Timestamp("2023-01-01"),pd.Timestamp("2023-05-01"),"next")[2]
        swapped = replace(q,prefix=tuple(replace(e,items=e.items[::-1]) for e in q.prefix))
        books = torch.randn(7,64)
        for mode in ["gru","mean","sasrec"]:
            model = EventNet(12,mode)
            batch = collate_queries([q,swapped],[False,False],[False,False])
            vectors = query_vectors(model,batch,books[batch[0]])
            torch.testing.assert_close(vectors[0],vectors[1])
            batch = collate_queries([q,q],[True,True],[False,True])
            self.assertEqual(batch[5].shape,(2,5))
            self.assertEqual(batch[2][1],0)
            self.assertTrue(torch.isfinite(query_vectors(model,batch,books[batch[0]])).all())


if __name__=="__main__":
    unittest.main()

