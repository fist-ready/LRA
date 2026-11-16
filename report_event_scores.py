"""逐人核验，并对评测起点后的首个有新借书的会话单列计分。"""
import numpy as np
import pandas as pd

from baseline import BASE,USER,TIME,load_interactions,attribute_value
from newcomer_replay import CALL,DEV,DIAG,GAP
from roster_replay import score

D = BASE/"outputs/event_sequence_v1/evaluation"


def verify():
    rows = pd.read_csv(D/"users_epoch4.csv",dtype={"user":str})
    data = load_interactions(BASE/"data/processed/borrowing_cleaned.csv")
    data[CALL] = data[CALL].map(attribute_value)
    out = []
    for date in [*DEV,DIAG]:
        start = pd.Timestamp(date)
        end = (data[TIME].max()+pd.Timedelta(1,"ns")) if date==DIAG else start+pd.DateOffset(months=2)
        history = data.loc[data[TIME]<start]
        seen = history.groupby(USER)[CALL].agg(set).to_dict()
        old = set(history.loc[history[TIME]<start-GAP,USER])
        future = data.loc[data[TIME].between(start,end,inclusive="left")&data[CALL].ne("")]
        truth,first = {},{}
        for u,g in future.groupby(USER):
            y = set(g[CALL])-seen.get(u,set())
            if not y:
                continue
            truth[u] = y
            split = g[TIME].diff().gt(pd.Timedelta(minutes=30)).cumsum()
            first[u] = next(z for _,group in g.groupby(split) if (z:=set(group[CALL])&y))
        for method,part in rows.loc[rows.window.eq(date)].groupby("method"):
            assert part.user.is_unique and set(part.user)==set(truth)
            for r in part.itertuples():
                top = r.pred.split("|")
                assert len(top)==len(set(top))==10 and not set(top)&seen.get(r.user,set())
                group = "老人" if r.user in old else "新人"
                assert np.isclose(score(top,truth[r.user],group)[-1],r.score)
                out.append(dict(window=date,method=method,group=group,
                                recall=len(set(top)&first[r.user])/len(first[r.user])))
    pd.DataFrame(out).groupby(["window","method","group"]).recall.mean().to_csv(D/"next_session_recall.csv",encoding="utf-8-sig")
    print("核验通过：",len(rows))


if __name__=="__main__":
    verify()

