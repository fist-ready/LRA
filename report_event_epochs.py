from pathlib import Path
import pandas as pd
from newcomer_replay import DEV

D = Path(__file__).resolve().parent / "outputs/event_sequence_v1/evaluation"
KEY = ["window","user","group","method"]


def load(name):
    df = pd.read_csv(D / name, dtype={"user": str})
    assert not df.duplicated(KEY).any()
    return df


def aggregate(df):
    g = df.groupby(["window","method","group"]).score.mean().unstack()
    return (.4*g["新人"]+.6*g["老人"]).unstack().reindex(DEV)


def main():
    orig = load("users_epoch4.csv")
    four = load("paired_rerun_epoch4.csv")
    eight = load("users_epoch8.csv")
    cols = ["truth","hit","recall","ndcg","score","pred"]
    z = four.merge(eight, on=KEY, suffixes=("_4","_8"), how="outer", validate="one_to_one", indicator=True)
    assert z._merge.eq("both").all() and z.truth_4.eq(z.truth_8).all()
    base = z.method.isin(["当前提交","前轮双塔轻融合"])
    assert z.loc[base,"pred_4"].eq(z.loc[base,"pred_8"]).all()
    assert set(z.window)==set(DEV)
    anchor = orig.loc[orig.window.isin(DEV)&orig.method.isin(["当前提交","前轮双塔轻融合"]),KEY+cols]
    assert anchor.sort_values(KEY).reset_index(drop=True).equals(four.loc[four.method.isin(["当前提交","前轮双塔轻融合"]),KEY+cols].sort_values(KEY).reset_index(drop=True))
    out = {}
    for key,df in [("原四轮",orig),("配对四轮",four),("配对八轮",eight)]:
        t = aggregate(df)
        t.loc["开发均值"] = t.mean()
        out[key] = t
    combined = pd.concat(out,axis=1)
    print(combined.to_string(float_format=lambda v: format(v,".4f")))
    combined.to_csv(D/"epoch_study_summary.csv",encoding="utf-8-sig")
    d = z.loc[~base].copy()
    d["新增命中读者"] = d.hit_8.gt(d.hit_4)
    d["失去命中读者"] = d.hit_8.lt(d.hit_4)
    changes = d.groupby(["window","method"])[["新增命中读者","失去命中读者"]].sum()
    print(changes)
    changes.to_csv(D/"epoch_study_hits.csv", encoding="utf-8-sig")


if __name__=="__main__":
    main()
