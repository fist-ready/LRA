"""Cold-newcomer portrait proxy: whole-reader folds, using only cutoff-visible attributes."""
import pandas as pd

from baseline import TIME, USER
from catalog_recommend import CALL, bucket
from event_encoder_b import read_data
from newcomer_portrait import ProfileRecommender, portrait
from newcomer_replay import DEV, DIAG
from prior_swap_folds import fold_data, horizon_ends
from prior_swap_study import DemandPrior, predictions, save
from roster_replay import score


def run():
    data = read_data()
    results = []
    for date in [*DEV,DIAG]:
        start = pd.Timestamp(date)
        full,_,_,old = fold_data(data,date,False)
        history = full.groupby(USER)[CALL].agg(set).to_dict()
        profile_cols = ["专业","单位","年级","读者级别","性别"]
        attrs = full.groupby(USER)[profile_cols].last().to_dict("index")
        future = {}
        for label,end in horizon_ends(date).items():
            view=data[(data[TIME]>=start)&(data[TIME]<end)]
            future[label] = {u:y for u,g in view.groupby(USER) if u in history and u not in old
                             and (y:=set(g[CALL])-history[u]-{""})}
        targets=set.union(*(set(x) for x in future.values()))
        for k in range(5):
            test={u for u in targets if bucket(u,5)==k}
            train=full[~full[USER].isin(test)]
            assert not test.intersection(train[USER])
            prior=DemandPrior(train,start)
            model=ProfileRecommender(train,start)
            for u in sorted(test):
                m,d,y,l,sex=(attrs[u][c] for c in profile_cols)
                public=portrait(d,m,y,l,sex,"新人",start)
                base,_=model.predict(public,[])
                extended=base+[c for c in model.base.index.popular if c not in base][:90]
                r=dict(release=extended, recent=[], attrs=[m,d,y,l,sex],event=[],user=u)
                variants=predictions(r,prior,{},set())
                for label,by_user in future.items():
                    if u not in by_user:
                        continue
                    for method in ["原版","时序先验","时序群体先验"]:
                        top=variants[method]
                        assert len(top)==len(set(top))==10 and set(top)<=set(extended)
                        hit,rec,_,pts=score(top,by_user[u],"新人")
                        results.append(dict(date=date,horizon=label,user=u,method=method,hit=hit,
                                      recall=rec,points=pts,changed=set(top)!=set(base)))
            prior.lift.cache_clear()
    df=pd.DataFrame(results)
    save(df,"proxy_readers.csv")
    save(df.groupby(["date","horizon","method"]).agg(readers=("user","size"),
         hit=("hit","mean"),recall=("recall","mean"),points=("points","mean"),
         changed=("changed","sum")).reset_index(),"proxy_scores.csv")


if __name__=="__main__":
    run()
