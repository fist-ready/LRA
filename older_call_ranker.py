"""固定新旧索书号候选并集，滚动训练老人新借书排序器。"""
import json
from collections import Counter

import numpy as np
import pandas as pd
from catboost import CatBoostRanker

from advanced_models import set_seed
from baseline import ITEM, TIME, USER, attribute_value, load_interactions
from call_number_replay import BASE, CALL, GROUP_GAP, fuse, metrics, unique_new
from older_call_recall import OlderCallRecall, WEIGHTS, blend
from ranking import DEVELOPMENT, DIAGNOSTIC, context_for
from ranking_models import CandidateFeatures, TextEncoder, fit_ranker, rank_candidates, training_data

ROOT = BASE / "outputs/ranking"
OUT = BASE / "outputs/older_call_ranker"
DATES = ["2023-03-01","2023-05-01",DEVELOPMENT[0],DEVELOPMENT[1],"2024-05-01",DEVELOPMENT[2],DIAGNOSTIC]
FEATURES = ["old_score","old_rr","new_rr","hot_rr","college_rr","cf_rr","content_rr",
            "opt_rr","readers","hot","local","category_share","author_share","history_size","idle_days"]
METHODS = ["原流程","新召回","等权融合","旧树全池","新借排序"]


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def rr(items):
    return {item: 1/(20+i) for i,item in enumerate(items,1)}


def snapshot(data, date, manifests, encoder, *, folder=None, barcode_model=None):
    start = pd.Timestamp(date)
    folder = folder or ROOT/"snapshots"/date
    meta = read(folder/"manifest.json")
    end = pd.Timestamp(meta["end"])
    assert pd.Timestamp(meta["history_max"]) < start
    history = data.loc[data[TIME] < start]
    target = data.loc[data[TIME].between(start,end,inclusive="left") & data[CALL].ne("")]
    old_users = set(history.loc[history[TIME] < start-GROUP_GAP,USER])
    index = OlderCallRecall(history,start)
    answers = {u:set(g[CALL])-index.books.get(u,set()) for u,g in target.groupby(USER) if u in old_users}
    answers = {u:y for u,y in answers.items() if y}
    users = sorted(answers)
    latest = history.groupby(USER)[TIME].last()
    departments = history.loc[history["单位"].ne("")].groupby(USER)["单位"].last().to_dict()
    mapping = history.loc[history[CALL].ne("")].drop_duplicates(ITEM,keep="last").set_index(ITEM)[CALL]
    copies = mapping.groupby(mapping).groups
    raw = read(folder/"channels.json")
    old_file = barcode_model or folder/"expanded.cbm"
    if barcode_model is not None and not old_file.is_file():
        raise FileNotFoundError(old_file)
    if old_file.exists():
        old = CatBoostRanker().load_model(str(old_file))
        columns = old.feature_names_
    else:
        train, _ = training_data(manifests,start,True)
        old,columns = fit_ranker(train,True,OUT/date/"barcode.cbm")
        del train
    stored = pd.read_pickle(folder/"features.pkl")
    ranked = rank_candidates(old,columns,stored.loc[stored[USER].isin(users)],raw["optimized"],True,100,200)
    del stored
    popular = sorted(index.readers,key=lambda i:(-len(index.readers[i]),i))
    mapping_dict = mapping.to_dict()
    records, barcode_pool = [], {}
    for u in users:
        seq = index.profile.get(u,[])
        seen = set(seq)
        channels = index.channels(seq,departments.get(u,""))
        opt = unique_new((mapping_dict.get(i) for i in raw["optimized"][u]),seen)
        legacy = fuse([opt,unique_new((mapping_dict.get(i) for i in ranked[u]),seen)])
        legacy = unique_new(legacy+popular,seen)[:100]
        fresh = blend(channels,WEIGHTS)
        pool = sorted(set(legacy)|set(fresh))
        assert len(pool)>=100 and not seen.intersection(pool)
        barcode_pool[u] = [barcode for item in pool for barcode in copies[item]]
        ranks = {name+"_rr":rr(items) for name,items in dict(channels,old=legacy,new=fresh,opt=opt).items()}
        categories = Counter(i.split("/")[0] for i in seq)
        authors = Counter(index.author[i] for i in seq if index.author[i])
        local = index.local.get(departments.get(u),{})
        for item in pool:
            records.append({USER:u,CALL:item,"query":date+"_"+u,"label":int(item in answers[u]),
                            "readers":len(index.readers[item]),"hot":index.hot[item],"local":local.get(item,0),
                            "history_size":len(seq),"idle_days":(start-latest[u]).days,
                            "category_share":categories[item.split("/")[0]]/max(1,len(seq)),
                            "author_share":authors[index.author[item]]/max(1,len(seq)),
                            **{key:table.get(item,0) for key,table in ranks.items()}})
    context = context_for(data,start,read(ROOT/"locked_plan.json")["previous_plan"])
    features,_ = CandidateFeatures(context,encoder).build(users,raw,candidate_codes=barcode_pool)
    features["old_score"] = old.predict(features[columns],ntree_end=100)
    features[CALL] = features[ITEM].map(mapping)
    best = features.groupby([USER,CALL]).old_score.max()
    frame = pd.DataFrame(records).join(best,on=[USER,CALL],validate="one_to_one")
    assert frame[FEATURES].notna().all().all()
    return frame,answers,end


def fit(train):
    # 没有正例的组不能提供组内排序信息；评测时仍保留全部读者。
    useful = train.groupby("query").label.transform("nunique").gt(1)
    examples = train.loc[useful].sort_values(["query",CALL])
    model = CatBoostRanker(iterations=80,depth=3,learning_rate=.04,l2_leaf_reg=12,random_seed=2026,
                           thread_count=4,verbose=False,allow_writing_files=False,
                           loss_function="YetiRank:mode=NDCG;top=10;dcg_denominator=LogPosition")
    model.fit(examples[FEATURES],examples.label,group_id=examples["query"])
    return model,dict(groups=examples["query"].nunique(),positives=int(examples.label.sum()))


def evaluate(frame,answers,model,date):
    frame = frame.copy()
    frame["新借排序"] = model.predict(frame[FEATURES])
    frame["旧树全池"] = frame.old_score
    frame["原流程"] = frame.old_rr
    frame["新召回"] = frame.new_rr
    frame["等权融合"] = frame.old_rr+frame.new_rr
    rows = []
    for user,g in frame.groupby(USER):
        truth = answers[user]
        new_only = set(g.loc[g.old_rr.eq(0),CALL])
        for method in METHODS:
            top = g.sort_values([method,CALL],ascending=[False,True])[CALL].head(10).tolist()
            assert len(set(top))==10
            m = metrics(top,truth)
            rows.append(dict(window=date,user=user,method=method,pred="|".join(top),truth=len(truth),
                             candidates=len(g),pool_hits=int(g.label.sum()),new_hits=len(set(top)&truth&new_only),
                             score=100*(.4*m["hit"]+.3*m["recall"]+.3*m["ndcg"]),**m))
    return rows


def main():
    set_seed(2026,4)
    OUT.mkdir(exist_ok=True,parents=True)
    data = load_interactions(BASE/"data/processed/borrowing_cleaned.csv")
    for key in [CALL,"单位"]:
        data[key] = data[key].map(attribute_value)
    manifests = [read(f) for f in sorted((ROOT/"snapshots").glob("*/manifest.json"))]
    encoder = TextEncoder(BASE/".model_deps/bge-small-zh-v1.5",ROOT/"embeddings")
    past,rows,audit = [],[],{}
    for date in DATES:
        print("构造",date,flush=True)
        directory = OUT/date
        directory.mkdir(exist_ok=True)
        frame,answers,end = snapshot(data,date,manifests,encoder)
        if date in [*DEVELOPMENT,DIAGNOSTIC]:
            allowed = [(d,f) for d,e,f in past if e<=pd.Timestamp(date)]
            model,stats = fit(pd.concat([f for _,f in allowed],ignore_index=True))
            model.save_model(str(directory/"ranker.cbm"))
            loaded = CatBoostRanker().load_model(str(directory/"ranker.cbm"))
            assert np.array_equal(model.predict(frame[FEATURES]),loaded.predict(frame[FEATURES]))
            rows.extend(evaluate(frame,answers,loaded,date))
            audit[date] = dict(train_dates=[d for d,_ in allowed],**stats)
        past.append((date,end,frame))
    details = pd.DataFrame(rows)
    details.to_csv(OUT/"per_user.csv",index=False,encoding="utf-8-sig")
    groups = details.groupby(["window","method"])
    summary = groups[["score","hit","recall","ndcg","candidates"]].mean()
    summary["pool_coverage"] = groups.pool_hits.sum()/groups.truth.sum()
    summary["new_hits"] = groups.new_hits.sum()
    for date in [*DEVELOPMENT,DIAGNOSTIC]:
        source = BASE/"outputs/older_call_recall"
        if date==DIAGNOSTIC:
            source /= "reused_diagnostic"
        prev = pd.read_csv(source/"per_user.csv",dtype={"user":str})
        for method,baseline in [("原流程","冻结模型"),("新召回","增加共现和内容")]:
            def preds(df,w,m):
                return df.loc[df.window.eq(w)&df.method.eq(m)].set_index("user").pred.sort_index()
            pd.testing.assert_series_equal(preds(details,date,method),preds(prev,date,baseline))
        old = pd.read_csv(source/"summary.csv")
        ref = old.loc[old.window.eq(date)&old.method.eq("冻结模型")].iloc[0]
        for method in METHODS:
            summary.loc[(date,method),"total"] = ref.total_100+.6*(summary.loc[(date,method),"score"]-ref.score)
    summary.to_csv(OUT/"summary.csv",encoding="utf-8-sig")
    (OUT/"training.json").write_text(json.dumps(audit,indent=2),encoding="utf8")
    print(summary.round(4))


if __name__ == "__main__":
    main()

