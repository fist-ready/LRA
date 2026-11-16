"""双塔挑战版：滚动训练、冷书阶段审计、CPU 检查点和独立正式格式预测。"""
import argparse
import json

import numpy as np
import pandas as pd
import torch

from baseline import BASE,USER,TIME,attribute_value,load_interactions
from catalog_recommend import Catalog,CALL,META,bucket
from catalog_experiment import PACKAGE,measure
from newcomer_replay import DEV,DIAG,GAP
from older_call_recall import blend
from ranking_models import TextEncoder
from tower_model import Towers,train,book_attrs,profile

OUT = BASE/"outputs/deep_challenger_v1"
OLD = BASE/"outputs/catalog_v1"
NAMES = ("基础","冻结BGE融合","双塔直排","双塔轻融合","双塔等权")


class MemoryEncoder:
    def __init__(self):
        self.backend = TextEncoder(BASE/".model_deps/bge-small-zh-v1.5",OLD/"embeddings")
        self.cache = {}

    def encode(self,texts):
        missing = list(dict.fromkeys(t for t in texts if t not in self.cache))
        if missing:
            self.cache.update(zip(missing,self.backend.encode(missing)))
        return np.stack([self.cache[t] for t in texts])


class Challenger:
    def __init__(self,h,start,meta,encoder,path,load=False):
        self.cat = Catalog(h,start,meta,encoder)
        cat = self.cat
        text = torch.cat([torch.zeros_like(cat.vectors[:1]),cat.vectors])
        attrs = torch.tensor([[0,0]]+[book_attrs(i,a) for i,a in zip(cat.items,cat.author)])
        self.ids = {i:j+1 for i,j in cat.ids.items()}
        if not load:
            model,stats = train(h,self.ids,text,attrs)
            torch.save({"state":model.state_dict(),"dim":text.shape[1]},path)
            path.with_suffix(".json").write_text(json.dumps(stats,indent=2),"utf8")
        checkpoint = torch.load(path,map_location="cpu",weights_only=True)
        self.model = Towers(checkpoint["dim"])
        self.model.load_state_dict(checkpoint["state"])
        self.model.eval()
        with torch.no_grad():
            self.books = self.model.books(text,attrs)
            if not load:
                assert torch.equal(self.books,model.books(text,attrs))

    @torch.no_grad()
    def predict(self,seq,titles,m,d,level,group):
        cat = self.cat
        blocked = {cat.ids[i] for i in seq if i in cat.ids}
        prior = cat.recommend(seq,titles,m,d,group)
        base = prior["基础"]
        vectors,raw = [],[]
        for j in range(max(0,len(seq)-8),len(seq)):
            i = seq[j]
            if i in self.ids:
                vectors.append(self.books[self.ids[i]])
                if cat.title_known[cat.ids[i]]:
                    raw.append(cat.vectors[cat.ids[i]])
            else:
                title = attribute_value(titles[j]) if j<len(titles) else ""
                v = torch.from_numpy(cat.encoder.encode(["题名："+title]))
                vectors.append(self.model.books(v,torch.tensor([book_attrs(i,"")]))[0])
                if title:
                    raw.append(v[0])
        sem = []
        if raw:
            sims = cat.vectors @ torch.stack(raw).T
            s = (.7*sims.max(1).values+.3*sims.mean(1)).numpy()
            s[~cat.title_known] = -np.inf
            sem = cat.order(s,blocked)
        attrs = profile(m,d,level)
        if vectors or any(attrs):
            hist = torch.zeros(1,8,48)
            mask = torch.zeros(1,8,dtype=torch.bool)
            if vectors:
                hist[0,-len(vectors):] = torch.stack(vectors)
                mask[0,-len(vectors):] = True
            query = self.model.users(hist,mask,torch.tensor([attrs]))
            deep = cat.order((self.books[1:] @ query.T).flatten().numpy(),blocked)
        else:
            deep = base
        lists = {"base":base,"deep":deep}
        return dict(zip(NAMES,[base,prior["画像书目与语义"],deep,
                               blend(lists,{"base":1.,"deep":.5}),blend(lists,{"base":1.,"deep":1.})])),sem


def replay(data,date,encoder):
    start = pd.Timestamp(date)
    end = data[TIME].max()+pd.Timedelta(1,"ns") if date==DIAG else start+pd.DateOffset(months=2)
    past = data.loc[data[TIME]<start]
    h = past.loc[past[CALL].ne("")]
    seen = h.groupby(USER)[CALL].agg(set).to_dict()
    old = set(past.loc[past[TIME]<start-GAP,USER])
    mask = {i for i in h[CALL].unique() if bucket(i,5)==0}
    truth = {u:set(g[CALL])-seen.get(u,set())-{""} for u,g in data.loc[data[TIME].between(start,end,inclusive="left")].groupby(USER)}
    rows = []
    for name in ["历史回放","冷书模拟"]:
        print(date,name,flush=True)
        visible = h if name=="历史回放" else h.loc[~h[CALL].isin(mask)]
        fields = [visible.loc[visible[k].map(attribute_value).ne("")].groupby(USER)[k].last().to_dict()
                  for k in ("专业","单位","读者级别")]
        challenger = Challenger(visible,start,h[META],encoder,OUT/(date+"_"+name+".pt"))
        for u,y in truth.items():
            if not y:
                continue
            group = "老人" if u in old else "新人"
            seq = challenger.cat.index.profile.get(u,[])
            plans,sem = challenger.predict(seq,[],*(f.get(u,"") for f in fields),group)
            for method,order in plans.items():
                pool = [i for i in order if i not in seen.get(u,set())]
                top = pool[:10]
                assert len(set(top))==10
                rows.append(dict(window=date,scenario=name,user=u,group=group,method=method,pred="|".join(top),
                                 score=measure(top,y,group),truth=len(y),hits=len(y.intersection(top)),
                                 masked_truth=len(y&mask),raw_sem_100=len(y.intersection(sem)&mask),
                                 pool_100=len(y.intersection(pool)&mask),top_10=len(y.intersection(top)&mask)))
    return pd.DataFrame(rows)


def export(data,encoder,load=False):
    start = pd.Timestamp("2025-09-01")
    h = data.loc[(data[TIME]<start)&data[CALL].ne("")]
    books = pd.read_csv(PACKAGE/"中文入藏（新书）.csv",dtype=str,keep_default_na=False,encoding="gb18030")
    books[CALL] = books[CALL].map(attribute_value)
    meta = pd.concat([books.loc[books[CALL].ne(""),META],h[META]])
    model = Challenger(h,start,meta,encoder,OUT/"official.pt",load=load)
    roster = pd.read_csv(PACKAGE/"验证集发放.csv",dtype=str,keep_default_na=False)
    plan = json.loads((OUT/"plan.json").read_text("utf8"))["challenger"]
    rows = []
    for r in roster.to_dict("records"):
        seq = [i.strip() for i in r["历史借阅索书号列表"].split("|") if i.strip()]
        titles = r["历史借阅题名列表"].split("|")
        plans,_ = model.predict(seq,titles if len(titles)==len(seq) else [],r["专业"],r["学院"],r["读者类别"],r["用户类型"])
        top = plans[plan][:10]
        assert len(set(top))==10 and set(top)<=model.cat.ids.keys()-set(seq)
        rows.append({"读者ID":r["读者ID"],"推荐索书号列表":"|".join(top)})
    assert len(rows)==len({r["读者ID"] for r in rows})==956
    pd.DataFrame(rows).to_csv(OUT/"验证集结果_双塔挑战.csv",index=False,encoding="utf-8-sig")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict-only",action="store_true",help="用保存的正式模型重跑预测")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True,parents=True)
    torch.set_num_threads(4)
    data = load_interactions(BASE/"data/processed/borrowing_cleaned.csv")
    data[CALL] = data[CALL].map(attribute_value)
    encoder = MemoryEncoder()
    if args.predict_only:
        export(data,encoder,load=True)
        return
    frames = []
    for date in [*DEV,DIAG]:
        if date==DIAG:
            dev = pd.concat(frames)
            g = dev.groupby(["scenario","window","method","group"]).score.mean().unstack()
            scores = (.4*g["新人"]+.6*g["老人"]).groupby(["scenario","method"]).mean().unstack()
            gain = scores.sub(scores["基础"],axis=0)
            chosen = max(NAMES[2:],key=lambda m:gain[m].min())
            (OUT/"plan.json").write_text(json.dumps({"challenger":chosen,"worst_dev_gain":gain[chosen].min()},ensure_ascii=False,indent=2),"utf8")
        frame = replay(data,date,encoder)
        frame.to_csv(OUT/(date+".csv"),index=False,encoding="utf-8-sig")
        frames.append(frame)
    full = pd.concat(frames)
    groups = full.groupby(["window","scenario","method","group"]).score.mean().unstack()
    groups["total"] = .4*groups["新人"]+.6*groups["老人"]
    groups.to_csv(OUT/"summary.csv",encoding="utf-8-sig")
    audit = full.loc[full.scenario.eq("冷书模拟")].groupby(["window","method"])[["masked_truth","raw_sem_100","pool_100","top_10"]].sum()
    audit.to_csv(OUT/"cold_audit.csv",encoding="utf-8-sig")
    print(groups.total.unstack(),flush=True)
    export(data,encoder)


if __name__=="__main__":
    main()

