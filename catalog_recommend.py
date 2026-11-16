"""面向匿名读者的索书号推荐，输入仅含预测前已知画像、历史及可用目录。"""
import hashlib
import re
from collections import Counter, defaultdict
from functools import lru_cache

import numpy as np
import pandas as pd
import torch

from baseline import USER, TIME, attribute_value
from newcomer_replay import college
from older_call_recall import CALL, OlderCallRecall, WEIGHTS, blend

META = [CALL,"题名","责任者"]
METHODS = ["基础","画像","画像与书目","画像书目与语义"]


def major(value):
    return re.sub(r"本科$","",attribute_value(value)).strip()


def clc(call):
    m = re.match(r"([A-Z]+\d{1,3})",call.upper())
    return m.group(1) if m else call.split("/")[0]


def bucket(value, n):
    return int(hashlib.sha256(value.encode()).hexdigest()[:8],16)%n


class Profile:
    def __init__(self,h):
        self.counts = defaultdict(Counter)
        self.support = Counter()
        fields = {USER:h[USER], "major":h["专业"].map(major),"dept":h["单位"].map(college),
                  "category":h[CALL].map(clc)}
        table = pd.DataFrame(fields)
        for key in ("major","dept"):
            for name,group in table.groupby(key):
                if name:
                    self.support[key,name] = group[USER].nunique()
                    self.counts[key,name] = Counter(group.drop_duplicates([USER,"category"]).category)
        counts = Counter(table.drop_duplicates([USER,"category"]).category)
        self.global_prob = {c:n/sum(counts.values()) for c,n in counts.items()}

    @lru_cache(maxsize=2048)
    def lifts(self,m,d):
        prob = self.global_prob.copy()
        for key in [("dept",college(d)),("major",major(m))]:
            if self.support[key]<5:
                continue
            counts = self.counts[key]
            denom = sum(counts.values())+50
            prob = {c:(counts[c]+50*v)/denom for c,v in prob.items()}
        return {c:np.clip((v/self.global_prob[c])**.5,.4,3.) for c,v in prob.items()}


class Catalog:
    def __init__(self,history,start,metadata,encoder):
        h = history.loc[(history[TIME]<start)&history[CALL].ne("")].copy()
        h["单位"] = h["单位"].map(college)
        self.index = OlderCallRecall(h,start)
        self.profile = Profile(h)
        # 候选编号排序决定所有并列分数的次序。
        books = metadata.drop_duplicates(CALL,keep="last").set_index(CALL).sort_index()
        self.items = books.index.tolist()
        self.ids = {item:i for i,item in enumerate(self.items)}
        self.category = np.array([clc(i) for i in self.items])
        self.exact = np.array([i.split("/")[0] for i in self.items])
        self.author = books["责任者"].map(attribute_value).to_numpy(dtype=str)
        self.count = np.array([len(self.index.readers.get(i,())) for i in self.items])
        self.hot = np.array([self.index.hot[i] for i in self.items])
        self.title_known = books["题名"].map(attribute_value).ne("").to_numpy(dtype=bool)
        self.vectors = torch.from_numpy(encoder.encode(["题名："+t for t in books["题名"].map(attribute_value)]))
        self.encoder = encoder

    def order(self,scores,blocked):
        x = np.array(scores,dtype=float)
        x[list(blocked)] = -np.inf
        order = np.argsort(-x,kind="stable")[:100]
        return [self.items[i] for i in order if np.isfinite(x[i])]

    def recommend(self,seq,titles,m,d,group,profile=None):
        seen = set(seq)
        blocked = {self.ids[i] for i in seq if i in self.ids}
        channels = self.index.channels(seq,college(d))
        if group=="老人" and seq:
            base = blend(channels,WEIGHTS)
        elif seq:
            channels["base"] = self.order(self.count,blocked)
            base = blend(channels,{"base":1.,"cf":.75,"content":1.})
        else:
            base = self.order(self.hot,blocked)
        lifts = (profile or self.profile).lifts(m,d)
        lift = np.fromiter((lifts.get(c,1.) for c in self.category),float)
        channels = {"base":base,"profile":self.order((self.hot+.25)*lift,blocked)}
        # 没有可用群体时画像实验精确退回基础排序。
        if not any(abs(x-1)>1e-8 for x in lifts.values()):
            channels["profile"] = base
        weights = {"base":1.,"profile":.7}
        plans = {"基础":base,"画像":blend(channels,weights)}
        weights = dict(weights)
        if seq:
            lexical = np.zeros(len(self.items))
            seeds = []
            for rank,pos in enumerate(range(len(seq)-1,max(-1,len(seq)-9),-1),1):
                code = seq[pos]
                idx = self.ids.get(code)
                author = self.author[idx] if idx is not None else ""
                lexical += ((self.exact==code.split("/")[0]).astype(float)+2*((self.author==author)&bool(author)))/np.sqrt(rank)
                if idx is not None and self.title_known[idx]:
                    seeds.append(self.vectors[idx])
                elif pos<len(titles) and attribute_value(titles[pos]):
                    seeds.append(torch.from_numpy(self.encoder.encode(["题名："+titles[pos]])[0]))
            if lexical.max()>0:
                channels["book"] = self.order(lexical*(1+.15*np.log1p(self.hot))*lift,blocked)
                weights["book"] = .7
            plans["画像与书目"] = blend(channels,weights)
            if seeds:
                sims = self.vectors @ torch.stack(seeds).T
                sim = (.7*sims.max(dim=1).values+.3*sims.mean(dim=1)).numpy()
                sim[~self.title_known] = -np.inf
                channels["semantic"] = self.order(sim,blocked)
                weights["semantic"] = .7
        plans.setdefault("画像与书目",plans["画像"])
        plans["画像书目与语义"] = blend(channels,weights)
        for items in plans.values():
            assert len(items)>=10 and not seen.intersection(items)
        return plans

