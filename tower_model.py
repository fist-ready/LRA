"""冻结题名向量，学习匿名读者与书目的匹配。0 号索引用于填充。"""
from bisect import bisect_left
from collections import defaultdict
from hashlib import blake2b

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from baseline import USER,TIME,attribute_value
from catalog_recommend import CALL,clc,major
from newcomer_replay import college


def hashed(value,size):
    if not value:
        return 0
    return 1+int.from_bytes(blake2b(value.encode(),digest_size=8).digest(),"little")%(size-1)


def profile(m,d,level):
    return [hashed(major(m),512),hashed(college(d),512),hashed(attribute_value(level),512)]


class Towers(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.a = nn.Embedding(2048,12,padding_idx=0)
        self.c = nn.Embedding(512,12,padding_idx=0)
        self.profiles = nn.ModuleList(nn.Embedding(512,8,padding_idx=0) for _ in range(3))
        self.book = nn.Sequential(nn.Linear(dim+24,96),nn.ReLU(),nn.Linear(96,48))
        self.user = nn.Sequential(nn.Linear(48+24,80),nn.ReLU(),nn.Linear(80,48))

    def books(self,text,attrs):
        return F.normalize(self.book(torch.cat([text,self.a(attrs[...,0]),self.c(attrs[...,1])],dim=-1)),dim=-1)

    def users(self,history,mask,profiles):
        weights = mask.float()*torch.arange(1,history.shape[1]+1,device=history.device).sqrt()
        interest = (history*weights.unsqueeze(-1)).sum(1)/weights.sum(1,keepdim=True).clamp_min(1)
        attrs = [emb(profiles[:,j]) for j,emb in enumerate(self.profiles)]
        return F.normalize(self.user(torch.cat([interest,*attrs],dim=1)),dim=1)


def book_attrs(call,author):
    return [hashed(attribute_value(author),2048),hashed(clc(call),512)]


def samples(history,ids):
    """同一时刻的多本借书共享之前的画像；负样本仅从此前可见且本人未借书中抽取。"""
    rng = np.random.default_rng(2026)
    first = history.drop_duplicates(CALL)
    pools = defaultdict(list)
    for code,t in first[[CALL,TIME]].itertuples(index=False):
        for key in ("*",clc(code)):
            pools[key].append((t.value,ids[code]))
    clock = {k:[x[0] for x in values] for k,values in pools.items()}
    examples = []
    for _,g in history.groupby(USER,sort=False):
        seen,seq = set(),[]
        prof = ["","",""]
        for t,session in g.groupby(TIME,sort=False):
            codes = list(dict.fromkeys(session[CALL]))
            forbidden = seen|{ids[x] for x in codes}
            for code in codes:
                if ids[code] in seen or not(seq or any(prof)):
                    continue
                negatives = []
                for key in [clc(code)]*4+["*"]*8:
                    stop = bisect_left(clock[key],t.value)
                    for _ in range(12 if stop else 0):
                        j = pools[key][rng.integers(stop)][1]
                        if j not in forbidden and j not in negatives:
                            negatives.append(j)
                            break
                    if len(negatives)==8:
                        break
                if len(negatives)==8:
                    seeds = seq[-8:]
                    examples.append(([0]*(8-len(seeds))+seeds,profile(*prof),[ids[code],*negatives]))
            for code in codes:
                j = ids[code]
                if j in seq:
                    seq.remove(j)
                seq.append(j)
                seen.add(j)
            for j,key in enumerate(["专业","单位","读者级别"]):
                valid = [v for v in session[key].map(attribute_value) if v]
                if valid:
                    prof[j] = valid[-1]
    return tuple(torch.tensor(x) for x in zip(*examples))


def train(history,ids,text,attrs):
    torch.manual_seed(2026)
    seeds,prof,choices = samples(history,ids)
    model = Towers(text.shape[1])
    opt = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    losses = []
    for _ in range(4):
        total = 0.
        for ix in torch.randperm(len(seeds)).split(128):
            hist = seeds[ix]
            hide = torch.rand(len(ix),1)<.2
            mask = hist.ne(0)&~hide
            query = model.users(model.books(text[hist],attrs[hist]),mask,prof[ix])
            books = model.books(text[choices[ix]],attrs[choices[ix]])
            logits = torch.einsum("bd,bnd->bn",query,books)*10
            loss = F.cross_entropy(logits,torch.zeros(len(ix),dtype=torch.long))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach())*len(ix)
        losses.append(total/len(seeds))
    return model.eval(),{"examples":len(seeds),"epoch_loss":losses}

