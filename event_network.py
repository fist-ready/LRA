"""共享书塔、无序事件池化、可选 GRU 和公开书单推断视图。"""
import json
import re

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from baseline import BASE, attribute_value
from catalog_recommend import CALL,clc
from event_sequences import NS_DAY
from tower_model import hashed


def read_labels(path):
    dictionary = json.loads((BASE/"book_topics_v2.json").read_text(encoding="utf8"))
    allowed = {int(k) for k in dictionary["topics"]} | {-1,0}
    labels = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        code,*tags = json.loads(line)
        if code in labels or not tags or len(set(tags))!=len(tags) or any(type(i)!=int or i not in allowed for i in tags):
            raise ValueError(f"标注不合法 {code}")
        if any(i in {-1,0,99} for i in tags) and len(tags)>1:
            raise ValueError(f"特殊主题不能混用 {code}")
        labels[code] = tags
    return labels


def make_bank(meta,labels,master,encoder,use_topics):
    books = meta.drop_duplicates(CALL,keep="last").set_index(CALL).sort_index()
    items = [""]+books.index.tolist()
    titles = books["题名"].map(attribute_value)
    vec = torch.from_numpy(encoder.encode(["题名："+v for v in titles]))
    text = torch.cat([torch.zeros_like(vec[:1]),vec])
    attrs = [[0]*4]
    topics = np.zeros((len(items),36),dtype=np.float32)
    for j,(code,row) in enumerate(books.iterrows(),1):
        year = re.search(r"(?:18|19|20)\d{2}",attribute_value(row.get("出版日期")))
        attrs.append([hashed(attribute_value(row.get("责任者")),2048),hashed(clc(code),512),
                      hashed(attribute_value(row.get("出版者")),1024),int(year[0])//5-350 if year else 0])
        if not use_topics:
            continue
        # 排除冲突、标题不匹配，以及由未来补全其他元数据得出的标签。
        record = master.get(code)
        if not record or record["冲突字段"] or any(attribute_value(record[k])!=attribute_value(row.get(col))
                for k,col in [("题名","题名"),("责任者","责任者")]):
            topics[j,35] = 1
            continue
        tags = labels.get(code,[0])
        for tag in tags:
            topics[j,tag-1 if 1<=tag<=33 else (33 if tag==99 else 34)] = 1/len(tags)
    return items,(text,torch.tensor(attrs),torch.from_numpy(topics))


class EventNet(nn.Module):
    def __init__(self,text_dim,encoder,*,aligned_init=False):
        super().__init__()
        self.encoder = encoder
        self.attrs = nn.ModuleList(nn.Embedding(n,8,padding_idx=0) for n in [2048,512,1024,128])
        self.book = nn.Sequential(nn.Linear(text_dim+32+36,96),nn.ReLU(),nn.Linear(96,64))
        self.project_event = nn.Linear(66,64)
        if encoder=="gru" or aligned_init:
            # 实验 B 中使后续共享层的随机初值和 GRU 组完全一致。
            recurrent = nn.GRU(64,64,batch_first=True)
            if encoder=="gru":
                self.gru = recurrent
        self.fields = nn.ModuleList(nn.Embedding(512,8,padding_idx=0) for _ in range(5))
        self.reader = nn.Sequential(nn.Linear(64*2+40+3,96),nn.ReLU(),nn.Linear(96,64))

        if encoder=="sasrec":
            self.position = nn.Embedding(32,64)
            nn.init.normal_(self.position.weight,std=.02)
            self.attention = nn.MultiheadAttention(64,2,dropout=0.,batch_first=True)
            self.attn_norm = nn.LayerNorm(64)
            self.ff_norm = nn.LayerNorm(64)
            self.out_norm = nn.LayerNorm(64)
            self.ff = nn.Sequential(nn.Linear(64,128),nn.ReLU(),nn.Linear(128,64))

    def attentive_states(self,events):
        width = events.size(1)
        positions = torch.arange(width,device=events.device)
        hidden = events + self.position(positions)
        normalized = self.attn_norm(hidden)
        # 右填充。每个有效位置仅可见自己及更早的有效位置。
        future = positions[None,:] > positions[:,None]
        attended,_ = self.attention(normalized,normalized,normalized,attn_mask=future,need_weights=False)
        hidden = hidden + attended
        return self.out_norm(hidden + self.ff(self.ff_norm(hidden)))

    def encode_books(self,features):
        text,attrs,topics = features
        x = [text,topics]+[emb(attrs[...,j]) for j,emb in enumerate(self.attrs)]
        return F.normalize(self.book(torch.cat(x,dim=-1)),dim=-1)

    def users(self,pooled,lengths,sizes,gaps,profile,extra):
        events = torch.tanh(self.project_event(torch.cat([pooled,sizes[...,None],gaps[...,None]],dim=-1)))
        positions = torch.arange(events.size(1),device=events.device)
        valid = positions[None,:]<lengths[:,None]
        weights = valid*(positions+1).float().sqrt()
        if self.encoder=="gru":
            states,_ = self.gru(events)
            h = states[torch.arange(len(events),device=events.device),(lengths-1).clamp_min(0)]
        elif self.encoder=="sasrec":
            states = self.attentive_states(events)
            h = states[torch.arange(len(events),device=events.device),(lengths-1).clamp_min(0)]
        else:
            h = (events*weights[...,None]).sum(1)/weights.sum(1,keepdim=True).clamp_min(1)
        last = pooled[torch.arange(len(events),device=events.device),(lengths-1).clamp_min(0)]
        h = h*(lengths>0)[:,None]
        p = [emb(profile[:,j]) for j,emb in enumerate(self.fields)]
        return F.normalize(self.reader(torch.cat([h,last,*p,extra],dim=-1)),dim=-1)


def collate_queries(qs,views,hide,*,per_book=False):
    events,extras = [],[]
    for q,official,empty in zip(qs,views,hide):
        if empty:
            seq = []
        elif official:
            seq = [(None,(i,)) for i in q.official]
        elif per_book:
            # 展开共同的事件前缀；同事件同一时间，不制造逐条时间信息。
            seq = [(e.end,(item,)) for e in q.prefix for item in e.ordered]
        else:
            seq = [(e.end,e.items) for e in q.prefix]
        events.append(seq)
        idle = np.log1p(max(0,(q.time-q.prefix[-1].end)/NS_DAY)) if seq and q.prefix else 0.
        extras.append([idle,np.log1p(len(q.seen)) if seq else 0.,float(official)])
    width = max(1,max(map(len,events)))
    lengths = torch.tensor(list(map(len,events)))
    keys,groups = [],[]
    size = torch.zeros(len(qs),width)
    gap = torch.zeros_like(size)
    for b,seq in enumerate(events):
        for j,(t,items) in enumerate(seq):
            keys.extend(items)
            groups.extend([b*width+j]*len(items))
            size[b,j] = np.log1p(len(items))
            if t is not None and j:
                gap[b,j] = np.log1p((t-seq[j-1][0])/NS_DAY)
    return (torch.tensor(keys,dtype=torch.long),torch.tensor(groups,dtype=torch.long),
            lengths,size,gap,torch.tensor([q.profile for q in qs]),torch.tensor(extras,dtype=torch.float32))


def query_vectors(model,batch,embeddings):
    _,groups,lengths,size,gap,prof,extra = batch
    b,w = size.shape
    pooled = embeddings.new_zeros(b*w,64)
    pooled.index_add_(0,groups,embeddings)
    counts = torch.bincount(groups,minlength=b*w).reshape(b,w,1).clamp_min(1)
    return model.users(pooled.reshape(b,w,64)/counts,lengths,size,gap,prof,extra)

