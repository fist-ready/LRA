"""训练样本：预测时的前缀和未来首次新借集合分开构造。"""
from dataclasses import dataclass
from bisect import bisect_left

import numpy as np
import pandas as pd

from baseline import USER, TIME, attribute_value
from catalog_recommend import CALL, clc, major
from newcomer_replay import college
from newcomer_portrait import stage
from tower_model import hashed

NS_DAY = 86400 * 10**9
PROFILE_COLS = ["专业","单位","年级","读者级别","性别"]
DEPTH = 8


@dataclass(frozen=True)
class Event:
    start: int
    end: int
    items: tuple
    profile: tuple
    ordered: tuple  # 仅给模拟官方书单的视图；同一时刻用编号打破并列。


def events(frame, ids, mode):
    result = []
    for u,g in frame.groupby(USER,sort=False):
        sequence = []
        last = None
        attrs = [""]*len(PROFILE_COLS)
        for time,code,*values in g[[TIME,CALL,*PROFILE_COLS]].itertuples(index=False,name=None):
            t = time.value
            boundary = (last is None or
                        (mode=="single" and t!=last) or
                        (mode=="gap30" and t-last>30*60*10**9) or
                        (mode=="day" and t//NS_DAY!=last//NS_DAY))
            if boundary:
                if last is not None:
                    sequence.append(Event(begin,last,tuple(sorted(items)),tuple(attrs),tuple(items)))
                begin,items = t,{}
            items[ids[code]] = None
            for j,value in enumerate(values):
                v = attribute_value(value)
                if v:
                    attrs[j] = v
            last = t
        if last is not None:
            sequence.append(Event(begin,last,tuple(sorted(items)),tuple(attrs),tuple(items)))
        result.append((u,sequence))
    return result


def profile_ids(values,t):
    m,d,y,level,sex = values
    return [hashed(x,512) for x in (major(m),college(d),stage(y,pd.Timestamp(t)),level,sex)]


@dataclass
class Query:
    user: str
    time: int
    prefix: tuple
    official: tuple
    profile: tuple
    positives: tuple
    seen: frozenset
    task: str


def make_query(user,t,history,targets,task,depth=DEPTH):
    ordered = tuple(dict.fromkeys(i for e in history for i in e.ordered))
    seen = frozenset(ordered)
    pos = tuple(sorted(set(targets)-seen))
    if not pos:
        return None
    attrs = history[-1].profile if history else ("",)*5
    return Query(user,t,tuple(history[-depth:]),ordered[-depth:],tuple(profile_ids(attrs,t)),pos,seen,task)


def queries(sequences, first_day, observed_end, objective, depth=DEPTH):
    """双月窗口从统一的日历起点开始；只取已完整观测的标签窗。"""
    result = []
    for u,seq in sequences:
        if objective in ("next","joint"):
            for j,e in enumerate(seq):
                q = make_query(u,e.start,seq[:j],e.items,"next",depth)
                if q:
                    result.append(q)
        if objective in ("window","joint"):
            for start in pd.date_range(first_day,observed_end,freq="2MS"):
                end = start+pd.DateOffset(months=2)
                if end.value>observed_end.value:
                    break
                # 跨边界的事件在采样切分点处只有部分可见，跳过歧义样本。
                if any(e.start < start.value <= e.end or e.start < end.value <= e.end for e in seq):
                    continue
                history = [e for e in seq if e.end<start.value]
                labels = {i for e in seq if start.value<=e.start and e.end<end.value for i in e.items}
                q = make_query(u,start.value,history,labels,"window",depth)
                if q:
                    result.append(q)
    return result


class NegativePool:
    def __init__(self,history,ids):
        self.pools = {}
        first = history.drop_duplicates(CALL)
        for code,t in first[[CALL,TIME]].itertuples(index=False):
            for key in ("*",clc(code)):
                self.pools.setdefault(key,[]).append((t.value,ids[code]))
        self.times = {k:[t for t,_ in v] for k,v in self.pools.items()}

    def draw(self,q,code,rng,count=12):
        excluded = q.seen.union(q.positives)
        got = {}
        for key in (clc(code),"*"):
            pool = self.pools.get(key,[])
            n = bisect_left(self.times.get(key,[]),q.time)
            if not n:
                continue
            limit = count//2 if key!="*" else count
            for j in rng.integers(n,size=count*8):
                i = pool[j][1]
                if i not in excluded:
                    got[i] = None
                    if len(got)>=limit:
                        break
        return list(got) if len(got)==count else []

