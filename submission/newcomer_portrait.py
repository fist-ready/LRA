"""首次公开名单的新人画像模型。权重在四窗回放前固定。"""
from collections import defaultdict
import re

import numpy as np
import pandas as pd

from data_io import TIME, USER, attribute_value, college
from older_call_recall import CALL

GAP = pd.Timestamp("2025-09-01") - pd.Timestamp("2025-01-17")


WEIGHTS = {"学院":.30,"专业":.32,"年级":.12,"读者类别":.15,"性别":.03,"用户类型":.08}
LEVEL = {"本科学生":"学生","校外读者":"校外人员","社会读者":"校外人员"}


def stage(value,t):
    year = attribute_value(value)
    if not re.fullmatch(r"\d{4}",year):
        return ""
    grade = t.year-int(t.month<9)-int(year)+1
    return str(min(grade,5)) if grade>0 else ""


def portrait(dept,major,grade,level,sex,kind,t):
    level = attribute_value(level)
    return dict(学院=college(dept),专业=attribute_value(major).removesuffix("本科"),
                年级=stage(grade,t),读者类别=LEVEL.get(level,level),性别=attribute_value(sex),用户类型=kind)


class ProfileRecommender:
    def __init__(self,h,start,index,sex_evidence=()):
        h = h.loc[h[TIME]<start].copy()
        self.codes = sorted(index.readers)
        self.ids = {x:i for i,x in enumerate(self.codes)}
        self.global_mass = np.array([index.hot[i] for i in self.codes],dtype=float)
        self.global_mass /= self.global_mass.sum()
        first = h.groupby(USER)[TIME].min()
        # 新生群体使用首次出现后不足 GAP 的借阅作为历史类比。
        entry = h[USER].map(first)
        h = h.loc[h[CALL].ne("")].copy()
        h["用户类型"] = np.where((h[TIME]-entry.loc[h.index])<GAP,"新人","老人")
        h["学院"] = h["单位"].map(college)
        h["专业"] = h["专业"].map(lambda v:attribute_value(v).removesuffix("本科"))
        h["年级"] = [stage(v,t) for v,t in zip(h["年级"],h[TIME])]
        h["读者类别"] = h["读者级别"].map(lambda v:LEVEL.get(attribute_value(v),attribute_value(v)))
        h["性别"] = h["性别"].map(attribute_value) if "性别" in h else ""
        self.tables = {}
        for k in WEIGHTS:
            pairs = h.loc[h[k].ne(""),[USER,CALL,TIME,k]].drop_duplicates([USER,CALL,k],keep="last")
            for val,g in pairs.groupby(k):
                wt = .2+.8*2**(-(start-g[TIME]).dt.days/120)
                self.add(k,val,g[USER].nunique(),wt.groupby(g[CALL]).sum().to_dict())
        supplemental = defaultdict(lambda:defaultdict(set))
        for u,sex,seq in sex_evidence:
            if attribute_value(sex):
                supplemental[sex][u].update(i for i in seq if i in self.ids)
        for sex,users in supplemental.items():
            counts = defaultdict(int)
            for books in users.values():
                for i in books:
                    counts[i] += 1
            if ("性别",sex) not in self.tables:
                self.add("性别",sex,sum(bool(v) for v in users.values()),counts)

    def add(self,key,value,support,counts):
        if support<5:
            return
        index = np.array([self.ids[i] for i in counts],dtype=int)
        values = np.array(list(counts.values()),dtype=float)
        self.tables[key,value] = (support,index,values)

    def predict(self,attrs):
        # 全局分布 + 平滑群体分布的凸组合；无支持的权重自动交还全局。
        scores = self.global_mass.copy()
        for key,w in WEIGHTS.items():
            table = self.tables.get((key,attrs[key]))
            if table is None:
                continue
            n,ids,counts = table
            effective = w*n/(n+20)*counts.sum()/(counts.sum()+40)
            scores -= effective*self.global_mass
            scores[ids] += effective*counts/counts.sum()
        return [self.codes[j] for j in np.argsort(-scores,kind="stable")[:10]]
