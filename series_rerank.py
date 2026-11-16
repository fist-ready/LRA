"""只在已借书与候选有明确书目关联时调整前十。"""
from collections import Counter,defaultdict
import json
import re
import unicodedata

import pandas as pd

from baseline import BASE,TIME,USER,attribute_value,load_interactions
from newcomer_replay import DEV,DIAG,CALL
from roster_replay import score

SRC = BASE/"outputs/catalog_v1"
OUT = BASE/"outputs/series_v1"
TRIALS = ["原排序","同系列","同系列或重复作者"]


def stem(title):
    t = unicodedata.normalize("NFKC",attribute_value(title)).lower()
    t = re.split(r"[.:：。]|第[一二三四五六七八九十百\d]+[部卷册辑季]",t)[0]
    t = re.sub(r"[（(]?[上中下][册部卷][）)]?$","",t)
    return re.sub(r"[^\w\u4e00-\u9fff]","",t)


class Links:
    def __init__(self,meta,history):
        self.by_series = defaultdict(set)
        self.by_author = defaultdict(set)
        self.meta = {}
        self.pop = Counter(history.drop_duplicates([USER,CALL])[CALL])
        for code,title,author in meta[[CALL,"题名","责任者"]].drop_duplicates(CALL,keep="last").itertuples(index=False):
            series = stem(title)
            author = re.sub(r"(编著|主编|编译|著|编)$","",attribute_value(author)).strip()
            self.meta[code] = (series,author)
            if len(series)>=3:
                self.by_series[series].add(code)
            if author:
                self.by_author[author,code.split("/")[0]].add(code)

    def choose(self,seq,titles):
        seen = set(seq)
        related = Counter()
        authors = Counter()
        for n,code in enumerate(reversed(seq[-12:]),1):
            name,author = self.meta.get(code,("",""))
            if not name and len(seq)==len(titles):
                name = stem(titles[-n])
            cls = code.split("/")[0]
            for other in self.by_series.get(name,set())-seen:
                _,a = self.meta[other]
                if other.split("/")[0]==cls and (not a or not author or a==author):
                    related[other] += 1/n
            if author:
                authors[author,cls] += 1
        secondary = Counter()
        for key,n in authors.items():
            if n>=2:
                for item in self.by_author[key]-seen:
                    secondary[item] += n
        def first(pool):
            return min(pool,key=lambda i:(-pool[i],-self.pop[i],i)) if pool else None
        match = first(related)
        return [None,match,match or first(secondary)]


def promote(order,item):
    if not item or item in order[:4]:
        return order
    result = [i for i in order if i!=item]
    return (result[:3]+[item]+result[3:])[:10]


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    data = load_interactions(BASE/"data/processed/borrowing_cleaned.csv")
    data[CALL] = data[CALL].map(attribute_value)
    from hashlib import sha256
    mask = {i for i in data[CALL].unique() if int(sha256(i.encode()).hexdigest()[:8],16)%5==0}
    rows = []
    selections = json.loads((SRC/"locked_plan.json").read_text("utf8"))
    for date in [*DEV,DIAG]:
        if date==DIAG:
            dev = pd.DataFrame(rows)
            g = dev.groupby(["scenario","window","method","group"]).score.mean().unstack()
            totals = (.4*g["新人"]+.6*g["老人"]).groupby(["scenario","method"]).mean().unstack()
            gains = totals.sub(totals["原排序"],axis=0)
            choice = max(TRIALS,key=lambda m:gains[m].min())
            (OUT/"locked_plan.json").write_text(json.dumps({"chosen":choice,"criterion":"开发窗真实回放和遮蔽模拟的平均增分取较小值；同分选原排序"},ensure_ascii=False),"utf8")
        base = pd.read_csv(SRC/(date+".csv"),dtype={"user":str})
        base = base.loc[base.method.eq(base.segment.map(selections))]
        start = pd.Timestamp(date)
        end = start+pd.DateOffset(months=2) if date!=DIAG else data[TIME].max()+pd.Timedelta(1,"ns")
        h = data.loc[(data[TIME]<start)&data[CALL].ne("")]
        seen = h.groupby(USER)[CALL].agg(set)
        truth = data.loc[data[TIME].between(start,end,inclusive="left")].groupby(USER)[CALL].agg(set)
        for scenario,part in base.groupby("scenario"):
            visible = h if scenario=="历史回放" else h.loc[~h[CALL].isin(mask)]
            index = Links(h,visible)
            sequences = visible.drop_duplicates([USER,CALL],keep="last").groupby(USER)[CALL].agg(list)
            for r in part.itertuples():
                seq = sequences.get(r.user,[])
                y = truth[r.user]-seen.get(r.user,set())-{""}
                candidates = index.choose(seq,[])
                for m,i in zip(TRIALS,candidates):
                    if i in seen.get(r.user,set()):
                        i = None
                    base_top = r.pred.split("|")
                    top = promote(base_top,i)
                    hrate,rec,nd,s = score(top,y,r.group)
                    assert len(set(top))==len(top)==10 and not set(top)&seen.get(r.user,set())
                    rows.append(dict(window=date,scenario=scenario,user=r.user,method=m,group=r.group,
                                     score=s,changed=top!=base_top,masked_hits=len(set(top)&mask&y),hits=len(set(top)&y)))
    f = pd.DataFrame(rows)
    f.to_csv(OUT/"per_user.csv",index=False,encoding="utf-8-sig")
    s = f.groupby(["window","scenario","method","group"]).score.mean().unstack()
    s["total"] = .4*s["新人"]+.6*s["老人"]
    s.to_csv(OUT/"summary.csv",encoding="utf-8-sig")
    directory = BASE.parent/"C题初赛验证集"
    books = pd.read_csv(directory/"中文入藏（新书）.csv",dtype=str,encoding="gb18030",keep_default_na=False)
    h = data.loc[(data[TIME]<pd.Timestamp("2025-09-01"))&data[CALL].ne("")]
    books[CALL] = books[CALL].map(attribute_value)
    index = Links(pd.concat([books.loc[books[CALL].ne("")],h]),h)
    roster = pd.read_csv(directory/"验证集发放.csv",dtype=str,keep_default_na=False).set_index("读者ID")
    base = pd.read_csv(SRC/"提交结果_选定.csv",dtype=str)
    result = []
    for u,encoded in base.itertuples(index=False):
        seq = [i.strip() for i in roster.at[u,"历史借阅索书号列表"].split("|") if i.strip()]
        choices = index.choose(seq,roster.at[u,"历史借阅题名列表"].split("|"))
        top = promote(encoded.split("|"),choices[TRIALS.index(choice)])
        assert len(set(top))==len(top)==10 and set(top)<=index.meta.keys() and not set(top)&set(seq)
        result.append({"读者ID":u,"推荐索书号列表":"|".join(top)})
    assert len({r["读者ID"] for r in result})==len(result)==956
    pd.DataFrame(result).to_csv(OUT/"提交结果_C题初赛.csv",encoding="utf-8-sig",index=False)
    print(choice);print(s.total.unstack().round(4))


if __name__=="__main__":
    main()

