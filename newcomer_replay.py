"""新人索书号推荐。只按开发窗选择配方，老人固定等权融合。"""
import json
from collections import Counter
from functools import lru_cache

import pandas as pd

from baseline import BASE, USER, TIME, attribute_value, load_interactions
from older_call_recall import CALL, OlderCallRecall, ranked, blend

OUT = BASE / "outputs/newcomer_replay"
DEV = ["2023-09-01","2024-03-01","2024-09-01"]
DIAG = "2024-11-01"
GAP = pd.Timestamp("2025-09-01")-pd.Timestamp("2025-01-17")
ALIASES = {"经管学院":"经济与管理学院","数计学院":"数学与计算机学院","智造学院":"智能制造学院",
           "土建学院":"土木与建筑工程学院","电气学院":"电气信息工程学院","生化学院":"生物与化学工程学院",
           "数学与计算机学院（大数据学院）":"数学与计算机学院","生物与化学工程学院（农学院）":"生物与化学工程学院"}
COLD = ["全局","近期","学期热门"]
WARM = COLD + ["学院平滑","个人历史","学院加历史"]


def college(value):
    name = attribute_value(value)
    return ALIASES.get(name,name)


class Newcomer:
    def __init__(self,data,start):
        h = data.loc[data[TIME]<start].copy()
        h["单位"] = h["单位"].map(college)
        self.depts = h.loc[h["单位"].ne("")].groupby(USER)["单位"].last().to_dict()
        self.index = OlderCallRecall(h,start)
        h = h.loc[h[CALL].ne("")]
        pairs = h.drop_duplicates([USER,CALL])
        self.pop = Counter(pairs[CALL])
        self.global_order = ranked(self.pop)
        self.departments = {d:Counter(g[CALL]) for d,g in h.drop_duplicates(["单位",USER,CALL]).groupby("单位") if d}
        self.total = sum(self.pop.values())
        # 当月与后一个月：只从过去年度的同月份借阅计数。
        months = [start.month,start.month%12+1]
        seasonal = h.loc[h[TIME].dt.month.isin(months)].drop_duplicates([USER,CALL])
        self.season = self.smoothed(Counter(seasonal[CALL]))
        self.dept_order = lru_cache(maxsize=256)(self._dept_order)

    def smoothed(self, counts):
        # 100 个读者-索书号对的全馆先验，归一化分母对排序为常数。
        return sorted(self.pop,key=lambda i:(-(counts[i]+100*self.pop[i]/self.total),i))

    def _dept_order(self,dept):
        counts = self.departments.get(college(dept))
        return self.smoothed(counts) if counts else self.global_order

    def predict(self,seq,dept,names):
        seen = set(seq)
        def unseen(order,k):
            from itertools import islice
            return list(islice((i for i in order if i not in seen),k))
        orders = {"全局":self.global_order,"近期":self.index.popular,"学期热门":self.season,
                  "学院平滑":self.dept_order(dept)}
        result = {n:unseen(orders[n],10) for n in names if n in orders}
        personal = [n for n in names if n in ("个人历史","学院加历史")]
        if personal:
            channels = self.index.channels(seq,dept)
            for n in personal:
                channels["base"] = unseen(orders["全局" if n=="个人历史" else "学院平滑"],100)
                result[n] = blend(channels,{"base":1,"cf":.75,"content":1})[:10]
        return result


def replay(data,date):
    start = pd.Timestamp(date)
    end = start+pd.DateOffset(months=2) if date!=DIAG else data[TIME].max()+pd.Timedelta(1,"ns")
    old = set(data.loc[data[TIME] < start-GAP,USER])
    target = data.loc[data[TIME].between(start,end,inclusive="left") & data[CALL].ne("") & ~data[USER].isin(old)]
    model = Newcomer(data,start)
    rows = []
    for u,g in target.groupby(USER):
        seq = model.index.profile.get(u,[])
        truth = set(g[CALL])-set(seq)
        if not truth:
            continue
        dept = model.depts.get(u,"")
        segment = "有历史" if seq else "无历史"
        for name,pred in model.predict(seq,dept,WARM if seq else COLD).items():
            assert len(pred)==len(set(pred))==10 and set(pred)<=model.pop.keys()-set(seq)
            hits = len(truth.intersection(pred))
            rows.append(dict(window=date,user=u,segment=segment,method=name,pred="|".join(pred),
                             truth=len(truth),hit=int(hits>0),recall=hits/len(truth)))
    frame = pd.DataFrame(rows)
    frame["score"] = 50*(frame.hit+frame.recall)
    return frame


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    data = load_interactions(BASE/"data/processed/borrowing_cleaned.csv")
    data[CALL] = data[CALL].map(attribute_value)
    dev = pd.concat([replay(data,d) for d in DEV],ignore_index=True)
    ref = pd.read_csv(BASE/"outputs/call_number_replay/per_user.csv",dtype={"user":str})
    keys = ["window","user"]
    expected = ref.loc[ref.group.eq("新人")&ref.method.eq("全局热门")].set_index(keys)
    actual = dev.loc[dev.method.eq("全局")].set_index(keys)
    for col in ["pred","hit","recall"]:
        pd.testing.assert_series_equal(actual[col].sort_index(),expected[col].sort_index(),check_dtype=False)
    # 以最终新人宏平均的贡献选方案；各开发窗等权，分群人数仍按该窗实际占比。
    sizes = dev.loc[dev.method.eq("全局")].groupby("window").size()
    plans = {}
    for group,names in [("无历史",COLD),("有历史",WARM)]:
        sums = dev.loc[dev.segment.eq(group)].groupby(["window","method"]).score.sum().unstack()
        values = sums.div(sizes,axis=0).mean()
        plans[group] = max(names,key=values.get)
    (OUT/"locked_plan.json").write_text(json.dumps(plans,ensure_ascii=False,indent=2),encoding="utf8")
    all_rows = pd.concat([dev,replay(data,DIAG)],ignore_index=True)
    all_rows.to_csv(OUT/"per_user.csv",encoding="utf-8-sig",index=False)
    groups = all_rows.groupby(["window","segment","method"])
    subs = groups[["hit","recall","score"]].mean()
    subs["users"] = groups.size()
    subs.to_csv(OUT/"segments.csv",encoding="utf-8-sig")
    old = pd.read_csv(BASE/"outputs/older_call_windows_v1/summary.csv").query("method=='等权融合+0'").set_index("window")
    rows = []
    for date,g in all_rows.groupby("window"):
        for name,chosen in [("对照",g.loc[g.method.eq("全局")]),("开发选定",g.loc[g.method.eq(g.segment.map(plans))])]:
            assert chosen.user.is_unique and len(chosen)==g.user.nunique()
            rows.append(dict(window=date,method=name,users=len(chosen),hit=chosen.hit.mean(),recall=chosen.recall.mean(),
                             newcomer=chosen.score.mean(),older=old.loc[date,"score"],
                             total=.4*chosen.score.mean()+.6*old.loc[date,"score"]))
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT/"summary.csv",encoding="utf-8-sig",index=False)
    print(plans);print(summary.round(4).to_string(index=False))


if __name__=="__main__":
    main()

