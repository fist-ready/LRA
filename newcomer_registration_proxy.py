"""模拟如果真正首次出现的读者已有独立登记画像，估计其潜在效果。"""
import pandas as pd
from evaluate_newcomer_portrait import OUT, per_user
from newcomer_portrait import ProfileRecommender, portrait
from newcomer_replay import CALL, DEV, DIAG
from baseline import USER,TIME,BASE,attribute_value,load_interactions


def main():
    data = load_interactions(BASE/"data/processed/borrowing_cleaned.csv")
    data[CALL] = data[CALL].map(attribute_value)
    rows = []
    for date in [*DEV,DIAG]:
        start = pd.Timestamp(date)
        end = data[TIME].max()+pd.Timedelta(1,"ns") if date==DIAG else start+pd.DateOffset(months=2)
        h = data.loc[data[TIME]<start]
        model = ProfileRecommender(h,start)
        y = data.loc[data[TIME].between(start,end,inclusive="left") & ~data[USER].isin(h[USER])]
        for u,g in y.groupby(USER):
            truth = set(g[CALL])-{""}
            if not truth:
                continue
            row = g.iloc[0]
            # 仅此反事实实验使用首条流水附带的属性模拟独立登记；读者自身流水不进训练。
            p = portrait(row["单位"],row["专业"],row["年级"],row["读者级别"],row.get("性别",""),"新人",start)
            rows.extend(per_user(u,date,"首借登记模拟","无历史",truth,model,p,[]))
    f = pd.DataFrame(rows)
    f.to_csv(OUT/"registration_per_user.csv",index=False,encoding="utf-8-sig")
    g = f.groupby(["window","method"])
    s = g[["hit","recall","score"]].mean()
    s["users"] = g.size()
    s.to_csv(OUT/"registration_summary.csv",encoding="utf-8-sig")
    print(s.round(4))


if __name__=="__main__":
    main()

