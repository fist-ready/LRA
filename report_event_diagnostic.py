from evaluate_event_suite import (BASE, REPORT, RUN, DIAG, CALL, prepare, evaluate, longer_run,
                                 load_interactions, attribute_value, load_model, torch)
import pandas as pd


def main():
    torch.set_num_threads(4)
    data=load_interactions(BASE/"data/processed/borrowing_cleaned.csv")
    data[CALL]=data[CALL].map(attribute_value)
    if "性别" not in data:
        data["性别"]=""
    ck,_=load_model(RUN/DIAG/"model.pt")
    records,h=prepare(data,DIAG,ck)
    four,eight=longer_run(ck,h,DIAG)
    tables=[]
    for n,m in [(4,four),(8,eight)]:
        part=evaluate(ck,m,records,DIAG,n)
        part.to_csv(REPORT/f"diagnostic_users_epoch{n}.csv",index=False,encoding="utf-8-sig")
        s=part.groupby(["method","group"]).score.mean().unstack()
        s["total"]=.4*s["新人"]+.6*s["老人"]
        tables.append(s.assign(epoch=n))
    pd.concat(tables).to_csv(REPORT/"diagnostic_epoch_study.csv",encoding="utf-8-sig")
    print(pd.concat(tables).round(4))


if __name__=="__main__":
    main()
