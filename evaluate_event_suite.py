"""回放事件模型；以相同全馆、读者、窗口、候选深度与冻结基准比较。"""
import argparse
import json
import numpy as np
import pandas as pd
import torch

from train_event_model import load_model, OUT, events, DEPTH, Query, profile_ids
from event_sequences import NegativePool, queries
from event_network import query_vectors, collate_queries, EventNet
from train_event_model import batch_loss
from baseline import BASE,TIME,USER,load_interactions,attribute_value
from newcomer_replay import DEV,DIAG,GAP,CALL,college
from older_call_recall import OlderCallRecall, WEIGHTS, blend
from roster_replay import score

REPORT = OUT/"evaluation"
RUN = OUT/"runs/gap30_gru_joint_topics"


def prepare(data,date,ck):
    depth = ck["manifest"].get("depth",DEPTH)
    start = pd.Timestamp(date)
    end = start+pd.DateOffset(months=2) if date!=DIAG else data[TIME].max()+pd.Timedelta(1,"ns")
    history = data.loc[data[TIME]<start]
    old = set(history.loc[history[TIME]<start-GAP,USER])
    h = history.loc[history[CALL].ne("")]
    index = OlderCallRecall(h.assign(单位=h["单位"].map(college)),start)
    ids = {code:i for i,code in enumerate(ck["codes"]) if i}
    assert set(ids)==set(h[CALL])==set(ck["legal"])
    seq = dict(events(h,ids,ck["manifest"]["events"]))
    pop = sorted(index.readers,key=lambda i:(-len(index.readers[i]),i))
    source = pd.read_csv(BASE/"outputs/catalog_v1"/(date+".csv"),dtype={"user":str})
    source = source.query("method=='基础' and scenario=='历史回放'").set_index("user")
    old_deep = pd.read_csv(BASE/"outputs/deep_challenger_v1"/(date+".csv"),dtype={"user":str})
    old_deep = old_deep.query("method=='双塔轻融合' and scenario=='历史回放'").set_index("user")
    portraits = pd.read_csv(BASE/"outputs/newcomer_portrait_v1/per_user.csv",dtype={"user":str})
    portraits = portraits.loc[(portraits.window==date)&(portraits['mode']=='常规')&
                             (portraits.method=='画像')&(portraits.segment=='无历史')].set_index("user")
    records = []
    for u,g in data.loc[data[TIME].between(start,end,inclusive="left")].groupby(USER):
        seen = index.books.get(u,set())
        y = set(g[CALL])-seen-{""}
        if not y:
            continue
        group = "老人" if u in old else "新人"
        e = seq.get(u,[])
        ordered = tuple(dict.fromkeys(i for event in e for i in event.ordered))
        prof = e[-1].profile if e else ("",)*5
        q = Query(u,start.value,tuple(e[-depth:]),ordered[-depth:],tuple(profile_ids(prof,start.value)),(),
                  frozenset(ordered),"infer")
        seeds = index.profile.get(u,[])
        channels = index.channels(seeds,college(prof[1]))
        if not seeds:
            base = index.popular[:100]
        elif group=="老人":
            base = blend(channels,WEIGHTS)
        else:
            channels["base"] = [i for i in pop if i not in seen][:100]
            base = blend(channels,{"base":1,"cf":.75,"content":1})
        assert base[:10]==source.at[u,"pred"].split("|")
        current = portraits.at[u,"pred"].split("|") if u in portraits.index else base[:10]
        records.append((q,group,y,base,current,old_deep.at[u,"pred"].split("|")))
    assert len(records)==len(source)
    return records,h


@torch.no_grad()
def evaluate(ck,model,records,date,epoch,*,official=True):
    model.eval()
    vec = model.encode_books(ck["bank"])
    rows = []
    for q,group,y,base,current,prior in records:
        raw = collate_queries([q],[official],[False],per_book=model.encoder=="book_mean")
        sim = (query_vectors(model,raw,vec[raw[0]])@vec.T)[0].numpy()
        sim[0] = -np.inf
        sim[list(q.seen)] = -np.inf
        ranked = [ck["codes"][i] for i in np.argsort(-sim,kind="stable")[:100]]
        fused = blend({"base":base,"event":ranked},{"base":1.,"event":.5})
        for name,items in [("当前提交",current),("前轮双塔轻融合",prior),("事件直排",ranked),("事件轻融合",fused)]:
            top = items[:10]
            assert len(set(top))==10 and not {ck["codes"][i] for i in q.seen}.intersection(top)
            hit,rec,nd,s = score(top,y,group)
            rows.append(dict(window=date,epoch=epoch,user=q.user,group=group,method=name,hit=hit,recall=rec,
                             ndcg=nd,score=s,truth=len(y),pred="|".join(top),
                             top100_hits=len(y.intersection(items)) if name.startswith("事件") else None))
    return pd.DataFrame(rows)


def iter_train_checkpoints(ck,h,date,objective,log_path,*,aligned_init=False,checkpoints=(4,8)):
    # 原文件未存优化器和 RNG；配对比较同一条新训练轨迹的第 4 和 8 轮。
    cfg = ck["manifest"]
    seed = cfg["seed"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    ids = {code:i for i,code in enumerate(ck["codes"]) if i}
    sequences = events(h,ids,cfg["events"])
    examples = queries(sequences,h[TIME].min().to_period("M").to_timestamp(),pd.Timestamp(cfg["last_label_bound"]),"joint",cfg.get("depth",DEPTH))
    from collections import defaultdict
    grouped = defaultdict(lambda:defaultdict(list))
    for q in examples:
        if objective == "joint" or q.task == objective:
            grouped[q.task][q.user].append(q)
    if not grouped or len(examples) != sum(cfg["queries"].values()):
        raise ValueError("训练样本与保存的窗口清单不一致")
    pools = [list(users.values()) for users in grouped.values()]
    sampler = NegativePool(h,ids)
    model = EventNet(ck["dim"],cfg["encoder"],aligned_init=aligned_init)
    opt = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    log = []
    for epoch in range(1,max(checkpoints)+1):
        losses = []
        # 全部目标使用联合样本规模计算预算，只统计实际完成的参数更新。
        steps = int(np.ceil(len(examples)/cfg["batch_size"]))
        while len(losses) < steps:
            batch = []
            for _ in range(cfg["batch_size"]):
                readers = pools[rng.integers(len(pools))]
                reader = readers[rng.integers(len(readers))]
                batch.append(reader[rng.integers(len(reader))])
            loss = batch_loss(model,ck["bank"],batch,sampler,ck["codes"],rng,cfg["official_view_rate"])
            if loss is not None:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),2.)
                opt.step()
                losses.append(loss.item())
        log.append(float(np.mean(losses)))
        if epoch in checkpoints:
            from copy import deepcopy
            yield epoch, deepcopy(model).eval()
        print(date,objective,epoch,round(log[-1],4),flush=True)
    log_path.write_text(json.dumps(log),"utf8")


def train_checkpoints(ck,h,date,objective,log_path,*,aligned_init=False,checkpoints=(4,8)):
    return tuple(model for _,model in iter_train_checkpoints(
        ck,h,date,objective,log_path,aligned_init=aligned_init,checkpoints=checkpoints))


def longer_run(ck,h,date):
    return train_checkpoints(ck,h,date,ck["manifest"]["objective"],REPORT/(date+"_epoch8_loss.json"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch-study",action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    data = load_interactions(BASE/"data/processed/borrowing_cleaned.csv")
    data[CALL] = data[CALL].map(attribute_value)
    if "性别" not in data:
        data["性别"] = ""
    REPORT.mkdir(parents=True,exist_ok=True)
    frames,paired_four = [],[]
    for date in (DEV if args.epoch_study else [*DEV,DIAG]):
        ck,model = load_model(RUN/date/"model.pt")
        records,h = prepare(data,date,ck)
        epochs = 4
        if args.epoch_study:
            four,model = longer_run(ck,h,date)
            part4 = evaluate(ck,four,records,date,4)
            part4.to_csv(REPORT/(date+"_paired_epoch4.csv"),index=False,encoding="utf-8-sig")
            paired_four.append(part4)
            epochs = 8
        part = evaluate(ck,model,records,date,epochs)
        if args.epoch_study:
            part.to_csv(REPORT/(date+"_paired_epoch8.csv"),index=False,encoding="utf-8-sig")
        frames.append(part)
        print("完成",date,flush=True)
    frame = pd.concat(frames)
    if paired_four:
        pd.concat(paired_four).to_csv(REPORT/"paired_rerun_epoch4.csv",index=False,encoding="utf-8-sig")
    frame.to_csv(REPORT/f"users_epoch{epochs}.csv",encoding="utf-8-sig",index=False)
    summary = frame.groupby(["window","method","group"])[["hit","recall","ndcg","score"]].mean()
    summary.to_csv(REPORT/f"groups_epoch{epochs}.csv",encoding="utf-8-sig")
    table = summary.score.unstack()
    table["total"] = .4*table["新人"]+.6*table["老人"]
    table.to_csv(REPORT/f"scores_epoch{epochs}.csv",encoding="utf-8-sig")
    print(table.total.unstack().round(4))


if __name__=="__main__":
    main()

