"""按全局截止时间训练事件双塔；不在训练命令中读取验证答案或选参。"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from baseline import BASE,TIME,USER,load_interactions,attribute_value
from catalog_recommend import CALL,bucket
from deep_challenger import MemoryEncoder
from event_sequences import DEPTH, Event, events,queries,NegativePool,Query,profile_ids
from event_network import EventNet,make_bank,read_labels,collate_queries,query_vectors
from newcomer_replay import DEV,DIAG

OFFICIAL = "2025-09-01"
DEFAULT_LABELS = BASE.parent/"全量书目_多主题标注结果_topic-v2.0.jsonl"
PACKAGE = BASE.parent/"C题初赛验证集"
OUT = BASE/"outputs/event_sequence_v1"
COLS = [CALL,"题名","责任者","出版者","出版日期"]


def read_inputs(args):
    h = load_interactions(BASE/"data/processed/borrowing_cleaned.csv")
    h[CALL] = h[CALL].map(attribute_value)
    h = h.loc[h[CALL].ne("")].copy()
    if "性别" not in h:
        h["性别"] = ""
    master = pd.read_csv(BASE/"data/book_catalog/索书号书目与主题标注.csv",dtype=str,keep_default_na=False).set_index(CALL)
    labels = read_labels(args.labels)
    if set(labels)!=set(master.index) or not master.index.is_unique:
        raise ValueError("标签与书目主表未一一对应")
    return h,labels,master.to_dict("index")


def official_meta(history):
    extra = pd.read_csv(PACKAGE/"中文入藏（新书）.csv",encoding="gb18030",dtype=str,keep_default_na=False)
    extra[CALL] = extra[CALL].map(attribute_value)
    legal = (set(history[CALL]) | set(extra[CALL])) - {""}
    roster = pd.read_csv(PACKAGE/"验证集发放.csv",dtype=str,keep_default_na=False)
    seeds = []
    for r in roster.to_dict("records"):
        codes = r["历史借阅索书号列表"].split("|")
        titles = r["历史借阅题名列表"].split("|")
        for j,code in enumerate(codes):
            if code.strip():
                seeds.append({CALL:code.strip(),"题名":titles[j] if len(titles)==len(codes) else ""})
    meta = pd.concat([pd.DataFrame(seeds),extra.loc[extra[CALL].ne(""),COLS],history[COLS]],ignore_index=True).fillna("")
    return meta,legal


def batch_loss(model,bank,qs,sampler,codes,rng,view_rate):
    chosen,options = [],[]
    for q in qs:
        pos = int(rng.choice(q.positives))
        neg = sampler.draw(q,codes[pos],rng)
        if neg:
            chosen.append(q)
            options.append([pos,*neg])
    if not chosen:
        return None
    batch = collate_queries(chosen,rng.random(len(chosen))<view_rate,rng.random(len(chosen))<.2,
                            per_book=model.encoder=="book_mean")
    device = bank[0].device
    batch = tuple(v.to(device) for v in batch)
    options = torch.tensor(options,device=device)
    all_ids,lookup = torch.unique(torch.cat([batch[0],options.flatten()]),return_inverse=True)
    books = model.encode_books(tuple(f[all_ids] for f in bank))
    n = batch[0].numel()
    readers = query_vectors(model,batch,books[lookup[:n]])
    candidates = books[lookup[n:].reshape_as(options)]
    logits = torch.einsum("bd,bnd->bn",readers,candidates)*10
    return F.cross_entropy(logits,torch.zeros(len(chosen),device=device,dtype=torch.long))


def fit_fold(data,labels,master,encoder,date,args):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    start = pd.Timestamp(date)
    history = data.loc[data[TIME]<start]
    if args.smoke:
        # 保留少量读者的完整时间跨度，以便烟测同时覆盖 next 和 window。
        history = history.loc[history[USER].map(lambda u: bucket(u,12)==0)]
    if history.empty:
        raise ValueError("截止前无训练数据")
    is_official = date==OFFICIAL and not args.smoke
    if is_official and args.cold_books:
        raise ValueError("冷书遮蔽是历史回放实验；正式截止日请使用全部已知流水。")
    meta,legal = official_meta(history) if is_official else (history[COLS],set(history[CALL]))
    end = min(start,data[TIME].max()+pd.Timedelta(1,"ns"))
    first = data[TIME].min().to_period("M").to_timestamp()
    if args.cold_books:
        hidden = {i for i in legal if bucket(i,5)==0}
        history = history.loc[~history[CALL].isin(hidden)]
    codes,cpu_bank = make_bank(meta,labels,master,encoder,not args.no_topics)
    ids = {code:j for j,code in enumerate(codes) if j}
    seq = events(history,ids,args.events)
    depth = getattr(args,"depth",DEPTH)
    examples = queries(seq,first,end,args.objective,depth)
    if not examples:
        raise ValueError("没有可用训练查询")
    # 每种目标总权重相同；同一目标内先等权抽读者，再抽其切分点。
    by_task = defaultdict(lambda:defaultdict(list))
    for q in examples:
        by_task[q.task][q.user].append(q)
    pools = [list(group.values()) for group in by_task.values()]
    sampler = NegativePool(history,ids)
    bank = tuple(f.to(args.device) for f in cpu_bank)
    model = EventNet(bank[0].shape[1],args.encoder,aligned_init=args.encoder=="sasrec").to(args.device)
    opt = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    name = f"{args.events}_{args.encoder}_{args.objective}_{'no_topics' if args.no_topics else 'topics'}"
    if depth!=DEPTH:
        name += f"_d{depth}"
    if args.cold_books:
        name += "_cold"
    directory = args.out/("smoke" if args.smoke else "runs")/name/date
    directory.mkdir(parents=True,exist_ok=True)
    model_path = directory/"model.pt"
    if model_path.exists() and not args.replace:
        raise FileExistsError(f"已有训练结果 {model_path}，换 --out 或显式 --replace")
    log = []
    for epoch in range(1,args.epochs+1):
        losses = []
        steps = 3 if args.smoke else int(np.ceil(len(examples)/args.batch_size))
        for _ in range(steps):
            batch = []
            for _ in range(args.batch_size):
                readers = pools[rng.integers(len(pools))]
                reader = readers[rng.integers(len(readers))]
                batch.append(reader[rng.integers(len(reader))])
            loss = batch_loss(model,bank,batch,sampler,codes,rng,args.view_rate)
            if loss is None:
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),2.)
            opt.step()
            losses.append(loss.item())
        if not losses or not np.isfinite(losses).all():
            raise RuntimeError("没有有效更新或损失非有限")
        log.append(float(np.mean(losses)))
        print(date,name,"epoch",epoch,"loss",round(log[-1],4),flush=True)
    manifest = dict(cutoff=date,history_max=str(history[TIME].max()),last_label_bound=str(end),
                    events=args.events,encoder=args.encoder,objective=args.objective,topics=not args.no_topics,
                    depth=depth,epochs=args.epochs,seed=args.seed,batch_size=args.batch_size,official_view_rate=args.view_rate,
                    smoke=args.smoke,cold_books=args.cold_books,history_rows=len(history),readers=len(seq),event_count=sum(len(s) for _,s in seq),
                    queries={k:sum(map(len,v.values())) for k,v in by_task.items()},loss=log,
                    label_sha256=hashlib.sha256(args.labels.read_bytes()).hexdigest())
    torch.save(dict(state=model.cpu().state_dict(),manifest=manifest,bank=cpu_bank,codes=codes,
                    legal=sorted(legal),dim=cpu_bank[0].shape[1]),directory/"model.tmp")
    (directory/"model.tmp").replace(model_path)
    (directory/"manifest.json").write_text(json.dumps(manifest,indent=2),"utf8")
    print("已保存",model_path,flush=True)
    return model_path


def load_model(path):
    ck = torch.load(path,map_location="cpu",weights_only=True)
    model = EventNet(ck["dim"],ck["manifest"]["encoder"])
    model.load_state_dict(ck["state"])
    model.eval()
    return ck,model


@torch.no_grad()
def rank(model,bookvec,q,legal,seen,official=True):
    batch = collate_queries([q],[official],[False],per_book=model.encoder=="book_mean")
    vec = query_vectors(model,batch,bookvec[batch[0]])
    scores = (bookvec@vec.T).squeeze(1).numpy()
    order = np.argsort(-scores,kind="stable")
    return [int(i) for i in order if i in legal and i not in seen][:10]


def export(path,output):
    ck,model = load_model(path)
    if ck["manifest"]["cutoff"]!=OFFICIAL or ck["manifest"]["smoke"]:
        raise ValueError("正式预测需要 2025-09-01 的全量权重")
    codes = ck["codes"]
    ids = {code:j for j,code in enumerate(codes)}
    legal = {ids[i] for i in ck["legal"]}
    t = pd.Timestamp(OFFICIAL)
    roster = pd.read_csv(PACKAGE/"验证集发放.csv",dtype=str,keep_default_na=False)
    with torch.no_grad():
        vec = model.encode_books(ck["bank"])
    rows = []
    for r in roster.to_dict("records"):
        history = [ids[i.strip()] for i in r["历史借阅索书号列表"].split("|") if i.strip()]
        last = pd.to_datetime(r["最近借阅日期"],errors="coerce")
        if pd.notna(last) and last>=t:
            raise ValueError("最近借阅超出预测截止")
        prefix = (Event(last.value,last.value,(),(),()),) if pd.notna(last) else ()
        prof = profile_ids([r[k] for k in ("专业","学院","年级","读者类别","性别")],t.value)
        q = Query(r["读者ID"],t.value,prefix,tuple(history[-ck["manifest"].get("depth",DEPTH):]),tuple(prof),(),frozenset(history),"infer")
        top = rank(model,vec,q,legal,q.seen)
        assert len(set(top))==10
        rows.append({"读者ID":q.user,"推荐索书号列表":"|".join(codes[i] for i in top)})
    assert len(rows)==len({r["读者ID"] for r in rows})==956
    output.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(output,index=False,encoding="utf-8-sig")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff",default=OFFICIAL)
    parser.add_argument("--suite",action="store_true",help="训练三开发窗、复用诊断和正式截止的独立模型")
    parser.add_argument("--events",choices=["single","gap30","day"],default="gap30")
    parser.add_argument("--encoder",choices=["mean","gru","book_mean","sasrec"],default="gru")
    parser.add_argument("--objective",choices=["next","window","joint"],default="joint")
    parser.add_argument("--no-topics",action="store_true")
    parser.add_argument("--cold-books",action="store_true",help="冷书模拟：保留书目，遮蔽约五分之一索书号的全部流水")
    parser.add_argument("--epochs",type=int,default=4)
    parser.add_argument("--batch-size",type=int,default=96)
    parser.add_argument("--depth",type=int,choices=[8,32],default=DEPTH)
    parser.add_argument("--seed",type=int,default=2026)
    parser.add_argument("--view-rate",type=float,default=.5)
    parser.add_argument("--out",type=Path,default=OUT)
    parser.add_argument("--labels",type=Path,default=DEFAULT_LABELS)
    parser.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--replace",action="store_true")
    parser.add_argument("--predict",type=Path,help="从正式权重导出，无需再次训练")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.predict:
        export(args.predict,args.out/"验证集结果_事件模型.csv")
        return
    if not (args.epochs>=1 and args.batch_size>=1 and 0<=args.view_rate<=1):
        parser.error("epochs、batch-size 或 view-rate 无效")
    data,labels,master = read_inputs(args)
    encoder = MemoryEncoder()
    dates = [*DEV,DIAG] if args.cold_books else [*DEV,DIAG,OFFICIAL]
    for date in (dates if args.suite else [pd.Timestamp(args.cutoff).strftime("%Y-%m-%d")]):
        path = fit_fold(data,labels,master,encoder,date,args)
        ck,m = load_model(path)
        if args.smoke:
            q = Query("test",pd.Timestamp(date).value,(),(),(0,)*5,(),frozenset(),"infer")
            with torch.no_grad():
                top = rank(m,m.encode_books(ck["bank"]),q,{j for j in range(1,len(ck["codes"]))},set())
            assert len(set(top))==10


if __name__=="__main__":
    main()

