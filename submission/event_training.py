import hashlib
import json
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from support import BASE, CALL, TIME, USER, attribute_value, bucket
from event_sequences import events, queries, NegativePool
from event_network import EventNet, make_bank, collate_queries, query_vectors

OFFICIAL="2025-09-01"
PACKAGE=BASE/"inputs"
COLS=[CALL,"题名","责任者","出版者","出版日期"]

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
    examples = queries(seq,first,end,args.objective)
    if not examples:
        raise ValueError("没有可用训练查询")
    # 每种目标总权重相同；同一目标内先等权抽读者，再抽其切分点。
    by_task = defaultdict(lambda:defaultdict(list))
    for q in examples:
        by_task[q.task][q.user].append(q)
    pools = [list(group.values()) for group in by_task.values()]
    sampler = NegativePool(history,ids)
    bank = tuple(f.to(args.device) for f in cpu_bank)
    # 使缓存命中、读取已有书向量和 BGE 首次初始化得到相同的双塔初值。
    torch.manual_seed(args.seed)
    model = EventNet(bank[0].shape[1],args.encoder).to(args.device)
    opt = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    name = f"{args.events}_{args.encoder}_{args.objective}_{'no_topics' if args.no_topics else 'topics'}"
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
                    epochs=args.epochs,seed=args.seed,batch_size=args.batch_size,official_view_rate=args.view_rate,
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
