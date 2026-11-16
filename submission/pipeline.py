"""原始借阅表、官方读者清单与新书目录的处理、正式训练及 Top-10 融合。"""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

import build_book_catalog
from base_rank import base_lists
from data_io import attribute_value, read_history, TIME
from event_network import read_labels, collate_queries, query_vectors
from event_sequences import Event, Query, DEPTH, profile_ids
from event_training import fit_fold, load_model, official_meta, OFFICIAL
from older_call_recall import blend, CALL
from preprocess import run as clean

ROOT = Path(__file__).resolve().parent
INPUT = ROOT/"inputs"
WORK = ROOT/"work"
CLEAN = WORK/"processed/borrowing_cleaned.csv"
MASTER = WORK/"catalog.csv"
CHECKPOINT = ROOT/"models/event.pt"
CONFIG = ROOT/"config.json"
RAW = INPUT/"2022-2025处理后借阅信息.xlsx"
ROSTER = INPUT/"验证集发放.csv"
LABELS = INPUT/"topics.jsonl"

class TitleEncoder:
    """题名编码使用随包冻结 BGE。相同语料、同版本编码器可复用结果。"""
    def encode(self,texts):
        from transformers import AutoModel, AutoTokenizer
        folder=ROOT/"models/bge"
        cache=WORK/"titles.npz"
        key=hashlib.sha256((folder/"provenance.json").read_bytes()+json.dumps(texts,ensure_ascii=False).encode()).hexdigest()
        self.title_key=key
        if cache.exists():
            with np.load(cache,allow_pickle=False) as stored:
                if stored["key"].item()==key:
                    return stored["vectors"]
        # 训练数据相同可直接复用随包参数内的冻结题名向量，双塔仍从随机初始化训练。
        features=ROOT/"models/title_manifest.json"
        if features.exists() and CHECKPOINT.exists():
            frozen=json.loads(features.read_text("utf8"))
            if frozen["title_key"]==key:
                bank=torch.load(CHECKPOINT,map_location="cpu",weights_only=True)
                return bank["bank"][0][1:].numpy().copy()
        tokenizer=AutoTokenizer.from_pretrained(folder,local_files_only=True)
        model=AutoModel.from_pretrained(folder,local_files_only=True).eval()
        unique=list(dict.fromkeys(texts))
        results=[]
        with torch.inference_mode():
            for i in range(0,len(unique),64):
                tokens=tokenizer(unique[i:i+64],padding=True,truncation=True,max_length=128,return_tensors="pt")
                cls=model(**tokens).last_hidden_state[:,0]
                results.append(torch.nn.functional.normalize(cls,dim=1).numpy())
        vectors=np.concatenate(results)
        indices={t:i for i,t in enumerate(unique)}
        out=vectors[[indices[t] for t in texts]]
        tmp=cache.with_suffix(".tmp")
        with tmp.open("wb") as f:
            np.savez(f,key=key,vectors=out)
        tmp.replace(cache)
        return out


def prepare(raw):
    clean(raw,WORK/"processed")
    with TemporaryDirectory(dir=WORK) as tmp:
        build_book_catalog.build(Path(tmp))
        (Path(tmp)/"索书号书目与主题标注.csv").replace(MASTER)
    fields=["读者条码","条形码","操作类型","处理时间","单位","专业","年级","读者级别","索书号","题名","责任者","出版者","出版日期"]
    cleaned=pd.read_csv(CLEAN,dtype=str,keep_default_na=False)
    if "性别" in cleaned:
        fields.append("性别")
    cleaned[fields].to_csv(CLEAN,index=False,encoding="utf-8-sig")
    hist=read_history(CLEAN)
    hist[CALL]=hist[CALL].map(attribute_value)
    hist=hist[hist[CALL].ne("")].copy()
    if "性别" not in hist:
        hist["性别"]=""
    labels=read_labels(LABELS)
    master=pd.read_csv(MASTER,dtype=str,keep_default_na=False).set_index(CALL)
    if not master.index.is_unique:
        raise ValueError("索书号目录有重复")
    print("合法索书号、读者书单及新入藏元数据已整合",len(master),"条；主题覆盖",len(set(labels)&set(master.index)),flush=True)
    return hist,labels,master.to_dict("index")


def train(raw,config):
    torch.manual_seed(config["seed"])
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    hist,labels,master=prepare(raw)
    with TemporaryDirectory(dir=WORK) as tmp:
        args=SimpleNamespace(seed=config["seed"],smoke=False,cold_books=False,events=config["events"],
                             encoder=config["encoder"],objective=config["objective"],
                             no_topics=False,device="cpu",out=Path(tmp),replace=False,epochs=config["epochs"],batch_size=96,
                             view_rate=.5,labels=LABELS)
        encoder=TitleEncoder()
        path=fit_fold(hist,labels,master,encoder,OFFICIAL,args)
        path.replace(CHECKPOINT)
        (path.parent/"manifest.json").replace(ROOT/"models/event_manifest.json")
        fingerprint={"title_key":encoder.title_key,"clean_sha256":hashlib.sha256(CLEAN.read_bytes()).hexdigest(),
                     "new_books_sha256":hashlib.sha256((INPUT/"中文入藏（新书）.csv").read_bytes()).hexdigest()}
        (ROOT/"models/title_manifest.json").write_text(json.dumps(fingerprint,indent=2),"utf8")
    predict(ROOT/"验证集结果.csv")


@torch.inference_mode()
def predict(output):
    torch.set_num_threads(1)
    ck,model=load_model(CHECKPOINT)
    cfg=json.loads(CONFIG.read_text("utf8"))
    cutoff=pd.Timestamp(OFFICIAL)
    manifest=ck["manifest"]
    if (manifest["cutoff"],manifest["epochs"],manifest["encoder"],manifest["objective"],manifest["smoke"])!=(OFFICIAL,cfg["epochs"],"gru","joint",False):
        raise ValueError("训练参数与本版配置不匹配")
    fingerprint=json.loads((ROOT/"models/title_manifest.json").read_text("utf8"))
    for file,key in [(CLEAN,"clean_sha256"),(INPUT/"中文入藏（新书）.csv","new_books_sha256")]:
        if hashlib.sha256(file.read_bytes()).hexdigest()!=fingerprint[key]:
            raise ValueError(f"{file.name} 与训练快照不一致，请先执行 train")
    hist=read_history(CLEAN)
    roster=pd.read_csv(ROSTER,dtype=str,keep_default_na=False)
    extra=pd.read_csv(INPUT/"中文入藏（新书）.csv",encoding="gb18030",dtype=str,keep_default_na=False)
    legal=(set(hist.loc[hist[TIME]<cutoff,CALL].map(attribute_value))|set(extra[CALL].map(attribute_value)))-{""}
    if legal!=set(ck["legal"]):
        raise ValueError("候选书库已变化，请执行 train 重建")
    codes=ck["codes"]
    ids={code:i for i,code in enumerate(codes)}
    positions=np.array(sorted(ids[c] for c in legal))
    vectors=model.encode_books(ck["bank"])
    base=base_lists(hist,roster,cutoff)
    results=[]
    for r,original in zip(roster.to_dict("records"),base,strict=True):
        seen=list(dict.fromkeys(v.strip() for v in r["历史借阅索书号列表"].split("|") if v.strip()))
        banned=set(seen)
        if seen:
            if any(code not in ids for code in seen):
                raise ValueError("出现权重书库之外的历史种子，请执行 train 重建")
            last=pd.to_datetime(r["最近借阅日期"],errors="coerce")
            if pd.notna(last) and last>=cutoff:
                raise ValueError("最近借阅时间越过截止日")
            prefix=(Event(last.value,last.value,(),(),()),) if pd.notna(last) else ()
            sequence=tuple(ids[c] for c in seen)
            attrs=profile_ids([r[k] for k in ("专业","学院","年级","读者类别","性别")],cutoff.value)
            q=Query(r["读者ID"],cutoff.value,prefix,sequence[-DEPTH:],tuple(attrs),(),frozenset(sequence),"infer")
            batch=collate_queries([q],[True],[False])
            sim=(query_vectors(model,batch,vectors[batch[0]])@vectors.T)[0].numpy()
            eligible=positions[~np.isin(positions,sequence)]
            order=eligible[np.argsort(-sim[eligible],kind="stable")[:cfg["fusion"]["depth"]]]
            event=[codes[i] for i in order]
            top=blend({"base":original,"event":event},{k:cfg["fusion"][k] for k in ("base","event")})[:10]
        else:
            top=original[:10]
        if len(set(top))!=10 or not set(top)<=legal or set(top)&banned:
            raise ValueError(f"Top-10 校验失败 {r['读者ID']}")
        results.append((r["读者ID"],"|".join(top)))
    frame=pd.DataFrame(results,columns=["读者ID","推荐索书号列表"])
    assert frame["读者ID"].is_unique and frame["读者ID"].tolist()==roster["读者ID"].tolist()
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    tmp=output.with_suffix(".tmp")
    frame.to_csv(tmp,index=False,encoding="utf-8-sig",lineterminator="\r\n")
    tmp.replace(output)
    print("已生成并核验",len(frame),"名读者的 Top-10，SHA256",hashlib.sha256(output.read_bytes()).hexdigest(),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=["prepare","train","predict"])
    parser.add_argument("--raw",type=Path,default=RAW)
    parser.add_argument("--output",type=Path,default=ROOT/"验证集结果.csv")
    args=parser.parse_args()
    if args.action=="train":
        train(args.raw,json.loads(CONFIG.read_text("utf8")))
    elif args.action=="prepare":
        torch.set_num_threads(1)
        h,_,_=prepare(args.raw)
        meta,_=official_meta(h[h[TIME]<pd.Timestamp(OFFICIAL)])
        titles=meta.drop_duplicates(CALL,keep="last").set_index(CALL).sort_index()["题名"].map(attribute_value)
        TitleEncoder().encode(["题名："+t for t in titles])
    else:
        predict(args.output)
