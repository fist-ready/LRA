"""构建更多互不重叠的双月窗口，所有实验产物写入独立目录。"""
import hashlib
import json
import pandas as pd

from advanced_models import set_seed
from baseline import TIME, USER, attribute_value, load_interactions
from call_number_replay import BASE, CALL, GROUP_GAP
from older_call_ranker import DATES, OUT as PREVIOUS, ROOT, read, snapshot
from ranking import atomic_json, context_for, legacy_channels
from ranking_models import CandidateFeatures, TextEncoder

OUT = BASE/"outputs/older_call_windows_v1"
EXTRA = ["2023-07-01","2023-11-01","2024-01-01","2024-07-01"]
ALL = sorted(DATES+EXTRA)


def model_at(date):
    source = max(d for d in DATES if d<=date)
    path = ROOT/"snapshots"/source/"expanded.cbm"
    return path if path.exists() else PREVIOUS/source/"barcode.cbm"


def build_extra(data,date,encoder,folder):
    start = pd.Timestamp(date)
    end = start+pd.DateOffset(months=2)
    history = data.loc[data[TIME]<start]
    old = set(history.loc[history[TIME]<start-GROUP_GAP,USER])
    future = data.loc[data[TIME].between(start,end,inclusive="left")]
    users = sorted(set(future[USER]) & old)
    context = context_for(data,start,read(ROOT/"locked_plan.json")["previous_plan"])
    plan = read(ROOT/"locked_plan.json")["previous_plan"]
    raw,_ = legacy_channels(context,users,plan,folder)
    frame,new = CandidateFeatures(context,encoder).build(users,raw)
    raw.update(new)
    frame.to_pickle(folder/"features.pkl")
    atomic_json(folder/"channels.json",raw)
    atomic_json(folder/"manifest.json",dict(start=start.isoformat(),end=end.isoformat(),
                                           history_max=history[TIME].max().isoformat()))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    data_path = BASE/"data/processed/borrowing_cleaned.csv"
    data = load_interactions(data_path)
    for key in (CALL,"单位"):
        data[key] = data[key].map(attribute_value)
    encoder = TextEncoder(BASE/".model_deps/bge-small-zh-v1.5",ROOT/"embeddings")
    for date in ALL:
        folder = OUT/date
        folder.mkdir(exist_ok=True)
        dest = folder/"calls.pkl"
        model = model_at(date)
        signature = {str(f.relative_to(BASE)):hashlib.sha256(f.read_bytes()).hexdigest()
                     for f in [data_path,model,BASE/"prepare_older_windows.py",BASE/"older_call_ranker.py",
                               BASE/"older_call_recall.py",BASE/"ranking_models.py",ROOT/"locked_plan.json"]}
        if dest.exists() and (folder/"signature.json").exists() and read(folder/"signature.json")==signature:
            continue
        set_seed(2026,4)
        print("准备",date,model,flush=True)
        source = ROOT/"snapshots"/date
        if date in EXTRA:
            source = folder
            build_extra(data,date,encoder,folder)
        frame,answers,end = snapshot(data,date,[],encoder,folder=source,barcode_model=model)
        pd.to_pickle((frame,answers,end),folder/"calls.tmp")
        (folder/"calls.tmp").replace(dest)
        atomic_json(folder/"signature.json",signature)
        print(date,len(frame),len(answers),"完成",flush=True)


if __name__=="__main__":
    main()

