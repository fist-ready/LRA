"""Replay the exact ZIP entry point, then export the cohort-prior challenger separately."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from zipfile import ZipFile


def export_release_ranks(root):
    # Isolate the ZIP's module names from research modules with the same names.
    sys.path.insert(0,str(root))
    import pipeline
    from older_call_recall import OlderCallRecall
    bases = []
    fused = []
    original_base = pipeline.base_lists
    original_blend = pipeline.blend

    def observe_base(hist, roster, cutoff):
        original = original_base(hist, roster, cutoff)
        popular = OlderCallRecall(hist,cutoff).popular
        for base in original:
            bases.append(base if len(base)>=30 else base+[c for c in popular if c not in base][:30])
        return original

    def observe_blend(channels,weights):
        result = original_blend(channels,weights)
        fused.append(result)
        return result

    pipeline.base_lists = observe_base
    pipeline.blend = observe_blend
    pipeline.predict(root/"原版.csv")
    roster = list(csv.DictReader(pipeline.ROSTER.open(encoding="utf-8-sig")))
    it = iter(fused)
    records = [(r["读者ID"], (next(it) if r["历史借阅索书号列表"].strip() else base)[:30])
               for r,base in zip(roster,bases,strict=True)]
    assert next(it,None) is None
    (root/"ranks.json").write_text(json.dumps(records,ensure_ascii=False),"utf8")


def main():
    from prior_swap_study import OUT, DemandPrior, major, college, stage, prior_rerank
    from baseline import BASE, TIME, attribute_value, load_interactions
    from catalog_recommend import CALL
    import pandas as pd
    import collections
    package = BASE.parent/"submit"/"可运行项目包.zip"
    result = BASE.parent/"submit"/"验证集结果.csv"
    if hashlib.sha256(package.read_bytes()).hexdigest() != "64b1a6d91a47efd97f6d042fed4c8d089a771c4c211e64cc0ef296481a064666":
        raise ValueError("提交 ZIP 版本已变，请重新核对基准")
    with tempfile.TemporaryDirectory(dir=OUT) as temp:
        with ZipFile(package) as z:
            z.extractall(temp)
        root = next(Path(temp).glob("*/pipeline.py")).parent.resolve()
        subprocess.run([sys.executable,"-X","utf8",__file__,"--extract-ranks",str(root)],check=True)
        assert (root/"原版.csv").read_bytes() == result.read_bytes()
        codes=dict(json.loads((root/"ranks.json").read_text("utf8")))
        src=root/"inputs"
        roster=pd.read_csv(src/"验证集发放.csv",dtype=str,keep_default_na=False)
        hist=load_interactions(root/"work/processed/borrowing_cleaned.csv")
        hist[CALL]=hist[CALL].map(attribute_value)
        cutoff=pd.Timestamp("2025-09-01")
        hist=hist[(hist[TIME]<cutoff)&hist[CALL].ne("")]
        prior=DemandPrior(hist,cutoff)
        extra=pd.read_csv(src/"中文入藏（新书）.csv",encoding="gb18030",dtype=str,keep_default_na=False)
        legal=set(hist[CALL])|set(extra[CALL].map(attribute_value))-{""}
        original=pd.read_csv(result,dtype=str,keep_default_na=False)
        assert list(roster["读者ID"])==list(original["读者ID"])
        predictions=[]
        audit=collections.defaultdict(collections.Counter)
        for r,prev in zip(roster.to_dict("records"),original["推荐索书号列表"],strict=True):
            base=codes[r["读者ID"]]
            assert base[:10]==prev.split("|")
            attrs=(major(r["专业"]),college(r["学院"]),stage(r["年级"],cutoff))
            top=prior_rerank(base,attrs,prior)
            seen=set(r["历史借阅索书号列表"].split("|"))-{""}
            assert len(top)==len(set(top))==10 and set(top)<=legal and not seen.intersection(top)
            counts=audit[r["用户类型"]+("有历史" if seen else "无历史")]
            counts["readers"]+=1
            counts["changed_set"]+=int(set(top)!=set(base[:10]))
            counts["changed_order_or_set"]+=int(top!=base[:10])
            predictions.append("|".join(top))
        assert len(predictions)==956 and original["读者ID"].is_unique
        original["推荐索书号列表"]=predictions
        dest=OUT/"验证集结果_时序群体先验_实验.csv"
        original.to_csv(dest,index=False,encoding="utf-8-sig")
        (OUT/"preview_audit.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2),"utf8")
        print(dict(audit),flush=True)


if __name__=="__main__":
    if len(sys.argv)==3 and sys.argv[1]=="--extract-ranks":
        export_release_ranks(Path(sys.argv[2]))
    else:
        main()
