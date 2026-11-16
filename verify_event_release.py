"""构建提交 ZIP，在独立目录快跑和原表重训，验证后发布两个文件。"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from zipfile import ZipFile, ZIP_DEFLATED

import pandas as pd

BASE=Path(__file__).resolve().parent
RELEASE=BASE/"outputs/release_event_v1"
ROOT=RELEASE/"C题初赛项目"
SUBMIT=BASE.parent/"submit"
CSV="验证集结果.csv"
ARCHIVE="可运行项目包.zip"

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def run(root,script,*args):
    subprocess.run([sys.executable,"-X","utf8",str(root/script),*args],cwd=root,check=True)

def main():
    frozen=ROOT/CSV
    target=RELEASE/ARCHIVE
    run(ROOT,"verify.py","--check-reference")
    files=list(ROOT.glob("*.py"))
    files += [ROOT/k for k in ["README.md","建模过程.md","requirements.txt","config.json","epoch_scores.csv","reference.sha256","book_topics_v2.json","work/processed/borrowing_cleaned.csv"]]
    files += [p for folder in ["inputs","models","evidence"] for p in (ROOT/folder).rglob("*") if p.is_file()]
    with ZipFile(target.with_suffix(".tmp"),"w",compression=ZIP_DEFLATED,compresslevel=5) as z:
        for f in sorted(files):
            z.write(f,(Path(ROOT.name)/f.relative_to(ROOT)).as_posix())
    target.with_suffix(".tmp").replace(target)
    with tempfile.TemporaryDirectory(prefix="verify_",dir=RELEASE) as dest:
        with ZipFile(target) as z:
            assert z.testzip() is None
            z.extractall(dest)
        t=Path(dest)/ROOT.name
        assert not (t/CSV).exists()
        run(t,"replay_epochs.py")
        run(t,"predict.py")
        run(t,"verify.py","--check-reference")
        run(t,"train.py")
        run(t,"verify.py","--check-reference")
        assert digest(t/CSV)==digest(frozen)
    prev=RELEASE/"previous_submit"
    old=pd.read_csv(prev/CSV,dtype=str,keep_default_na=False).set_index("读者ID")
    new=pd.read_csv(frozen,dtype=str,keep_default_na=False).set_index("读者ID")
    readers=pd.read_csv(ROOT/"inputs/验证集发放.csv",dtype=str,keep_default_na=False).set_index("读者ID")
    assert old.index.equals(new.index) and new.index.equals(readers.index)
    has=readers["历史借阅索书号列表"].str.strip().ne("")
    changed=old.iloc[:,0].ne(new.iloc[:,0])
    assert not changed[~has].any()
    report={"zip_bytes":target.stat().st_size,"zip_sha256":digest(target),"csv_sha256":digest(frozen),
            "history_readers":int(has.sum()),"changed_lists":int(changed.sum()),
            "changed_sets":sum(set(a.split("|"))!=set(b.split("|")) for a,b in zip(new.iloc[:,0],old.iloc[:,0])),
            "verified_load":True,"verified_retrain":True}
    for source in [target,frozen]:
        tmp=SUBMIT/(source.name+".tmp")
        shutil.copy2(source,tmp)
        assert digest(tmp)==digest(source)
    replaced=[]
    try:
        for source in [target,frozen]:
            (SUBMIT/(source.name+".tmp")).replace(SUBMIT/source.name)
            replaced.append(source.name)
        assert digest(SUBMIT/CSV)==report["csv_sha256"] and digest(SUBMIT/ARCHIVE)==report["zip_sha256"]
    except Exception:
        for name in replaced:
            shutil.copy2(prev/name,SUBMIT/name)
        raise
    (RELEASE/"verified.json").write_text(json.dumps(report,indent=2),"utf8")
    print(report,flush=True)

if __name__=="__main__":
    main()
