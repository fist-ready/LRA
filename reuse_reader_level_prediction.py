"""仅在选定关闭且历史训练特征完全相同时，复用已核验的全历史产物。"""
import hashlib
import json
import shutil
from pathlib import Path
import pandas as pd
from baseline import BASE, source_info, write_json


def main():
    root = BASE / 'outputs/reader_level_v2'
    plan = json.loads((root/'locked_plan.json').read_text(encoding='utf-8'))
    old = json.loads((BASE/'outputs/ranking/locked_plan.json').read_text(encoding='utf-8'))
    assert plan['reader_level']['enabled'] is False
    assert plan['recipe'] == old['recipe'] and plan['previous_plan'] == old['previous_plan']
    assert plan['input'] == source_info(BASE/'data/2022-2025处理后借阅信息.xlsx') == old['input']
    for name, digest in plan['source_code'].items():
        assert hashlib.sha256((BASE/name).read_bytes()).hexdigest() == digest
    paths = sorted((root/'off/snapshots').glob('*/features.pkl'))
    assert len(paths) == 8
    for path in paths:
        pd.testing.assert_frame_equal(pd.read_pickle(path),pd.read_pickle(BASE/'outputs/ranking/snapshots'/path.parent.name/'features.pkl'))
    source = BASE/'outputs/ranking_prediction'
    meta = json.loads((source/'prediction_summary.json').read_text(encoding='utf-8'))
    assert meta['recipe'] == plan['recipe'] and meta['input'] == plan['input']
    target = BASE/'outputs/reader_level_prediction'
    target.mkdir(parents=True,exist_ok=True)
    files = ('result.csv','ranker.cbm','ranker.features.json','ranker_training.json')
    for name in files:
        shutil.copyfile(source/name,target/name)
    write_json(target/'prediction_summary.json',{**meta,'reader_level':plan['reader_level'],
        'reused_equivalent_off_artifacts':True,'source_directory':str(source)})
    write_json(target/'reuse_provenance.json',{'reason':'development selected off; all eight training feature tables identical',
        'feature_tables_compared':len(paths),'source_directory':str(source),
        'sha256':{name:hashlib.sha256((target/name).read_bytes()).hexdigest() for name in files}})
    print('关闭方案全部8个训练特征表与原方案完全一致，已复用全历史模型与推荐；下一步运行独立核验。')


if __name__ == '__main__':
    main()
