"""独立重算开关实验指标、验证时间边界及推荐结果；不重新训练。"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from baseline import BASE, USER, ITEM, TIME, load_interactions, source_info, ground_truth, ranking_metrics, write_json


def verify(output, prediction=None):
    output=Path(output)
    report=json.loads((output/'experiment_report.json').read_text(encoding='utf-8'))
    plan=json.loads((output/'locked_plan.json').read_text(encoding='utf-8'))
    source=BASE/'data/2022-2025处理后借阅信息.xlsx'
    assert source_info(source)==report['protocol']['input']==plan['input']
    data=load_interactions(source)
    files=0
    for start, modes in report['windows'].items():
        for mode,result in modes.items():
            folder=output/mode/'snapshots'/start
            manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
            cutoff=pd.Timestamp(start)
            catalog=set(data.loc[data[TIME].lt(cutoff),ITEM])
            truth=ground_truth(data.loc[data[TIME].ge(cutoff)&data[TIME].lt(pd.Timestamp(manifest['end']))])
            rows=pd.read_csv(folder/'result.csv',dtype=str,keep_default_na=False)
            assert rows.shape==(len(truth),11) and rows[USER].is_unique and set(rows[USER])==set(truth)
            metrics=[]
            for row in rows.itertuples(index=False,name=None):
                assert len(set(row[1:]))==10 and set(row[1:])<=catalog
                metrics.append(ranking_metrics(list(row[1:]),truth[row[0]],10))
            for metric in ('recall@10','ndcg@10','score'):
                assert abs(np.mean([x[metric] for x in metrics])-result['metrics']['all'][metric])<1e-12
            assert all(pd.Timestamp(s['end'])<=cutoff for s in result['training'])
            files+=1
    snapshots=0
    for off_path in (output/'off/snapshots').glob('*/features.pkl'):
        on_path=output/'on/snapshots'/off_path.parent.name/'features.pkl'
        off,on=pd.read_pickle(off_path),pd.read_pickle(on_path)
        pd.testing.assert_frame_equal(off,on[off.columns])
        assert set(on)-set(off)=={'level_item_probability','level_category_probability','level_group_available'}
        assert np.isfinite(on.select_dtypes('number').to_numpy()).all()
        assert on['level_group_available'].isin([0,1]).all()
        manifest=json.loads((off_path.parent/'manifest.json').read_text(encoding='utf-8'))
        cutoff=pd.Timestamp(manifest['start'])
        catalog=set(data.loc[data[TIME].lt(cutoff),ITEM])
        assert set(on[ITEM])<=catalog and not on.duplicated([USER,ITEM]).any()
        truth=ground_truth(data.loc[data[TIME].ge(cutoff)&data[TIME].lt(pd.Timestamp(manifest['end']))])
        labels=[int(i in truth[u]) for u,i in on[[USER,ITEM]].itertuples(index=False,name=None)]
        np.testing.assert_array_equal(labels,on['label'])
        snapshots+=1
    means=report['selection']['development_means']
    assert report['selection']['selected_mode']==('on' if means['on']>means['off'] else 'off')
    summary={'source_unchanged':True,'paired_snapshots_verified':snapshots,'recommendation_files_recomputed':files,
             'base_features_exactly_equal':True,'training_boundaries_valid':True,'off_exactly_reproduced':report['off_exactly_reproduced']}
    if prediction:
        prediction=Path(prediction)
        meta=json.loads((prediction/'prediction_summary.json').read_text(encoding='utf-8'))
        assert meta['input']==plan['input'] and meta['reader_level']==plan['reader_level']
        cutoff=pd.Timestamp(meta['cutoff'])
        catalog=set(data.loc[data[TIME].lt(cutoff),ITEM])
        rec=pd.read_csv(prediction/'result.csv',dtype=str,keep_default_na=False)
        assert rec.shape==(meta['readers'],11) and rec[USER].is_unique
        if meta.get('reused_equivalent_off_artifacts'):
            assert set(rec[USER]) == set(data.loc[data[TIME].lt(cutoff), USER])
        for row in rec.itertuples(index=False,name=None):
            assert len(set(row[1:]))==10 and set(row[1:])<=catalog
        training=json.loads((prediction/'ranker_training.json').read_text(encoding='utf-8'))
        assert all(pd.Timestamp(s['end'])<=cutoff for s in training)
        summary['prediction_readers']=len(rec)
        summary['prediction_training_windows']=len(training)
    write_json(output/'verification_summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',default=str(BASE/'outputs/reader_level_v2'))
    parser.add_argument('--prediction-dir')
    args=parser.parse_args()
    verify(args.output_dir,args.prediction_dir)
