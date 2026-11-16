"""固定第三轮结构的读者级别开关对照；重建派生特征，保留不可变召回缓存。"""
import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd
from advanced_models import clc_prefix, set_seed
from advanced import paired_bootstrap
from baseline import BASE, USER, ITEM, TIME, load_interactions, source_info, ground_truth, evaluate_rankings, recommendations_frame, save_csv
from ranking import atomic_json, context_for, strategy_predictions, DEVELOPMENT, DIAGNOSTIC
from ranking_models import training_data, fit_ranker, rank_candidates
from reader_level import ReaderLevelConfig, ReaderLevelPreference


def extend_features(frame, context, config, external=None):
    result = frame.copy()
    if not config.enabled:
        return result
    categories = [clc_prefix(context.metadata.get(i, {}).get('索书号', '')) for i in context.items]
    preference = ReaderLevelPreference(context, config, categories)
    for column in ('level_item_probability', 'level_category_probability', 'level_group_available'):
        result[column] = np.float32(0)
    for user, rows in result.groupby(USER, sort=False):
        item, category, active = preference.probabilities(user, (external or {}).get(user))
        ids = rows[ITEM].map(context.item_index).to_numpy()
        result.loc[rows.index, 'level_item_probability'] = item[ids].astype(np.float32)
        result.loc[rows.index, 'level_category_probability'] = category[ids].astype(np.float32)
        result.loc[rows.index, 'level_group_available'] = np.float32(active)
    return result


def run(args):
    source = Path(args.source)
    output = Path(args.output_dir)
    if output.resolve() == source.resolve() or source.resolve() in output.resolve().parents:
        raise ValueError('对照输出必须独立于旧结果目录。')
    output.mkdir(parents=True, exist_ok=True)
    old = json.loads((source/'locked_plan.json').read_text(encoding='utf-8'))
    old_report = json.loads((source/'experiment_report.json').read_text(encoding='utf-8'))
    if source_info(args.input)['sha256'] != old['input']['sha256']:
        raise ValueError('输入与原实验不一致。')
    config = ReaderLevelConfig(True, args.min_users, args.strength, args.prior_weight)
    code = {name: hashlib.sha256((BASE/name).read_bytes()).hexdigest() for name in
            ('reader_level.py','reader_level_experiment.py','ranking_models.py','advanced_models.py','optimized_models.py')}
    protocol = {'version': 'reader-level-v1', 'source_identity': old['identity'], 'input': old['input'],
        'config': asdict(config), 'code': code, 'recipe': old['recipe'],
        'selection': 'three development windows mean; ties prefer off',
        'diagnostic_reused_not_blind': True, 'new_categorical_feature': False,
        'cache_policy': 'immutable legacy candidate/feature inputs; rebuild derived numeric features and retrain both rankers',
        'parameters_fixed_before_evaluation': True}
    identity = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    if (output/'protocol.json').exists():
        if json.loads((output/'protocol.json').read_text(encoding='utf-8')) != protocol:
            raise ValueError('源码或协议变更，请另选输出目录。')
    atomic_json(output/'protocol.json', protocol)
    data = load_interactions(args.input)
    snapshots = {'off': [], 'on': []}
    reports, detail = {}, {}
    recipe = old['recipe']
    manifests = list(old['training_snapshots'])
    manifests.append(json.loads((source/'snapshots'/DIAGNOSTIC/'manifest.json').read_text(encoding='utf-8')))
    selected = None
    plans = {}
    for original in manifests:
        start = original['start'][:10]
        if original['identity'] != old['identity']:
            raise ValueError('原快照身份不符。')
        context = context_for(data, start, old['previous_plan'])
        historical_levels = ReaderLevelPreference(context, config, ['']*len(context.items)).profiles
        frame = pd.read_pickle(original['path'])
        if not set(frame[ITEM]) <= context.baseline.items:
            raise ValueError('候选包含未来图书。')
        original_folder = Path(original['directory'])
        raw = json.loads((original_folder/'channels.json').read_text(encoding='utf-8'))
        target = data.loc[data[TIME].ge(pd.Timestamp(original['start'])) & data[TIME].lt(pd.Timestamp(original['end']))]
        truth = ground_truth(target)
        for mode in ('off','on'):
            cfg = ReaderLevelConfig(**{**asdict(config), 'enabled': mode == 'on'})
            folder = output/mode/'snapshots'/start
            folder.mkdir(parents=True, exist_ok=True)
            path = folder/'features.pkl'
            manifest = {**original, 'identity': identity+':'+mode, 'path': str(path), 'directory':str(folder),
                'reader_level':asdict(cfg), 'source_features_sha256':hashlib.sha256(Path(original['path']).read_bytes()).hexdigest()}
            if not (folder/'manifest.json').exists():
                rebuilt = extend_features(frame, context, cfg)
                pd.testing.assert_frame_equal(rebuilt[frame.columns], frame)
                rebuilt.to_pickle(path)
                atomic_json(folder/'manifest.json',manifest)
            else:
                if json.loads((folder/'manifest.json').read_text(encoding='utf-8')) != manifest:
                    raise ValueError('派生缓存不匹配。')
            if start in DEVELOPMENT or start == DIAGNOSTIC:
                atomic_json(output/'progress.json', {'stage':'training','window':start,'mode':mode})
                if not (folder/'evaluation.json').exists():
                    set_seed(2026,4)
                    train, used = training_data(snapshots[mode], start, True)
                    print(f'{start} {mode}: {len(train)} candidates', flush=True)
                    # 完整训练250棵后使用前100棵，与旧流程一致。
                    model, columns = fit_ranker(train, True, folder/'ranker.cbm', iterations=250)
                    features = pd.read_pickle(path)
                    ranked = rank_candidates(model, columns, features, raw['optimized'], True, 100, 200)
                    prediction = strategy_predictions(raw,{(True,100):ranked},recipe)
                    if any(len(v)!=10 or len(set(v))!=10 or not set(v)<=context.baseline.items for v in prediction.values()):
                        raise ValueError('输出列表不合法。')
                    metrics, per_user = evaluate_rankings(prediction,truth,context.baseline,10)
                    per_user['reader_level'] = per_user[USER].map(historical_levels).fillna('未知')
                    groups = {str(k):{'users':len(g), **{m:float(g[m].mean()) for m in ('recall@10','ndcg@10','score')}}
                              for k,g in per_user.groupby('reader_level')}
                    save_csv(per_user,folder/'per_user.csv')
                    save_csv(recommendations_frame(prediction,10),folder/'result.csv')
                    atomic_json(folder/'evaluation.json',{'metrics':metrics,'by_level':groups,'training':used})
                result = json.loads((folder/'evaluation.json').read_text(encoding='utf-8'))
                if mode == 'off':
                    previous = old_report['diagnostic'] if start == DIAGNOSTIC else next(d for d in old_report['development'] if d['start'][:10]==start)
                    for metric in ('recall@10','ndcg@10','score'):
                        if abs(result['metrics']['all'][metric]-previous['methods'][recipe['name']]['all'][metric])>1e-12:
                            raise ValueError('关闭臂未精确复现旧方案，停止比较。')
                reports.setdefault(start,{})[mode] = result
                detail[(start,mode)] = pd.read_csv(folder/'per_user.csv',dtype={USER:str})
                print(start,mode,result['metrics']['all'],flush=True)
            snapshots[mode].append(manifest)
        if start == DEVELOPMENT[-1]:
            means = {mode:float(np.mean([reports[d][mode]['metrics']['all']['score'] for d in DEVELOPMENT])) for mode in snapshots}
            selected = 'on' if means['on'] > means['off'] else 'off'
            for mode in snapshots:
                plans[mode] = {**old,'schema':2,'identity':identity+':'+mode,'reader_level':{**asdict(config),'enabled':mode=='on'},
                    'feature_version':'reader-level-v1','source_code':code,'training_snapshots':list(snapshots[mode]),
                    'snapshot_root':str(output/mode),'embedding_cache':str(source/'embeddings'), 'fit_iterations':250,
                    'selection':{'selected_mode':selected,'development_means':means,'based_on':'development only'}}
                atomic_json(output/mode/'locked_plan.json',plans[mode])
            atomic_json(output/'locked_plan.json',plans[selected])
            print('开发选定',selected,means,flush=True)
    comparison = paired_bootstrap(detail[(DIAGNOSTIC,'off')],detail[(DIAGNOSTIC,'on')],samples=5000)
    atomic_json(output/'experiment_report.json',{'protocol':protocol,'selection':plans[selected]['selection'],
        'windows':reports,'paired_comparison':comparison,'off_exactly_reproduced':True})
    save_csv(pd.DataFrame([{'window':w,'mode':mode,**r['metrics']['all']} for w,values in reports.items() for mode,r in values.items()]),output/'comparison.csv')
    atomic_json(output/'progress.json',{'stage':'evaluation_complete','selected':selected})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',default=str(BASE/'outputs/ranking'))
    parser.add_argument('--input',default=str(BASE/'data/2022-2025处理后借阅信息.xlsx'))
    parser.add_argument('--output-dir',default=str(BASE/'outputs/reader_level_v2'))
    parser.add_argument('--min-users',type=int,default=3)
    parser.add_argument('--strength',type=float,default=20.)
    parser.add_argument('--prior-weight',type=float,default=.25)
    run(parser.parse_args())
