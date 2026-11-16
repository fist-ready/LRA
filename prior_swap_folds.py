"""Rebuild fold-specific GRU-8 rankings with and without interactions for masked books."""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from baseline import BASE, TIME, USER
from catalog_recommend import CALL, bucket
from event_encoder_b import read_data
from event_network import collate_queries, query_vectors
from event_sequences import DEPTH, Query, events, profile_ids, queries
from evaluate_event_suite import RUN, iter_train_checkpoints, load_model
from newcomer_portrait import ProfileRecommender, portrait
from newcomer_replay import DEV, DIAG, GAP, college
from older_call_recall import OlderCallRecall, WEIGHTS, blend

OUT = BASE / "outputs" / "prior_swap_study"
LONG_SPAN = pd.Timestamp("2026-10-30") - pd.Timestamp("2025-09-01")


def horizon_ends(date):
    start = pd.Timestamp(date)
    short = start + pd.DateOffset(months=2)
    if date == DIAG:
        short = pd.Timestamp("2025-01-17")
    ends = {"short": short}
    if date == DEV[0]:
        ends["long"] = start + LONG_SPAN
    return ends


def fold_data(data, date, masked):
    start = pd.Timestamp(date)
    before = data[data[TIME] < start]
    full = before[before[CALL].ne("")]
    hidden = {c for c in full[CALL].unique() if masked and bucket(c, 5) == 0}
    visible = full[~full[CALL].isin(hidden)]
    old = set(before.loc[before[TIME] < start - GAP, USER])
    return full, visible, hidden, old


def run(date, masked, seed=2026):
    start = pd.Timestamp(date)
    scenario = "masked" if masked else "normal"
    directory = OUT if seed == 2026 else OUT / str(seed)
    directory.mkdir(exist_ok=True)
    dest = directory / f"{date}_{scenario}.jsonl"
    if dest.exists():
        return
    data = read_data()
    full, visible, hidden, old = fold_data(data, date, masked)
    ck, _ = load_model(RUN / date / "model.pt")
    ids = {c: j for j, c in enumerate(ck["codes"]) if j}
    assert set(ids) == set(full[CALL]) and not hidden.intersection(visible[CALL])
    seqs = dict(events(visible, ids, "gap30"))
    if masked:
        train_queries = queries(list(seqs.items()), visible[TIME].min().to_period("M").to_timestamp(), start, "joint")
        ck["manifest"]["queries"] = dict(Counter(q.task for q in train_queries))
        del train_queries
    ck["manifest"]["seed"] = seed
    for _, model in iter_train_checkpoints(ck, visible, date, "joint", dest.with_suffix(".loss.json"), checkpoints=(8,)):
        pass
    index = OlderCallRecall(visible.assign(单位=visible["单位"].map(college)), start)
    pop = sorted(index.readers, key=lambda i: (-len(index.readers[i]), i))
    profiles = ProfileRecommender(visible, start)
    cold_list, _ = profiles.predict(portrait("", "", "", "", "", "新人", start), [])
    original_seen = full.groupby(USER)[CALL].agg(set).to_dict()
    end = max(horizon_ends(date).values())
    roster = data.loc[(data[TIME] >= start) & (data[TIME] < end)].groupby(USER)
    with torch.inference_mode(), dest.with_suffix(".tmp").open("w", encoding="utf8") as stream:
        vec = model.encode_books(ck["bank"])
        for u, future in roster:
            banned = original_seen.get(u, set())
            if not set(future[CALL]) - banned - {""}:
                continue
            group = "老人" if u in old else "新人"
            seq = seqs.get(u, [])
            ordered = tuple(dict.fromkeys(i for e in seq for i in e.ordered))
            attrs = seq[-1].profile if seq else ("",) * 5
            seeds = index.profile.get(u, [])
            channels = index.channels(seeds, college(attrs[1])) if seeds else {}
            if not seeds:
                original = cold_list if group == "新人" else index.popular
                base = list(original) + [c for c in index.popular if c not in original]
            elif group == "老人":
                base = blend(channels, WEIGHTS)
            else:
                channels["base"] = [i for i in pop if i not in banned][:100]
                base = blend(channels, {"base": 1, "cf": .75, "content": 1})
            base = [c for c in base if c not in banned][:100]
            q = Query(u, start.value, tuple(seq[-DEPTH:]), ordered[-DEPTH:],
                      tuple(profile_ids(attrs, start.value)), (), frozenset(ordered), "infer")
            batch = collate_queries([q], [True], [False])
            sim = (query_vectors(model, batch, vec[batch[0]]) @ vec.T)[0].numpy()
            sim[0] = -np.inf
            sim[[ids[c] for c in banned]] = -np.inf
            event = [ck["codes"][i] for i in np.argsort(-sim, kind="stable")[:100]]
            release = blend({"base": base, "event": event}, {"base": 1, "event": .5}) if seeds else base
            row = dict(user=u, group=group, base=base, event=event, release=release,
                       visible_history=len(ordered), recent=[ck["codes"][i] for i in ordered[-DEPTH:]], attrs=attrs)
            for key in ("base", "event", "release"):
                assert len(row[key]) == len(set(row[key])) == 100 and not banned.intersection(row[key])
            print(json.dumps(row, ensure_ascii=False), file=stream)
    dest.with_suffix(".tmp").replace(dest)
    print(date, scenario, "rankings complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", choices=["normal","masked"])
    parser.add_argument("--seed", type=int, choices=[2026,2027], default=2026)
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True, parents=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    for date in [*DEV, DIAG]:
        run(date, args.scenario == "masked", args.seed)
