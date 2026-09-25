"""Train the pairwise matcher and score the full decision layer offline.

Splits by ENTITY, never by pair: the same entity's candidates appearing in both
train and validation would leak. The corpus stays whole for both splits, so the
validation entities face the true distractor density.

The disjoint rows only see conflicts among the sampled validation entities, so a
small n_val understates them. n_val = 0 takes every entity not used for training,
which is the closest this gets to predict.py's whole-country competition.

Usage: python train_eval.py [country] [n_train] [n_val]
"""
import gc
import pathlib
import pickle
import sys
import time

import lightgbm as lgb
import numpy as np
import pyarrow.parquet as pq
from sklearn.isotonic import IsotonicRegression

import blocking
import disjoint
import features
import strfeatures
from metric import choose_k, f05
from textnorm import norm

ROOT = "data/parquet"
CAND_CAP = 60


def cap_candidates(q, c, chan, cap=CAND_CAP):
    """Keep the best `cap` candidates per entity by max channel score."""
    best = chan.max(1)
    starts = np.flatnonzero(np.r_[True, q[1:] != q[:-1]])
    ends = np.r_[starts[1:], len(q)]
    keep = []
    for s, e in zip(starts, ends):
        if e - s <= cap:
            keep.append(np.arange(s, e))
        else:
            sel = np.argpartition(best[s:e], -cap)[-cap:]
            keep.append(s + np.sort(sel))
    k = np.concatenate(keep)
    return q[k], c[k], chan[k]


def run_block(names, addrs, corpus, tag):
    t0 = time.time()
    q, c, chan = blocking.generate(None, None, names, addrs, corpus, verbose=False)
    order = np.argsort(q, kind="stable")
    q, c, chan = q[order], c[order], chan[order]
    q, c, chan = cap_candidates(q, c, chan)
    print(f"  {tag}: {len(q):,} pairs ({len(q)/max(len(names),1):.1f}/entity) "
          f"in {time.time()-t0:.0f}s")
    return q, c, chan


def prepare_country(country, n_tr, n_va, gtm, rng):
    """Blocking + features for one country. Frees the corpus before returning,
    so peak memory is one shard regardless of how many countries we train on."""
    s1_tab, corpus_tab = blocking.load_shard(ROOT, "train", country)
    s1_ids = s1_tab.column("entity_id").to_pylist()
    c_ids = np.array(corpus_tab.column("entity_id").to_pylist(), dtype=object)
    print(f"\n{country}: {len(s1_ids):,} S1, {len(c_ids):,} corpus")

    t0 = time.time()
    corpus = blocking.build_corpus(corpus_tab, verbose=False)
    del corpus_tab
    print(f"  indexed in {time.time()-t0:.0f}s")

    if n_va <= 0:
        n_va = max(len(s1_ids) - n_tr, 1)
    pick = rng.choice(len(s1_ids), min(n_tr + n_va, len(s1_ids)), replace=False)
    split = int(len(pick) * n_tr / (n_tr + n_va))
    # Over the whole country's S1, not the sample: counted within a 15k sample
    # a chain's other branches are mostly missing, so train and validation
    # (and predict.py) would each see a different feature.
    dup = features.name_dup_counts(
        [norm(x) for x in s1_tab.column("business_name").to_pylist()])
    idf_lut = {t: float(corpus["index"]["idf"][i]) for i, t in
               enumerate(corpus["index"]["vocab"].to_pylist())
               if corpus["index"]["idf"][i] > 0}

    def prep(idx, tag):
        sub = s1_tab.take(idx)
        names, addrs = blocking.normalise(sub)
        ids = [s1_ids[i] for i in idx]
        q, c, chan = run_block(names, addrs, corpus, f"{country} {tag}")
        truth = [set((gtm.get(i) or "").split(",")) - {""} for i in ids]
        cand_ids = c_ids[c]
        y = np.fromiter((cand_ids[j] in truth[q[j]] for j in range(len(q))),
                        np.int8, len(q))
        is_s3 = np.fromiter((s.startswith("S3-") for s in cand_ids), bool, len(q))
        Xs = strfeatures.build(names, addrs, q,
                               corpus["names_arr"].take(c).to_pylist(),
                               corpus["addrs_arr"].take(c).to_pylist(), idf_lut)
        X = np.hstack([features.build(q, chan, is_s3,
                                      dup[idx]), Xs])
        return X, y, q, cand_ids, truth, ids

    out = (prep(pick[:split], "train"), prep(pick[split:], "valid"))
    del corpus, s1_tab, c_ids, idf_lut
    gc.collect()
    return out


def main():
    countries = (sys.argv[1] if len(sys.argv) > 1 else "India").split(",")
    n_tr = int(sys.argv[2]) if len(sys.argv) > 2 else 15000
    n_va = int(sys.argv[3]) if len(sys.argv) > 3 else 8000
    rng = np.random.default_rng(0)

    gt = pq.read_table(pathlib.Path(ROOT) / "train_ground_truth.parquet")
    gtm = dict(zip(gt.column("source1_entity_id").to_pylist(),
                   gt.column("matched_entity_ids").to_pylist()))
    del gt

    tr_parts, va_parts = [], []
    for ctry in countries:
        tr, va = prepare_country(ctry, n_tr, n_va, gtm, rng)
        tr_parts.append(tr)
        va_parts.append((ctry,) + va)

    Xtr = np.vstack([t[0] for t in tr_parts])
    ytr = np.concatenate([t[1] for t in tr_parts])
    del tr_parts
    gc.collect()

    # Validation entities are renumbered so several countries can share one
    # evaluation pass without their entity indices colliding.
    Xva_l, yva_l, qva_l, cva_l, truth_va, ids_va, ctry_va = [], [], [], [], [], [], []
    offset = 0
    for ctry, X, y, q, cand, truth, ids in va_parts:
        Xva_l.append(X); yva_l.append(y); qva_l.append(q + offset); cva_l.append(cand)
        truth_va += truth; ids_va += ids; ctry_va += [ctry] * len(ids)
        offset += len(ids)
    Xva = np.vstack(Xva_l); yva = np.concatenate(yva_l)
    qva = np.concatenate(qva_l); cva = np.concatenate(cva_l)
    del Xva_l, yva_l, qva_l, cva_l, va_parts
    gc.collect()
    print(f"\ntraining on {', '.join(countries)}: {len(ytr):,} pairs, "
          f"{len(ids_va):,} validation entities")
    print(f"  positives: train {ytr.mean():.2%}, valid {yva.mean():.2%}")

    t0 = time.time()
    model = lgb.train(
        dict(objective="binary", learning_rate=0.06, num_leaves=63,
             min_data_in_leaf=50, feature_fraction=0.9, bagging_fraction=0.8,
             bagging_freq=1, verbose=-1, num_threads=8),
        lgb.Dataset(Xtr, ytr, feature_name=features.NAMES + strfeatures.NAMES), num_boost_round=350)
    raw = model.predict(Xva)
    print(f"model trained in {time.time()-t0:.0f}s")

    # Calibration matters more than ranking here: the stopping rule consumes
    # probabilities, so a good ranker that is badly calibrated stops in the
    # wrong place. Fit isotonic on the first half of validation ENTITIES and
    # score only the second half, so the reported numbers are out-of-sample.
    # The calibration half still takes part in conflict resolution below: its
    # entities compete for the same records in the real pipeline.
    n_cal = len(ids_va) // 2
    cal = qva < n_cal
    iso = IsotonicRegression(out_of_bounds="clip").fit(raw[cal], yva[cal])
    p = iso.predict(raw)

    with open("model.pkl", "wb") as fh:
        pickle.dump({"model": model, "iso": iso}, fh)
    print("saved model.pkl")

    print("\nfeature importance:")
    for n, g in sorted(zip(features.NAMES + strfeatures.NAMES, model.feature_importance("gain")),
                       key=lambda x: -x[1])[:10]:
        print(f"  {n:12s} {g:12,.0f}")

    with open("valstate.pkl", "wb") as fh:
        pickle.dump({"p": p, "cand": cva, "q": qva, "truth": truth_va,
                     "ids": ids_va, "ctry": ctry_va}, fh)
    print("saved valstate.pkl (decision-rule tuning needs no re-blocking)")

    # ---- decision layer ----
    # Scored on the held-out half only; see the calibration note above.
    ev = np.arange(n_cal, len(ids_va))
    starts = np.flatnonzero(np.r_[True, qva[1:] != qva[:-1]])
    ends = np.r_[starts[1:], len(qva)]
    per_entity = {}
    for s, e in zip(starts, ends):
        per_entity[qva[s]] = (p[s:e], cva[s:e])

    def evaluate(policy, label):
        sc = np.empty(len(ev))
        npred = np.zeros(len(ev))
        for j, i in enumerate(ev):
            probs, cand = per_entity.get(i, (np.empty(0), np.empty(0, object)))
            order = np.argsort(-probs)
            probs, cand = probs[order], cand[order]
            k = policy(probs)
            pred = set(cand[:k])
            npred[j] = k
            sc[j] = f05(len(pred & truth_va[i]), len(truth_va[i]), k)
        print(f"  {label:26s} macro F0.5 = {sc.mean():.4f}   "
              f"mean k = {npred.mean():.2f}")
        return sc.mean()

    n_true = np.array([len(truth_va[i]) for i in ev])
    print(f"\nvalidation (held-out half): {len(ev):,} entities, "
          f"mean true n = {n_true.mean():.2f}, singletons {(n_true==0).mean():.2%}")
    best_fixed = max(evaluate(lambda pr, k=k: min(k, len(pr)), f"fixed top-{k}")
                     for k in (1, 2, 3, 4))
    for thr in (0.3, 0.5, 0.7):
        evaluate(lambda pr, t=thr: int((pr > t).sum()), f"threshold {thr}")
    adaptive = evaluate(
        lambda pr: choose_k(pr, float(np.prod(1.0 - pr)) if len(pr) else 1.0),
        "adaptive-k (p > 0.8*F_k)")

    print(f"\n  adaptive - best_fixed = {adaptive - best_fixed:+.4f}")
    # Walk the entity groups once; `qva == i` per entity would be O(n_ent*n_pairs).
    hit = np.zeros(len(ids_va))
    for s, e in zip(starts, ends):
        hit[qva[s]] = len(truth_va[qva[s]] & set(cva[s:e]))
    print(f"  blocking ceiling      = {f05(hit[ev], n_true, hit[ev]).mean():.4f}")

    # ---- disjointness (disjoint.py) ----
    # Same adaptive-k decision, but made jointly: every validation entity
    # (both halves) competes for records, and each record keeps one owner.
    # Record IDs are unique across countries, so one integer index covers all.
    _, rec = np.unique(cva, return_inverse=True)

    def decide(prob):
        acc = np.zeros(len(prob), bool)
        for s, e in zip(starts, ends):
            o = s + np.argsort(-prob[s:e], kind="stable")
            acc[o[:choose_k(prob[o], float(np.prod(1.0 - prob[o])))]] = True
        return acc

    def score_mask(acc, label):
        is_ev = qva >= n_cal
        kk = np.bincount(qva[acc & is_ev], minlength=len(ids_va))[ev]
        cc = np.zeros(len(ids_va))
        for j in np.flatnonzero(acc & is_ev):
            cc[qva[j]] += cva[j] in truth_va[qva[j]]
        owners = np.bincount(rec[acc])
        shared = (owners >= 2).sum() / max((owners >= 1).sum(), 1)
        f = f05(cc[ev], n_true, kk).mean()
        print(f"  {label:26s} macro F0.5 = {f:.4f}   mean k = {kk.mean():.2f}   "
              f"records shared = {shared:.2%}")
        return f

    print("\ndisjointness (one owner per record):")
    acc = decide(p)
    base = score_mask(acc, "adaptive-k, per entity")
    res = score_mask(disjoint.resolve_conflicts(qva, rec, p, acc),
                     "+ resolve_conflicts")
    red = score_mask(disjoint.resolve_conflicts(qva, rec, p, acc, redecide=True),
                     "+ resolve, redecide")
    pn = disjoint.sinkhorn_normalise(qva, rec, p)
    sk = score_mask(disjoint.resolve_conflicts(qva, rec, pn, decide(pn)),
                    "sinkhorn + resolve")
    print(f"\n  resolve - base        = {res - base:+.4f}")
    print(f"  redecide - base       = {red - base:+.4f}")
    print(f"  sinkhorn+resolve - base = {sk - base:+.4f}")


if __name__ == "__main__":
    main()
