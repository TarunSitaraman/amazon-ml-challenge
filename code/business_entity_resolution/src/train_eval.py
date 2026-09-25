"""Train the pairwise matcher and score the full decision layer offline.

Splits by ENTITY, never by pair: the same entity's candidates appearing in both
train and validation would leak. The corpus stays whole for both splits, so the
validation entities face the true distractor density.

Usage: python train_eval.py [country] [n_train] [n_val]
"""
import pathlib
import sys
import time

import lightgbm as lgb
import numpy as np
import pyarrow.parquet as pq
from sklearn.isotonic import IsotonicRegression

import blocking
import features
from metric import choose_k, f05

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


def main():
    country = sys.argv[1] if len(sys.argv) > 1 else "India"
    n_tr = int(sys.argv[2]) if len(sys.argv) > 2 else 15000
    n_va = int(sys.argv[3]) if len(sys.argv) > 3 else 8000
    rng = np.random.default_rng(0)

    s1_tab, corpus_tab = blocking.load_shard(ROOT, "train", country)
    s1_ids = s1_tab.column("entity_id").to_pylist()
    c_ids = np.array(corpus_tab.column("entity_id").to_pylist(), dtype=object)
    gt = pq.read_table(pathlib.Path(ROOT) / "train_ground_truth.parquet")
    gtm = dict(zip(gt.column("source1_entity_id").to_pylist(),
                   gt.column("matched_entity_ids").to_pylist()))
    print(f"{country}: {len(s1_ids):,} S1, {len(c_ids):,} corpus")

    t0 = time.time()
    corpus = blocking.build_corpus(corpus_tab, verbose=False)
    print(f"corpus indexed in {time.time()-t0:.0f}s")

    pick = rng.choice(len(s1_ids), n_tr + n_va, replace=False)
    tr_idx, va_idx = pick[:n_tr], pick[n_tr:]

    def prep(idx, tag):
        sub = s1_tab.take(idx)
        names, addrs = blocking.normalise(sub)
        ids = [s1_ids[i] for i in idx]
        q, c, chan = run_block(names, addrs, corpus, tag)
        truth = [set((gtm.get(i) or "").split(",")) - {""} for i in ids]
        cand_ids = c_ids[c]
        y = np.fromiter((cand_ids[j] in truth[q[j]] for j in range(len(q))),
                        np.int8, len(q))
        is_s3 = np.fromiter((s.startswith("S3-") for s in cand_ids), bool, len(q))
        X = features.build(q, chan, is_s3)
        return X, y, q, cand_ids, truth, ids

    Xtr, ytr, *_ = prep(tr_idx, "train")
    Xva, yva, qva, cva, truth_va, ids_va = prep(va_idx, "valid")
    print(f"  positives: train {ytr.mean():.2%}, valid {yva.mean():.2%}")

    t0 = time.time()
    model = lgb.train(
        dict(objective="binary", learning_rate=0.06, num_leaves=63,
             min_data_in_leaf=50, feature_fraction=0.9, bagging_fraction=0.8,
             bagging_freq=1, verbose=-1, num_threads=8),
        lgb.Dataset(Xtr, ytr, feature_name=features.NAMES), num_boost_round=350)
    raw = model.predict(Xva)
    print(f"model trained in {time.time()-t0:.0f}s")

    # Calibration matters more than ranking here: the stopping rule consumes
    # probabilities, so a good ranker that is badly calibrated stops in the
    # wrong place. Fit isotonic on a held-out half of validation.
    half = len(raw) // 2
    iso = IsotonicRegression(out_of_bounds="clip").fit(raw[:half], yva[:half])
    p = iso.predict(raw)

    print("\nfeature importance:")
    for n, g in sorted(zip(features.NAMES, model.feature_importance("gain")),
                       key=lambda x: -x[1])[:8]:
        print(f"  {n:12s} {g:12,.0f}")

    # ---- decision layer ----
    starts = np.flatnonzero(np.r_[True, qva[1:] != qva[:-1]])
    ends = np.r_[starts[1:], len(qva)]
    per_entity = {}
    for s, e in zip(starts, ends):
        per_entity[qva[s]] = (p[s:e], cva[s:e])

    def evaluate(policy, label):
        sc = np.empty(len(ids_va))
        npred = np.zeros(len(ids_va))
        for i in range(len(ids_va)):
            probs, cand = per_entity.get(i, (np.empty(0), np.empty(0, object)))
            order = np.argsort(-probs)
            probs, cand = probs[order], cand[order]
            k = policy(probs)
            pred = set(cand[:k])
            npred[i] = k
            sc[i] = f05(len(pred & truth_va[i]), len(truth_va[i]), k)
        print(f"  {label:26s} macro F0.5 = {sc.mean():.4f}   "
              f"mean k = {npred.mean():.2f}")
        return sc.mean()

    n_true = np.array([len(t) for t in truth_va])
    print(f"\nvalidation: {len(ids_va):,} entities, mean true n = {n_true.mean():.2f}, "
          f"singletons {(n_true==0).mean():.2%}")
    best_fixed = max(evaluate(lambda pr, k=k: min(k, len(pr)), f"fixed top-{k}")
                     for k in (1, 2, 3, 4))
    for thr in (0.3, 0.5, 0.7):
        evaluate(lambda pr, t=thr: int((pr > t).sum()), f"threshold {thr}")
    adaptive = evaluate(
        lambda pr: choose_k(pr, float(np.prod(1.0 - pr)) if len(pr) else 1.0),
        "adaptive-k (p > 0.8*F_k)")

    print(f"\n  adaptive - best_fixed = {adaptive - best_fixed:+.4f}")
    ceiling = f05(np.array([len(t & set(cva[qva == i])) for i, t in enumerate(truth_va)]),
                  n_true,
                  np.array([len(t & set(cva[qva == i])) for i, t in enumerate(truth_va)])).mean()
    print(f"  blocking ceiling      = {ceiling:.4f}")


if __name__ == "__main__":
    main()
