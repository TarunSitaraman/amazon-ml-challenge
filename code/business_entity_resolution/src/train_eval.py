"""Train the pairwise matcher and score the full decision layer offline.

Splits by ENTITY, never by pair: the same entity's candidates appearing in both
train and validation would leak. The corpus stays whole for both splits, so the
validation entities face the true distractor density.

The disjoint rows only see conflicts among the sampled validation entities, so a
small n_val understates them. n_val = 0 takes every entity not used for training,
which is the closest this gets to predict.py's whole-country competition.

Singleton head (singleton.py): P(n=0) per entity, replacing prod(1 - p) in
choose_k. Its training features must come from OUT-OF-FOLD pair scores, so the
training ENTITIES are split into OOF_K folds, a pair model is trained on the
other folds and scores the held-out one, and the head is fit on those scores.
It is calibrated on the validation calibration half and gated on the held-out
half: it is saved for predict.py --singleton on only if its precision on the
entities it makes abstain is at least SINGLETON_GATE and its macro F0.5 is no
lower than the product rule's, else the product rule stays.

--profile prints wall clock, throughput and peak RSS per pipeline stage, per
country (blocking and features) and overall (profiling.py). Default off.

Pre-ranker (preranker.py): PRERANK=1 makes cap_candidates keep the top CAND_CAP
per entity by a learned score instead of chan.max(1); PRERANK_KIND=lgb|logreg
picks the model (default lgb). Default PRERANK=0, the previous behaviour. Both
kinds are always fit, on the TRAINING entities' uncapped pairs only, and the
blocking ceiling is reported at every cap in CURVE_CAPS under chan.max(1) and
both kinds, so the comparison needs no second run. With several countries each
country's validation is cut by a pre-ranker fit on the training entities of the
countries processed so far; the one saved to model.pkl is fit on all of them.

Usage: python train_eval.py [country] [n_train] [n_val] [--profile]
"""
import gc
import os
import pathlib
import pickle
import sys
import time

import lightgbm as lgb
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
from sklearn.isotonic import IsotonicRegression

import blocking
import disjoint
import features
import preranker
import singleton
import strfeatures
from metric import choose_k, f05
from profiling import PROF, pop_flag, stage
from textnorm import norm

ROOT = "data/parquet"
# Per-entity candidate cap. This, not the df cap, is what binds recall once the
# channels retrieve widely: raising DF_CAP 10x produced 60.0 candidates/entity
# against 59.4, because everything above this is trimmed. Env-overridable so the
# recall/cost frontier can actually be swept.
CAND_CAP = int(os.environ.get("CAND_CAP", 60))
PRERANK = os.environ.get("PRERANK", "0") == "1"
PRERANK_KIND = os.environ.get("PRERANK_KIND", "lgb")
# Caps the ceiling curves are reported at; CAND_CAP is added if missing.
CURVE_CAPS = (10, 20, 30, 40, 50, 60, 80, 100, 120, 150, 200)
# Uniform subsample of each country's training pairs the pre-rankers are fit on.
PRERANK_FIT_PAIRS = 1_000_000
OOF_K = 4
# break_even_precision(0.9) = 0.474 is the knife edge; 0.55 leaves margin for
# noise in both the precision estimate and f_alt (singleton.py).
SINGLETON_GATE = 0.55
PAIR_PARAMS = dict(objective="binary", learning_rate=0.06, num_leaves=63,
                   min_data_in_leaf=50, feature_fraction=0.9, bagging_fraction=0.8,
                   bagging_freq=1, verbose=-1, num_threads=8)
PAIR_ROUNDS = 350
CHAN_COLS = slice(0, len(blocking.CHANNELS))      # features.build puts chan first
IS_S3_COL = features.NAMES.index("is_s3")


def cap_keep(q, score, cap=CAND_CAP):
    """Indices of the best `cap` pairs per entity by score, in their original
    order. q sorted ascending."""
    starts = np.flatnonzero(np.r_[True, q[1:] != q[:-1]])
    ends = np.r_[starts[1:], len(q)]
    keep = []
    for s, e in zip(starts, ends):
        if e - s <= cap:
            keep.append(np.arange(s, e))
        else:
            sel = np.argpartition(score[s:e], -cap)[-cap:]
            keep.append(s + np.sort(sel))
    return np.concatenate(keep) if keep else np.empty(0, np.int64)


def cap_candidates(q, c, chan, cap=CAND_CAP, score=None):
    """Keep the best `cap` candidates per entity by `score`, or by max channel
    score when score is None (PRERANK=0)."""
    k = cap_keep(q, chan.max(1) if score is None else score, cap)
    return q[k], c[k], chan[k]


def train_pair_model(X, y):
    return lgb.train(PAIR_PARAMS, lgb.Dataset(X, y, feature_name=features.NAMES
                                              + strfeatures.NAMES),
                     num_boost_round=PAIR_ROUNDS)


def head_features(q, p, X, text):
    """Singleton-head features for entities 0..len(text)-1, from their pairs'
    calibrated scores p and pair feature matrix X (q sorted by entity)."""
    Xe, names = singleton.entity_features(q, p, X[:, CHAN_COLS], X[:, IS_S3_COL] > 0.5,
                                          len(text), chan_names=blocking.CHANNELS)
    return singleton.with_text(Xe, text), names


def zero_prob_product(p, starts, ends, n_ent, q):
    """The product rule prod(1 - p) per entity; 1.0 for an entity with no pairs."""
    pz = np.ones(n_ent)
    for s, e in zip(starts, ends):
        pz[q[s]] = float(np.prod(1.0 - p[s:e]))
    return pz


def run_block(names, addrs, corpus):
    """Retrieval only, uncapped, sorted by entity; the cut is the caller's."""
    q, c, chan = blocking.generate(None, None, names, addrs, corpus, verbose=False)
    order = np.argsort(q, kind="stable")
    return q[order], c[order], chan[order]


def fit_prerankers(pool):
    F = np.vstack(pool["F"]); y = np.concatenate(pool["y"])
    with stage("prerank fit", n=len(y), unit="pairs"):
        return {k: preranker.PreRanker(k).fit(F, y) for k in ("logreg", "lgb")}


def ceiling_hits(q, y, score, n_ent, caps):
    """True matches kept per entity at each cap -> (n_ent x len(caps))."""
    out = np.zeros((n_ent, len(caps)))
    for j, cap in enumerate(caps):
        k = cap_keep(q, score, cap)
        out[:, j] = np.bincount(q[k], weights=y[k], minlength=n_ent)
    return out


def prepare_country(country, n_tr, n_va, gtm, rng, pool=None):
    """Blocking + features for one country. Frees the corpus before returning,
    so peak memory is one shard regardless of how many countries we train on.
    pool accumulates pre-ranker training pairs across countries; the fitted
    pre-rankers are pool["rankers"]."""
    pool = {"F": [], "y": []} if pool is None else pool
    PROF.begin(country)
    with stage("load_shard"):
        s1_tab, corpus_tab = blocking.load_shard(ROOT, "train", country)
    s1_ids = s1_tab.column("entity_id").to_pylist()
    c_ids = np.array(corpus_tab.column("entity_id").to_pylist(), dtype=object)
    c_is_s3 = pc.starts_with(corpus_tab.column("entity_id"), "S3-").to_numpy()
    print(f"\n{country}: {len(s1_ids):,} S1, {len(c_ids):,} corpus")

    t0 = time.time()
    corpus = blocking.build_corpus(corpus_tab, verbose=False,
                                   string_features=True)
    del corpus_tab
    print(f"  indexed in {time.time()-t0:.0f}s")

    if n_va <= 0:
        n_va = max(len(s1_ids) - n_tr, 1)
    pick = rng.choice(len(s1_ids), min(n_tr + n_va, len(s1_ids)), replace=False)
    split = int(len(pick) * n_tr / (n_tr + n_va))
    # Over the whole country's S1, not the sample: counted within a 15k sample
    # a chain's other branches are mostly missing, so train and validation
    # (and predict.py) would each see a different feature.
    with stage("name_dup", n=len(s1_ids), unit="records"):
        dup = features.name_dup_counts(
            [norm(x) for x in s1_tab.column("business_name").to_pylist()])
    with stage("idf lut"):
        idf_lut = {t: float(corpus["index"]["idf"][i]) for i, t in
                   enumerate(corpus["index"]["vocab"].to_pylist())
                   if corpus["index"]["idf"][i] > 0}
        s_idf = singleton.idf_from_index(corpus["index"])

    def prep(idx, tag):
        with stage("normalise S1", n=len(idx), unit="records"):
            sub = s1_tab.take(idx)
            names, addrs = blocking.normalise(sub)
        ids = [s1_ids[i] for i in idx]
        t0 = time.time()
        q, c, chan = run_block(names, addrs, corpus)
        n_ret = len(q)
        # Labels on the uncapped list: the pre-ranker is fit on it and the
        # ceiling curves count what each cut keeps of it.
        with stage("labels", n=len(q), unit="pairs"):
            truth = [set((gtm.get(i) or "").split(",")) - {""} for i in ids]
            cand_all = c_ids[c]
            y = np.fromiter((cand_all[j] in truth[q[j]] for j in range(len(q))),
                            np.int8, len(q))
            del cand_all
        tf = time.perf_counter()
        with stage("prerank features", n=len(q), unit="pairs"):
            F = preranker.build(q, chan, c_is_s3[c])
        t_feat = time.perf_counter() - tf
        curves, scores = None, {}
        if tag == "train":
            sel = np.arange(len(y))
            if len(sel) > PRERANK_FIT_PAIRS:
                sel = np.sort(fit_rng.choice(len(y), PRERANK_FIT_PAIRS, replace=False))
            pool["F"].append(F[sel]); pool["y"].append(y[sel])
            pool["rankers"] = fit_prerankers(pool)
        else:
            # Every validation entity is out-of-sample for the pre-rankers.
            caps = sorted(set(CURVE_CAPS) | {CAND_CAP})
            curves = {"caps": caps, "n_pairs": len(q),
                      "all": np.bincount(q, weights=y, minlength=len(idx)),
                      "ns_features": 1e9 * t_feat / max(len(q), 1)}
            with stage("ceiling curves", n=len(q), unit="pairs"):
                curves["chan.max"] = ceiling_hits(q, y, chan.max(1), len(idx), caps)
                for k, r in pool["rankers"].items():
                    ts = time.perf_counter()
                    scores[k] = r.score(F)
                    curves["ns_" + k] = 1e9 * (time.perf_counter() - ts) / max(len(q), 1)
                    curves[k] = ceiling_hits(q, y, scores[k], len(idx), caps)
        if PRERANK and PRERANK_KIND not in scores:
            with stage("prerank score", n=len(q), unit="pairs"):
                scores[PRERANK_KIND] = pool["rankers"][PRERANK_KIND].score(F)
        with stage("cap_candidates", n=len(q), unit="pairs"):
            k = cap_keep(q, scores[PRERANK_KIND] if PRERANK else chan.max(1))
            q, c, chan, y = q[k], c[k], chan[k], y[k]
        del F, scores
        cand_ids = c_ids[c]
        print(f"  {country} {tag}: {len(q):,} pairs ({len(q)/max(len(idx),1):.1f}/entity) "
              f"cut from {n_ret:,} ({n_ret/max(len(idx),1):.1f}/entity) by "
              f"{PRERANK_KIND if PRERANK else 'chan.max'} in {time.time()-t0:.0f}s")
        is_s3 = c_is_s3[c]
        with stage("strfeatures.build", n=len(q), unit="pairs"):
            Xs = strfeatures.build(corpus["recs"], names, addrs, q, c, idf_lut)
        with stage("features.build", n=len(q), unit="pairs"):
            X = np.hstack([features.build(q, chan, is_s3,
                                          dup[idx]), Xs])
        with stage("singleton text features", n=len(idx), unit="entities"):
            text = singleton.text_features(names, addrs, s_idf)
        return X, y, q, cand_ids, truth, ids, text, curves

    fit_rng = np.random.default_rng(1)    # the main rng's draws stay as before
    out = (prep(pick[:split], "train"), prep(pick[split:], "valid"))
    with stage("free corpus + gc"):
        del corpus, s1_tab, c_ids, c_is_s3, idf_lut, s_idf
        gc.collect()
    PROF.end()
    return out


def report_curves(curves_va, ev, n_true, rankers):
    """Blocking ceiling vs CAND_CAP under chan.max(1) and each pre-ranker, over
    the held-out validation entities (the same ones as "blocking ceiling")."""
    caps = curves_va[0]["caps"]
    kinds = ["chan.max", "logreg", "lgb"]
    hits = {k: np.vstack([cv[k] for cv in curves_va])[ev] for k in kinds}
    ceil = {k: np.array([f05(hits[k][:, j], n_true, hits[k][:, j]).mean()
                         for j in range(len(caps))]) for k in kinds}
    per = {k: hits[k].mean(0) for k in kinds}
    allh = np.concatenate([cv["all"] for cv in curves_va])[ev]
    print(f"\ncandidate cut: blocking ceiling by CAND_CAP ({len(ev):,} held-out "
          f"entities; pre-rankers fit on training entities only)")
    print(f"  {'':5s}  {'ceiling':^31s}   {'true matches kept / entity':^31s}")
    print(f"  {'cap':>5s}  " + " ".join(f"{k:>10s}" for k in kinds)
          + "   " + " ".join(f"{k:>10s}" for k in kinds))
    for j, cap in enumerate(caps):
        print(f"  {cap:>5d}  " + " ".join(f"{ceil[k][j]:10.4f}" for k in kinds)
              + "   " + " ".join(f"{per[k][j]:10.3f}" for k in kinds)
              + ("   <- CAND_CAP" if cap == CAND_CAP else ""))
    print(f"  {'all':>5s}  {f05(allh, n_true, allh).mean():10.4f} (uncapped, any "
          f"ranker){'':9s} {allh.mean():10.3f}")
    base = ceil["chan.max"]
    jc = caps.index(CAND_CAP)
    for k in ("logreg", "lgb"):
        d = ceil[k] - base
        reach = [cap for j, cap in enumerate(caps) if ceil[k][j] >= base[jc]]
        print(f"  {k}: ceiling - chan.max at CAND_CAP {CAND_CAP} = {d[jc]:+.4f}; "
              f"best gain {d.max():+.4f} at cap {caps[int(d.argmax())]}; reaches "
              f"chan.max@{CAND_CAP} ({base[jc]:.4f}) at cap "
              f"{reach[0] if reach else '> ' + str(caps[-1])}")
    n_pairs = sum(cv["n_pairs"] for cv in curves_va)
    ns = {k: np.average([cv["ns_" + k] for cv in curves_va],
                        weights=[cv["n_pairs"] for cv in curves_va])
          for k in ("features", "logreg", "lgb")}
    print(f"  pre-ranker cost on {n_pairs:,} uncapped validation pairs: features "
          f"{ns['features']:.0f} ns/pair, score logreg {ns['logreg']:.0f} ns/pair, "
          f"lgb {ns['lgb']:.0f} ns/pair")
    for k in ("logreg", "lgb"):
        print(f"  top {k} features: " + ", ".join(
            f"{n} {v:.2f}" for n, v in rankers[k].importance()[:6]))
    print(f"  cut used for the matcher below: "
          f"{PRERANK_KIND if PRERANK else 'chan.max'} (PRERANK={int(PRERANK)})")


def main():
    if pop_flag(sys.argv):
        PROF.enable()
    countries = (sys.argv[1] if len(sys.argv) > 1 else "India").split(",")
    n_tr = int(sys.argv[2]) if len(sys.argv) > 2 else 15000
    n_va = int(sys.argv[3]) if len(sys.argv) > 3 else 8000
    rng = np.random.default_rng(0)

    with stage("load ground truth"):
        gt = pq.read_table(pathlib.Path(ROOT) / "train_ground_truth.parquet")
        gtm = dict(zip(gt.column("source1_entity_id").to_pylist(),
                       gt.column("matched_entity_ids").to_pylist()))
        del gt

    tr_parts, va_parts = [], []
    pool = {"F": [], "y": []}
    for ctry in countries:
        tr, va = prepare_country(ctry, n_tr, n_va, gtm, rng, pool)
        tr_parts.append(tr)
        va_parts.append((ctry,) + va)

    Xtr = np.vstack([t[0] for t in tr_parts])
    ytr = np.concatenate([t[1] for t in tr_parts])
    # training entities renumbered across countries, as validation is below
    ent_off = np.cumsum([0] + [len(t[5]) for t in tr_parts])
    qtr = np.concatenate([t[2] + o for t, o in zip(tr_parts, ent_off)])
    text_tr = np.vstack([t[6] for t in tr_parts])
    ysing_tr = singleton.singleton_labels([s for t in tr_parts for s in t[4]])
    del tr_parts
    rankers = pool["rankers"]
    del pool
    gc.collect()

    # Validation entities are renumbered so several countries can share one
    # evaluation pass without their entity indices colliding.
    Xva_l, yva_l, qva_l, cva_l, truth_va, ids_va, ctry_va = [], [], [], [], [], [], []
    text_va, curves_va = [], []
    offset = 0
    for ctry, X, y, q, cand, truth, ids, text, curves in va_parts:
        Xva_l.append(X); yva_l.append(y); qva_l.append(q + offset); cva_l.append(cand)
        text_va.append(text); curves_va.append(curves)
        truth_va += truth; ids_va += ids; ctry_va += [ctry] * len(ids)
        offset += len(ids)
    Xva = np.vstack(Xva_l); yva = np.concatenate(yva_l)
    qva = np.concatenate(qva_l); cva = np.concatenate(cva_l)
    text_va = np.vstack(text_va)
    del Xva_l, yva_l, qva_l, cva_l, va_parts
    gc.collect()
    print(f"\ntraining on {', '.join(countries)}: {len(ytr):,} pairs, "
          f"{len(ids_va):,} validation entities")
    print(f"  positives: train {ytr.mean():.2%}, valid {yva.mean():.2%}")

    t0 = time.time()
    with stage("lgb.train", n=len(ytr), unit="pairs"):
        model = train_pair_model(Xtr, ytr)
    with stage("model.predict", n=len(yva), unit="pairs"):
        raw = model.predict(Xva)
    print(f"model trained in {time.time()-t0:.0f}s")

    # ---- out-of-fold pair scores for the singleton head ----
    # Folds are over training ENTITIES: an entity's pairs all land in one fold,
    # so no fold model has seen any candidate of the entities it scores.
    t0 = time.time()
    n_tr_ent = len(text_tr)
    fold = rng.permutation(n_tr_ent) % OOF_K
    pair_fold = fold[qtr]
    raw_oof = np.empty(len(ytr))
    for f in range(OOF_K):
        m = pair_fold == f
        with stage("lgb.train (OOF)", n=int((~m).sum()), unit="pairs"):
            fm = train_pair_model(Xtr[~m], ytr[~m])
        with stage("model.predict (OOF)", n=int(m.sum()), unit="pairs"):
            raw_oof[m] = fm.predict(Xtr[m])
        del fm
    # Calibrated like the real scores: the head reads probabilities. Isotonic
    # is fit on the OOF scores themselves, which are all out-of-sample.
    with stage("isotonic (OOF)", n=len(ytr), unit="pairs"):
        p_oof = (IsotonicRegression(out_of_bounds="clip").fit(raw_oof, ytr)
                 .predict(raw_oof))
    del raw_oof, pair_fold
    with stage("singleton head fit", n=n_tr_ent, unit="entities"):
        Xe_tr, e_names = head_features(qtr, p_oof, Xtr, text_tr)
        head = singleton.SingletonHead(feature_names=e_names).fit(Xe_tr, ysing_tr)
    del Xe_tr, p_oof
    print(f"singleton head: {OOF_K}-fold OOF over {n_tr_ent:,} training entities "
          f"({ysing_tr.mean():.2%} singletons) in {time.time()-t0:.0f}s")

    # Calibration matters more than ranking here: the stopping rule consumes
    # probabilities, so a good ranker that is badly calibrated stops in the
    # wrong place. Fit isotonic on the first half of validation ENTITIES and
    # score only the second half, so the reported numbers are out-of-sample.
    # The calibration half still takes part in conflict resolution below: its
    # entities compete for the same records in the real pipeline.
    n_cal = len(ids_va) // 2
    cal = qva < n_cal
    with stage("isotonic", n=len(raw), unit="pairs"):
        iso = IsotonicRegression(out_of_bounds="clip").fit(raw[cal], yva[cal])
        p = iso.predict(raw)

    # Validation entity groups (qva is sorted: per-country sorted, offsets increasing).
    starts = np.flatnonzero(np.r_[True, qva[1:] != qva[:-1]])
    ends = np.r_[starts[1:], len(qva)]

    # Head P(n=0) on validation, from the full model's scores as at test time.
    # Isotonic on the calibration half corrects the shift from OOF-model scores
    # (3/4 of the data) to full-model scores; the held-out half stays clean.
    with stage("singleton head calibrate+predict", n=len(ids_va), unit="entities"):
        ysing_va = singleton.singleton_labels(truth_va)
        Xe_va, _ = head_features(qva, p, Xva, text_va)
        head.calibrate(Xe_va[:n_cal], ysing_va[:n_cal])
        pz_head = head.predict_proba(Xe_va)
    with stage("product rule P(n=0)", n=len(p), unit="pairs"):
        pz_prod = zero_prob_product(p, starts, ends, len(ids_va), qva)
    del Xe_va

    print("\nfeature importance:")
    for n, g in sorted(zip(features.NAMES + strfeatures.NAMES, model.feature_importance("gain")),
                       key=lambda x: -x[1])[:10]:
        print(f"  {n:12s} {g:12,.0f}")

    with stage("valstate.pkl write"), open("valstate.pkl", "wb") as fh:
        pickle.dump({"p": p, "cand": cva, "q": qva, "truth": truth_va,
                     "ids": ids_va, "ctry": ctry_va, "p_zero_head": pz_head}, fh)
    print("saved valstate.pkl (decision-rule tuning needs no re-blocking)")

    # ---- decision layer ----
    # Scored on the held-out half only; see the calibration note above.
    ev = np.arange(n_cal, len(ids_va))
    per_entity = {}
    for s, e in zip(starts, ends):
        per_entity[qva[s]] = (p[s:e], cva[s:e])

    def run_policy(policy):
        """policy(sorted probs, entity index) -> k. -> per-ev-entity (F0.5, k)."""
        with stage("choose_k eval", n=len(ev), unit="entities"):
            sc = np.empty(len(ev))
            npred = np.zeros(len(ev))
            for j, i in enumerate(ev):
                probs, cand = per_entity.get(i, (np.empty(0), np.empty(0, object)))
                order = np.argsort(-probs)
                probs, cand = probs[order], cand[order]
                k = policy(probs, i)
                pred = set(cand[:k])
                npred[j] = k
                sc[j] = f05(len(pred & truth_va[i]), len(truth_va[i]), k)
        return sc, npred

    def evaluate(policy, label):
        sc, npred = run_policy(policy)
        print(f"  {label:26s} macro F0.5 = {sc.mean():.4f}   "
              f"mean k = {npred.mean():.2f}")
        return sc.mean()

    n_true = np.array([len(truth_va[i]) for i in ev])
    print(f"\nvalidation (held-out half): {len(ev):,} entities, "
          f"mean true n = {n_true.mean():.2f}, singletons {(n_true==0).mean():.2%}")
    best_fixed = max(evaluate(lambda pr, i, k=k: min(k, len(pr)), f"fixed top-{k}")
                     for k in (1, 2, 3, 4))
    for thr in (0.3, 0.5, 0.7):
        evaluate(lambda pr, i, t=thr: int((pr > t).sum()), f"threshold {thr}")
    adaptive = evaluate(lambda pr, i: choose_k(pr, pz_prod[i]),
                        "adaptive-k (p > 0.8*F_k)")

    print(f"\n  adaptive - best_fixed = {adaptive - best_fixed:+.4f}")

    # Walk the entity groups once; `qva == i` per entity would be O(n_ent*n_pairs).
    hit = np.zeros(len(ids_va))
    for s, e in zip(starts, ends):
        hit[qva[s]] = len(truth_va[qva[s]] & set(cva[s:e]))
    print(f"  blocking ceiling      = {f05(hit[ev], n_true, hit[ev]).mean():.4f}")
    report_curves(curves_va, ev, n_true, rankers)

    # ---- singleton head gate ----
    # The operating point is the head's own decision: an entity with candidates
    # is flagged when choose_k(probs, head P(n=0)) returns 0. Entities with no
    # candidates get [] under either rule, so they are left out of precision.
    sc_prod, k_prod = run_policy(lambda pr, i: choose_k(pr, pz_prod[i]))
    sc_head, k_head = run_policy(lambda pr, i: choose_k(pr, pz_head[i]))
    has = np.array([i in per_entity for i in ev])
    is_single = n_true == 0
    flag = has & (k_head == 0)
    flip = flag & (k_prod > 0)          # abstentions the product rule would not make
    prec = is_single[flag].mean() if flag.any() else float("nan")
    prec_flip = is_single[flip].mean() if flip.any() else float("nan")
    f_alt = sc_prod[has & ~is_single].mean()
    # Precision over all abstentions includes the ones the product rule already
    # makes, and ignores singletons the head stops abstaining on, so it alone
    # can pass a head that lowers the score. Require the measured gain too.
    use_head = bool(prec >= SINGLETON_GATE) and sc_head.mean() >= sc_prod.mean()
    print(f"\nsingleton head (held-out half, {has.sum():,} entities with candidates, "
          f"{is_single[has].sum():,} singletons):")
    print(f"  flagged (k=0)         = {flag.sum():,}   precision = {prec:.3f}   "
          f"singleton recall = {flag[is_single & has].mean():.3f}")
    print(f"  flipped vs product    = {flip.sum():,}   precision = {prec_flip:.3f}")
    print(f"  mean P(n=0) head {pz_head[ev].mean():.4f}   product "
          f"{pz_prod[ev].mean():.4f}   true rate {is_single.mean():.4f}")
    print(f"  break-even at measured f_alt {f_alt:.3f} = "
          f"{singleton.break_even_precision(f_alt):.3f}   gate = {SINGLETON_GATE}")
    print(f"  adaptive-k, product P(n=0)  macro F0.5 = {sc_prod.mean():.4f}")
    print(f"  adaptive-k, head P(n=0)     macro F0.5 = {sc_head.mean():.4f}   "
          f"({sc_head.mean() - sc_prod.mean():+.4f})")
    print(f"  gate (precision >= {SINGLETON_GATE} and head F0.5 >= product) "
          f"{'PASSED: head saved for predict.py --singleton on' if use_head else 'FAILED: product rule stays'}")
    print("  top head features:", ", ".join(n for n, _ in head.importance()[:6]))

    with stage("model.pkl write"), open("model.pkl", "wb") as fh:
        pickle.dump({"model": model, "iso": iso,
                     "singleton": head if use_head else None,
                     "singleton_precision": prec,
                     # the matcher was trained on lists cut this way
                     "prerank": PRERANK_KIND if PRERANK else "off",
                     "prerankers": rankers}, fh)
    print("saved model.pkl")

    # ---- disjointness (disjoint.py) ----
    # Same adaptive-k decision, but made jointly: every validation entity
    # (both halves) competes for records, and each record keeps one owner.
    # Record IDs are unique across countries, so one integer index covers all.
    _, rec = np.unique(cva, return_inverse=True)

    # P(n=0) is the gated choice: the head if it passed, else the product rule
    # (p_zero=None / computed from the scores actually decided on, as before).
    pz_use = pz_head if use_head else None

    def decide(prob, pz=None):
        if pz is None:
            with stage("product rule P(n=0)", n=len(prob), unit="pairs"):
                pz = zero_prob_product(prob, starts, ends, len(ids_va), qva)
        with stage("choose_k", n=len(prob), unit="pairs"):
            acc = np.zeros(len(prob), bool)
            for s, e in zip(starts, ends):
                o = s + np.argsort(-prob[s:e], kind="stable")
                acc[o[:choose_k(prob[o], pz[qva[s]])]] = True
        return acc

    def resolve(prob, acc, **kw):
        with stage("disjoint resolve", n=len(prob), unit="pairs"):
            return disjoint.resolve_conflicts(qva, rec, prob, acc, p_zero=pz_use,
                                              **kw)

    def score_mask(acc, label):
        with stage("disjoint scoring", n=len(acc), unit="pairs"):
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

    print(f"\ndisjointness (one owner per record, P(n=0) from "
          f"{'singleton head' if use_head else 'product rule'}):")
    acc = decide(p, pz_use)
    base = score_mask(acc, "adaptive-k, per entity")
    res = score_mask(resolve(p, acc), "+ resolve_conflicts")
    red = score_mask(resolve(p, acc, redecide=True), "+ resolve, redecide")
    with stage("disjoint sinkhorn", n=len(p), unit="pairs"):
        pn = disjoint.sinkhorn_normalise(qva, rec, p)
    sk = score_mask(resolve(pn, decide(pn, pz_use)), "sinkhorn + resolve")
    print(f"\n  resolve - base        = {res - base:+.4f}")
    print(f"  redecide - base       = {red - base:+.4f}")
    print(f"  sinkhorn+resolve - base = {sk - base:+.4f}")
    PROF.report()


if __name__ == "__main__":
    main()
