"""Entity-level singleton head: P(n=0 | entity).

Why a separate head. 5.58% of S1 entities have no matches at all. For them,
predicting [] scores exactly 1.0 and predicting anything scores 0.0, so a
pipeline that never abstains is capped at 0.9442. The decision layer abstains
through one number: `metric.choose_k` accepts the first candidate iff

    p1 > 1.533 * P(n=0)

so P(n=0) moving from the 0.056 prior to 0.50 moves the entry bar from 0.086 to
0.766. It is the most leveraged number in the decision layer.

Why not the product. Zero-match is correlated ACROSS sources (RESEARCH.md s3):
P(n_S2=0) = 13.04%, P(n_S3=0) = 12.07%, so independence predicts both-zero at
1.57% but 5.58% is observed -- a 3.55x lift. A latent per-entity "obscurity"
governs whether an entity appears anywhere. Computing P(n=0) = prod(1 - p_i)
from pairwise probabilities assumes the absence events are independent, and so
SYSTEMATICALLY underestimates singleton probability for exactly the entities
that are singletons. This head reads entity-level features (both sources' best
scores, margins, candidate density, name rarity) and is fit directly on the
entity label n == 0, so the correlation is learned rather than assumed away.

Country-agnostic by construction. France has zero labels, so no feature is a
country indicator: everything is a score, a count or a text statistic that
means the same thing in every shard. Train pooled across labelled countries and
apply anywhere.

Leakage warning. The pair scores fed to `entity_features` must be OUT-OF-FOLD
for the training entities. In-sample pair-model scores are overconfident on the
very entities the head learns from, which teaches it that singletons look
cleaner than they do at test time.

Wiring:
    Xe, names = entity_features(q, p, chan, is_s3, n_ent, names=..., addrs=...,
                                idf=idf, chan_names=blocking.CHANNELS)
    head = SingletonHead(feature_names=names).fit(Xe_tr, y_tr)
    p_zero = head.predict_proba(Xe)          # one P(n=0) per entity
    k = choose_k(sorted_probs_i, p_zero[i])

Run this module directly for a self-test on synthetic data.
"""
import math

import lightgbm as lgb
import numpy as np
from sklearn.isotonic import IsotonicRegression


def break_even_precision(f_alt):
    """Precision a singleton flag needs before abstaining pays.

    Flagging an entity (predicting []) gains 1.0 when it truly is a singleton
    (vs 0.0 for any non-empty prediction) and loses f_alt -- the F_0.5 the
    pipeline would otherwise have scored -- when it is not. With precision P
    over flagged entities, abstaining pays iff P * 1 > (1 - P) * f_alt, i.e.

        P > f_alt / (1 + f_alt)

    At f_alt = 0.9 break-even is 0.474. That is the knife edge, not the target:
    precision is estimated on validation and f_alt on matched entities varies,
    so the head is only worth wiring in once it holds ~0.55 precision or better
    on held-out entities at the threshold it will run at.
    """
    return f_alt / (1.0 + f_alt)


# --------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------

BASE_NAMES = ["n_cand", "n_s2", "n_s3",
              "best", "best_s2", "best_s3", "top2", "gap12",
              "n_above", "sum_p", "indep_logp0"]
TEXT_NAMES = ["name_rarity_mean", "name_rarity_max", "name_tokens", "addr_tokens"]


def build_idf(docs):
    """Token IDF over a corpus of normalised strings (space-tokenised).

    -> (idf dict, idf of an unseen token). Build it from the S2+S3 corpus names
    of the same shard: a name whose tokens are rare there is one the corpus is
    unlikely to contain.
    """
    df = {}
    for d in docs:
        for t in set((d or "").split()):
            df[t] = df.get(t, 0) + 1
    n = len(docs)
    idf = {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}
    return idf, math.log(n + 1) + 1.0


def _text_features(names, addrs, idf):
    n = len(names) if names is not None else len(addrs)
    out = np.full((n, len(TEXT_NAMES)), np.nan, np.float32)
    if names is not None:
        table, unseen = idf if idf is not None else ({}, np.nan)
        for i, s in enumerate(names):
            toks = (s or "").split()
            out[i, 2] = len(toks)
            if toks and idf is not None:
                w = [table.get(t, unseen) for t in toks]
                out[i, 0] = sum(w) / len(w)
                out[i, 1] = max(w)
    if addrs is not None:
        out[:, 3] = [len((s or "").split()) for s in addrs]
    return out


def entity_features(q, score, chan, is_s3, n_entities, names=None, addrs=None,
                    idf=None, chan_names=None, thr=0.5):
    """Entity-level features from per-pair arrays. -> (X, feature_names).

    q:      entity index per pair, 0..n_entities-1 (need not be sorted)
    score:  calibrated pair match probability in [0, 1] (out-of-fold for
            training entities -- see module docstring)
    chan:   (n_pairs x n_channels) per-channel similarities
    is_s3:  True where the candidate comes from Source 3
    names, addrs: optional normalised S1 name/address strings, one per entity
    idf:    optional output of build_idf, for name rarity

    Every entity gets a row. An entity with no candidates gets zeros in the
    score columns and n_cand = 0, which is itself the strongest singleton cue.
    Text columns are NaN when not supplied; LightGBM treats that as missing.
    """
    q = np.asarray(q, np.int64)
    score = np.asarray(score, np.float64)
    is_s3 = np.asarray(is_s3, bool)
    chan = np.asarray(chan, np.float64)
    if chan.ndim == 1:
        chan = chan[:, None]
    n_ch = chan.shape[1]
    chan_names = list(chan_names) if chan_names is not None else [f"chan{i}" for i in range(n_ch)]
    assert len(chan_names) == n_ch
    feat_names = BASE_NAMES + [f"best_{c}" for c in chan_names] + TEXT_NAMES

    X = np.zeros((n_entities, len(BASE_NAMES) + n_ch), np.float64)
    if len(q):
        # group by entity, best score first within each group
        o = np.lexsort((-score, q))
        q, score, is_s3, chan = q[o], score[o], is_s3[o], chan[o]
        starts = np.flatnonzero(np.r_[True, q[1:] != q[:-1]])
        sizes = np.diff(np.r_[starts, len(q)])
        ent = q[starts]

        top1 = score[starts]
        top2 = np.where(sizes >= 2, score[np.minimum(starts + 1, len(q) - 1)], 0.0)
        best_s2 = np.maximum.reduceat(np.where(is_s3, -np.inf, score), starts)
        best_s3 = np.maximum.reduceat(np.where(is_s3, score, -np.inf), starts)
        n_s3 = np.add.reduceat(is_s3.astype(np.int64), starts)
        log_absent = np.log1p(-np.clip(score, 0.0, 1.0 - 1e-6))

        X[ent, 0] = sizes
        X[ent, 1] = sizes - n_s3
        X[ent, 2] = n_s3
        X[ent, 3] = top1
        X[ent, 4] = np.where(np.isfinite(best_s2), best_s2, 0.0)
        X[ent, 5] = np.where(np.isfinite(best_s3), best_s3, 0.0)
        X[ent, 6] = top2
        X[ent, 7] = top1 - top2
        X[ent, 8] = np.add.reduceat((score > thr).astype(np.int64), starts)
        X[ent, 9] = np.add.reduceat(score, starts)
        # the independence estimate log prod(1 - p): given to the head as a
        # feature so it can learn the correction instead of rediscovering it
        X[ent, 10] = np.add.reduceat(log_absent, starts)
        X[ent, len(BASE_NAMES):] = np.maximum.reduceat(chan, starts, axis=0)

    if names is None and addrs is None:
        text = np.full((n_entities, len(TEXT_NAMES)), np.nan, np.float32)
    else:
        text = _text_features(names, addrs, idf)
        assert len(text) == n_entities
    return np.hstack([X.astype(np.float32), text]), feat_names


def singleton_labels(truth):
    """truth: per-entity sets of matched ids -> 1 where the entity has none."""
    return np.fromiter((len(t) == 0 for t in truth), np.int8, len(truth))


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------

class SingletonHead:
    """LightGBM binary classifier for P(n=0 | entity).

    Plain log-loss with no class reweighting: `choose_k` consumes this as a
    probability, so calibration matters more than recall. `calibrate` adds an
    optional isotonic pass on a held-out split.
    """

    PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31,
                  min_data_in_leaf=100, feature_fraction=0.9,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  verbose=-1, num_threads=8, seed=0)

    def __init__(self, n_trees=200, feature_names=None, **params):
        self.n_trees = n_trees
        self.feature_names = feature_names
        self.params = {**self.PARAMS, **params}
        self.booster = None
        self.iso = None

    def fit(self, X, y):
        y = np.asarray(y).astype(np.float32)
        ds = lgb.Dataset(np.asarray(X, np.float32), y,
                         feature_name=self.feature_names or "auto")
        self.booster = lgb.train(self.params, ds, num_boost_round=self.n_trees)
        self.iso = None
        return self

    def calibrate(self, X, y):
        """Isotonic recalibration on entities NOT used in fit."""
        raw = self.booster.predict(np.asarray(X, np.float32))
        self.iso = IsotonicRegression(y_min=0.0, y_max=1.0,
                                      out_of_bounds="clip").fit(raw, y)
        return self

    def predict_proba(self, X):
        """-> 1-D array of P(n=0), one per row (not sklearn's (n, 2) shape,
        because the decision layer wants exactly this column)."""
        p = self.booster.predict(np.asarray(X, np.float32))
        return self.iso.predict(p) if self.iso is not None else p

    def importance(self):
        names = self.booster.feature_name()
        gain = self.booster.feature_importance("gain")
        return sorted(zip(names, gain), key=lambda x: -x[1])


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------

def _logloss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _synthetic(n_ent, rng):
    """Entities with a latent obscurity that drives zero-match in BOTH sources,
    reproducing the cross-source correlation measured in RESEARCH.md s3."""
    obscure = rng.random(n_ent) < 0.15
    p_src_zero = np.where(obscure, 0.70, 0.05)
    n2 = np.where(rng.random(n_ent) < p_src_zero, 0, 1 + rng.poisson(1.0, n_ent))
    n3 = np.where(rng.random(n_ent) < p_src_zero, 0, 1 + rng.poisson(1.0, n_ent))
    n_true = n2 + n3
    n_dis = rng.poisson(np.where(obscure, 5, 10))

    tot = n_true + n_dis
    q = np.repeat(np.arange(n_ent), tot)
    first = np.repeat(np.cumsum(np.r_[0, tot[:-1]]), tot)
    pos = np.arange(len(q)) - first
    is_true = pos < np.repeat(n_true, tot)
    is_s3 = np.where(pos < np.repeat(n2, tot), False,
                     np.where(is_true, True, rng.random(len(q)) < 0.5))
    obs_p = np.repeat(obscure, tot)
    raw = np.where(is_true,
                   np.where(obs_p, rng.beta(3, 3, len(q)), rng.beta(4, 2, len(q))),
                   rng.beta(1.5, 6, len(q)))
    miss = rng.random((len(q), 3)) < 0.3
    chan = np.where(miss, 0.0,
                    np.clip(raw[:, None] + rng.normal(0, 0.15, (len(q), 3)), 0, 1))

    # obscure entities carry rarer name tokens and shorter addresses
    names, addrs = [], []
    for i in range(n_ent):
        k = rng.zipf(1.5, rng.integers(2, 5)) + (1000 if obscure[i] else 0)
        names.append(" ".join(f"t{x}" for x in k))
        addrs.append(" ".join(["a"] * int(rng.poisson(3 if obscure[i] else 6))))
    return dict(q=q, raw=raw, chan=chan, is_s3=is_s3, is_true=is_true,
                n_true=n_true, n2=n2, n3=n3, names=names, addrs=addrs)


def _slice(d, lo, hi):
    """Entities [lo, hi) of a synthetic set, re-indexed from 0."""
    m = (d["q"] >= lo) & (d["q"] < hi)
    out = {k: d[k][m] for k in ("q", "raw", "chan", "is_s3", "is_true")}
    out["q"] = out["q"] - lo
    for k in ("n_true", "n2", "n3"):
        out[k] = d[k][lo:hi]
    out["names"], out["addrs"] = d["names"][lo:hi], d["addrs"][lo:hi]
    return out


def _self_test():
    from metric import choose_k, f05

    ok = True

    def check(cond, msg):
        nonlocal ok
        print(f"  [{'ok' if cond else 'FAIL'}] {msg}")
        ok &= bool(cond)

    # ---- break-even ----
    print("break-even precision:")
    be = break_even_precision(0.9)
    check(abs(be - 0.9 / 1.9) < 1e-12 and round(be, 3) == 0.474,
          f"f_alt=0.9 -> {be:.4f} (expected 0.474)")
    check(break_even_precision(1.0) == 0.5, "f_alt=1.0 -> 0.5")

    # ---- features on a hand-computed example ----
    print("\nentity_features, hand example:")
    q = [1, 0, 0, 0]
    sc = [0.2, 0.9, 0.4, 0.7]
    s3 = [True, False, True, False]
    ch = [[0.1, 0.3], [0.8, 0.2], [0.5, 0.5], [0.6, 0.9]]
    idf = build_idf(["acme corp", "acme ltd"])
    X, fn = entity_features(q, sc, ch, s3, 3, names=["acme corp", "zzq", ""],
                            addrs=["1 main st", "", "x"], idf=idf,
                            chan_names=["name", "addr"])
    r = [dict(zip(fn, row)) for row in X]
    e0 = dict(n_cand=3, n_s2=2, n_s3=1, best=0.9, best_s2=0.9, best_s3=0.4,
              top2=0.7, gap12=0.2, n_above=2, sum_p=2.0,
              indep_logp0=math.log(0.1 * 0.6 * 0.3), best_name=0.8, best_addr=0.9,
              name_rarity_mean=(1.0 + math.log(1.5) + 1.0) / 2,
              name_rarity_max=math.log(1.5) + 1.0, name_tokens=2, addr_tokens=3)
    e1 = dict(n_cand=1, n_s2=0, n_s3=1, best=0.2, best_s2=0.0, best_s3=0.2,
              top2=0.0, gap12=0.2, n_above=0, sum_p=0.2, best_name=0.1,
              best_addr=0.3, name_rarity_mean=math.log(3) + 1.0, addr_tokens=0)
    for i, exp in enumerate((e0, e1)):
        bad = {k: (r[i][k], v) for k, v in exp.items()
               if not np.isclose(r[i][k], v, atol=1e-5)}
        check(not bad, f"entity {i} matches by hand" + (f": {bad}" if bad else ""))
    check(all(r[2][k] == 0 for k in BASE_NAMES) and np.isnan(r[2]["name_rarity_mean"])
          and r[2]["addr_tokens"] == 1, "entity 2 (no candidates) is all-zero")
    Xn, _ = entity_features([], [], np.zeros((0, 2)), [], 2)
    check(Xn.shape == (2, len(fn)) and np.isnan(Xn[:, -1]).all(),
          "no pairs, no text -> zero/NaN rows")

    # ---- synthetic: correlated zero-match ----
    rng = np.random.default_rng(0)
    n_tr, n_va = 12000, 8000
    d = _synthetic(n_tr + n_va, rng)
    z2, z3 = d["n2"] == 0, d["n3"] == 0
    lift = (z2 & z3).mean() / (z2.mean() * z3.mean())
    print(f"\nsynthetic: {n_tr + n_va:,} entities, P(S2=0)={z2.mean():.2%} "
          f"P(S3=0)={z3.mean():.2%} both={(z2 & z3).mean():.2%} "
          f"indep={z2.mean() * z3.mean():.2%}  lift={lift:.2f}x")
    check(lift > 2.0, "zero-match is correlated across sources")

    tr, va = _slice(d, 0, n_tr), _slice(d, n_tr, n_tr + n_va)

    # the "pair model": raw score calibrated on training pairs only
    iso = IsotonicRegression(out_of_bounds="clip").fit(tr["raw"], tr["is_true"].astype(float))
    idf = build_idf(d["names"])

    def feats(s, n):
        p = iso.predict(s["raw"])
        X, fn = entity_features(s["q"], p, s["chan"], s["is_s3"], n,
                                names=s["names"], addrs=s["addrs"], idf=idf)
        return X, fn, p

    Xtr, fn, _ = feats(tr, n_tr)
    Xva, _, pva = feats(va, n_va)
    ytr = (tr["n_true"] == 0).astype(np.int8)
    yva = (va["n_true"] == 0).astype(np.int8)

    head = SingletonHead(feature_names=fn).fit(Xtr, ytr)
    p_head = head.predict_proba(Xva)
    p_ind = np.exp(Xva[:, fn.index("indep_logp0")])
    check(p_head.shape == (n_va,) and ((p_head >= 0) & (p_head <= 1)).all(),
          "predict_proba -> one probability per entity")

    print(f"\n  {'':22s} {'head':>8s} {'prod(1-p)':>10s}")
    ll_h, ll_i = _logloss(yva, p_head), _logloss(yva, p_ind)
    print(f"  {'log-loss':22s} {ll_h:8.4f} {ll_i:10.4f}")
    print(f"  {'mean P(n=0)':22s} {p_head.mean():8.4f} {p_ind.mean():10.4f}"
          f"   (true rate {yva.mean():.4f})")
    s = yva == 1
    print(f"  {'mean P(n=0) | n=0':22s} {p_head[s].mean():8.4f} {p_ind[s].mean():10.4f}")
    be = break_even_precision(0.9)
    for lab, pz in (("head", p_head), ("prod(1-p)", p_ind)):
        flag = pz > be
        prec = yva[flag].mean() if flag.any() else float("nan")
        print(f"  flag P(n=0)>{be:.3f} [{lab:9s}]  flagged={flag.sum():5d}  "
              f"precision={prec:.3f}  singleton recall={flag[s].mean():.3f}")
    check(ll_h < ll_i, "head beats the independence product on log-loss")
    check(abs(p_head.mean() - yva.mean()) < 0.02, "head is calibrated in the large")

    # end-to-end through the real decision rule
    starts = np.flatnonzero(np.r_[True, va["q"][1:] != va["q"][:-1]])
    ends = np.r_[starts[1:], len(va["q"])]
    groups = {va["q"][a]: (pva[a:b], va["is_true"][a:b]) for a, b in zip(starts, ends)}

    def macro(p_zero):
        sc = np.empty(n_va)
        for i in range(n_va):
            pr, t = groups.get(i, (np.empty(0), np.empty(0, bool)))
            o = np.argsort(-pr)
            k = choose_k(pr[o], float(p_zero[i]))
            sc[i] = f05(t[o][:k].sum(), va["n_true"][i], k)
        return sc.mean()

    print(f"\n  choose_k macro F0.5: head {macro(p_head):.4f}   "
          f"prod(1-p) {macro(p_ind):.4f}   prior 0.056 {macro(np.full(n_va, 0.056)):.4f}")

    print("\n  top features:", ", ".join(n for n, _ in head.importance()[:6]))
    print("\n" + ("PASS" if ok else "FAIL"))
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if _self_test() else 1)
