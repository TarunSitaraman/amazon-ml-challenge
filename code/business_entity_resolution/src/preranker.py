"""Learned pre-ranker for the candidate cut (two-stage retrieval, stage one).

CAND_CAP is a fixed per-entity budget, so what decides WHICH retrieved pairs
survive the cut decides the blocking ceiling. chan.max(1), one channel's score,
spends that budget crudely: widening retrieval 10x (DF_CAP) at the same budget
dropped the ceiling 0.9391 -> 0.8323, because distractors with a high score in
one channel outranked true matches that several channels agreed on.

This scores every retrieved pair with a small model on signals that already
exist before the cut, and train_eval.cap_candidates keeps the top CAND_CAP by
that score instead. It never reads strfeatures: those are paid for AFTER the
cut, and avoiding them for most pairs is the point of cutting.

Features (all over the entity's UNCAPPED list, as seen at cut time):
  per-channel scores, n channels fired, max / sum of channels,
  candidate count, rank of max score within entity, gap and ratio to the
  entity's best, per-channel gap to the entity's best in that channel for the
  continuous channels, per-channel rank for c2/c6, and is_s3 (a per-corpus
  bool array computed once per country, then indexed).

Two model kinds, both cheap:
  lgb     LightGBM, 50 trees x 15 leaves (the default)
  logreg  logistic regression on standardised features; scoring is one matvec

q must be sorted ascending (entity groups contiguous), as everywhere else.
"""
import numpy as np

from blocking import CHANNELS

# continuous channels; c1/c5 are 0/1 keys, so gap and rank carry nothing there
_CONT = [CHANNELS.index(n) for n in ("c2_name", "c6_addr", "c3_pair")]
_RANKED = [CHANNELS.index(n) for n in ("c2_name", "c6_addr")]

NAMES = (CHANNELS
         + ["n_chan", "max_chan", "sum_chan", "log_n_cand", "log_rank_max",
            "gap_top", "ratio_top"]
         + [f"gap_{CHANNELS[j]}" for j in _CONT]
         + [f"log_rank_{CHANNELS[j]}" for j in _RANKED]
         + ["is_s3"])

LGB_PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=15,
                  min_data_in_leaf=200, feature_fraction=1.0, verbose=-1,
                  num_threads=4)
LGB_ROUNDS = 50


def groups(q):
    starts = np.flatnonzero(np.r_[True, q[1:] != q[:-1]])
    ends = np.r_[starts[1:], len(q)]
    return starts, ends


def group_rank(q, s, starts, ends):
    """Rank of s within its entity, descending, ties sharing the best rank
    (= count of strictly greater scores in the entity), as features.build does.
    s must be non-negative float32 (channel scores are): its bit pattern then
    sorts like the value, so one int64 key (entity, -score) replaces a lexsort
    at about a quarter of the cost."""
    n = len(q)
    bits = np.ascontiguousarray(s, np.float32).view(np.uint32).astype(np.int64)
    order = np.argsort((q.astype(np.int64) << 32) | (0xFFFFFFFF - bits), kind="stable")
    ss, qq = s[order], q[order]
    new = np.ones(n, bool)
    new[1:] = (qq[1:] != qq[:-1]) | (ss[1:] != ss[:-1])
    first = np.maximum.accumulate(np.where(new, np.arange(n), 0))
    rank = np.empty(n, np.float32)
    rank[order] = first - np.repeat(starts, ends - starts)
    return rank


def build(q, chan, is_s3):
    """-> (n_pairs x len(NAMES)) float32, Fortran-ordered: filled a column at a
    time, which is several times cheaper than strided writes into row-major.
    q sorted; chan (n_pairs x n_channels), non-negative; is_s3 per pair."""
    out = np.empty((len(NAMES), len(q)), np.float32)
    if not len(q):
        return out.T
    starts, ends = groups(q)
    size = ends - starts
    cT = np.ascontiguousarray(chan.T)
    nc = len(cT)
    max_chan = cT.max(0)
    grp_max = np.repeat(np.maximum.reduceat(max_chan, starts), size)
    out[:nc] = cT
    out[nc] = (cT > 0).sum(0)
    out[nc + 1] = max_chan
    out[nc + 2] = cT.sum(0)
    out[nc + 3] = np.repeat(np.log1p(size.astype(np.float32)), size)
    out[nc + 4] = np.log1p(group_rank(q, max_chan, starts, ends))
    out[nc + 5] = grp_max - max_chan
    out[nc + 6] = max_chan / np.maximum(grp_max, 1e-6)
    j = nc + 7
    for ch in _CONT:
        out[j] = np.repeat(np.maximum.reduceat(cT[ch], starts), size) - cT[ch]
        j += 1
    for ch in _RANKED:
        # a pair the channel did not retrieve (0) ranks after all it did
        out[j] = np.log1p(group_rank(q, cT[ch], starts, ends))
        j += 1
    out[j] = is_s3
    return out.T


class PreRanker:
    """P(match) from build()'s features. score() returns a ranking score
    (raw margin, not a probability): only its order within an entity matters."""

    def __init__(self, kind="lgb"):
        if kind not in ("lgb", "logreg"):
            raise ValueError(f"unknown pre-ranker kind {kind!r}")
        self.kind = kind

    def fit(self, F, y):
        self.const = len(y) == 0 or y.min() == y.max()
        if self.const:
            # one label class (a toy shard): nothing to learn, rank all equal
            return self
        if self.kind == "lgb":
            import lightgbm as lgb
            self.model = lgb.train(LGB_PARAMS, lgb.Dataset(F, y, feature_name=NAMES),
                                   num_boost_round=LGB_ROUNDS)
        else:
            from sklearn.linear_model import LogisticRegression
            self.mu = F.mean(0)
            self.sd = F.std(0) + 1e-6
            m = LogisticRegression(C=1.0, max_iter=500).fit((F - self.mu) / self.sd, y)
            # folded into one matvec: (F - mu)/sd @ w + b == F @ (w/sd) + b'
            w = m.coef_[0] / self.sd
            self.w = w.astype(np.float32)
            self.b = float(m.intercept_[0] - self.mu @ w)
        return self

    def score(self, F):
        if not len(F) or self.const:
            return np.zeros(len(F), np.float32)
        if self.kind == "lgb":
            return self.model.predict(F, raw_score=True).astype(np.float32)
        return F @ self.w + np.float32(self.b)

    def importance(self):
        if self.const:
            return [(n, 0.0) for n in NAMES]
        if self.kind == "lgb":
            g = self.model.feature_importance("gain")
            return sorted(zip(NAMES, g / max(g.sum(), 1e-12)), key=lambda x: -x[1])
        return sorted(zip(NAMES, np.abs(self.w * self.sd)), key=lambda x: -x[1])
