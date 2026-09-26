"""Training on 200k-500k entities in bounded RAM: negative subsampling with
importance weights, and a float32 feature matrix on disk.

At ~87 candidates per entity, 200k training entities is ~17M pairs x 31
features; with the corpus index resident that does not fit a 16GB laptop. Two
levers, both off by default (train_eval.py behaves exactly as before):

NEG_KEEP (default 1.0): keep every positive, and keep each negative with
probability NEG_KEEP, weighting the kept ones 1/NEG_KEEP. The weighted loss is
an unbiased estimate of the full-data loss, so the fitted probabilities stay
calibrated -- dropping negatives WITHOUT the weight would inflate every
probability by roughly 1/NEG_KEEP in odds. Some negatives are always kept at
weight 1, because they are the ones that decide rankings:
  - every negative that outranks some positive of its entity, i.e. scores at
    least the entity's lowest positive on max_chan (the blocking score
    cap_candidates ranks by; the model's own ranking does not exist yet), and
  - the entity's top NEG_TOP negatives by that score (default 3), so entities
    with no positive -- singletons -- keep the candidates that tempt the
    decision layer into a wrong pick.
Sampling happens after features.build, whose rank / n_cand / gap_top columns
are computed over the entity's FULL candidate list as at test time, and before
strfeatures.build, the expensive per-pair stage.

The singleton head reads entity aggregates (candidate counts, sum of p, the
product rule), which subsampling would change. So the first HEAD_ENT training
entities of each country (default 20000; they are a random sample, the pick is
shuffled) are never subsampled, and the head and its OOF isotonic are fit on
them only.

TRAIN_CHUNK (default 0 = off): build the training features TRAIN_CHUNK entities
at a time and append each chunk as float32 to one flat file under TRAIN_DIR
(default cache/). LightGBM bins it through a lightgbm.Sequence that reads
batches with pread, so the float matrix is never resident: only LightGBM's
binned copy (1 byte per value) is, plus a second one while a weighted model
trains on all rows. Reads go through pread rather than np.memmap
so that --profile's peak RSS counts real allocations only, not file pages the
kernel can drop. The OOF fold models train on Dataset.subset of the one binned
Dataset, and prediction streams the file in batches. The file is deleted when
training ends.

Run this module directly for the calibration self-test on synthetic pairs.
"""
import os
import pathlib

import lightgbm as lgb
import numpy as np

from profiling import stage


def config():
    """-> dict, parsed from the environment once per run."""
    keep = float(os.environ.get("NEG_KEEP", 1.0))
    if not 0.0 < keep <= 1.0:
        raise ValueError(f"NEG_KEEP={keep}: want 0 < NEG_KEEP <= 1")
    cfg = {"keep": keep,
           "top": int(os.environ.get("NEG_TOP", 3)),
           "head_ent": int(os.environ.get("HEAD_ENT", 20000)),
           "chunk": int(os.environ.get("TRAIN_CHUNK", 0)),
           "dir": os.environ.get("TRAIN_DIR", "cache")}
    if cfg["chunk"] < 0 or cfg["head_ent"] < 0 or cfg["top"] < 0:
        raise ValueError("TRAIN_CHUNK, HEAD_ENT and NEG_TOP must be >= 0")
    return cfg


def neg_sample(q, y, score, rank, full, keep, top, rng):
    """Which pairs to keep, and their importance weights.

    q: entity per pair, 0..n-1. y: labels. score: max_chan. rank: features'
    rank column (0 = best, ties share). full: bool per entity, never sampled.
    -> (bool keep mask, float32 weight per KEPT pair).
    """
    q = np.asarray(q, np.int64)
    pos = y == 1
    n_ent = len(full)
    lowest_pos = np.full(n_ent, np.inf)
    np.minimum.at(lowest_pos, q[pos], score[pos])
    forced = (pos | (score >= lowest_pos[q]) | (rank < top)
              | np.asarray(full, bool)[q])
    kept = forced | (rng.random(len(q)) < keep)
    w = np.where(forced, 1.0, 1.0 / keep).astype(np.float32)
    return kept, w[kept]


# --------------------------------------------------------------------------
# float32 feature matrix on disk
# --------------------------------------------------------------------------

class DiskMatrix:
    """Row-major float32 (n x n_cols) matrix in one flat file, appended in
    chunks. np.memmap(path, np.float32, "r", shape=(n, n_cols)) opens it."""

    def __init__(self, directory, n_cols, name="train_X.f32"):
        d = pathlib.Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.path = d / name
        self.n_cols, self.n = n_cols, 0
        self._out = open(self.path, "wb")
        self._fd = None
        self._base = None               # the one binned lgb.Dataset

    def __len__(self):
        return self.n

    @property
    def shape(self):
        return (self.n, self.n_cols)

    def append(self, X):
        X = np.ascontiguousarray(X, np.float32)
        assert X.ndim == 2 and X.shape[1] == self.n_cols, X.shape
        with stage("memmap write", n=len(X), unit="pairs"):
            X.tofile(self._out)
        self.n += len(X)

    def finish(self):
        self._out.close()
        self._fd = os.open(self.path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
        return self

    def read(self, lo, hi):
        """Rows lo..hi-1 as a fresh float32 array."""
        row = self.n_cols * 4
        buf = os.pread(self._fd, (hi - lo) * row, lo * row) \
            if hasattr(os, "pread") else self._seek_read(lo, hi)
        return np.frombuffer(buf, np.float32).reshape(hi - lo, self.n_cols)

    def _seek_read(self, lo, hi):              # Windows has no os.pread
        row = self.n_cols * 4
        os.lseek(self._fd, lo * row, os.SEEK_SET)
        return os.read(self._fd, (hi - lo) * row)

    def batches(self, rows=None, size=1 << 20):
        """-> (lo, hi, X block, local row mask or None)."""
        for lo in range(0, self.n, size):
            hi = min(lo + size, self.n)
            m = None if rows is None else rows[lo:hi]
            if m is not None and not m.any():
                continue
            yield lo, hi, self.read(lo, hi), m

    def columns(self, cols, rows=None):
        """X[rows][:, cols] without reading more than one batch at a time."""
        out = [X[:, cols] if m is None else X[m][:, cols]
               for _, _, X, m in self.batches(rows)]
        return (np.vstack(out) if out
                else np.empty((0, len(range(self.n_cols)[cols])), np.float32))

    def predict(self, model, rows=None):
        """model.predict(X[rows]), streamed."""
        out = [model.predict(X if m is None else X[m])
               for _, _, X, m in self.batches(rows)]
        return np.concatenate(out) if out else np.empty(0)

    def dataset(self, y, w, rows, params, feature_name):
        """lgb.Dataset for X[rows] (rows: bool mask or None) weighted by w.

        The float rows are binned once, into an unweighted base Dataset; every
        other request is a Dataset.subset of it. LightGBM ignores set_weight on
        a subset before construction, cannot unset weights after it, and treats
        all-ones weights as none, so the base stays unweighted and weights go
        on the constructed subset. A weighted request for all rows therefore
        holds a second binned copy (1 byte per value) while it trains.
        """
        if self._base is None:
            with stage("lgb.Dataset (memmap)", n=self.n, unit="pairs"):
                self._base = lgb.Dataset(_Seq(self), label=y, params=params,
                                         feature_name=feature_name,
                                         free_raw_data=True).construct()
        if w is None and rows is None:
            return self._base
        idx = np.arange(self.n) if rows is None else np.flatnonzero(rows)
        sub = self._base.subset(idx.astype(np.int32), params=params)
        if w is not None:
            sub.construct()
            sub.set_weight(w[idx])
        return sub

    def close(self):
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        if not self._out.closed:
            self._out.close()
        self._base = None
        try:
            self.path.unlink()
        except OSError:
            pass


class _Seq(lgb.Sequence):
    """lightgbm.Sequence over a DiskMatrix: batch reads for binning, single
    rows (as float64, which LightGBM's sampler requires) for bin boundaries."""

    batch_size = 1 << 16

    def __init__(self, m):
        self.m = m

    def __len__(self):
        return self.m.n

    def __getitem__(self, i):
        if isinstance(i, slice):
            lo, hi, step = i.indices(self.m.n)
            assert step == 1
            return self.m.read(lo, hi)
        if isinstance(i, (list, np.ndarray)):
            return np.stack([self.m.read(j, j + 1)[0] for j in i])
        return self.m.read(int(i), int(i) + 1)[0].astype(np.float64)


# --------------------------------------------------------------------------
# helpers train_eval calls on either an in-RAM ndarray or a DiskMatrix
# --------------------------------------------------------------------------

def take(X, rows):
    return X if rows is None else X[rows]


def predict(model, X, rows=None):
    if isinstance(X, DiskMatrix):
        return X.predict(model, rows)
    return model.predict(take(X, rows))


def columns(X, cols, rows=None):
    if isinstance(X, DiskMatrix):
        return X.columns(cols, rows)
    return take(X, rows)[:, cols]


# --------------------------------------------------------------------------
# self-test: subsampled + weighted training stays calibrated
# --------------------------------------------------------------------------

def reliability(p, y, bins=10):
    """(ECE, max bin gap) over equal-mass bins of p: |mean p - observed rate|."""
    o = np.argsort(p, kind="stable")
    gaps, sizes = [], []
    for part in np.array_split(o, bins):
        gaps.append(abs(p[part].mean() - y[part].mean()))
        sizes.append(len(part))
    gaps, sizes = np.array(gaps), np.array(sizes)
    return float((gaps * sizes).sum() / sizes.sum()), float(gaps.max())


def synthetic_pairs(n_ent, rng, n_feat=8):
    """Retrieval-shaped pairs: 40-130 candidates per entity, ~5% positive,
    ~6% of entities with no positive; max_chan-like score correlated with y."""
    sizes = rng.integers(40, 131, n_ent)
    q = np.repeat(np.arange(n_ent), sizes)
    n = len(q)
    ent_diff = rng.normal(0, 1, n_ent)[q]       # per-entity difficulty
    X = rng.normal(0, 1, (n, n_feat)).astype(np.float32)
    logit = (1.6 * X[:, 0] + 0.8 * X[:, 1] - 0.6 * X[:, 2] * X[:, 3]
             + 0.7 * ent_diff - 4.6)
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(np.int8)
    score = (X[:, 0] + 0.5 * rng.normal(0, 1, n)).astype(np.float32)
    rank = np.empty(n, np.float32)
    starts = np.r_[0, np.cumsum(sizes)[:-1]]
    for s, e in zip(starts, starts + sizes):
        m = -score[s:e]
        rank[s:e] = np.searchsorted(np.sort(m), m, "left")
    X = np.column_stack([X, score, rank]).astype(np.float32)
    return q, X, y, score, rank


def self_test(n_train=6000, n_test=6000, keep=0.1, top=3, rounds=250, seed=0,
              verbose=True):
    """Train unsubsampled, subsampled+weighted and subsampled-unweighted on the
    same synthetic entities; score reliability of RAW LightGBM probabilities
    (no isotonic, which would hide the bias) on unsubsampled test pairs.
    -> dict of (ECE, max gap) per model and the kept share."""
    rng = np.random.default_rng(seed)
    q, X, y, score, rank = synthetic_pairs(n_train, rng)
    _, Xt, yt, _, _ = synthetic_pairs(n_test, rng)
    params = dict(objective="binary", learning_rate=0.06, num_leaves=63,
                  min_data_in_leaf=50, feature_fraction=0.9, bagging_fraction=0.8,
                  bagging_freq=1, verbose=-1, num_threads=4, seed=seed)

    def fit(Xs, ys, w=None):
        return lgb.train(params, lgb.Dataset(Xs, ys, weight=w), rounds).predict(Xt)

    kept, w = neg_sample(q, y, score, rank, np.zeros(n_train, bool), keep, top,
                         np.random.default_rng(seed + 1))
    res = {"kept_share": float(kept.mean()),
           "full": reliability(fit(X, y), yt),
           "weighted": reliability(fit(X[kept], y[kept], w), yt),
           "unweighted": reliability(fit(X[kept], y[kept]), yt),
           "positives_kept": float(kept[y == 1].mean())}
    if verbose:
        print(f"synthetic: {len(y):,} train pairs ({y.mean():.2%} positive), "
              f"NEG_KEEP={keep}, NEG_TOP={top}: kept {res['kept_share']:.1%} "
              f"of pairs")
        for k in ("full", "weighted", "unweighted"):
            print(f"  {k:11s} ECE {res[k][0]:.4f}   max bin gap {res[k][1]:.4f}")
    return res


if __name__ == "__main__":
    for keep in (0.3, 0.1, 0.05):
        r = self_test(keep=keep)
        ok = abs(r["weighted"][0] - r["full"][0]) <= 0.02
        print(f"  weighted within 0.02 of unsubsampled: {'PASS' if ok else 'FAIL'}\n")
