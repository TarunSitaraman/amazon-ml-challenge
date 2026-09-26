"""preranker.py and the train_eval cut it plugs into. Synthetic only."""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import features  # noqa: E402
import preranker  # noqa: E402
import train_eval  # noqa: E402
from blocking import CHANNELS  # noqa: E402


def make_lists(n_ent, rng, lo=5, hi=200):
    """Retrieval-shaped pairs: q sorted, sparse channels, 0/1 key channels,
    ~2% positives that tend to fire several channels at a moderate score while
    some distractors have one very high channel."""
    size = rng.integers(lo, hi, n_ent)
    q = np.repeat(np.arange(n_ent), size)
    n = len(q)
    y = (rng.random(n) < 0.02).astype(np.int8)
    fire = rng.random((n, len(CHANNELS))) < np.where(y[:, None], 0.6, 0.25)
    chan = np.where(fire, rng.random((n, len(CHANNELS))), 0).astype(np.float32)
    for j in (CHANNELS.index("c1_canon"), CHANNELS.index("c5_postal")):
        chan[:, j] = (chan[:, j] > 0).astype(np.float32)
    spike = (rng.random(n) < 0.1) & (y == 0)
    chan[spike, 0] = 0.9 + 0.1 * rng.random(spike.sum())
    chan[~fire.any(1), 0] = 0.01           # every retrieved pair fired somewhere
    is_s3 = rng.random(n) < 0.3
    return q, chan, y, is_s3


def _old_cap(q, c, chan, cap):
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


def test_default_cut_is_unchanged():
    rng = np.random.default_rng(0)
    q, chan, _, _ = make_lists(300, rng)
    c = rng.integers(0, 10**6, len(q))
    for cap in (1, 30, 60, 500):
        for a, b in zip(train_eval.cap_candidates(q, c, chan, cap), _old_cap(q, c, chan, cap)):
            assert np.array_equal(a, b)


def test_group_rank_matches_features_rank_with_ties():
    rng = np.random.default_rng(1)
    q, chan, _, _ = make_lists(200, rng)
    chan = np.round(chan, 1)                        # plenty of ties
    X = features.build(q, chan, np.zeros(len(q), bool))
    starts, ends = preranker.groups(q)
    r = preranker.group_rank(q, chan.max(1), starts, ends)
    assert np.array_equal(r, X[:, features.NAMES.index("rank")])


def test_build_shape_and_empty():
    rng = np.random.default_rng(2)
    q, chan, _, is_s3 = make_lists(50, rng)
    F = preranker.build(q, chan, is_s3)
    assert F.shape == (len(q), len(preranker.NAMES)) and F.dtype == np.float32
    assert np.isfinite(F).all()
    assert preranker.build(q[:0], chan[:0], is_s3[:0]).shape == (0, len(preranker.NAMES))


@pytest.mark.parametrize("kind", ["logreg", "lgb"])
def test_prerankers_beat_chan_max_where_agreement_matters(kind):
    rng = np.random.default_rng(3)
    q, chan, y, s3 = make_lists(1500, rng)
    r = preranker.PreRanker(kind).fit(preranker.build(q, chan, s3), y)
    qv, cv, yv, s3v = make_lists(1500, rng)
    sc = r.score(preranker.build(qv, cv, s3v))
    base = train_eval.ceiling_hits(qv, yv, cv.max(1), 1500, [20]).sum()
    ours = train_eval.ceiling_hits(qv, yv, sc, 1500, [20]).sum()
    assert ours > base


def test_logreg_score_is_sklearn_decision_function():
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(4)
    q, chan, y, s3 = make_lists(300, rng)
    F = preranker.build(q, chan, s3)
    r = preranker.PreRanker("logreg").fit(F, y)
    mu, sd = F.mean(0), F.std(0) + 1e-6
    m = LogisticRegression(C=1.0, max_iter=500).fit((F - mu) / sd, y)
    assert np.allclose(r.score(F), m.decision_function((F - mu) / sd), atol=1e-3)


def test_one_class_fits_constant():
    rng = np.random.default_rng(5)
    q, chan, _, s3 = make_lists(20, rng)
    F = preranker.build(q, chan, s3)
    for kind in ("logreg", "lgb"):
        r = preranker.PreRanker(kind).fit(F, np.zeros(len(q), np.int8))
        assert not r.score(F).any()


def test_ceiling_hits_uncapped_equals_all():
    rng = np.random.default_rng(6)
    q, chan, y, _ = make_lists(100, rng)
    h = train_eval.ceiling_hits(q, y, chan.max(1), 100, [10, 10**6])
    assert np.array_equal(h[:, 1], np.bincount(q, weights=y, minlength=100))
    assert (h[:, 0] <= h[:, 1]).all()
