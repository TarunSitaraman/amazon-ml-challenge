"""bigtrain.py: negative sampling, the on-disk training matrix, the calibration
self-test, and the train_eval wiring on the tiny fixture (no data/ needed)."""
import pathlib
import sys

import lightgbm as lgb
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import bigtrain  # noqa: E402
from test_hard_negatives import GTM  # noqa: E402
from test_silent_bugs import _write  # noqa: E402


def test_neg_sample_keeps_positives_outranking_and_top_negatives():
    # entity 0: positive at score 0.5; entity 1: no positive; entity 2: full
    q = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2])
    y = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0])
    score = np.array([0.9, 0.5, 0.5, 0.1, 0.8, 0.3, 0.2, 0.1, 0.1])
    rank = np.array([0, 1, 1, 3, 0, 1, 2, 0, 0], np.float32)
    full = np.array([False, False, True])
    # keep ~0: only forced pairs survive
    kept, w = bigtrain.neg_sample(q, y, score, rank, full, 1e-9, 1,
                                  np.random.default_rng(0))
    assert kept.tolist() == [True, True, True, False, True, False, False, True, True]
    assert (w == 1).all()


def test_neg_sample_weights_unbiased():
    rng = np.random.default_rng(0)
    q = np.repeat(np.arange(2000), 50)
    y = np.zeros(len(q), np.int8)
    score = rng.random(len(q))
    rank = np.tile(np.arange(50), 2000).astype(np.float32)
    kept, w = bigtrain.neg_sample(q, y, score, rank, np.zeros(2000, bool), 0.2, 3,
                                  rng)
    assert kept[rank < 3].all()
    assert set(np.unique(w).tolist()) == {1.0, np.float32(5.0)}
    assert abs(w.sum() / len(q) - 1) < 0.01          # weights restore the count


def test_disk_matrix_roundtrip_and_training_matches_numpy(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5000, 4)).astype(np.float32)
    y = (X[:, 0] + rng.normal(size=5000) > 0).astype(float)
    w = rng.uniform(0.5, 2, 5000)
    m = bigtrain.DiskMatrix(tmp_path, 4)
    for lo in range(0, 5000, 1300):
        m.append(X[lo:lo + 1300])
    m.finish()
    assert np.array_equal(np.memmap(m.path, np.float32, "r", shape=m.shape), X)
    rows = rng.random(5000) < 0.3
    assert np.array_equal(bigtrain.columns(m, slice(1, 3), rows), X[rows][:, 1:3])
    params = dict(objective="binary", verbose=-1, num_threads=1, deterministic=True)
    names = [f"f{i}" for i in range(4)]
    full = lgb.train(params, m.dataset(y, w, None, params, names), 10)
    ref = lgb.train(params, lgb.Dataset(X, y, weight=w, feature_name=names), 10)
    assert np.allclose(bigtrain.predict(full, m), ref.predict(X))
    assert np.allclose(bigtrain.predict(full, m, rows), ref.predict(X[rows]))
    # weighted and unweighted subsets really differ
    a = lgb.train(params, m.dataset(y, w, rows, params, names), 10)
    b = lgb.train(params, m.dataset(y, None, rows, params, names), 10)
    assert not np.allclose(a.predict(X), b.predict(X))
    m.close()
    assert not m.path.exists()


def test_self_test_weighted_is_calibrated_unweighted_is_not():
    r = bigtrain.self_test(n_train=3000, n_test=3000, keep=0.1, rounds=150,
                           verbose=False)
    assert r["positives_kept"] == 1.0
    assert abs(r["weighted"][0] - r["full"][0]) <= 0.02
    assert r["unweighted"][0] > r["weighted"][0] + 0.01


def test_prepare_country_chunked_matches_in_ram(tmp_path, monkeypatch):
    import train_eval
    _write(tmp_path / "parquet", "train")
    monkeypatch.setattr(train_eval, "ROOT", str(tmp_path / "parquet"))
    ref, _ = train_eval.prepare_country("India", 3, 1, GTM, np.random.default_rng(0))
    monkeypatch.setenv("TRAIN_CHUNK", "1")
    store = bigtrain.DiskMatrix(tmp_path / "cache", ref[0].shape[1])
    tr, _ = train_eval.prepare_country("India", 3, 1, GTM, np.random.default_rng(0),
                                       store)
    assert tr[0] is store and tr[3] is None and tr[8] is None
    store.finish()
    assert np.array_equal(store.read(0, store.n), ref[0])
    for j in (1, 2):
        assert np.array_equal(tr[j], ref[j])
    assert tr[4] == ref[4] and tr[5] == ref[5] and np.array_equal(tr[6], ref[6], equal_nan=True)
    store.close()


def test_prepare_country_neg_keep_keeps_positives(tmp_path, monkeypatch):
    import train_eval
    _write(tmp_path / "parquet", "train")
    monkeypatch.setattr(train_eval, "ROOT", str(tmp_path / "parquet"))
    ref, rva = train_eval.prepare_country("India", 3, 1, GTM, np.random.default_rng(0))
    monkeypatch.setenv("NEG_KEEP", "0.01")
    monkeypatch.setenv("NEG_TOP", "0")
    monkeypatch.setenv("HEAD_ENT", "1")
    tr, va = train_eval.prepare_country("India", 3, 1, GTM, np.random.default_rng(0))
    smp = tr[8]
    assert smp["full"].tolist() == [True, False, False]
    assert tr[1].sum() == ref[1].sum()                       # every positive
    assert (tr[2] == 0).sum() == (ref[2] == 0).sum()         # head entity whole
    assert len(smp["w"]) == len(tr[1])
    assert np.array_equal(va[0], rva[0]) and va[8] is None   # validation untouched


def test_hard_neg_mix_measured_in_importance_weight():
    import hard_negatives as hn
    P, B, S = hn.POSITIVE, hn.BLOCKER, hn.SIBLING
    # 10 blockers sampled at 0.1 (weight 10) stand for 100; 20 siblings kept whole
    y = np.array([1] * 5 + [0] * 30)
    cat = np.array([P] * 5 + [B] * 10 + [S] * 20, np.int8)
    base = np.r_[np.ones(5), np.full(10, 10.0), np.ones(20)]
    cfg = {"mix": np.array([0.5, 0.5, 0.0]), "max_w": 100.0}
    w = hn.mix_weights(y, cat, cfg, base=base)
    assert (w[:5] == 1).all()
    neg = cat != P
    assert np.isclose(w[neg].sum(), 120)                     # balance unchanged
    assert np.isclose(w[cat == B].sum(), 60) and np.isclose(w[cat == S].sum(), 60)
    assert np.array_equal(hn.mix_weights(y, cat, {"mix": None}, base=base), base)
