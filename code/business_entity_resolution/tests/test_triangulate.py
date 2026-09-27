"""triangulate.py: the stage-2 features against their loop oracle, the edge
cases, and the synthetic 'weak 3rd match' self-test (no data/ needed)."""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import strfeatures  # noqa: E402
import triangulate as tri  # noqa: E402

COL = {k: i for i, k in enumerate(tri.NAMES)}


def _recs():
    names = ["sharma medical store", "sharma medicl store", "sharma med store",
             "gupta traders", "kumar bakery", "sharma medical store"]
    addrs = ["12 mg road 560001", "12 mg rd 560001", "", "5 fort road 400001",
             "12 hill road 560009", "99 park street 560002"]
    return strfeatures.precompute_records(names, addrs)


def test_self_test():
    tri._self_test()


def test_edge_cases_match_reference():
    recs = _recs()
    # entity 0: 5 candidates; entity 1: one candidate; entity 2: tied p
    q = np.array([0, 0, 0, 0, 0, 1, 2, 2, 2])
    c = np.array([0, 1, 2, 3, 4, 5, 0, 5, 1])
    p = np.array([0.9, 0.6, 0.3, 0.3, 0.1, 0.8, 0.5, 0.5, 0.2])
    s3 = np.array([0, 1, 0, 1, 0, 1, 0, 1, 1], bool)
    for tj in (1, 2, 10):
        a = tri.features(q, c, p, s3, recs, tj)
        assert np.array_equal(a, tri.reference(q, c, p, s3, recs, tj))
    T = tri.features(q, c, p, s3, recs)
    # the lone candidate has no anchors
    assert T[5, COL["tri_nm_simp"]] == 0 and T[5, COL["tri_nm_p"]] == 0
    assert T[5, COL["tri_rank"]] == 0 and T[5, COL["tri_xsrc_top"]] == 0
    # ties share the best rank
    assert T[6, COL["tri_rank"]] == T[7, COL["tri_rank"]] == 0
    # the weak copy (row 2) resembles the top candidate, the bakery does not
    assert T[2, COL["tri_nm_simp"]] > 0.3 > T[4, COL["tri_nm_simp"]]
    assert T[2, COL["tri_nm_p"]] == np.float32(0.9)
    # row 1 shares digits with the confident top (the other anchors have p <=
    # 0.5); row 4 shares "12" with both confident anchors (p 0.9 and 0.6)
    assert T[1, COL["tri_n_conf_dg"]] == 1 and T[4, COL["tri_n_conf_dg"]] == 2
    assert T[3, COL["tri_n_conf_dg"]] == 0
    # cross source relative to the top: row 1 is S3 vs the S2 top
    assert T[1, COL["tri_xsrc_top"]] == 1 and T[0, COL["tri_xsrc_top"]] == 0


def test_empty_and_unsorted():
    recs = _recs()
    e = np.zeros(0, np.int64)
    assert tri.features(e, e, np.zeros(0), np.zeros(0, bool), recs).shape == \
        (0, len(tri.NAMES))
    with pytest.raises(ValueError):
        tri.features(np.array([1, 0]), np.array([0, 1]), np.array([.5, .5]),
                     np.zeros(2, bool), recs)
    with pytest.raises(ValueError):
        tri.features(np.array([0]), np.array([0]), np.array([.5]),
                     np.zeros(1, bool), recs, top_j=33)


def test_prepare_country_keeps_candidate_sets(tmp_path, monkeypatch):
    """TRI=1: prep() keeps just its candidates' record sets, and features on
    them equal features on the whole corpus's Records."""
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import blocking
    import train_eval
    from test_silent_bugs import _write
    _write(tmp_path / "parquet", "train")
    monkeypatch.setattr(train_eval, "ROOT", str(tmp_path / "parquet"))
    monkeypatch.setattr(tri, "ENABLED", False)
    off = train_eval.prepare_country("India", 2, 2, {}, np.random.default_rng(0))
    assert off[0][8] is None and off[1][8] is None
    monkeypatch.setattr(tri, "ENABLED", True)
    tr, va = train_eval.prepare_country("India", 2, 2, {}, np.random.default_rng(0))
    _, corpus_tab = blocking.load_shard(str(tmp_path / "parquet"), "train", "India")
    corpus = blocking.build_corpus(corpus_tab, verbose=False, string_features=True)
    c_ids = corpus_tab.column("entity_id").to_pylist()
    for part in (tr, va):
        X, q, cand, (sets, c_local) = part[0], part[2], part[3], part[8]
        c = np.array([c_ids.index(x) for x in cand])
        p = np.linspace(0.9, 0.1, len(q))
        s3 = X[:, train_eval.IS_S3_COL] > 0.5
        assert np.array_equal(tri.features(q, c_local, p, s3, sets),
                              tri.features(q, c, p, s3, corpus["recs"]))
