"""hard_negatives.py: owners, categories, mix weights, and the train_eval wiring
on the tiny chain fixture from test_silent_bugs (no data/ needed)."""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import hard_negatives as hn  # noqa: E402
from test_silent_bugs import _write  # noqa: E402

B, S, N, P = hn.BLOCKER, hn.SIBLING, hn.SAME_NAME, hn.POSITIVE


def test_record_owner_maps_gold_and_leaves_distractors():
    c_ids = np.array(["S2-1", "S2-2", "S3-9", "S3-1"], object)
    gtm = {"a": "S2-1,S3-1", "b": "S2-2", "c": "", "zz": "S2-1"}
    owner = hn.record_owner(["a", "b", "c"], c_ids, gtm)
    assert owner.tolist() == [0, 1, -1, 0]


def test_name_ids_never_pair_empty_names():
    ids = hn.name_ids(["x", "", "x", "", "y"])
    assert ids[0] == ids[2] and ids[1] != ids[3] and ids[0] != ids[4]
    assert ids[1] not in (ids[0], ids[4])


def test_categorise():
    # entity 0 ("x"): pairs to its own gold, a distractor, a same-name entity's
    # gold, its nearest neighbour's gold, and an unrelated entity's gold
    owner = np.array([0, -1, 1, 2, 3])
    nid = np.array([7, 7, 8, 9])
    nbr = np.array([[2, -1], [-1, -1], [-1, -1], [-1, -1]])
    y = np.array([1, 0, 0, 0, 0])
    cat = hn.categorise(y, np.zeros(5, np.int64), np.arange(5), owner, nid, nbr)
    assert cat.tolist() == [P, B, N, S, B]


def test_same_name_wins_over_sibling():
    owner, nid = np.array([1]), np.array([7, 7])
    nbr = np.array([[1], [0]])
    cat = hn.categorise(np.array([0]), np.array([0]), np.array([0]), owner, nid, nbr)
    assert cat.tolist() == [N]


def test_mix_weights_hits_the_target_shares():
    y = np.array([1] * 5 + [0] * 95)
    cat = np.array([P] * 5 + [B] * 80 + [S] * 10 + [N] * 5, np.int8)
    cfg = {"mix": np.array([0.5, 0.25, 0.25]), "max_w": 100.0}
    w = hn.mix_weights(y, cat, cfg)
    assert (w[:5] == 1).all()
    neg = cat != P
    assert w[neg].sum() == pytest.approx(95)                  # balance unchanged
    for k, share in enumerate(cfg["mix"]):
        assert w[cat == k].sum() / w[neg].sum() == pytest.approx(share)


def test_mix_weights_cap_and_missing_category():
    y = np.array([0] * 100)
    cat = np.array([B] * 99 + [S], np.int8)                   # no same_name at all
    w = hn.mix_weights(y, cat, {"mix": np.array([0.5, 0.25, 0.25]), "max_w": 20.0})
    assert w[-1] == 20.0                                      # 1/3 * 100 capped
    assert w[:99] == pytest.approx(np.full(99, (2 / 3) * 100 / 99))


def test_natural_mix_is_uniform_weights():
    y = np.array([1, 0, 0, 0, 0])
    cat = np.array([P, B, B, S, N], np.int8)
    w = hn.mix_weights(y, cat, {"mix": np.array([0.5, 0.25, 0.25]), "max_w": 20.0})
    assert w.tolist() == [1, 1, 1, 1, 1]


def test_config(monkeypatch):
    monkeypatch.delenv("HARD_NEG", raising=False)
    assert hn.config() is None
    monkeypatch.setenv("HARD_NEG", "off")
    assert hn.config() is None
    monkeypatch.setenv("HARD_NEG", "2,1,1")
    assert hn.config()["mix"].tolist() == [0.5, 0.25, 0.25]
    monkeypatch.setenv("HARD_NEG", "natural")
    cfg = hn.config()
    assert cfg["mix"] is None
    assert (hn.mix_weights(np.array([0, 1]), np.array([S, P], np.int8), cfg) == 1).all()
    monkeypatch.setenv("HARD_NEG", "1,1")
    with pytest.raises(ValueError):
        hn.config()


def test_neighbours_exclude_self_and_pad():
    import blocking
    names = ["sharma medical store", "sharma medical store", "sharma medical",
             "verma textiles"]
    nb = hn.neighbours(blocking.build_index(names), names, [0, 3], 2)
    assert 0 not in nb[0] and set(nb[0]) == {1, 2}
    assert nb[1].tolist() == [-1, -1]


GTM = {"S1-1": "S2-1,S3-1", "S1-2": "S2-2", "S1-3": "S3-2"}


def test_prepare_country_off_is_unchanged(tmp_path, monkeypatch):
    import train_eval
    _write(tmp_path / "parquet", "train")
    monkeypatch.setattr(train_eval, "ROOT", str(tmp_path / "parquet"))
    monkeypatch.delenv("HARD_NEG", raising=False)
    tr, va = train_eval.prepare_country("India", 2, 2, GTM, np.random.default_rng(0))
    assert tr[7] is None and va[7] is None


def test_prepare_country_categorises_chain_negatives(tmp_path, monkeypatch):
    import train_eval
    _write(tmp_path / "parquet", "train")
    monkeypatch.setattr(train_eval, "ROOT", str(tmp_path / "parquet"))
    monkeypatch.delenv("HARD_NEG", raising=False)
    off = train_eval.prepare_country("India", 2, 2, GTM, np.random.default_rng(0))
    monkeypatch.setenv("HARD_NEG", "0.5,0.25,0.25")
    tr, va = train_eval.prepare_country("India", 2, 2, GTM, np.random.default_rng(0))
    for part in (tr, va):
        X, y, q, cand, truth, ids, text, hni, _ = part
        cat = hni["cat"]
        assert ((cat == P) == (y == 1)).all()
        for j in np.flatnonzero(cat == N):
            # a chain branch's gold record, offered to a different branch
            assert cand[j] not in GTM.get(ids[q[j]], "").split(",")
            assert any(cand[j] in GTM[e].split(",") for e in ("S1-1", "S1-2", "S1-3"))
        assert hni["orig"].all()
    # the features are untouched by the flag
    assert np.array_equal(off[0][0], tr[0])


def test_prepare_country_add_neighbours(tmp_path, monkeypatch):
    import train_eval
    _write(tmp_path / "parquet", "train")
    monkeypatch.setattr(train_eval, "ROOT", str(tmp_path / "parquet"))
    monkeypatch.setenv("HARD_NEG", "0.5,0.25,0.25")
    monkeypatch.setenv("HARD_NEG_ADD", "1")
    # seed 1 samples S1-2 to train and S1-3 to validate. S1-2's nearest names
    # are its three branches (cosine 1, ties to the lowest index): S1-1 is
    # added, S1-3 may not be, since it is a validation entity.
    tr, va = train_eval.prepare_country("India", 1, 1, GTM, np.random.default_rng(1))
    assert tr[5] == ["S1-2", "S1-1"] and va[5] == ["S1-3"]
    X, y, q, cand, truth, ids, text, hni, _ = tr
    assert (hni["orig"] == (q == 0)).all() and not hni["orig"].all()
    # S2-2 is S1-2's gold, so for the added S1-1 it is a same-name negative
    added = q == 1
    assert (hni["cat"][added & (cand == "S2-2")] == N).all()
    assert (added & (cand == "S2-2")).any()
