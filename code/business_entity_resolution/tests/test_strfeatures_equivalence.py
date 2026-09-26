"""strfeatures.build (precomputed CSR records) must equal the original per-pair
loop, build_reference, bit for bit.

The synthetic pairs are shaped like real retrieval output, not random strings:
blocking retrieves a candidate BECAUSE it shares tokens with the query, so each
candidate keeps ~70% of its query's name and address tokens, and each corpus
record sits in many entities' candidate lists. Random non-overlapping text
would only exercise the empty-intersection path.
"""
import pathlib
import sys

import numpy as np
import pyarrow as pa
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import strfeatures  # noqa: E402
from textnorm import norm  # noqa: E402

KEEP = 0.85


def _word(rng):
    return "".join(rng.choice(list("abcdefghijklmnopqrstuvwxyz"), rng.integers(3, 10)))


def make_pairs(n_entities, per_entity, rng, slots_per_record=10):
    """-> (q_names, q_addrs, q_idx, c_names, c_addrs, c) with c indexing the
    corpus lists.

    Entities come in groups of slots_per_record that share a base name and
    street (a chain, or a common word plus a locality), which is what makes
    blocking retrieve the same records for all of them. Each group owns
    per_entity corpus records, each a mutation of one member keeping ~70% of
    its tokens, and every member's candidate list is the group's records. So a
    record fills slots_per_record lists, like ~104M slots over ~10M records."""
    words = [_word(rng) for _ in range(4000)]
    zipf = 1.0 / np.arange(1, len(words) + 1)
    zipf /= zipf.sum()

    def words_(k):
        return [words[i] for i in rng.choice(len(words), k, p=zipf)]

    q_names, q_addrs, group = [], [], []
    n_groups = -(-n_entities // slots_per_record)
    for g in range(n_groups):
        base = words_(rng.integers(2, 4))
        street, city = words_(rng.integers(2, 4)), words_(1)
        pin = str(rng.integers(100000, 999999))
        for _ in range(min(slots_per_record, n_entities - len(q_names))):
            nm = base + words_(rng.integers(0, 2))
            if rng.random() < 0.1:
                nm.append(_word(rng))          # a token the corpus may never see
            if rng.random() < 0.05:
                nm.append(str(rng.integers(1, 99)))
            ad = [str(rng.integers(1, 999))] + street + city
            if rng.random() < 0.6:
                ad.append(pin)
            if rng.random() < 0.2:
                ad.insert(1, f"{rng.integers(1, 99)}{'abc'[rng.integers(3)]}")
            q_names.append(nm)
            q_addrs.append([] if rng.random() < 0.05 else ad)
            group.append(g)

    def mutate(toks):
        if not toks:
            return []
        kept = [t for t in toks if rng.random() < KEEP] or [toks[0]]
        kept += words_(rng.poisson(0.5))
        r = rng.random()
        if r < 0.15:
            rng.shuffle(kept)              # same token set, different order
        elif r < 0.2:
            kept.append(kept[0])           # repeated token
        return kept

    group = np.array(group)
    members = [np.flatnonzero(group == g) for g in range(n_groups)]
    c_names, c_addrs, recs_of = [], [], []
    for g in range(n_groups):
        ids = []
        for _ in range(per_entity):
            e = rng.choice(members[g])
            r = rng.random()
            if r < 0.15:                   # exact copy (chains, true matches)
                nm = list(q_names[e])
            elif r < 0.2:
                nm = []                    # empty after normalisation
            else:
                nm = mutate(q_names[e])
            ad = [] if rng.random() < 0.05 else mutate(q_addrs[e])
            ids.append(len(c_names))
            c_names.append(" ".join(nm))
            c_addrs.append(" ".join(ad))
        recs_of.append(np.array(ids))

    # Some raw punctuation / case so norm() does real work.
    def raw(t):
        return (t.upper() if rng.random() < 0.1 else t) + \
            ("." if rng.random() < 0.05 else "")
    q_names = [norm(" ".join(raw(t) for t in x)) for x in q_names]
    q_addrs = [norm(", ".join(x)) for x in q_addrs]
    c_names = [norm(x) for x in c_names]
    c_addrs = [norm(x) for x in c_addrs]

    q_idx = np.repeat(np.arange(n_entities), per_entity)
    c = np.concatenate([recs_of[g] for g in group])
    return q_names, q_addrs, q_idx, c_names, c_addrs, c


def _both(q_names, q_addrs, q_idx, c_names, c_addrs, c, idf_lut, grammar=None):
    # grammar passed explicitly: a local data/corruption_grammar.json must not
    # leak into build() through its default
    recs = strfeatures.precompute_records(c_names, c_addrs)
    new = strfeatures.build(recs, q_names, q_addrs, q_idx, c, idf_lut, grammar)
    na, aa = pa.array(c_names), pa.array(c_addrs)
    old = strfeatures.build_reference(q_names, q_addrs, q_idx,
                                      na.take(c).to_pylist(),
                                      aa.take(c).to_pylist(), idf_lut, grammar)
    return old, new


def _assert_identical(old, new):
    assert old.dtype == new.dtype and old.shape == new.shape
    diff = old.view(np.uint32) != new.view(np.uint32)
    if diff.any():
        cols = sorted({strfeatures.NAMES[k] for k in np.flatnonzero(diff.any(0))})
        pytest.fail(f"{int(diff.any(1).sum())} pairs differ, in {cols}")


def test_bit_identical_on_realistic_pairs():
    rng = np.random.default_rng(0)
    data = make_pairs(2500, 24, rng)            # 60,000 pairs
    assert len(data[2]) >= 50_000
    vocab = sorted({t for s in data[3] for t in s.split()})
    idf_lut = {t: float(rng.gamma(2.0, 2.0)) for t in vocab if rng.random() < 0.9}
    old, new = _both(*data, idf_lut)
    _assert_identical(old, new)

    col = {k: i for i, k in enumerate(strfeatures.NAMES)}
    # The shape is what makes the test meaningful: most pairs overlap.
    assert (old[:, col["nm_jac"]] > 0).mean() > 0.6
    assert (old[:, col["nm_4gram"]] > 0).mean() > 0.6
    assert (old[:, col["dg_shared"]] > 0).mean() > 0.3
    assert 0.02 < old[:, col["nm_exact"]].mean() < 0.5
    assert (old[:, col["nm_rare_shared"]] > 0).mean() > 0.5


def test_bit_identical_without_idf_and_edge_cases():
    q_names = [norm(x) for x in ["Acme Ltd", "", "a b", "Studio 54", "x"]]
    q_addrs = [norm(x) for x in ["12 Main St", "", "5 5 5", "", "unseen 77"]]
    c_names = [norm(x) for x in ["Acme Ltd", "Ltd Acme", "", "a b", "b a",
                                 "Studio 54", "Studio 54 54", "xyzw", "x"]]
    c_addrs = [norm(x) for x in ["12 Main St", "", "", "5", "", "54",
                                 "Main 12", "77", "unseen"]]
    q_idx = np.repeat(np.arange(len(q_names)), len(c_names))
    c = np.tile(np.arange(len(c_names)), len(q_names))
    for lut in (None, {"acme": 3.5, "54": 1.0, "ltd": 0.0}):
        _assert_identical(*_both(q_names, q_addrs, q_idx, c_names, c_addrs, c, lut))


def test_rejects_unnormalised_text():
    with pytest.raises(ValueError):
        strfeatures.precompute_records(["Acme Ltd"], [""])
    with pytest.raises(ValueError):
        strfeatures.precompute_records(["acme  ltd"], [""])


def test_empty_candidate_list():
    recs = strfeatures.precompute_records(["acme"], [""])
    X = strfeatures.build(recs, ["acme"], [""], np.zeros(0, int), np.zeros(0, int),
                          None, None)
    assert X.shape == (0, len(strfeatures.NAMES))


def test_bit_identical_across_chunk_boundaries(monkeypatch):
    # Production corpora span many 250k-record precompute chunks and many
    # pair chunks; shrink both so this corpus crosses dozens of boundaries.
    monkeypatch.setattr(strfeatures, "_CHUNK", 97)
    monkeypatch.setattr(strfeatures, "_PAIR_CHUNK", 1013)
    rng = np.random.default_rng(3)
    data = make_pairs(400, 24, rng)
    vocab = sorted({t for s in data[3] for t in s.split()})
    idf_lut = {t: float(rng.gamma(2.0, 2.0)) for t in vocab}
    _assert_identical(*_both(*data, idf_lut))


def test_accepts_chunked_arrow_columns():
    names = pa.chunked_array([["acme ltd"], ["zeta 12"]])
    addrs = pa.chunked_array([["12 main st", ""]])
    recs = strfeatures.precompute_records(names, addrs)
    ref = strfeatures.precompute_records(["acme ltd", "zeta 12"], ["12 main st", ""])
    for k in ("name_ids", "gram_ids", "dig_ids", "seq_ids"):
        assert np.array_equal(getattr(recs, k), getattr(ref, k))
