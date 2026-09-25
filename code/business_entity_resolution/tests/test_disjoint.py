import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from disjoint import resolve_conflicts, sinkhorn_normalise  # noqa: E402


def a(*xs):
    return np.array(xs)


# ---- resolve_conflicts -------------------------------------------------------

def owners(q, c, p, acc):
    m = resolve_conflicts(q, c, p, acc)
    return sorted(zip(q[m].tolist(), c[m].tolist()))


def test_resolve_empty_and_nothing_accepted():
    assert resolve_conflicts(a(), a(), a(), a()).tolist() == []
    q, c, p = a(0, 1), a(10, 10), a(0.9, 0.8)
    assert not resolve_conflicts(q, c, p, a(False, False)).any()


def test_resolve_no_conflict_is_unchanged():
    q, c, p = a(0, 0, 1), a(10, 11, 12), a(0.9, 0.8, 0.7)
    acc = a(True, True, True)
    assert resolve_conflicts(q, c, p, acc).tolist() == acc.tolist()


def test_resolve_output_is_subset_of_accepted():
    q, c, p = a(0, 0, 1, 1), a(10, 11, 10, 11), a(0.9, 0.3, 0.8, 0.6)
    acc = a(True, False, True, True)
    m = resolve_conflicts(q, c, p, acc)
    assert not (m & ~acc).any()


def test_resolve_symmetric_conflict_goes_to_higher_probability():
    # identical entities except the probability on the shared record
    q, c = a(0, 0, 1, 1), a(10, 11, 10, 12)
    p = a(0.6, 0.9, 0.9, 0.9)
    acc = np.ones(4, bool)
    assert owners(q, c, p, acc) == [(0, 11), (1, 10), (1, 12)]


def test_resolve_prefers_bigger_f_gain_over_bigger_probability():
    # entity 0: the shared record is its only pick, dropping it costs a lot.
    # entity 1: already has two strong records, the shared one adds little.
    q = a(0, 1, 1, 1)
    c = a(10, 10, 11, 12)
    p = a(0.70, 0.75, 0.95, 0.95)
    acc = np.ones(4, bool)
    assert owners(q, c, p, acc) == [(0, 10), (1, 11), (1, 12)]


def test_resolve_drops_record_when_nobody_gains():
    # both claimants are better off without a weak record
    q = a(0, 0, 0, 1, 1, 1)
    c = a(10, 11, 99, 12, 13, 99)
    p = a(0.95, 0.95, 0.05, 0.95, 0.95, 0.05)
    acc = np.ones(6, bool)
    assert 99 not in [rec for _, rec in owners(q, c, p, acc)]


def test_resolve_tie_goes_to_lowest_entity():
    q, c, p = a(2, 1, 3), a(10, 10, 10), a(0.8, 0.8, 0.8)
    acc = np.ones(3, bool)
    assert owners(q, c, p, acc) == [(1, 10)]


def test_resolve_tie_is_independent_of_row_order():
    q, c, p = a(2, 1, 3, 1), a(10, 10, 10, 11), a(0.8, 0.8, 0.8, 0.9)
    acc = np.ones(4, bool)
    base = owners(q, c, p, acc)
    rng = np.random.default_rng(0)
    for _ in range(20):
        o = rng.permutation(4)
        assert owners(q[o], c[o], p[o], acc[o]) == base


def test_resolve_at_most_one_owner_random():
    rng = np.random.default_rng(1)
    q = rng.integers(0, 60, 4000)
    c = rng.integers(0, 400, 4000)
    p = rng.integers(1, 20, 4000) / 20      # coarse, many ties
    acc = rng.random(4000) < 0.5
    m = resolve_conflicts(q, c, p, acc)
    assert np.bincount(c[m]).max() <= 1
    assert not (m & ~acc).any()


def test_redecide_loser_picks_up_next_candidate():
    # entity 1 keeps only A (10) at first; A goes to entity 0, whose only
    # candidate it is. Re-deciding lets entity 1 fall back to B (11).
    q = a(0, 1, 1)
    c = a(10, 10, 11)
    p = a(0.90, 0.70, 0.40)
    acc = a(True, True, False)
    assert owners(q, c, p, acc) == [(0, 10)]
    m = resolve_conflicts(q, c, p, acc, redecide=True)
    assert sorted(zip(q[m].tolist(), c[m].tolist())) == [(0, 10), (1, 11)]


def test_redecide_never_takes_a_record_owned_by_another_entity():
    # entity 1 loses 10; its fallback 12 is already owned by entity 2
    q = a(0, 1, 1, 2)
    c = a(10, 10, 12, 12)
    p = a(0.90, 0.70, 0.40, 0.90)
    acc = a(True, True, False, True)
    m = resolve_conflicts(q, c, p, acc, redecide=True)
    assert sorted(zip(q[m].tolist(), c[m].tolist())) == [(0, 10), (2, 12)]


def test_redecide_at_most_one_owner_random():
    rng = np.random.default_rng(3)
    q = rng.integers(0, 60, 4000)
    c = rng.integers(0, 400, 4000)
    p = rng.integers(1, 20, 4000) / 20
    acc = rng.random(4000) < 0.5
    m = resolve_conflicts(q, c, p, acc, redecide=True)
    assert np.bincount(c[m]).max() <= 1


# ---- sinkhorn_normalise ------------------------------------------------------

def test_sinkhorn_single_suitor_keeps_probability():
    q, c, p = a(0, 1, 2), a(10, 11, 12), a(0.2, 0.5, 0.9)
    assert np.allclose(sinkhorn_normalise(q, c, p), p)


def test_sinkhorn_record_mass_at_most_one():
    rng = np.random.default_rng(2)
    q = rng.integers(0, 50, 3000)
    c = rng.integers(0, 200, 3000)
    p = rng.random(3000)
    out = sinkhorn_normalise(q, c, p)
    assert np.bincount(c, out).max() <= 1.0 + 1e-9
    assert (out <= p + 1e-12).all()          # never manufactures mass


def test_sinkhorn_contested_record_uses_null_sink():
    # two suitors at 0.5 each: odds 1, 1 against a null of 1 -> 1/3 each
    out = sinkhorn_normalise(a(0, 1), a(10, 10), a(0.5, 0.5))
    assert np.allclose(out, [1 / 3, 1 / 3])


def test_sinkhorn_tie_is_symmetric():
    out = sinkhorn_normalise(a(0, 1, 2), a(10, 10, 10), a(0.7, 0.7, 0.7))
    assert np.allclose(out, out[0])


def test_sinkhorn_row_budget_caps_entity_and_frees_the_record():
    q, c, p = a(0, 0, 1), a(10, 11, 10), a(0.9, 0.9, 0.9)
    free = sinkhorn_normalise(q, c, p)
    capped = sinkhorn_normalise(q, c, p, row_budget=a(0.5, 10.0))
    assert np.bincount(q, capped)[0] <= 0.5 + 1e-6
    assert capped[2] > free[2]               # rival picks up what entity 0 gave back


def test_sinkhorn_row_budget_longer_than_candidates():
    # entity 2 exists in the batch but has no candidates
    out = sinkhorn_normalise(a(0, 1), a(10, 11), a(0.5, 0.5), row_budget=a(1.0, 1.0, 1.0))
    assert np.allclose(out, [0.5, 0.5])


def test_sinkhorn_empty():
    assert sinkhorn_normalise(a(), a(), a()).tolist() == []
