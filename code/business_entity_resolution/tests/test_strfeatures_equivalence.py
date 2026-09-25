"""strfeatures.build must equal the per-pair loop it replaced, value for value.

build_loop below is the previous implementation, copied verbatim, kept as the
reference. Throughput, measured with bench() on one core of a 4-vCPU cloud
sandbox (numpy 2.x, scipy 1.x), 100k entities, 2M pairs with realistic token
counts, candidates recurring ~20x: ~190k pairs/s end to end (~6 min for
70M pairs) against ~11k pairs/s for the loop (~1.8 h), about 17x. Tokenising
each distinct record in Python is now the larger cost, not the pair stage.
"""
import pathlib
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import strfeatures  # noqa: E402
from textnorm import ngrams, norm  # noqa: E402


# ---- previous implementation (reference) ------------------------------------

NAMES = strfeatures.NAMES

def _prep(texts):
    """-> (token sets, digit-token sets, 4-gram sets) for already-normalised text."""
    toks, digs, grams = [], [], []
    for t in texts:
        s = set(t.split())
        toks.append(s)
        digs.append({x for x in s if x[:1].isdigit()})
        grams.append(ngrams(t, 4))
    return toks, digs, grams


def _jac(a, b):
    if not a and not b:
        return 0.0
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def _cont(a, b):
    m = min(len(a), len(b))
    return len(a & b) / m if m else 0.0


def build_loop(q_names, q_addrs, q_idx, c_names, c_addrs, idf_lut=None):
    """q_*: per-entity normalised text. q_idx: entity index per pair.
    c_*: per-PAIR normalised candidate text (already gathered)."""
    n = len(c_names)
    out = np.zeros((n, len(NAMES)), np.float32)
    qn_t, qn_d, qn_g = _prep(q_names)
    qa_t, qa_d, _ = _prep(q_addrs)

    for i in range(n):
        e = q_idx[i]
        cn, ca = c_names[i], c_addrs[i]
        ct = set(cn.split())
        at = set(ca.split())
        # both sides take digits from name AND address, as qd does
        cd = {x for x in at | ct if x[:1].isdigit()}
        qt, qg = qn_t[e], qn_g[e]
        qa, qd = qa_t[e], qn_d[e] | qa_d[e]

        out[i, 0] = _jac(qt, ct)
        out[i, 1] = _cont(qt, ct)
        out[i, 2] = _jac(qg, ngrams(cn, 4))
        out[i, 3] = float(q_names[e] == cn and bool(cn))
        lq, lc = len(q_names[e]), len(cn)
        out[i, 4] = min(lq, lc) / max(lq, lc) if max(lq, lc) else 0.0
        out[i, 5] = _jac(qa, at)
        out[i, 6] = _cont(qa, at)
        out[i, 7] = _jac(qd, cd)
        out[i, 8] = len(qd & cd)
        out[i, 9] = float(bool(qd) and bool(cd))
        out[i, 10] = float(not qa)
        out[i, 11] = float(not at)
        if idf_lut is not None:
            shared = qt & ct
            out[i, 12] = max((idf_lut.get(t, 0.0) for t in shared), default=0.0)
    return out


# ---- equivalence ------------------------------------------------------------

POOL_N = ["Acme Ltd", "Acme Pvt Ltd", "ACME LTD.", "Zen Cafe", "", "3M", "A&B Stores 2",
          "Studio 54", "M/s. A.B.C. Traders", "Sœur Marie", "राम स्टोर", "7 Eleven",
          "Café Rouge", "  ", "Joe's B & B", "Sharma Medical Store"]
POOL_A = ["12 Main St", "", "Plot 4 Sector 9", "12 Main Street", "9 9 9", "99 Main St",
          "4 Rue de Rivoli 75001", "12 एमजी रोड", "Shop 3, 2nd Floor, 560001", " ",
          "C/O A.K. Sharma 12", "Main St"]


def _random_case(seed, n_ent, n_pairs):
    rng = np.random.default_rng(seed)
    pick = lambda pool, k: [norm(pool[j]) for j in rng.integers(0, len(pool), k)]
    qn, qa = pick(POOL_N, n_ent), pick(POOL_A, n_ent)
    q = rng.integers(0, n_ent, n_pairs)
    cn, ca = pick(POOL_N, n_pairs), pick(POOL_A, n_pairs)
    same = rng.random(n_pairs) < 0.3             # plenty of exact and chain pairs
    cn = [qn[e] if s else x for e, s, x in zip(q, same, cn)]
    ca = [qa[e] if s and rng.random() < 0.5 else x for e, s, x in zip(q, same, ca)]
    return qn, qa, q, cn, ca


IDF = {"acme": 5.0, "ltd": 0.1, "main": 2.0, "st": 0.5, "store": 1.5, "sharma": 3.0}


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("idf", [None, IDF])
def test_matches_loop(seed, idf):
    qn, qa, q, cn, ca = _random_case(seed, 40, 3000)
    ref = build_loop(qn, qa, q, cn, ca, idf)
    for chunk in (1, 7, 1_000_000):
        got = strfeatures.build(qn, qa, q, cn, ca, idf, chunk=chunk)
        assert got.dtype == np.float32 and got.shape == ref.shape
        bad = np.argwhere(got != ref)
        assert not len(bad), [(strfeatures.NAMES[j], got[i, j], ref[i, j]) for i, j in bad[:5]]


def test_empty_inputs():
    assert strfeatures.build([], [], np.array([], int), [], []).shape == (0, 13)
    X = strfeatures.build([""], [""], np.array([0]), [""], [""], IDF)
    assert np.array_equal(X, build_loop([""], [""], np.array([0]), [""], [""], IDF))


def bench(n_ent=100_000, n_pairs=2_000_000, loop_pairs=100_000):
    rng = np.random.default_rng(1)
    words = np.array(["".join(w) for w in rng.choice(list("abcdefghijklmnop"), (30_000, 6))])
    mk = lambda k: " ".join(rng.choice(words, k))
    qn = [mk(rng.integers(1, 5)) for _ in range(n_ent)]
    qa = [mk(rng.integers(3, 8)) + f" {rng.integers(1, 999)} {rng.integers(10000, 99999)}"
          for _ in range(n_ent)]
    pool = rng.integers(0, n_ent, n_pairs // 20)     # candidates recur across pairs
    q = np.sort(rng.integers(0, n_ent, n_pairs))
    c = pool[rng.integers(0, len(pool), n_pairs)]
    cn, ca = [qn[j] for j in c], [qa[j] for j in c]
    t = time.perf_counter(); strfeatures.build(qn, qa, q, cn, ca, IDF)
    fast = n_pairs / (time.perf_counter() - t)
    t = time.perf_counter(); build_loop(qn, qa, q[:loop_pairs], cn[:loop_pairs], ca[:loop_pairs], IDF)
    slow = loop_pairs / (time.perf_counter() - t)
    print(f"vectorised {fast:,.0f} pairs/s   loop {slow:,.0f} pairs/s   ({fast / slow:.0f}x)")


if __name__ == "__main__":
    bench()
