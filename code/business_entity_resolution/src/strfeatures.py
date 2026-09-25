"""String-similarity features for candidate pairs.

The blocking-geometry features tell the matcher how a pair was *retrieved*, not
whether the two records look alike. These supply the latter.

Weighting note, from the audit: 39.28% of S1 entities share a normalised name
with a DIFFERENT S1 entity (chains, franchises -- S1 is deduplicated, so they
are provably distinct businesses). For that population the name is useless and
the address digits are the only discriminator: true pairs share an address digit
token 78% of the time against 1.8% for random pairs. Hence the digit features
are kept separate and explicit rather than folded into a general address score.

Performance. predict.py and train_eval.py pass ~70M pairs per run, and the
original per-pair Python loop measured ~11k pairs/s (~1.8 hours). This version
gives the same output for every pair, but:
  1. each distinct S1 entity and each distinct candidate (name, addr) record is
     tokenised once, as a sorted row of int32 token ids in a scipy CSR matrix;
  2. per chunk of pairs, Q[q_idx] and C[c] are C-level row gathers, and their
     elementwise product is a merge of each pair's two sorted rows, so
     |A n B| is just the product's row lengths. No Python runs per pair.
Measured ~17x faster than the loop (~190k vs ~11k pairs/s on one core); see
tests/test_strfeatures_equivalence.py, which also checks equality value for value.
"""
import numpy as np
import scipy.sparse as sp

from textnorm import ngrams, norm

NAMES = ["nm_jac", "nm_cont", "nm_4gram", "nm_exact", "nm_lenratio",
         "ad_jac", "ad_cont", "dg_jac", "dg_shared", "dg_both_have",
         "ad_empty_q", "ad_empty_c", "nm_rare_shared"]

_CHUNK = 1_000_000      # pairs per step; bounds the gathered rows to a few hundred MB


class _Vocab:
    """str -> dense id, shared by both sides so ids are comparable."""

    def __init__(self):
        self.ids = {}

    def rows(self, sets):
        """Iterable of str-sets -> (offsets, flat ids), ids sorted within each row."""
        ids = self.ids
        lens, flat = [], []
        for s in sets:
            row = sorted({ids.setdefault(t, len(ids)) for t in s})
            lens.append(len(row))
            flat.extend(row)
        off = np.zeros(len(lens) + 1, np.int64)
        np.cumsum(lens, out=off[1:])
        return off, np.asarray(flat, np.int32)


def _digits(s):
    return {x for x in s if x[:1].isdigit()}


def _mat(rows, width):
    off, flat = rows
    m = sp.csr_matrix((np.ones(len(flat), np.float64), flat, off),
                      shape=(len(off) - 1, max(width, 1)))
    m.has_sorted_indices = True
    return m


def _jac(la, lb, i):
    u = la + lb - i
    return np.divide(i, u, out=np.zeros_like(i), where=u > 0)


def _cont(la, lb, i):
    m = np.minimum(la, lb)
    return np.divide(i, m, out=np.zeros_like(i), where=m > 0)


def _overlap(A, B, qi, ci):
    """Per pair: |A|, |B|, |A n B| as float64, plus the product matrix."""
    a, b = A[qi], B[ci]
    both = a.multiply(b).tocsr()
    return (np.diff(a.indptr).astype(np.float64), np.diff(b.indptr).astype(np.float64),
            np.diff(both.indptr).astype(np.float64), both)


def build(q_names, q_addrs, q_idx, c_names, c_addrs, idf_lut=None, chunk=_CHUNK):
    """q_*: per-entity normalised text. q_idx: entity index per pair.
    c_*: per-PAIR normalised candidate text (already gathered)."""
    n = len(c_names)
    out = np.zeros((n, len(NAMES)), np.float32)
    if n == 0:
        return out
    q_idx = np.asarray(q_idx, np.int64)

    # candidates repeat across pairs; tokenise each distinct (name, addr) once
    ukey, c = {}, np.empty(n, np.int64)
    for i, k in enumerate(zip(c_names, c_addrs)):
        c[i] = ukey.setdefault(k, len(ukey))
    cn_u = [k[0] for k in ukey]
    ca_u = [k[1] for k in ukey]

    tv, gv, sid = _Vocab(), _Vocab(), {}
    qn_t = [set(t.split()) for t in q_names]
    qa_t = [set(t.split()) for t in q_addrs]
    cn_t = [set(t.split()) for t in cn_u]
    ca_t = [set(t.split()) for t in ca_u]
    rows = {
        "qn": tv.rows(qn_t), "qa": tv.rows(qa_t),
        "qd": tv.rows(_digits(x) | _digits(y) for x, y in zip(qn_t, qa_t)),
        "qg": gv.rows(ngrams(t, 4) for t in q_names),
        "cn": tv.rows(cn_t), "ca": tv.rows(ca_t),
        "cd": tv.rows(_digits(x) | _digits(y) for x, y in zip(cn_t, ca_t)),
        "cg": gv.rows(ngrams(t, 4) for t in cn_u),
    }
    M = {k: _mat(r, len(gv.ids) if k[1] == "g" else len(tv.ids)) for k, r in rows.items()}
    # nm_exact compares strings; an empty candidate name never matches
    q_sid = np.fromiter((sid.setdefault(t, len(sid)) for t in q_names), np.int64, len(q_names))
    c_sid = np.fromiter((sid.setdefault(t, len(sid)) if t else -1 for t in cn_u),
                        np.int64, len(cn_u))
    q_len = np.fromiter(map(len, q_names), np.float64, len(q_names))
    c_len = np.fromiter(map(len, cn_u), np.float64, len(cn_u))
    w = None
    if idf_lut is not None:
        w = np.zeros(max(len(tv.ids), 1), np.float64)
        for t, j in tv.ids.items():
            w[j] = idf_lut.get(t, 0.0)

    for s in range(0, n, chunk):
        qi, ci = q_idx[s:s + chunk], c[s:s + chunk]
        X = out[s:s + len(qi)]
        la, lb, i, both = _overlap(M["qn"], M["cn"], qi, ci)
        X[:, 0], X[:, 1] = _jac(la, lb, i), _cont(la, lb, i)
        X[:, 2] = _jac(*_overlap(M["qg"], M["cg"], qi, ci)[:3])
        X[:, 3] = q_sid[qi] == c_sid[ci]
        lq, lc = q_len[qi], c_len[ci]
        hi = np.maximum(lq, lc)
        X[:, 4] = np.divide(np.minimum(lq, lc), hi, out=np.zeros_like(hi), where=hi > 0)
        la, lb, i, _ = _overlap(M["qa"], M["ca"], qi, ci)
        X[:, 5], X[:, 6] = _jac(la, lb, i), _cont(la, lb, i)
        X[:, 10], X[:, 11] = la == 0, lb == 0
        la, lb, i, _ = _overlap(M["qd"], M["cd"], qi, ci)
        X[:, 7], X[:, 8], X[:, 9] = _jac(la, lb, i), i, (la > 0) & (lb > 0)
        if w is not None:
            # max IDF over the shared name tokens, 0 when none are shared
            has = np.flatnonzero(np.diff(both.indptr))
            if len(has):
                X[has, 12] = np.maximum.reduceat(w[both.indices], both.indptr[has])
    return out


def _self_test():
    qn = [norm("Acme Ltd"), norm("Acme Ltd")]
    qa = [norm("12 Main St, Springfield"), norm("12 Main St, Springfield")]
    cn = [norm("Acme Ltd"), norm("Acme Ltd")]
    ca = [norm("12 Main St, Springfield"), norm("99 Main St, Springfield")]
    X = build(qn, qa, np.array([0, 1]), cn, ca)
    col = {k: i for i, k in enumerate(NAMES)}
    same, chain = X[0], X[1]
    assert same[col["nm_exact"]] == 1.0 and same[col["nm_jac"]] == 1.0
    assert chain[col["nm_exact"]] == 1.0 and chain[col["nm_jac"]] == 1.0, \
        "same chain must be indistinguishable by NAME"
    assert chain[col["dg_jac"]] < same[col["dg_jac"]], \
        "digit features must separate two branches of one chain"
    disjoint = build([norm("Acme Ltd")], [norm("12 Main St")], np.array([0]),
                     [norm("Zeta Foods")], [norm("77 Oak Rd")])[0]
    assert disjoint[col["nm_jac"]] == 0.0 and disjoint[col["dg_jac"]] == 0.0
    empty = build([norm("Acme")], [""], np.array([0]), [norm("Acme")], [""])[0]
    assert empty[col["ad_empty_q"]] == 1.0 and empty[col["ad_empty_c"]] == 1.0
    print("PASS -- chain separated by digits, not by name")


if __name__ == "__main__":
    _self_test()
