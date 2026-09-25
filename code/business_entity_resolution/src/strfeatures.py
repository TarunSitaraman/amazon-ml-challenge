"""String-similarity features for candidate pairs.

Why these (AUDIT.md, true links vs random same-country records):
  name token Jaccard            0.635 vs 0.031
  one name a subset of other    57.5% vs 0.05%
  address shares a digit token  78%   vs 1.8%
But 39.28% of S1 entities share a normalised name with a DIFFERENT S1 entity
(chains), so every name feature is blind on those pairs. The address digit
tokens -- house, plot, suite, PIN/ZIP -- are what tells two branches of one chain
apart, and the metric weights precision 2x, so a chain false merge is the
expensive error. The digit block therefore carries three views (Jaccard, match
count, and an explicit conflict flag) rather than one; a tree model can split on
"names equal AND digits conflict" directly instead of having to discover it.

Performance. This runs over ~70M pairs on CPU, where a Python loop at ~50us/pair
is an hour. So:
  1. each RECORD is normalised and encoded once, as sorted unique int32 token ids
     (a records x vocab CSR row). Python cost scales with records (~12M), not
     pairs (~70M); a record reused by 40 pairs is encoded once;
  2. per chunk of pairs, A[q] and B[c] are C-level CSR row gathers and A[q]*B[c]
     (elementwise) is a linear merge of each pair's two sorted rows, so
     |A n B| is the product's row lengths. No per-pair Python anywhere.
Throughput assumption, MEASURED with the synthetic benchmark
(`python strfeatures.py --bench`: 200k records, 4M pairs, realistic token
counts; one core of a 4-vCPU cloud sandbox, numpy 2.4 / scipy):
  pair stage   ~1.07M pairs/s   -> ~65s for 70M pairs
  encoding     ~30k records/s   -> ~6.5 min for ~12M records (textnorm-bound)
A pure-numpy ragged-gather + searchsorted kernel measured ~190k pairs/s on the
same box (~6 min for 70M) -- memory-bandwidth bound at ~20 array passes per
feature. Re-measure on the real shard; the bench prints both rates.
Output is float32 n_pairs x k (~13 cols): 70M pairs is ~3.6GB, so call it per
country shard.
"""
import sys

import numpy as np
import scipy.sparse as sp

from textnorm import ngrams, norm, nums, toks

_CHUNK = 1_000_000      # pairs per step: row gathers of ~25 tokens/pair/side -> a few hundred MB peak


# ------------------------------------------------------------------ encoding

class _Vocab:
    """String -> dense int32 id, shared by both sides so ids are comparable."""

    def __init__(self):
        self.ids = {}

    def encode(self, sets):
        """Iterable of str-sets -> CSR (offsets int64, flat int32, ids sorted per row)."""
        ids = self.ids
        lens, flat = [], []
        for s in sets:
            row = sorted({ids.setdefault(t, len(ids)) for t in s})
            lens.append(len(row))
            flat.extend(row)
        off = np.zeros(len(lens) + 1, np.int64)
        np.cumsum(lens, out=off[1:])
        return off, np.asarray(flat, np.int32)


def _grams(n):
    # A name shorter than 4 chars has no 4-grams; treat the whole string as its
    # one gram so "3m" vs "3m" is 1.0, not an empty-set 0.
    g = ngrams(n)
    t = n.replace(" ", "")
    return g if g or not t else {t}


def _encode_side(names, addrs, vocab, gvocab, strid):
    # norm() once per field; textnorm.toks/nums/ngrams re-normalise, but norm is
    # idempotent and its fast path on already-clean ASCII is cheap.
    nn = [norm(x) for x in names]
    na = [norm(x) for x in addrs]
    return {
        "name": vocab.encode(toks(x) for x in nn),
        "gram": gvocab.encode(_grams(x) for x in nn),
        "addr": vocab.encode(toks(x) for x in na),
        "num": vocab.encode(nums(x) for x in na),
        "nid": np.fromiter((strid.setdefault(x, len(strid)) if x else -1 for x in nn),
                           np.int64, len(nn)),
        "nlen": np.fromiter((len(x) for x in nn), np.float32, len(nn)),
        "aempty": np.fromiter((not x for x in na), np.float32, len(na)),
    }


# ------------------------------------------------------------ pair kernels

def _csr(enc, n_vocab, data=None):
    """(offsets, flat) -> binary (or weighted) records x vocab CSR, indices sorted."""
    off, flat = enc
    vals = np.ones(len(flat), np.float32) if data is None else data[flat]
    m = sp.csr_matrix((vals, flat, off), shape=(len(off) - 1, n_vocab))
    m.has_sorted_indices = True      # _Vocab.encode sorts; skip scipy's re-check
    return m


def _intersect(A, B, qi, ci, Aw=None, Bw=None):
    """Per-pair |A|, |B|, |A n B| (and IDF mass of A, B, A n B when Aw/Bw given).

    A[qi] and B[ci] are C-level row gathers, and the elementwise product of two
    canonical CSR matrices is a linear merge of each pair's two sorted rows, so
    |A n B| is just the row lengths of the product. Measured ~15x faster than an
    equivalent pure-numpy ragged gather + searchsorted kernel on 4-grams, because
    it makes one pass over memory instead of ~20.
    """
    Aq, Bc = A[qi], B[ci]
    inter = np.diff(Aq.multiply(Bc).tocsr().indptr).astype(np.float32)
    out = [np.diff(Aq.indptr).astype(np.float32), np.diff(Bc.indptr).astype(np.float32), inter]
    if Aw is not None:
        Awq = Aw[qi]
        row = lambda m: np.asarray(m.sum(axis=1), np.float64).ravel()
        out += [row(Awq), row(Bw[ci]), row(Awq.multiply(Bc))]
    return out


def _div(a, b):
    """a / b with 0 where b == 0 (empty-vs-empty carries no evidence)."""
    a = np.asarray(a, np.float32)
    return np.divide(a, b, out=np.zeros_like(a), where=np.asarray(b) > 0)


def _jacc(la, lb, i):
    return _div(i, la + lb - i)


# ------------------------------------------------------------------- public

BASE_NAMES = [
    "name_jacc",          # name token Jaccard
    "name_contain",       # |A n B| / min(|A|,|B|): 1.0 iff one name's tokens ⊆ other's
    "name_gram4_jacc",    # char 4-gram Jaccard (spelling / transliteration drift)
    "name_exact",         # normalised names identical
    "name_len_ratio",     # min/max normalised length
    "addr_jacc",          # address token Jaccard
    "addr_num_jacc",      # digit-token Jaccard          -- chain discriminator
    "addr_num_match",     # count of shared digit tokens -- chain discriminator
    "addr_num_conflict",  # both have digits, none shared -- chain discriminator
    "s1_addr_empty",
    "cand_addr_empty",
]
IDF_NAMES = [
    "name_idf_shared",    # IDF mass of shared name tokens
    "name_idf_jacc",      # IDF-weighted name Jaccard ("limited", "llc" count ~0)
]


def build(s1_names, s1_addrs, cand_names, cand_addrs, q=None, c=None, idf=None,
          chunk=_CHUNK):
    """String-similarity features. Returns (float32 matrix n_pairs x k, NAMES).

    Two layouts:
      q, c None   the four sequences are pair-aligned: row i vs row i.
      q, c given  the sequences are per-record tables and pair k compares S1
                  record q[k] with candidate record c[k]. Use this in the
                  pipeline: each record is then encoded once, however many
                  pairs it is in.
    idf: optional mapping token -> idf weight (e.g. dict(zip(vocab, idf)) from
    blocking.build_index). Tokens absent from it are treated as maximally rare.
    The IDF columns are only emitted when it is given; use the returned NAMES.
    """
    vocab, gvocab, strid = _Vocab(), _Vocab(), {}
    A = _encode_side(s1_names, s1_addrs, vocab, gvocab, strid)
    B = _encode_side(cand_names, cand_addrs, vocab, gvocab, strid)

    if q is None:
        if len(s1_names) != len(cand_names):
            raise ValueError("pair-aligned mode needs equal-length sides")
        q = c = np.arange(len(s1_names), dtype=np.int64)
    q = np.asarray(q, np.int64)
    c = np.asarray(c, np.int64)

    w = None
    names = list(BASE_NAMES)
    if idf is not None:
        # dense weight per vocab id; unseen tokens get the rarest weight
        vals = np.fromiter(idf.values(), np.float64, len(idf)) if len(idf) else np.ones(1)
        w = np.full(len(vocab.ids), vals.max(), np.float64)
        for t, i in vocab.ids.items():
            v = idf.get(t)
            if v is not None:
                w[i] = v
        names += IDF_NAMES

    # records x vocab matrices, built once; every chunk only gathers rows
    nv, ng = max(len(vocab.ids), 1), max(len(gvocab.ids), 1)
    M = {side: {"name": _csr(E["name"], nv), "gram": _csr(E["gram"], ng),
                "addr": _csr(E["addr"], nv), "num": _csr(E["num"], nv),
                "name_w": None if w is None else _csr(E["name"], nv, w)}
         for side, E in (("a", A), ("b", B))}
    Ma, Mb = M["a"], M["b"]

    X = np.zeros((len(q), len(names)), np.float32)
    for s in range(0, len(q), chunk):
        qi, ci = q[s:s + chunk], c[s:s + chunk]
        cols = []

        nm = _intersect(Ma["name"], Mb["name"], qi, ci, Ma["name_w"], Mb["name_w"])
        la, lb, i = nm[:3]
        cols += [_jacc(la, lb, i), _div(i, np.minimum(la, lb))]

        cols.append(_jacc(*_intersect(Ma["gram"], Mb["gram"], qi, ci)))

        ida, idb = A["nid"][qi], B["nid"][ci]
        cols.append(((ida == idb) & (ida >= 0)).astype(np.float32))

        ra, rb = A["nlen"][qi], B["nlen"][ci]
        cols.append(_div(np.minimum(ra, rb), np.maximum(ra, rb)))

        cols.append(_jacc(*_intersect(Ma["addr"], Mb["addr"], qi, ci)))

        la, lb, i = _intersect(Ma["num"], Mb["num"], qi, ci)
        cols += [_jacc(la, lb, i), i,
                 ((la > 0) & (lb > 0) & (i == 0)).astype(np.float32)]

        cols += [A["aempty"][qi], B["aempty"][ci]]

        if w is not None:
            wa, wb, wi = nm[3:]
            cols += [wi.astype(np.float32), _div(wi, wa + wb - wi)]

        X[s:s + len(qi)] = np.column_stack(cols)
    return X, names


# ---------------------------------------------------------------- self-test

def _selftest():
    col = lambda X, names, n: X[:, names.index(n)]
    sims = ["name_jacc", "name_contain", "name_gram4_jacc", "name_exact",
            "name_len_ratio", "addr_jacc", "addr_num_jacc", "name_idf_jacc"]
    idf = {"acme": 5.0, "ltd": 0.1, "main": 2.0, "st": 0.5}

    # identical -> every similarity 1.0 (strings need a digit for the digit block)
    same = [("Acme Ltd", "12 Main St"), ("Café Rouge", "4 Rue de Rivoli 75001"),
            ("3M", "Plot 7, Sector 5"), ("राम स्टोर", "12 एमजी रोड")]
    n, a = zip(*same)
    X, names = build(n, a, n, a, idf=idf)
    for s in sims:
        assert np.allclose(col(X, names, s), 1.0), (s, col(X, names, s))
    assert np.all(col(X, names, "addr_num_conflict") == 0)
    assert np.all(col(X, names, "s1_addr_empty") == 0)

    # disjoint -> every overlap similarity 0.0 (length ratio compares lengths,
    # not content, so it is legitimately non-zero here)
    X, names = build(["Acme Ltd"], ["12 Main St"], ["Zenith Bakery"], ["99 Oak Rd"], idf=idf)
    for s in sims:
        if s == "name_len_ratio":
            continue
        assert col(X, names, s)[0] == 0.0, (s, col(X, names, s))
    assert col(X, names, "addr_num_conflict")[0] == 1.0

    # chain: same name, same street, different building. Name features must NOT
    # separate the true pair from the chain sibling; digit features must.
    s1n, s1a = ["Acme Ltd"], ["12 Main St"]
    cn, ca = ["Acme Ltd", "Acme Ltd"], ["12 Main St", "99 Main St"]
    X, names = build(s1n, s1a, cn, ca, q=[0, 0], c=[0, 1], idf=idf)
    true, chain = X[0], X[1]
    for s in names:
        if s.startswith("name_"):
            assert true[names.index(s)] == chain[names.index(s)], s
    j = names.index
    assert true[j("addr_num_jacc")] == 1.0 and chain[j("addr_num_jacc")] == 0.0
    assert true[j("addr_num_match")] == 1.0 and chain[j("addr_num_match")] == 0.0
    assert true[j("addr_num_conflict")] == 0.0 and chain[j("addr_num_conflict")] == 1.0
    assert chain[j("addr_jacc")] == 0.5          # {main, st} / {12, 99, main, st}

    # empty strings: no division by zero, no NaN, flags set
    X, names = build(["", "Acme"], ["", ""], ["", "Acme"], ["", "1 A St"])
    assert np.isfinite(X).all()
    assert col(X, names, "name_exact")[0] == 0.0          # empty == empty is not evidence
    assert col(X, names, "s1_addr_empty").tolist() == [1.0, 1.0]
    assert col(X, names, "cand_addr_empty").tolist() == [1.0, 0.0]
    assert "name_idf_jacc" not in names

    # indexed layout == pair-aligned layout
    rng = np.random.default_rng(0)
    pool_n = ["Acme Ltd", "Acme Pvt Ltd", "Zen Cafe", "", "A&B Stores 2"]
    pool_a = ["12 Main St", "", "Plot 4 Sector 9", "12 Main Street", "9 9 9"]
    q, c = rng.integers(0, 5, 200), rng.integers(0, 5, 200)
    X1, _ = build(pool_n, pool_a, pool_n, pool_a, q=q, c=c, idf=idf, chunk=37)
    X2, _ = build([pool_n[i] for i in q], [pool_a[i] for i in q],
                  [pool_n[i] for i in c], [pool_a[i] for i in c], idf=idf)
    assert np.array_equal(X1, X2)

    # kernel vs plain Python sets straight from textnorm
    from textnorm import toks as T
    j = names.index
    for k in range(200):
        a, b = T(pool_n[q[k]]), T(pool_n[c[k]])
        ref = len(a & b) / len(a | b) if a | b else 0.0
        assert np.isclose(X1[k, j("name_jacc")], ref), (k, ref)
        a, b = nums(pool_a[q[k]]), nums(pool_a[c[k]])
        assert X1[k, j("addr_num_match")] == len(a & b)
    print("strfeatures self-test OK")


def _bench(n_rec=200_000, n_pairs=2_000_000):
    """Synthetic throughput check; realistic token counts, not real data."""
    import time
    rng = np.random.default_rng(1)
    # letters only: a digit inside a synthetic word would make it a "number" token
    words = np.array(["".join(w) for w in rng.choice(list("abcdefghijklmnopqrstuvwxyz"),
                                                        (50_000, 6))])
    mk = lambda k: " ".join(rng.choice(words, k))
    names = [mk(rng.integers(1, 5)) for _ in range(n_rec)]
    addrs = [mk(rng.integers(3, 8)) + f" {rng.integers(1, 999)} {rng.integers(10000, 99999)}"
             for _ in range(n_rec)]
    q = np.sort(rng.integers(0, n_rec, n_pairs))
    c = rng.integers(0, n_rec, n_pairs)
    t0 = time.perf_counter()
    build(names, addrs, names, addrs, q=q[:1], c=c[:1])
    t1 = time.perf_counter()
    build(names, addrs, names, addrs, q=q, c=c)
    t2 = time.perf_counter()
    enc = t1 - t0
    print(f"encode: {2 * n_rec / enc:,.0f} records/s   "
          f"pairs: {n_pairs / (t2 - t1 - enc):,.0f} pairs/s")


if __name__ == "__main__":
    _selftest()
    if "--bench" in sys.argv:
        _bench()
