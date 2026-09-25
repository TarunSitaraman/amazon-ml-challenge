"""String-similarity features for candidate pairs.

The blocking-geometry features tell the matcher how a pair was *retrieved*, not
whether the two records look alike. These supply the latter.

Weighting note, from the audit: 39.28% of S1 entities share a normalised name
with a DIFFERENT S1 entity (chains, franchises -- S1 is deduplicated, so they
are provably distinct businesses). For that population the name is useless and
the address digits are the only discriminator: true pairs share an address digit
token 78% of the time against 1.8% for random pairs. Hence the digit features
are kept separate and explicit rather than folded into a general address score.

Layout: a corpus record sits in many entities' candidate lists, so its token,
digit and 4-gram sets are built once per country by precompute_records() and
build() only gathers ranges of them by candidate index. Every set is a CSR pair
of flat int32 arrays (offsets, sorted unique ids): 10M Python sets would not fit
in memory. build_reference() is the original per-pair loop, kept as the oracle
that build() must match exactly (see _self_test and the equivalence test).
"""
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from textnorm import ngrams

NAMES = ["nm_jac", "nm_cont", "nm_4gram", "nm_exact", "nm_lenratio",
         "ad_jac", "ad_cont", "dg_jac", "dg_shared", "dg_both_have",
         "ad_empty_q", "ad_empty_c", "nm_rare_shared"]

# norm() output is lower-case [0-9a-z] tokens joined by single spaces. The fast
# path depends on that: it splits on " " (== str.split() on such text), compares
# names by token sequence, and codes 4-grams in base 36. Anything else raises.
_NOT_CANONICAL = r"[^0-9a-z ]|^ | $|  "

# 4-gram code: base 36 over [0-9a-z], so 36**4 = 1,679,616 codes fit in int32.
# This is the "hash" of a 4-gram, and on norm() output it is injective: there
# are no collisions at all, which is why nm_4gram matches the string sets bit
# for bit. (A lossy int32 hash would also have been fine for a similarity
# feature, since a rare collision only nudges one Jaccard, but it could not
# have been proved identical to the old code.)
_B36 = np.zeros(256, np.int32)
_B36[np.frombuffer(b"0123456789", np.uint8)] = np.arange(10)
_B36[np.frombuffer(b"abcdefghijklmnopqrstuvwxyz", np.uint8)] = np.arange(10, 36)
_GRAM_SPACE = 36 ** 4

_CHUNK = 250_000            # records per precompute step (bounds transient RAM)
_PAIR_CHUNK = 50_000        # pairs per build step (cache-sized; also bounds RAM)


class Records:
    """Per-record token, digit and 4-gram sets for one corpus, CSR layout.

    For each set kind X: X_off (int32, n+1) and X_ids (int32, sorted unique
    within each record). name_seq is the name's token ids in order, for the
    exact-name test; name_len the name's length in characters.
    """
    __slots__ = ("n", "vocab", "is_digit", "name_len",
                 "name_off", "name_ids", "addr_off", "addr_ids",
                 "dig_off", "dig_ids", "gram_off", "gram_ids",
                 "seq_off", "seq_ids", "_idf")

    def nbytes(self):
        """Bytes held, vocabulary included."""
        arrays = [self.is_digit, self.name_len, self.name_off, self.name_ids,
                  self.addr_off, self.addr_ids, self.dig_off, self.dig_ids,
                  self.gram_off, self.gram_ids, self.seq_off, self.seq_ids]
        return sum(a.nbytes for a in arrays) + self.vocab.nbytes


def _check(arr, what):
    bad = pc.sum(pc.match_substring_regex(arr, _NOT_CANONICAL)).as_py()
    if bad:
        raise ValueError(f"{bad} {what} strings are not norm() output")


def _csr(rec, ids, n, space):
    """(record, id) pairs -> offsets, ids sorted and unique within each record."""
    key = np.sort(rec.astype(np.int64) * space + ids)
    key = key[np.r_[True, key[1:] != key[:-1]]] if len(key) else key
    rec, ids = np.divmod(key, space)
    return np.bincount(rec, minlength=n), ids.astype(np.int32)


def _split(arr):
    """-> (record index, token string array) with empty tokens dropped."""
    lists = pc.split_pattern(arr, " ")
    counts = pc.list_value_length(lists).to_numpy(zero_copy_only=False)
    flat = pc.list_flatten(lists)
    rec = np.repeat(np.arange(len(arr), dtype=np.int64), counts)
    keep = pc.greater(pc.utf8_length(flat), 0).to_numpy(zero_copy_only=False)
    return rec[keep], flat.filter(pa.array(keep))


def _grams(arr):
    """-> (record index, 4-gram code) for every 4-gram of each space-free name."""
    s = pc.replace_substring(arr, " ", "").cast(pa.large_string())
    off = np.frombuffer(s.buffers()[1], np.int64)[s.offset:s.offset + len(s) + 1]
    data = np.frombuffer(s.buffers()[2], np.uint8) if s.buffers()[2] else \
        np.zeros(0, np.uint8)
    ng = np.maximum(np.diff(off) - 3, 0)
    rec = np.repeat(np.arange(len(s), dtype=np.int64), ng)
    pos = np.repeat(off[:-1] - np.r_[0, np.cumsum(ng)[:-1]], ng) + \
        np.arange(ng.sum(), dtype=np.int64)
    d = _B36[data]
    code = ((d[pos] * 36 + d[pos + 1]) * 36 + d[pos + 2]) * 36 + d[pos + 3]
    return rec, code


def _offsets(counts):
    off = np.zeros(len(counts) + 1, np.int64)
    np.cumsum(counts, out=off[1:])
    if off[-1] > np.iinfo(np.int32).max:
        raise OverflowError("CSR too large for int32 offsets")
    return off.astype(np.int32)


def precompute_records(names, addrs):
    """names, addrs: normalised text per record (list of str or Arrow array).
    -> Records, the per-record half of every string feature, built once."""
    names = pa.array(names, pa.string()) if not isinstance(names, pa.Array) else names
    addrs = pa.array(addrs, pa.string()) if not isinstance(addrs, pa.Array) else addrs
    names, addrs = names.fill_null(""), addrs.fill_null("")
    n = len(names)
    if len(addrs) != n:
        raise ValueError("names and addrs differ in length")
    _check(names, "name")
    _check(addrs, "address")

    # One vocabulary for names and addresses: digit tokens come from both.
    uniq = [pc.unique(_split(a.slice(lo, _CHUNK))[1])
            for a in (names, addrs) for lo in range(0, n, _CHUNK)]
    vocab = pc.unique(pa.chunked_array(uniq, pa.string()).combine_chunks()) \
        if uniq else pa.array([], pa.string())
    V = max(len(vocab), 1)
    is_digit = pc.match_substring_regex(vocab, "^[0-9]").to_numpy(
        zero_copy_only=False).astype(bool)

    parts = {k: ([], []) for k in ("name", "addr", "dig", "gram", "seq")}

    def put(kind, counts, ids):
        parts[kind][0].append(counts)
        parts[kind][1].append(ids)

    for lo in range(0, n, _CHUNK):
        nm, ad = names.slice(lo, _CHUNK), addrs.slice(lo, _CHUNK)
        m = len(nm)
        rn, tn = _split(nm)
        ra, ta = _split(ad)
        idn = pc.index_in(tn, value_set=vocab).to_numpy(zero_copy_only=False)
        ida = pc.index_in(ta, value_set=vocab).to_numpy(zero_copy_only=False)
        idn, ida = idn.astype(np.int32), ida.astype(np.int32)
        put("seq", np.bincount(rn, minlength=m), idn)
        put("name", *_csr(rn, idn, m, V))
        put("addr", *_csr(ra, ida, m, V))
        dn, da = is_digit[idn], is_digit[ida]
        put("dig", *_csr(np.r_[rn[dn], ra[da]], np.r_[idn[dn], ida[da]], m, V))
        put("gram", *_csr(*_grams(nm), m, _GRAM_SPACE))

    r = Records()
    r.n, r.vocab, r.is_digit, r._idf = n, vocab, is_digit, None
    r.name_len = pc.utf8_length(names).to_numpy(zero_copy_only=False).astype(np.int32)
    for kind, (counts, ids) in parts.items():
        c = np.concatenate(counts) if counts else np.zeros(0, np.int64)
        setattr(r, kind + "_off", _offsets(c))
        setattr(r, kind + "_ids", np.concatenate(ids).astype(np.int32)
                if ids else np.zeros(0, np.int32))
    return r


def _idf_array(recs, idf_lut):
    """idf per vocabulary id, cached on the Records for this idf_lut object."""
    if recs._idf is None or recs._idf[0] is not idf_lut:
        vals = np.fromiter((idf_lut.get(t, 0.0) for t in recs.vocab.to_pylist()),
                           np.float64, len(recs.vocab))
        recs._idf = (idf_lut, vals)
    return recs._idf[1]


def _gather(off, ids, j):
    """Concatenated ids of records j -> (values, segment lengths)."""
    lo = off[j].astype(np.int64)
    ln = off[j + 1].astype(np.int64) - lo
    tot = int(ln.sum())
    start = np.cumsum(ln) - ln
    idx = np.repeat(lo - start, ln) + np.arange(tot, dtype=np.int64)
    return ids[idx], ln


def _seg_sum(x, ln):
    cs = np.zeros(len(x) + 1, np.int64)
    np.cumsum(x, out=cs[1:])
    end = np.cumsum(ln)
    return cs[end] - cs[end - ln]


def _intersect(qkeys, space, e, off, ids, j):
    """|Q_e & C_j| per pair by probing each candidate id in the entity's sorted
    keys. -> (intersection size, candidate set size, hit mask, candidate ids)."""
    vals, ln = _gather(off, ids, j)
    key = np.repeat(e.astype(np.int64), ln) * space + vals
    pos = np.searchsorted(qkeys, key)
    hit = np.zeros(len(key), bool)
    if len(qkeys):
        np.minimum(pos, len(qkeys) - 1, out=pos)
        hit = qkeys[pos] == key
    return _seg_sum(hit, ln), ln, hit, vals


def _qkeys(qr, kind, remap, space):
    """Query-side keys e*space + global id, sorted, unknown tokens dropped."""
    off, ids = getattr(qr, kind + "_off"), getattr(qr, kind + "_ids")
    e = np.repeat(np.arange(qr.n, dtype=np.int64), np.diff(off))
    g = remap[ids] if remap is not None else ids.astype(np.int64)
    ok = g >= 0
    return np.sort(e[ok] * space + g[ok]), np.diff(off).astype(np.int64)


def _div(a, b):
    out = np.zeros(len(a), np.float64)
    np.divide(a, b, out=out, where=b > 0)
    return out


def build(recs, q_names, q_addrs, q_idx, c, idf_lut=None):
    """recs: precompute_records() of the corpus. q_*: per-entity normalised text.
    q_idx: entity index per pair. c: corpus record index per pair.
    Output equals build_reference() on the gathered candidate text."""
    q_idx = np.asarray(q_idx, np.int64)
    c = np.asarray(c, np.int64)
    n = len(c)
    out = np.zeros((n, len(NAMES)), np.float32)
    if n == 0:
        return out
    qr = precompute_records(q_names, q_addrs)
    # query vocabulary -> corpus vocabulary; -1 for a token the corpus lacks,
    # which counts toward the query's set size but can never be shared
    remap = pc.fill_null(pc.index_in(qr.vocab, value_set=recs.vocab), -1) \
        .to_numpy(zero_copy_only=False).astype(np.int64)
    V = max(len(recs.vocab), 1)
    kn, qn = _qkeys(qr, "name", remap, V)
    ka, qa = _qkeys(qr, "addr", remap, V)
    kd, qd = _qkeys(qr, "dig", remap, V)
    kg, qg = _qkeys(qr, "gram", None, _GRAM_SPACE)
    q_seq = remap[qr.seq_ids]
    q_seqn = np.diff(qr.seq_off).astype(np.int64)
    idf = _idf_array(recs, idf_lut) if idf_lut is not None else None

    for lo in range(0, n, _PAIR_CHUNK):
        e, j = q_idx[lo:lo + _PAIR_CHUNK], c[lo:lo + _PAIR_CHUNK]
        o = out[lo:lo + _PAIR_CHUNK]

        i_n, c_n, hit, vals = _intersect(kn, V, e, recs.name_off, recs.name_ids, j)
        a_n = qn[e]
        o[:, 0] = _div(i_n, a_n + c_n - i_n)
        o[:, 1] = _div(i_n, np.minimum(a_n, c_n))
        if idf is not None:
            v = np.where(hit, idf[vals], -np.inf)
            has = i_n > 0
            if has.any():
                starts = (np.cumsum(c_n) - c_n)[has]
                m = np.zeros(len(e), np.float64)
                m[has] = np.maximum.reduceat(v, starts)
                o[:, 12] = m

        i_g, c_g, _, _ = _intersect(kg, _GRAM_SPACE, e, recs.gram_off, recs.gram_ids, j)
        a_g = qg[e]
        o[:, 2] = _div(i_g, a_g + c_g - i_g)

        lq, lc = qr.name_len[e].astype(np.int64), recs.name_len[j].astype(np.int64)
        o[:, 4] = _div(np.minimum(lq, lc), np.maximum(lq, lc))

        # Exact name: on norm() output, equal strings <=> equal token sequences.
        seqn = np.diff(recs.seq_off)[j].astype(np.int64)
        cand = np.flatnonzero((lq == lc) & (lc > 0) & (q_seqn[e] == seqn))
        if len(cand):
            cv, ln = _gather(recs.seq_off, recs.seq_ids, j[cand])
            qv, _ = _gather(qr.seq_off, q_seq, e[cand])
            o[cand, 3] = (_seg_sum(cv != qv, ln) == 0)

        i_a, c_a, _, _ = _intersect(ka, V, e, recs.addr_off, recs.addr_ids, j)
        a_a = qa[e]
        o[:, 5] = _div(i_a, a_a + c_a - i_a)
        o[:, 6] = _div(i_a, np.minimum(a_a, c_a))

        i_d, c_d, _, _ = _intersect(kd, V, e, recs.dig_off, recs.dig_ids, j)
        a_d = qd[e]
        o[:, 7] = _div(i_d, a_d + c_d - i_d)
        o[:, 8] = i_d
        o[:, 9] = (a_d > 0) & (c_d > 0)
        o[:, 10] = a_a == 0
        o[:, 11] = c_a == 0
    return out


# ---- reference implementation (the original per-pair loop) ------------------

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


def build_reference(q_names, q_addrs, q_idx, c_names, c_addrs, idf_lut=None):
    """The original per-pair loop. c_*: per-PAIR candidate text (gathered).
    Slow; kept only as the oracle build() is tested against."""
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


def _self_test():
    from textnorm import norm

    def one(qn, qa, cn, ca):
        cn, ca = [norm(x) for x in cn], [norm(x) for x in ca]
        recs = precompute_records(cn, ca)
        return build(recs, [norm(x) for x in qn], [norm(x) for x in qa],
                     np.zeros(len(cn), np.int64), np.arange(len(cn)))

    X = one(["Acme Ltd"], ["12 Main St, Springfield"], ["Acme Ltd", "Acme Ltd"],
            ["12 Main St, Springfield", "99 Main St, Springfield"])
    col = {k: i for i, k in enumerate(NAMES)}
    same, chain = X[0], X[1]
    assert same[col["nm_exact"]] == 1.0 and same[col["nm_jac"]] == 1.0
    assert chain[col["nm_exact"]] == 1.0 and chain[col["nm_jac"]] == 1.0, \
        "same chain must be indistinguishable by NAME"
    assert chain[col["dg_jac"]] < same[col["dg_jac"]], \
        "digit features must separate two branches of one chain"
    disjoint = one(["Acme Ltd"], ["12 Main St"], ["Zeta Foods"], ["77 Oak Rd"])[0]
    assert disjoint[col["nm_jac"]] == 0.0 and disjoint[col["dg_jac"]] == 0.0
    empty = one(["Acme"], [""], ["Acme"], [""])[0]
    assert empty[col["ad_empty_q"]] == 1.0 and empty[col["ad_empty_c"]] == 1.0
    print("PASS -- chain separated by digits, not by name")


if __name__ == "__main__":
    _self_test()
