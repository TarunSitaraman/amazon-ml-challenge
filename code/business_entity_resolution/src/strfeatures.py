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

Corruption grammar (GRAMMAR_NAMES): mine_corruption.py writes the generator's
abbreviation, region-code and forbidden-swap tables to a JSON. Those string
tables are turned into integer token ids once per country (against the
corpus vocabulary) and per build() call (against the query vocabulary), so the
per-pair work is the same sorted-key probing as the overlap features. An
absent grammar file leaves those four columns at 0.
"""
import hashlib
import json
import os
import pathlib
import re
import sys

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from profiling import stage
from textnorm import ngrams

GRAMMAR_NAMES = ["abbrev_matched_tokens", "abbrev_adjusted_jaccard",
                 "forbidden_hits", "region_code_matched"]
NAMES = ["nm_jac", "nm_cont", "nm_4gram", "nm_exact", "nm_lenratio",
         "ad_jac", "ad_cont", "dg_jac", "dg_shared", "dg_both_have",
         "ad_empty_q", "ad_empty_c", "nm_rare_shared"] + GRAMMAR_NAMES

# Where mine_corruption.py writes by default, relative to the working directory
# (code/business_entity_resolution). CORRUPTION_GRAMMAR overrides it.
GRAMMAR_PATH = os.environ.get("CORRUPTION_GRAMMAR", "data/corruption_grammar.json")

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
                 "seq_off", "seq_ids", "_idf", "_gram")

    def nbytes(self):
        """Bytes held, vocabulary included."""
        arrays = [self.is_digit, self.name_len, self.name_off, self.name_ids,
                  self.addr_off, self.addr_ids, self.dig_off, self.dig_ids,
                  self.gram_off, self.gram_ids, self.seq_off, self.seq_ids]
        return sum(a.nbytes for a in arrays) + self.vocab.nbytes


def _as_array(x):
    if isinstance(x, pa.ChunkedArray):
        x = x.combine_chunks()
    return x.cast(pa.string()) if isinstance(x, pa.Array) else pa.array(x, pa.string())


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
    names, addrs = _as_array(names), _as_array(addrs)
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
    r.n, r.vocab, r.is_digit, r._idf, r._gram = n, vocab, is_digit, None, None
    r.name_len = pc.utf8_length(names).to_numpy(zero_copy_only=False).astype(np.int32)
    for kind, (counts, ids) in parts.items():
        c = np.concatenate(counts) if counts else np.zeros(0, np.int64)
        setattr(r, kind + "_off", _offsets(c))
        setattr(r, kind + "_ids", np.concatenate(ids).astype(np.int32)
                if ids else np.zeros(0, np.int32))
    return r


def _idf_array(recs, idf_lut):
    """idf per vocabulary id, cached on the Records for this idf_lut object.
    Keyed by identity: pass a new dict, not a mutated one, to change idf."""
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


# ---- corruption grammar ------------------------------------------------------
#
# Every table entry is an equivalence between S1-side tokens (the query) and
# S2/S3-side tokens (the candidate): ("corporation",) -> ("corp",), or a region
# acronym ("uttar", "pradesh") -> ("up",). An entry fires on a pair when all its
# query tokens are in the query but NOT in the candidate, and all its candidate
# tokens are in the candidate but NOT in the query: it explains tokens the exact
# overlap features count as unmatched, and never a token both sides share.
#
# A forbidden entry is the exception: it fires only on tokens NO other entry
# explains. The forbidden table is a cross product of variable tokens, so
# "road -> hr" is in it; without that rule "12 road haryana" vs "12 rd hr" would
# count two legitimate abbreviations as evidence against the match. _COVER
# entries (address abbreviations that are not region codes, and acronyms) take
# part only in that rule.

_ABBREV, _FORBID, _REGION, _COVER = 0, 1, 2, 3
_TOKEN = re.compile(r"^[0-9a-z]+$")


class Grammar:
    """The string tables of a corruption_grammar.json, per field: a list of
    (query tokens, candidate tokens, kind, weight). kind is _ABBREV (weight
    P(s2 | s1)), _FORBID, _REGION or _COVER (weight 1)."""

    def __init__(self, tables, sha256=None):
        self.tables = tables
        self.sha256 = sha256

    @classmethod
    def from_json(cls, obj, sha256=None):
        tables = {"name": [], "address": []}
        seen = set()

        def add(field, q, c, kind, w):
            q = tuple(dict.fromkeys(q))
            c = tuple(dict.fromkeys(c))
            # tokens are norm() output; anything else could never match
            if not q or not c or not all(_TOKEN.match(t) for t in q + c):
                return
            if set(q) & set(c) or (field, q, c) in seen:
                return
            seen.add((field, q, c))
            tables[field].append((q, c, kind, float(np.float32(w))))

        def acronym(field, e, kind):
            if e.get("direction") == "contract":     # S1 long -> S2 initials
                add(field, e["long"].split(), (e["short"],), kind, 1.0)
            elif e.get("direction") == "expand":     # S1 initials -> S2 long
                add(field, (e["short"],), e["long"].split(), kind, 1.0)

        ab, acr = obj.get("abbreviations", {}), obj.get("acronyms", {})
        for e in ab.get("name", []):
            add("name", (e["s1"],), (e["s2"],), _ABBREV, e["p_s2_given_s1"])
        # region codes first: an address abbreviation they already hold is
        # skipped as a duplicate below, so it stays _REGION
        for e in obj.get("region_codes", {}).get("entries", []):
            if "s1" in e:                       # a single-token abbreviation
                add("address", (e["s1"],), (e["s2"],), _REGION, 1.0)
            else:
                acronym("address", e, _REGION)
        for e in ab.get("address", []):
            add("address", (e["s1"],), (e["s2"],), _COVER, 1.0)
        for field in ("name", "address"):
            for e in acr.get(field, []):
                acronym(field, e, _COVER)
        for field in ("name", "address"):
            for e in obj.get("forbidden", {}).get(field, []):
                add(field, (e["s1"],), (e["s2"],), _FORBID, 1.0)
        return cls(tables, sha256)

    def __len__(self):
        return sum(len(v) for v in self.tables.values())


_LOADED = {}


def load_grammar(path=None, verbose=True):
    """-> Grammar from mine_corruption.py's JSON, read once per path, or None
    when the file is absent or unreadable (the grammar features are then 0)."""
    path = pathlib.Path(path or GRAMMAR_PATH)
    key = str(path.resolve())
    if key not in _LOADED:
        g = None
        if path.is_file():
            try:
                raw = path.read_bytes()
                g = Grammar.from_json(json.loads(raw), hashlib.sha256(raw).hexdigest())
            except (OSError, ValueError, KeyError, TypeError, AttributeError) as ex:
                print(f"WARNING: {path} unreadable ({ex}); grammar features are 0",
                      file=sys.stderr, flush=True)
        elif verbose:
            print(f"note: no corruption grammar at {path}; grammar features are 0",
                  file=sys.stderr, flush=True)
        _LOADED[key] = g
    return _LOADED[key]


class _Table:
    """One field's entries as flat integer arrays, bound to one corpus vocab.
    Entries whose candidate tokens are not all in the corpus can never fire and
    are dropped here.

    Most entries are one token to one token. Those are also indexed as a key
    s_q * len(s_cids) + s_c over their own small vocabularies (s_qstr, s_cids),
    so a pair looks them up token against token, however many entries a token
    has: the forbidden table gives a common token hundreds. The few multi-token
    entries (acronyms) are expanded per entity instead (_activate)."""
    __slots__ = ("kind", "w", "nq", "nc", "q_off", "q_str", "q_cid",
                 "c_off", "c_ids", "multi", "s_qstr", "s_cids", "s_keys", "s_ent",
                 "is_sc", "is_mc")


def _ranges(off, r):
    """Flat indices of CSR segments r. -> (indices, segment lengths)."""
    lo = off[r]
    ln = off[r + 1] - lo
    return np.repeat(lo - (np.cumsum(ln) - ln), ln) + np.arange(ln.sum()), ln


def _compile(entries, vocab):
    def ids(strs):
        a = pa.array(strs, pa.string())
        return pc.fill_null(pc.index_in(a, value_set=vocab), -1) \
            .to_numpy(zero_copy_only=False).astype(np.int64)

    c_all = ids([t for _, c, _, _ in entries for t in c])
    nc = np.array([len(c) for _, c, _, _ in entries], np.int64)
    c_ent = np.repeat(np.arange(len(entries)), nc)
    keep = np.bincount(c_ent[c_all < 0], minlength=len(entries)) == 0
    entries = [x for x, k in zip(entries, keep) if k]
    t = _Table()
    t.kind = np.array([x[2] for x in entries], np.int8)
    t.w = np.array([x[3] for x in entries], np.float64)
    t.nq = np.array([len(x[0]) for x in entries], np.int64)
    t.nc = nc[keep]
    t.q_off = _offsets(t.nq).astype(np.int64)
    t.c_off = _offsets(t.nc).astype(np.int64)
    t.q_str = pa.array([s for x in entries for s in x[0]], pa.string())
    t.q_cid = ids(t.q_str.to_pylist()) if len(t.q_str) else np.zeros(0, np.int64)
    t.c_ids = c_all[np.repeat(keep, nc)]
    t.multi = (t.nq > 1) | (t.nc > 1)
    one = np.flatnonzero(~t.multi)
    qs = np.array(t.q_str.to_pylist(), object)[t.q_off[one]] if len(one) else \
        np.zeros(0, object)
    uq, s_q = np.unique(qs.astype(str), return_inverse=True)
    t.s_qstr = pa.array(uq.tolist(), pa.string())
    t.s_cids, s_c = np.unique(t.c_ids[t.c_off[one]], return_inverse=True)
    key = s_q.astype(np.int64) * max(len(t.s_cids), 1) + s_c
    o = np.argsort(key)
    t.s_keys, t.s_ent = key[o], one[o]
    # candidate-side token masks over the corpus vocabulary (1 byte per token):
    # most candidate tokens are in no entry, and a mask drops them in O(1)
    t.is_sc = np.zeros(max(len(vocab), 1), bool)
    t.is_sc[t.s_cids] = True
    t.is_mc = np.zeros(max(len(vocab), 1), bool)
    ci, _ = _ranges(t.c_off, np.flatnonzero(t.multi))
    t.is_mc[t.c_ids[ci]] = True
    return t


def _grammar_tables(recs, grammar):
    """Per-country compile, cached on the Records for this Grammar object."""
    if recs._gram is None or recs._gram[0] is not grammar:
        with stage("grammar compile", n=len(grammar), unit="entries"):
            recs._gram = (grammar, {f: _compile(v, recs.vocab)
                                    for f, v in grammar.tables.items()})
    return recs._gram[1]


def _activate(t, qr, kind, V):
    """Multi-token entries whose query tokens are all in entity e's `kind`
    set, expanded to sorted probe keys e*V + candidate token id. -> (keys,
    entry per key, query-local id of every flat query token, -1 if the query
    lacks it; that last covers all entries, single-token ones included)."""
    E = len(t.kind)
    q_lid = pc.fill_null(pc.index_in(t.q_str, value_set=qr.vocab), -1) \
        .to_numpy(zero_copy_only=False).astype(np.int64) if E else \
        np.zeros(0, np.int64)
    empty = np.zeros(0, np.int64)
    if not E or not qr.n:
        return empty, empty, q_lid
    ent = np.repeat(np.arange(E), t.nq)
    ok = (q_lid >= 0) & t.multi[ent]
    lid, ent = q_lid[ok], ent[ok]
    o = np.argsort(lid, kind="stable")
    lid, ent = lid[o], ent[o]
    cnt = np.bincount(lid, minlength=len(qr.vocab)).astype(np.int64)
    start = np.cumsum(cnt) - cnt

    off, tok = getattr(qr, kind + "_off"), getattr(qr, kind + "_ids")
    e_tok = np.repeat(np.arange(qr.n, dtype=np.int64), np.diff(off))
    k = cnt[tok]
    sel = k > 0
    e_tok, tok, k = e_tok[sel], tok[sel], k[sel]
    idx = np.repeat(start[tok] - (np.cumsum(k) - k), k) + np.arange(k.sum())
    key, n = np.unique(np.repeat(e_tok, k) * E + ent[idx], return_counts=True)
    key = key[n == t.nq[key % E]]
    e_act, r = key // E, key % E

    ci, m = _ranges(t.c_off, r)
    keys = np.repeat(e_act, m) * V + t.c_ids[ci]
    o = np.argsort(keys, kind="stable")
    return keys[o], np.repeat(r, m)[o], q_lid


def _single_prep(t, qr, kind, remap):
    """Per entity, its tokens that are the query side of a single-token entry.
    -> (CSR offsets, s_q index, corpus id or -1)."""
    off, tok = getattr(qr, kind + "_off"), getattr(qr, kind + "_ids")
    lid = pc.fill_null(pc.index_in(t.s_qstr, value_set=qr.vocab), -1) \
        .to_numpy(zero_copy_only=False).astype(np.int64)
    to_g = np.full(max(len(qr.vocab), 1), -1, np.int64)
    ok = lid >= 0
    to_g[lid[ok]] = np.flatnonzero(ok)
    g = to_g[tok]
    keep = g >= 0
    e = np.repeat(np.arange(qr.n, dtype=np.int64), np.diff(off))[keep]
    e_off = np.zeros(qr.n + 1, np.int64)
    np.cumsum(np.bincount(e, minlength=qr.n), out=e_off[1:])
    return e_off, g[keep], remap[tok[keep]]


def _fire_single(t, prep, e, hit, vals, ln, V):
    """Single-token entries that fire on each pair of this chunk: the pair's
    unshared query tokens that start an entry, crossed with its unshared
    candidate tokens that end one. -> (pair, entry)."""
    empty = np.zeros(0, np.int64)
    if not len(t.s_keys):
        return empty, empty
    e_off, e_g, e_cid = prep
    m = len(e)
    p = np.repeat(np.arange(m, dtype=np.int64), ln)
    cm = ~hit & t.is_sc[vals]
    cp, cv = p[cm], vals[cm]
    pos = np.minimum(np.searchsorted(t.s_cids, cv), len(t.s_cids) - 1)
    ok = t.s_cids[pos] == cv
    cp, cg = cp[ok], pos[ok]
    if not len(cp):
        return empty, empty
    qi, k = _ranges(e_off, e)
    qp, qg, qc = np.repeat(np.arange(m, dtype=np.int64), k), e_g[qi], e_cid[qi]
    shared = p[hit] * V + vals[hit]
    ok = ~((qc >= 0) & _member(shared, qp * V + qc))
    qp, qg = qp[ok], qg[ok]
    ncp = np.bincount(cp, minlength=m)
    kq = ncp[qp]
    sel = kq > 0
    qp, qg, kq = qp[sel], qg[sel], kq[sel]
    idx = np.repeat((np.cumsum(ncp) - ncp)[qp] - (np.cumsum(kq) - kq), kq) + \
        np.arange(kq.sum())
    key = np.repeat(qg, kq) * max(len(t.s_cids), 1) + cg[idx]
    pos = np.minimum(np.searchsorted(t.s_keys, key), len(t.s_keys) - 1)
    hit_e = t.s_keys[pos] == key
    return np.repeat(qp, kq)[hit_e], t.s_ent[pos[hit_e]]


def _fire(t, akeys, aent, e, hit, vals, ln, V):
    """Multi-token entries that fire on each pair of this chunk. -> (pair, entry)."""
    E = len(t.kind)
    empty = np.zeros(0, np.int64)
    if not len(akeys):
        return empty, empty
    p = np.repeat(np.arange(len(e), dtype=np.int64), ln)
    miss = ~hit & t.is_mc[vals]
    pm = p[miss]
    probe = e[pm] * V + vals[miss]
    lo = np.searchsorted(akeys, probe, "left")
    n = np.searchsorted(akeys, probe, "right") - lo
    s = n > 0
    pm, lo, n = pm[s], lo[s], n[s]
    if not len(pm):
        return empty, empty
    idx = np.repeat(lo - (np.cumsum(n) - n), n) + np.arange(n.sum())
    key, cnt = np.unique(np.repeat(pm, n) * E + aent[idx], return_counts=True)
    key = key[cnt == t.nc[key % E]]
    pp, r = key // E, key % E
    # ... and none of the entry's query tokens may be shared with the candidate
    shared = p[hit] * V + vals[hit]          # sorted: p, then ids ascending
    qi, m = _ranges(t.q_off, r)
    qc = t.q_cid[qi]
    look = np.repeat(pp, m) * V + qc
    in_c = (qc >= 0) & _member(shared, look)
    bad = np.bincount(np.repeat(np.arange(len(pp)), m)[in_c],
                      minlength=len(pp)) > 0
    return pp[~bad], r[~bad]


def _both_fire(t, akeys, aent, sprep, e, hit, vals, ln, V):
    p1, r1 = _fire_single(t, sprep, e, hit, vals, ln, V)
    p2, r2 = _fire(t, akeys, aent, e, hit, vals, ln, V)
    return np.r_[p1, p2], np.r_[r1, r2]


def _member(keys, probe):
    """probe in keys (keys sorted)."""
    if not len(keys):
        return np.zeros(len(probe), bool)
    pos = np.minimum(np.searchsorted(keys, probe), len(keys) - 1)
    return keys[pos] == probe


def _unexplained(t, pp, r, q_lid, V, Vq):
    """Drop the forbidden fires whose query or candidate token another fired
    entry explains. -> (pair, entry)."""
    fb = t.kind[r] == _FORBID
    if not fb.any():
        return pp, r
    ok_p, ok_r = pp[~fb], r[~fb]
    qi, m = _ranges(t.q_off, ok_r)
    cov_q = np.unique(np.repeat(ok_p, m) * Vq + q_lid[qi])
    ci, m = _ranges(t.c_off, ok_r)
    cov_c = np.unique(np.repeat(ok_p, m) * V + t.c_ids[ci])
    fp, fr = pp[fb], r[fb]                     # single-token entries
    keep = ~(_member(cov_q, fp * Vq + q_lid[t.q_off[fr]]) |
             _member(cov_c, fp * V + t.c_ids[t.c_off[fr]]))
    return np.r_[ok_p, fp[keep]], np.r_[ok_r, fr[keep]]


def _n_distinct(pp, tok, space, n):
    """Distinct tokens per pair. -> counts, length n."""
    u = np.unique(pp * space + tok)
    return np.bincount(u // space, minlength=n)


_DEFAULT = object()


def build(recs, q_names, q_addrs, q_idx, c, idf_lut=None, grammar=_DEFAULT):
    """recs: precompute_records() of the corpus. q_*: per-entity normalised text.
    q_idx: entity index per pair. c: corpus record index per pair.
    grammar: a Grammar, None for none, or by default load_grammar().
    Output equals build_reference() on the gathered candidate text."""
    if grammar is _DEFAULT:
        grammar = load_grammar()
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
    col = {k: i for i, k in enumerate(NAMES)}
    if grammar is not None:
        tabs = _grammar_tables(recs, grammar)
        with stage("grammar activate", n=qr.n, unit="entities"):
            act = {f: _activate(tabs[f], qr, k, V) + (_single_prep(tabs[f], qr, k, remap),)
                   for f, k in (("name", "name"), ("address", "addr"))}
        Vq = max(len(qr.vocab), 1)

    for lo in range(0, n, _PAIR_CHUNK):
        e, j = q_idx[lo:lo + _PAIR_CHUNK], c[lo:lo + _PAIR_CHUNK]
        o = out[lo:lo + _PAIR_CHUNK]

        i_n, c_n, hit_n, vals_n = _intersect(kn, V, e, recs.name_off, recs.name_ids, j)
        a_n = qn[e]
        o[:, 0] = _div(i_n, a_n + c_n - i_n)
        o[:, 1] = _div(i_n, np.minimum(a_n, c_n))
        if idf is not None:
            v = np.where(hit_n, idf[vals_n], -np.inf)
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
        seqn = recs.seq_off[j + 1].astype(np.int64) - recs.seq_off[j]
        cand = np.flatnonzero((lq == lc) & (lc > 0) & (q_seqn[e] == seqn))
        if len(cand):
            cv, ln = _gather(recs.seq_off, recs.seq_ids, j[cand])
            qv, _ = _gather(qr.seq_off, q_seq, e[cand])
            o[cand, 3] = (_seg_sum(cv != qv, ln) == 0)

        i_a, c_a, hit_a, vals_a = _intersect(ka, V, e, recs.addr_off, recs.addr_ids, j)
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

        if grammar is None:
            continue
        with stage("strfeatures.grammar", n=len(e), unit="pairs"):
            m = len(e)
            t, (ak, ae, q_lid, sp) = tabs["name"], act["name"]
            pp, r = _unexplained(t, *_both_fire(t, ak, ae, sp, e, hit_n, vals_n, c_n, V),
                                 q_lid, V, Vq)
            ab = t.kind[r] == _ABBREV
            pa_, ra = pp[ab], r[ab]
            o[:, col["abbrev_matched_tokens"]] = np.bincount(
                pa_, weights=t.w[ra], minlength=m)
            ci, k = _ranges(t.c_off, ra)
            mc = _n_distinct(np.repeat(pa_, k), t.c_ids[ci], V, m)
            qi, k = _ranges(t.q_off, ra)
            mq = _n_distinct(np.repeat(pa_, k), q_lid[qi], Vq, m)
            eq = np.minimum(mc, mq)
            o[:, col["abbrev_adjusted_jaccard"]] = _div(i_n + eq, a_n + c_n - i_n - eq)
            forb = np.bincount(pp[t.kind[r] == _FORBID], minlength=m)

            t, (ak, ae, q_lid, sp) = tabs["address"], act["address"]
            pp, r = _unexplained(t, *_both_fire(t, ak, ae, sp, e, hit_a, vals_a, c_a, V),
                                 q_lid, V, Vq)
            forb += np.bincount(pp[t.kind[r] == _FORBID], minlength=m)
            o[:, col["forbidden_hits"]] = forb
            o[:, col["region_code_matched"]] = np.bincount(
                pp[t.kind[r] == _REGION], minlength=m)
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


def _fired(entries, qt, ct):
    """Entries that fire on one pair, by the rules above _ABBREV."""
    qx, cx = qt - ct, ct - qt
    fired = [x for x in entries if set(x[0]) <= qx and set(x[1]) <= cx]
    eq = [x for x in fired if x[2] != _FORBID]
    cq = {t for x in eq for t in x[0]}
    cc = {t for x in eq for t in x[1]}
    return eq + [x for x in fired if x[2] == _FORBID
                 and not (set(x[0]) & cq or set(x[1]) & cc)]


def build_reference(q_names, q_addrs, q_idx, c_names, c_addrs, idf_lut=None,
                    grammar=None):
    """The original per-pair loop. c_*: per-PAIR candidate text (gathered).
    Slow; kept only as the oracle build() is tested against."""
    col = {k: i for i, k in enumerate(NAMES)}
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
        if grammar is not None:
            fn = _fired(grammar.tables["name"], qt, ct)
            fa = _fired(grammar.tables["address"], qa, at)
            ab = [x for x in fn if x[2] == _ABBREV]
            out[i, col["abbrev_matched_tokens"]] = sum(x[3] for x in ab)
            eq = min(len({t for x in ab for t in x[0]}),
                     len({t for x in ab for t in x[1]}))
            inter = len(qt & ct)
            den = len(qt) + len(ct) - inter - eq
            out[i, col["abbrev_adjusted_jaccard"]] = (inter + eq) / den if den else 0.0
            out[i, col["forbidden_hits"]] = sum(x[2] == _FORBID for x in fn + fa)
            out[i, col["region_code_matched"]] = sum(x[2] == _REGION for x in fa)
    return out


def _self_test():
    from textnorm import norm

    def one(qn, qa, cn, ca, grammar=None):
        cn, ca = [norm(x) for x in cn], [norm(x) for x in ca]
        recs = precompute_records(cn, ca)
        return build(recs, [norm(x) for x in qn], [norm(x) for x in qa],
                     np.zeros(len(cn), np.int64), np.arange(len(cn)), None, grammar)

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

    g = Grammar.from_json({
        "abbreviations": {"name": [{"s1": "corporation", "s2": "corp",
                                    "p_s2_given_s1": 0.45}]},
        "region_codes": {"entries": [{"s1": "haryana", "s2": "hr"}]},
        "forbidden": {"name": [{"s1": "incorporated", "s2": "corp"}]}})
    abbr, none = one(["Acme Corporation"], ["5 MG Rd, Haryana"],
                     ["Acme Corp", "Acme Inc"], ["5 MG Rd HR", ""], g)
    forb = one(["Acme Incorporated"], [""], ["Acme Corp"], [""], g)[0]
    assert abbr[col["nm_jac"]] < abbr[col["abbrev_adjusted_jaccard"]] == 1.0, \
        "Corp/Corporation must raise adjusted Jaccard"
    assert abbr[col["abbrev_matched_tokens"]] == np.float32(0.45)
    assert abbr[col["region_code_matched"]] == 1.0
    assert none[col["abbrev_adjusted_jaccard"]] == none[col["nm_jac"]]
    assert forb[col["forbidden_hits"]] == 1.0 and abbr[col["forbidden_hits"]] == 0.0, \
        "incorporated -> corp must be flagged"
    off = one(["Acme Corporation"], [""], ["Acme Corp"], [""], None)[0]
    assert not off[[col[k] for k in GRAMMAR_NAMES]].any()
    print("PASS -- grammar: corp/corporation matched, forbidden swap flagged")


if __name__ == "__main__":
    _self_test()
