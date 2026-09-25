"""Candidate generation. Country-sharded, multi-channel, unioned.

Channels (union, never intersection -- each reaches pairs the others cannot):
  C1  canonical-name exact key   cheap; also the normaliser sanity check
  C2  IDF-weighted rare name token, as an inverted index    the backbone
  C3  rarest-token-pair composite key   rare by construction, so reach without cap cost
  C5  postal x house-number composite key   the only channel independent of name

Country is a verified-safe hard block (0 cross-country links in 7.6M), so each
country is an independent job. Document frequency is fit on the corpus being
searched -- transductively on test at inference time -- which is unsupervised,
uses only provided files, and is what makes an unseen country work without
anyone knowing its language.
"""
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import scipy.sparse as sp

from textnorm import norm

import gc
import itertools
import os

# The df cap is the cost knob: ops = sum over keys of df_query * df_corpus.
# It is also the dominant recall constraint (see diag_dfcap.py), so these are
# env-overridable to make the recall/cost frontier sweepable.
DF_CAP = int(os.environ.get("DF_CAP", 2000))
ADDR_DF_CAP = int(os.environ.get("ADDR_DF_CAP", 10000))
C1_DF_CAP = 5       # a canonical name shared by more than a few entities is a chain
# C3: a pair key shared by more corpus records than this is a chain name
# ("state bank") and retrieves siblings, not matches. Tokens below the floor
# are excluded before picking the rarest three: a df-1 token is almost always a
# typo, and any pair it could reach C2 already reaches at the top of its list.
C3_DF_CAP = int(os.environ.get("C3_DF_CAP", 50))
C3_DF_FLOOR = int(os.environ.get("C3_DF_FLOOR", 2))
C3_N_RAREST = 3
TOP_K = int(os.environ.get("TOP_K", 40))
ADDR_TOP_K = int(os.environ.get("ADDR_TOP_K", 60))


def load_shard(root, split, country):
    """Returns (s1_table, corpus_table). Corpus = S2 + S3 for this country."""
    def rd(src):
        d = ds.dataset(f"{root}/{split}_source{src}", format="parquet", partitioning="hive")
        return d.to_table(columns=["entity_id", "business_name", "business_address"],
                          filter=pc.field("country") == country)
    return rd(1), pa.concat_tables([rd(2), rd(3)])


def normalise(table):
    """Normalised name and address as plain Python lists (needed for regex/keys)."""
    return ([norm(x) for x in table.column("business_name").to_pylist()],
            [norm(x) for x in table.column("business_address").to_pylist()])


def _tokenise(texts):
    """-> (doc_ids, token_ids, n_vocab), deduplicated within a document."""
    arr = pa.array(texts)
    lists = pc.split_pattern(arr, " ")
    counts = pc.list_value_length(lists).to_numpy(zero_copy_only=False)
    flat = lists.flatten()
    enc = flat.dictionary_encode()
    tok = enc.indices.to_numpy(zero_copy_only=False).astype(np.int64)
    doc = np.repeat(np.arange(len(texts), dtype=np.int64), counts)
    keep = tok >= 0
    doc, tok = doc[keep], tok[keep]
    # dedupe (doc, token) so df counts documents, not occurrences
    key = doc * (tok.max() + 2) + tok
    _, first = np.unique(key, return_index=True)
    return doc[first], tok[first], len(enc.dictionary), enc.dictionary


def build_index(corpus_names, df_cap=DF_CAP):
    """Token-major CSR (vocab x n_corpus): an IDF-ready inverted index."""
    doc, tok, vocab_n, vocab = _tokenise(corpus_names)
    df = np.bincount(tok, minlength=vocab_n)
    alive = df <= df_cap
    m = alive[tok]
    doc, tok = doc[m], tok[m]
    idf = np.zeros(vocab_n, np.float32)
    live = df > 0
    idf[live] = np.log(len(corpus_names) / df[live]).astype(np.float32)
    idf[~alive] = 0.0
    # L2-normalise each corpus record so retrieval scores cosine, not a raw IDF
    # sum. Without this, a record with many address tokens outranks the true
    # match purely by having more chances to collide.
    w = idf[tok]
    sq = np.bincount(doc, weights=w.astype(np.float64) ** 2,
                     minlength=len(corpus_names))
    nrm = np.sqrt(sq)
    nrm[nrm == 0] = 1.0
    inv = sp.csr_matrix(((w / nrm[doc]).astype(np.float32), (tok, doc)),
                        shape=(vocab_n, len(corpus_names)))
    return {"inv": inv, "idf": idf, "vocab": vocab, "df": df, "n": len(corpus_names)}


def _query_matrix(names, index):
    """Queries as an L2-normalised (n_query x vocab) CSR with corpus IDF weights."""
    lut = {v: i for i, v in enumerate(index["vocab"].to_pylist())}
    rows, cols = [], []
    for i, nm in enumerate(names):
        for t in set(nm.split()):
            j = lut.get(t)
            if j is not None and index["idf"][j] > 0:
                rows.append(i)
                cols.append(j)
    rows = np.asarray(rows, np.int64)
    cols = np.asarray(cols, np.int64)
    if not len(cols):
        return sp.csr_matrix((len(names), index["inv"].shape[0]), dtype=np.float32)
    data = index["idf"][cols]
    sq = np.bincount(rows, weights=data.astype(np.float64) ** 2, minlength=len(names))
    nrm = np.sqrt(sq)
    nrm[nrm == 0] = 1.0
    return sp.csr_matrix(((data / nrm[rows]).astype(np.float32), (rows, cols)),
                         shape=(len(names), index["inv"].shape[0]))


def c2_query(names, index, top_k=TOP_K, batch=4000):
    """IDF-weighted rare-token retrieval. -> (query_idx, corpus_idx, score)."""
    Q = _query_matrix(names, index)
    qi, ci, sc = [], [], []
    for lo in range(0, len(names), batch):
        hi = min(lo + batch, len(names))
        S = (Q[lo:hi] @ index["inv"]).tocsr()
        for r in range(S.shape[0]):
            s, e = S.indptr[r], S.indptr[r + 1]
            if s == e:
                continue
            cols, vals = S.indices[s:e], S.data[s:e]
            if e - s > top_k:
                sel = np.argpartition(vals, -top_k)[-top_k:]
                cols, vals = cols[sel], vals[sel]
            qi.append(np.full(len(cols), lo + r, np.int64))
            ci.append(cols.astype(np.int64))
            sc.append(vals)
    if not qi:
        return (np.empty(0, np.int64),) * 2 + (np.empty(0, np.float32),)
    return np.concatenate(qi), np.concatenate(ci), np.concatenate(sc)


def _key_join(left_keys, right_keys, right_cap):
    """Join two lists-of-key-lists on shared keys. -> (left_idx, right_idx)."""
    lv, li = left_keys
    rv, ri = right_keys
    if not len(lv) or not len(rv):
        return np.empty(0, np.int64), np.empty(0, np.int64)
    # suppress keys that are too common on the right to be discriminative
    uniq, cnt = np.unique(rv, return_counts=True)
    good = uniq[cnt <= right_cap]
    lm, rm = np.isin(lv, good), np.isin(rv, good)
    li, lv, ri, rv = li[lm], lv[lm], ri[rm], rv[rm]
    order = np.argsort(rv, kind="stable")
    rv, ri = rv[order], ri[order]
    lo = np.searchsorted(rv, lv, "left")
    hi = np.searchsorted(rv, lv, "right")
    out_l, out_r = [], []
    for i in range(len(lv)):
        if hi[i] > lo[i]:
            out_r.append(ri[lo[i]:hi[i]])
            out_l.append(np.full(hi[i] - lo[i], li[i], np.int64))
    if not out_l:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    return np.concatenate(out_l), np.concatenate(out_r)


def _hash64(s):
    return np.int64(hash(s) & 0x7FFFFFFFFFFFFFFF)


def c1_keys(names):
    """Canonical name: order-invariant bag of tokens. -> (values, owner_idx)."""
    vals, own = [], []
    for i, n in enumerate(names):
        if n:
            vals.append(_hash64(" ".join(sorted(set(n.split())))))
            own.append(i)
    return np.asarray(vals, np.int64), np.asarray(own, np.int64)


def c5_keys(addrs):
    """Address digit keys.

    Country-agnostic: a postal code is just a digit run of length 5-6, which
    holds for US, India and France without naming any of them.

    Measured: emitting every digit run as its own key added 40 minutes of wall
    clock for +0.01% recall, because C6 already indexes address tokens and those
    include the digits. Only the postal|number composite earns its place.
    """
    vals, own = [], []
    for i, a in enumerate(addrs):
        digits = [t for t in a.split() if t.isdigit()]
        posts = [d for d in digits if 5 <= len(d) <= 6]
        for p in posts:
            for o in (d for d in digits if len(d) < 5):
                vals.append(_hash64(p + "|" + o))
                own.append(i)
    return np.asarray(vals, np.int64), np.asarray(own, np.int64)


def c3_keys(texts, df, vocab_lut, floor=C3_DF_FLOOR, n_rarest=C3_N_RAREST):
    """Rarest-token-pair keys. -> (values, owner_idx).

    Each record keeps its n_rarest lowest-df tokens and emits every pair of them
    as one key. Two mid-frequency tokens rarely co-occur, so the pair's df is far
    below either token's -- recall past the C2 df cap without paying
    df_query * df_corpus on the tokens themselves.

    df and vocab_lut are the corpus name index's df array and vocab (the pyarrow
    string array whose position is the token id), i.e. index["df"] and
    index["vocab"] from build_index. Both sides of the join MUST use the same
    pair: rarity is ranked by corpus df, with token id breaking ties, so a query
    and its match pick the same tokens whenever they share them.

    The key is the sorted id pair packed as lo * V + hi: order-invariant and
    exact rather than hashed, so it cannot collide. Tokens not in the corpus
    vocab can never join and are dropped before ranking, so they do not use up
    a slot. Records with fewer than 2 surviving tokens emit nothing.

    Fully vectorised: no per-record Python objects on a 4M-record corpus.
    """
    empty = (np.empty(0, np.int64), np.empty(0, np.int64))
    if not len(texts):
        return empty
    lists = pc.split_pattern(pa.array(texts, pa.string()), " ")
    counts = pc.fill_null(pc.list_value_length(lists), 0).to_numpy(zero_copy_only=False)
    flat = lists.flatten()
    tok = pc.fill_null(pc.index_in(flat, value_set=vocab_lut), -1)
    tok = tok.to_numpy(zero_copy_only=False).astype(np.int64)
    # a double space yields an empty token; it is in the vocab but is not content
    blank = pc.equal(pc.binary_length(flat), 0).to_numpy(zero_copy_only=False)
    tok[blank] = -1
    doc = np.repeat(np.arange(len(texts), dtype=np.int64), counts)
    keep = tok >= 0
    doc, tok = doc[keep], tok[keep]
    d = np.asarray(df)[tok]
    keep = d >= floor
    doc, tok, d = doc[keep], tok[keep], d[keep]
    if not len(doc):
        return empty

    # rarest first within each record; one sort also makes repeats adjacent
    order = np.lexsort((tok, d, doc))
    doc, tok = doc[order], tok[order]
    new = np.r_[True, (doc[1:] != doc[:-1]) | (tok[1:] != tok[:-1])]
    doc, tok = doc[new], tok[new]
    start = np.r_[True, doc[1:] != doc[:-1]]
    grp = np.cumsum(start) - 1
    rank = np.arange(len(doc)) - np.flatnonzero(start)[grp]
    owner = doc[start]
    keep = rank < n_rarest
    rarest = np.full((len(owner), n_rarest), -1, np.int64)
    rarest[grp[keep], rank[keep]] = tok[keep]

    V = np.int64(len(df))
    vals, own = [], []
    for a, b in itertools.combinations(range(n_rarest), 2):
        ok = rarest[:, b] >= 0          # rank b present implies rank a < b is too
        ta, tb = rarest[ok, a], rarest[ok, b]
        vals.append(np.minimum(ta, tb) * V + np.maximum(ta, tb))
        own.append(owner[ok])
    return np.concatenate(vals), np.concatenate(own)


def build_corpus(corpus_tab, verbose=True):
    """Index the corpus once; reused across every query batch."""
    c_names, c_addrs = normalise(corpus_tab)
    empty = sum(1 for x in c_names if not x)
    if verbose and empty:
        print(f"    {empty:,} corpus names empty after normalisation "
              f"({empty/len(c_names):.1%}) -- reachable only via address")
    # Build each index then drop its source text immediately. Holding the
    # normalised strings for a 4M-record corpus costs ~500MB for something
    # nothing downstream reads, and on a 16GB box that is the difference
    # between running and paging.
    #
    # Folded and consonant-skeleton name indices were measured and removed: their
    # MARGINAL recall contribution was +0.06% (71.83% -> 71.89%), because the
    # links they reach are already caught by the address channel. Their apparent
    # per-channel attribution (28.65%, 15.90%) is overlap, not new reach. They
    # cost two extra inverted indices plus Python-level string generation over
    # every corpus record, which is what pushed this shard into swap.
    out = {"index": build_index(c_names),
           "aindex": build_index(c_addrs, ADDR_DF_CAP),
           "c1": c1_keys(c_names), "c5": c5_keys(c_addrs)}
    out["c3"] = c3_keys(c_names, out["index"]["df"], out["index"]["vocab"])
    # Keep the normalised text for string features, but as Arrow arrays: a
    # contiguous buffer plus int32 offsets is ~140MB for 4.1M records, where the
    # equivalent Python list of str objects is ~3x that.
    out["names_arr"] = pa.array(c_names)
    out["addrs_arr"] = pa.array(c_addrs)
    del c_names, c_addrs
    gc.collect()
    return out


def generate(s1_tab, corpus_tab, s1_names=None, s1_addrs=None,
             corpus=None, top_k=TOP_K, verbose=True):
    """Union of all channels.

    Returns (q_idx, c_idx, chan_scores) deduplicated, where chan_scores is a
    (n_pairs x n_channels) float32 matrix of each channel's similarity, 0 where
    the channel did not retrieve the pair. Keeping the channels separate rather
    than fusing them hands the matcher one real feature per channel for free --
    agreement between independent channels is exactly the signal a fused
    scalar destroys.

    Arrays rather than dict-of-dicts: at full scale this is tens of millions of
    pairs, where a Python dict would cost both a rewrite and most of the RAM.
    """
    if s1_names is None:
        s1_names, s1_addrs = normalise(s1_tab)
    if corpus is None:
        corpus = build_corpus(corpus_tab, verbose)
    index = corpus["index"]
    if verbose:
        alive = int((index["df"] > 0).sum())
        capped = int((index["df"] > DF_CAP).sum())
        print(f"    {len(s1_names):,} S1 vs {index['n']:,} corpus | "
              f"vocab {alive:,}, {capped:,} above df cap {DF_CAP}")

    parts = []

    def add(qi, ci, sc, tag):
        parts.append((qi, ci, sc.astype(np.float32)))
        if verbose:
            print(f"    {tag:14s} {len(qi):>10,} pairs")

    add(*c2_query(s1_names, index, top_k), "C2 name-token")
    # C6 is the workhorse for India and the ONLY channel that reaches records
    # whose name is in another script (audit: 99.88% on script mismatch).
    add(*c2_query(s1_addrs, corpus["aindex"], ADDR_TOP_K), "C6 addr-token")
    l, r = _key_join(c1_keys(s1_names), corpus["c1"], C1_DF_CAP)
    add(l, r, np.ones(len(l)), "C1 canonical")
    l, r = _key_join(c5_keys(s1_addrs), corpus["c5"], 50)
    add(l, r, np.ones(len(l)), "C5 postal-num")
    # Appended last so the existing channel columns keep their positions. A
    # pair can share up to 3 keys; the share is the score, since all three
    # rarest tokens agreeing is much stronger evidence than one pair.
    l, r = _key_join(c3_keys(s1_names, index["df"], index["vocab"]),
                     corpus["c3"], C3_DF_CAP)
    k, n_shared = np.unique(l * np.int64(index["n"]) + r, return_counts=True)
    add(k // index["n"], k % index["n"], n_shared / C3_N_RAREST, "C3 token-pair")

    q = np.concatenate([p[0] for p in parts])
    c = np.concatenate([p[1] for p in parts])
    src = np.concatenate([np.full(len(p[0]), i, np.int8)
                          for i, p in enumerate(parts)])
    s = np.concatenate([p[2] for p in parts])
    if not len(q):
        return q, c, np.zeros((0, len(parts)), np.float32)

    key = q * np.int64(index["n"]) + c
    order = np.argsort(key, kind="stable")
    key, q, c, s, src = key[order], q[order], c[order], s[order], src[order]
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    pair_of = np.cumsum(np.r_[True, key[1:] != key[:-1]]) - 1
    chan = np.zeros((len(starts), len(parts)), np.float32)
    chan[pair_of, src] = s          # last write wins; a channel emits a pair once
    return q[starts], c[starts], chan


CHANNELS = ["c2_name", "c6_addr", "c1_canon", "c5_postal", "c3_pair"]


if __name__ == "__main__":
    # Synthetic self-test: python blocking.py (no data/ needed).
    common = ["sharma", "medical", "store", "traders"]
    corpus_names = (["sharma kirana"] * 30 + ["medical centre"] * 30 +
                    ["store house"] * 30 + ["traders union"] * 30 +
                    ["sharma medical store"] * 2 + ["sharma traders"] * 3 +
                    ["sharma medcal store", "zzuniq alone", "", "solo"])
    index = build_index(corpus_names)
    df, vocab = index["df"], index["vocab"]
    tid = {t: i for i, t in enumerate(vocab.to_pylist())}

    # order-invariance, including repeated tokens and extra spaces
    a = np.sort(c3_keys(["sharma medical store"], df, vocab)[0])
    for v in ["store sharma medical", "medical  store sharma sharma"]:
        assert np.array_equal(a, np.sort(c3_keys([v], df, vocab)[0])), v
    assert len(a) == 3

    # a pair of common tokens is rarer than either token alone
    cv, co = c3_keys(corpus_names, df, vocab)
    key_df = dict(zip(*np.unique(cv, return_counts=True)))
    V = len(df)
    for x, y in [("sharma", "medical"), ("sharma", "store"), ("medical", "store")]:
        i, j = sorted((tid[x], tid[y]))
        kdf = key_df[i * V + j]
        assert kdf < min(df[tid[x]], df[tid[y]]), (x, y, kdf)
        print(f"  df[{x}]={df[tid[x]]:>3} df[{y}]={df[tid[y]]:>3}  pair df={kdf}")

    # only the 3 rarest tokens are paired: the most common of four is dropped
    four = c3_keys(["sharma medical store traders"], df, vocab)[0]
    assert len(four) == 3
    worst = max(common, key=lambda t: df[tid[t]])
    assert not any(tid[worst] in (k // V, k % V) for k in four), worst

    # <2 content tokens emits nothing: empty, one token, df-1 typo, unknown token
    for v in ["", "solo sharma", "sharma", "sharma medcal", "sharma qqqq"]:
        vv, oo = c3_keys([v], df, vocab)
        assert not len(vv) and not len(oo), v

    # end to end: a typo in one token still joins on the surviving clean pair
    qv, qo = c3_keys(["medical sharma store", "sharma"], df, vocab)
    assert qo.dtype == np.int64 and qv.dtype == np.int64
    l, r = _key_join((qv, qo), (cv, co), C3_DF_CAP)
    hits = {corpus_names[i] for i in r[l == 0]}
    assert "sharma medcal store" in hits and "sharma medical store" in hits, hits
    assert not len(r[l == 1])
    print("c3_keys self-test passed")
