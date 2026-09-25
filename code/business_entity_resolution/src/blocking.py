"""Candidate generation. Country-sharded, multi-channel, unioned.

Channels (union, never intersection -- each reaches pairs the others cannot):
  C1  canonical-name exact key   cheap; also the normaliser sanity check
  C2  IDF-weighted rare name token, as an inverted index    the backbone
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
import os

# The df cap is the cost knob: ops = sum over keys of df_query * df_corpus.
# It is also the dominant recall constraint (see diag_dfcap.py), so these are
# env-overridable to make the recall/cost frontier sweepable.
DF_CAP = int(os.environ.get("DF_CAP", 2000))
ADDR_DF_CAP = int(os.environ.get("ADDR_DF_CAP", 10000))
C1_DF_CAP = 5       # a canonical name shared by more than a few entities is a chain
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
    than fusing them hands the matcher six real features for free -- agreement
    between independent channels is exactly the signal a fused scalar destroys.

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


CHANNELS = ["c2_name", "c6_addr", "c1_canon", "c5_postal"]
