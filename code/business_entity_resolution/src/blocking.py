"""Candidate generation. Country-sharded, multi-channel, unioned.

Channels (union, never intersection -- each reaches pairs the others cannot):
  C1  canonical-name exact key   cheap; also the normaliser sanity check
  C2  IDF-weighted rare name token, as an inverted index    the backbone
  C3  rarest-token-pair composite key   rare by construction, so reach without cap cost
  C5  postal x house-number composite key   the only channel independent of name
  C4  rarest-address-token-pair composite key   C3 over addresses (C4=1, default off)

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

import strfeatures
from profiling import stage
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
# C4: C3 over address tokens, with tokens containing a digit ranked ahead of
# words. Reaches links whose shared address tokens are all above ADDR_DF_CAP
# (1.31% of true links, the largest unreachable category). Default off: its
# pairs compete for the same CAND_CAP slots, so it can push found matches out
# (the mechanism of EXPERIMENTS.md #4). train_eval.py prints the ceiling with
# and without it. Switching it changes CHANNELS, so model.pkl must be retrained.
C4 = os.environ.get("C4", "0") == "1"
C4_DF_CAP = int(os.environ.get("C4_DF_CAP", 50))
C4_DF_FLOOR = int(os.environ.get("C4_DF_FLOOR", 2))
C4_N_RAREST = 3
# Records per _pair_keys chunk. Keys are per record, so chunking never changes
# them; it bounds the flattened-token arrays, which for 4M addresses of ~10
# tokens each would otherwise be several 40M-element int64 arrays at once.
PAIR_CHUNK = int(os.environ.get("PAIR_CHUNK", 500_000))
TOP_K = int(os.environ.get("TOP_K", 40))
ADDR_TOP_K = int(os.environ.get("ADDR_TOP_K", 60))
# Peak-memory budget for one c2_query batch product (Q[lo:hi] @ inv). On a 16GB
# box shared with a browser, this intermediate -- not the index -- is what caps
# DF_CAP/ADDR_DF_CAP, so batches are sized to fit it (see _batch_bounds).
MEM_BUDGET_MB = int(os.environ.get("MEM_BUDGET_MB", 1500))
# scipy CSR float32 product: 4 bytes data + 4 bytes int32 column index per nnz.
# int32 indices hold while a batch stays under 2^31 nnz (16GB), which any sane
# budget guarantees.
BYTES_PER_NNZ = 8


def load_shard(root, split, country):
    """Returns (s1_table, corpus_table). Corpus = S2 + S3 for this country."""
    def rd(src):
        d = ds.dataset(f"{root}/{split}_source{src}", format="parquet", partitioning="hive")
        return d.to_table(columns=["entity_id", "business_name", "business_address"],
                          filter=pc.field("country") == country)
    return rd(1), pa.concat_tables([rd(2), rd(3)])


# Deduplicating costs a hash pass and a gather; measured on synthetic strings it
# is ~10% slower than plain norm() when nothing repeats and pays from roughly
# 10% repeats up. Above this distinct fraction the column is normalised directly.
DEDUP_MAX_DISTINCT = 0.9


def norm_column(col):
    """[norm(x) for x in col.to_pylist()], normalising each distinct value once.
    -> (list, n_distinct). Repeated values share one output string object."""
    col = pc.fill_null(col, "")                     # norm(None) == norm("")
    uniq = pc.unique(col)
    if len(uniq) > DEDUP_MAX_DISTINCT * len(col):
        return [norm(x) for x in col.to_pylist()], len(uniq)
    out = np.array([norm(x) for x in uniq.to_pylist()], dtype=object)
    return out[pc.index_in(col, value_set=uniq).to_numpy()].tolist(), len(uniq)


def normalise(table, stats=None):
    """Normalised name and address as plain Python lists (needed for regex/keys).
    stats, if a dict, receives the distinct count of each column."""
    names, n_names = norm_column(table.column("business_name"))
    addrs, n_addrs = norm_column(table.column("business_address"))
    if stats is not None:
        stats.update(names=n_names, addrs=n_addrs)
    return names, addrs


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


def _batch_bounds(Q, df, budget_nnz, max_batch):
    """Row ranges [lo, hi) of Q whose product with the inverted index fits budget_nnz.

    A query row's product row has at most sum(df[t]) nonzeros over its surviving
    tokens t (exact unless one corpus record shares several tokens with it), so
    the cumulative df sum is a conservative nnz estimate. Each batch is the
    longest run from lo, up to max_batch rows, whose estimate fits the budget;
    this is recomputed per batch because query token rarity varies hugely (a
    run of chain names costs orders of magnitude more than a run of rare ones).
    A single row over budget still runs alone: its cost is bounded by
    n_tokens * df_cap, so it cannot blow up by itself.
    """
    row_cost = np.zeros(Q.shape[0] + 1, np.int64)
    tok_cost = np.cumsum(np.r_[0, df[Q.indices].astype(np.int64)])
    np.cumsum(tok_cost[Q.indptr[1:]] - tok_cost[Q.indptr[:-1]], out=row_cost[1:])
    bounds, lo, n = [], 0, Q.shape[0]
    while lo < n:
        # largest hi with row_cost[hi] - row_cost[lo] <= budget, i.e. shrink
        # from max_batch until the estimate fits
        hi = int(np.searchsorted(row_cost, row_cost[lo] + budget_nnz, "right")) - 1
        hi = min(max(hi, lo + 1), lo + max_batch, n)
        bounds.append((lo, hi))
        lo = hi
    return bounds


def c2_query(names, index, top_k=TOP_K, batch=4000, mem_budget_mb=None):
    """IDF-weighted rare-token retrieval. -> (query_idx, corpus_idx, score).

    batch is now an upper bound on rows per batch; the actual size adapts so the
    estimated product nnz stays under mem_budget_mb (default MEM_BUDGET_MB).
    Total work (sum of product nnz) is identical however the rows are split, so
    runtime should stay roughly flat: only peak memory falls, at the cost of
    more, smaller sparse products. Batching never changes output, since each
    product row depends only on its own query row.
    """
    budget_mb = MEM_BUDGET_MB if mem_budget_mb is None else mem_budget_mb
    budget_nnz = int(budget_mb * 2**20 // BYTES_PER_NNZ)
    Q = _query_matrix(names, index)
    qi, ci, sc = [], [], []
    for lo, hi in _batch_bounds(Q, index["df"], budget_nnz, batch):
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
        del S
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
    return _pair_keys(texts, df, vocab_lut, floor, n_rarest, prefer_digits=False)


def c4_keys(texts, df, vocab_lut, floor=C4_DF_FLOOR, n_rarest=C4_N_RAREST):
    """Rarest-address-token-pair keys: c3_keys over addresses. -> (values, owner_idx).

    df and vocab_lut are the address index's (corpus["aindex"]). Same packed key,
    same df floor, same rarest-first ranking by corpus df on both sides, with one
    change: tokens containing a digit (house, plot, PIN numbers; the definition
    textnorm.nums uses) rank ahead of every word token. At most n_rarest - 1
    digit tokens are kept, so a record with both digits and words always emits
    a number x word pair: a house number plus a locality is the most specific
    pair an address has, even when both are individually common.
    """
    return _pair_keys(texts, df, vocab_lut, floor, n_rarest, prefer_digits=True)


def _pair_keys(texts, df, vocab_lut, floor, n_rarest, prefer_digits):
    """Shared body of c3_keys and c4_keys, chunked by record (PAIR_CHUNK)."""
    vals, own = [], []
    for lo in range(0, len(texts), PAIR_CHUNK):
        v, o = _pair_keys_chunk(texts[lo:lo + PAIR_CHUNK], df, vocab_lut, floor,
                                n_rarest, prefer_digits)
        vals.append(v)
        own.append(o + lo)
    if not vals:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    return np.concatenate(vals), np.concatenate(own)


def _pair_keys_chunk(texts, df, vocab_lut, floor, n_rarest, prefer_digits):
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
    # 0 for a token containing a digit, 1 otherwise; all 0 when not preferring
    word = (np.zeros(len(tok), np.int8) if not prefer_digits else
            np.asarray(pc.invert(pc.match_substring_regex(flat, r"\p{Nd}"))
                       .to_numpy(zero_copy_only=False), np.int8))
    doc = np.repeat(np.arange(len(texts), dtype=np.int64), counts)
    keep = tok >= 0
    doc, tok, word = doc[keep], tok[keep], word[keep]
    d = np.asarray(df)[tok]
    keep = d >= floor
    doc, tok, d, word = doc[keep], tok[keep], d[keep], word[keep]
    if not len(doc):
        return empty

    # rarest first within each record (digit tokens first when preferred); one
    # sort also makes repeats adjacent
    order = np.lexsort((tok, d, word, doc))
    doc, tok, word = doc[order], tok[order], word[order]
    new = np.r_[True, (doc[1:] != doc[:-1]) | (tok[1:] != tok[:-1])]
    doc, tok, word = doc[new], tok[new], word[new]
    if prefer_digits:
        # keep at most n_rarest - 1 digit tokens so a word always gets a slot;
        # digits sort first, so their rank within the record is their digit rank
        start = np.r_[True, doc[1:] != doc[:-1]]
        grp = np.cumsum(start) - 1
        rank = np.arange(len(doc)) - np.flatnonzero(start)[grp]
        keep = (word == 1) | (rank < n_rarest - 1)
        doc, tok = doc[keep], tok[keep]
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


def build_corpus(corpus_tab, verbose=True, string_features=False):
    """Index the corpus once; reused across every query batch.
    string_features=True also adds out["recs"] for strfeatures.build."""
    stats = {}
    with stage("normalise corpus", n=len(corpus_tab), unit="records"):
        c_names, c_addrs = normalise(corpus_tab, stats)
    if verbose and c_names:
        print(f"    distinct raw strings: names {stats['names']/len(c_names):.1%}, "
              f"addresses {stats['addrs']/len(c_addrs):.1%}")
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
    n = len(c_names)
    out = {}
    with stage("build_index c2_name", n=n, unit="records"):
        out["index"] = build_index(c_names)
    with stage("build_index c6_addr", n=n, unit="records"):
        out["aindex"] = build_index(c_addrs, ADDR_DF_CAP)
    with stage("build_index c1_canon", n=n, unit="records"):
        out["c1"] = c1_keys(c_names)
    with stage("build_index c5_postal", n=n, unit="records"):
        out["c5"] = c5_keys(c_addrs)
    with stage("build_index c3_pair", n=n, unit="records"):
        out["c3"] = c3_keys(c_names, out["index"]["df"], out["index"]["vocab"])
    if C4:
        with stage("build_index c4_addr_pair", n=n, unit="records"):
            out["c4"] = c4_keys(c_addrs, out["aindex"]["df"], out["aindex"]["vocab"])
    # Per-record half of the string features, built once here instead of per
    # candidate pair: flat int32 token, digit and 4-gram sets (strfeatures.py).
    if string_features:
        with stage("precompute_records", n=n, unit="records"):
            out["recs"] = strfeatures.precompute_records(c_names, c_addrs)
    with stage("drop corpus text + gc", n=n, unit="records"):
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

    nq = len(s1_names)
    with stage("query c2_name", n=nq, unit="queries"):
        add(*c2_query(s1_names, index, top_k), "C2 name-token")
    # C6 is the workhorse for India and the ONLY channel that reaches records
    # whose name is in another script (audit: 99.88% on script mismatch).
    with stage("query c6_addr", n=nq, unit="queries"):
        add(*c2_query(s1_addrs, corpus["aindex"], ADDR_TOP_K), "C6 addr-token")
    with stage("query c1_canon", n=nq, unit="queries"):
        l, r = _key_join(c1_keys(s1_names), corpus["c1"], C1_DF_CAP)
        add(l, r, np.ones(len(l)), "C1 canonical")
    with stage("query c5_postal", n=nq, unit="queries"):
        l, r = _key_join(c5_keys(s1_addrs), corpus["c5"], 50)
        add(l, r, np.ones(len(l)), "C5 postal-num")
    # Appended last so existing channel columns keep their positions. A pair can
    # share up to 3 keys; the share is the score, since all three rarest tokens
    # agreeing is much stronger evidence than one pair agreeing.
    with stage("query c3_pair", n=nq, unit="queries"):
        l, r = _key_join(c3_keys(s1_names, index["df"], index["vocab"]),
                         corpus["c3"], C3_DF_CAP)
        k, n_shared = np.unique(l * np.int64(index["n"]) + r, return_counts=True)
        add(k // index["n"], k % index["n"], n_shared / C3_N_RAREST, "C3 token-pair")
    # Appended after C3 for the same reason, and scored the same way.
    if C4:
        aindex = corpus["aindex"]
        with stage("query c4_addr_pair", n=nq, unit="queries"):
            l, r = _key_join(c4_keys(s1_addrs, aindex["df"], aindex["vocab"]),
                             corpus["c4"], C4_DF_CAP)
            k, n_shared = np.unique(l * np.int64(index["n"]) + r,
                                    return_counts=True)
            add(k // index["n"], k % index["n"], n_shared / C4_N_RAREST,
                "C4 addr-pair")

    with stage("union channels", n=nq, unit="queries"):
        return _union(parts, index["n"])


def _union(parts, n_corpus):
    """Deduplicate the channels' pairs into (q, c, chan); see generate()."""
    q = np.concatenate([p[0] for p in parts])
    c = np.concatenate([p[1] for p in parts])
    src = np.concatenate([np.full(len(p[0]), i, np.int8)
                          for i, p in enumerate(parts)])
    s = np.concatenate([p[2] for p in parts])
    if not len(q):
        return q, c, np.zeros((0, len(parts)), np.float32)

    key = q * np.int64(n_corpus) + c
    order = np.argsort(key, kind="stable")
    key, q, c, s, src = key[order], q[order], c[order], s[order], src[order]
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    pair_of = np.cumsum(np.r_[True, key[1:] != key[:-1]]) - 1
    chan = np.zeros((len(starts), len(parts)), np.float32)
    chan[pair_of, src] = s          # last write wins; a channel emits a pair once
    return q[starts], c[starts], chan


CHANNELS = (["c2_name", "c6_addr", "c1_canon", "c5_postal", "c3_pair"]
            + (["c4_addr_pair"] if C4 else []))


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

    # ---- C4: address token pairs ----
    # Two address tokens each far above the address df cap (so C6 drops both)
    # whose pair is rare: house number 12 is on every street, koramangala is a
    # whole locality, "12 koramangala" is a handful of buildings.
    street = ["mg road", "church street", "brigade road", "residency road"]
    corpus_addrs = ([f"12 {s} indiranagar" for s in street for _ in range(10)] +
                    [f"{n} koramangala {s}" for n in (7, 9, 44, 81)
                     for s in street for _ in range(10)] +
                    ["12 koramangala 5th block", "12 koramangala hosur road",
                     "flat 3 12 koramangala 560034", "  ", "zzuniq"])
    small_cap = 30                            # stands in for ADDR_DF_CAP
    aidx = build_index(corpus_addrs, df_cap=small_cap)
    adf, avocab = aidx["df"], aidx["vocab"]
    atid = {t: i for i, t in enumerate(avocab.to_pylist())}
    AV = len(adf)
    for t in ("12", "koramangala"):
        assert adf[atid[t]] > small_cap, (t, adf[atid[t]])
    av, ao = c4_keys(corpus_addrs, adf, avocab)
    akey_df = dict(zip(*np.unique(av, return_counts=True)))
    i, j = sorted((atid["12"], atid["koramangala"]))
    pair_df = akey_df[i * AV + j]
    print(f"  df[12]={adf[atid['12']]} df[koramangala]={adf[atid['koramangala']]}"
          f" (cap {small_cap})  pair df={pair_df}")
    assert pair_df <= C4_DF_CAP and pair_df < small_cap, pair_df

    # C6 cannot reach it: every shared token is above the cap, so the query
    # has no live token in common with the match
    query = ["koramangala near 12 bus stop"]
    Qa = _query_matrix(query, aidx)
    assert (Qa @ aidx["inv"]).nnz == 0
    l, r = _key_join(c4_keys(query, adf, avocab), (av, ao), C4_DF_CAP)
    hits = {corpus_addrs[x] for x in r}
    assert {"12 koramangala 5th block", "12 koramangala hosur road",
            "flat 3 12 koramangala 560034"} <= hits, hits
    assert not any(h.endswith("indiranagar") for h in hits), hits

    # digit tokens rank ahead of rarer words, but never take every slot
    # (own corpus: every token here needs df >= the floor to be a candidate)
    pref = (corpus_addrs + ["flat 3 560034 hosur 5th block"] * 2 +
            ["12 hosur road", "block road"])
    pidx = build_index(pref, df_cap=small_cap)
    pdf, pvocab = pidx["df"], pidx["vocab"]
    PV = len(pdf)
    ptid = {t: i for i, t in enumerate(pvocab.to_pylist())}

    def picked(fn, text):
        v, _ = fn([text], pdf, pvocab)
        return {pvocab[int(t)].as_py() for k in v for t in (k // PV, k % PV)}
    three = picked(c4_keys, "flat 3 12 koramangala 560034")  # 3 digit tokens
    assert three == {"3", "560034", "flat"}, three             # 2 digits + a word
    # rarest-first alone drops the house number for three rarer words;
    # preferring digits keeps it
    text = "12 koramangala 5th block hosur road"
    assert "12" not in picked(c3_keys, text), picked(c3_keys, text)
    assert {"12", "5th"} <= picked(c4_keys, text), picked(c4_keys, text)
    print(f"  {text!r}: C3 picks {sorted(picked(c3_keys, text))}, "
          f"C4 picks {sorted(picked(c4_keys, text))}")

    # order-invariant, and chunking never changes the keys
    a = np.sort(c4_keys(["12 koramangala hosur road"], adf, avocab)[0])
    assert np.array_equal(a, np.sort(c4_keys(["road hosur  koramangala 12 12"],
                                             adf, avocab)[0]))
    PAIR_CHUNK, keep_chunk = 7, PAIR_CHUNK
    small = sorted(zip(*c4_keys(corpus_addrs, adf, avocab)))
    PAIR_CHUNK = keep_chunk
    assert small == sorted(zip(av, ao))

    # end to end through generate(): C4 is appended as the last column and
    # reaches the pair C6 cannot, while C4=0 leaves the output unchanged
    ctab = pa.table({"business_name": [f"shop {k}" for k in range(len(corpus_addrs))],
                     "business_address": corpus_addrs})
    qtab = pa.table({"business_name": ["unrelated name"],
                     "business_address": query})
    keep_cap, ADDR_DF_CAP = ADDR_DF_CAP, small_cap
    off = generate(qtab, ctab, verbose=False)
    C4, keep_ch = True, CHANNELS
    CHANNELS = CHANNELS + ["c4_addr_pair"]
    on = generate(qtab, ctab, verbose=False)
    C4, CHANNELS, ADDR_DF_CAP = False, keep_ch, keep_cap
    assert on[2].shape[1] == off[2].shape[1] + 1
    had = {(int(q), int(c)) for q, c in zip(off[0], off[1])}
    c4_only = [int(c) for q, c, ch in zip(*on) if (q, c) not in had]
    assert {corpus_addrs[x] for x in c4_only} == hits, c4_only
    old = {(int(q), int(c)): tuple(ch) for q, c, ch in zip(*off)}
    for q, c, ch in zip(*on):
        if (int(q), int(c)) in old:
            assert tuple(ch[:-1]) == old[int(q), int(c)]
    print(f"  C4 adds {len(c4_only)} pairs C6 cannot reach")
    print("c4_keys self-test passed")

    # c2_query adaptive batching: batch size shrinks with query df, output doesn't move
    rng = np.random.default_rng(0)
    V = 400
    # Zipf-ish corpus: token t appears in roughly 20000/(t+1) of 20000 records,
    # so low ids are common (df in the thousands) and high ids are rare.
    p = 1.0 / np.arange(1, V + 1)
    p /= p.sum()
    toks = [f"t{i}" for i in range(V)]
    cnames = [" ".join(toks[j] for j in rng.choice(V, 4, p=p)) for _ in range(20000)]
    cindex = build_index(cnames, df_cap=10**9)
    cdf = cindex["df"]
    ctid = {t: i for i, t in enumerate(cindex["vocab"].to_pylist())}
    by_df = sorted(ctid, key=lambda t: cdf[ctid[t]])
    rare, common = by_df[:200], by_df[-20:]
    q_low = [" ".join(rng.choice(rare, 3)) for _ in range(3000)]
    q_high = [" ".join(rng.choice(common, 3)) for _ in range(3000)]

    budget_mb = 2.0                   # 262,144 nnz: forces many batches
    budget_nnz = int(budget_mb * 2**20 // BYTES_PER_NNZ)
    sizes = {}
    for tag, qs in [("low", q_low), ("high", q_high)]:
        Qm = _query_matrix(qs, cindex)
        b = _batch_bounds(Qm, cdf, budget_nnz, 4000)
        assert b[0][0] == 0 and b[-1][1] == len(qs)
        assert all(x[1] == y[0] for x, y in zip(b, b[1:]))
        for lo, hi in b:              # the estimate is an upper bound and fits
            nnz = (Qm[lo:hi] @ cindex["inv"]).nnz
            est = int(cdf[Qm[lo:hi].indices].sum())
            assert nnz <= est and (est <= budget_nnz or hi - lo == 1), (lo, hi)
        sizes[tag] = np.mean([hi - lo for lo, hi in b])
        print(f"  {tag}-df queries: {len(b)} batches, mean {sizes[tag]:.0f} rows")
    assert sizes["high"] < sizes["low"], sizes

    for qs in (q_low, q_high, q_low + q_high):
        ref = c2_query(qs, cindex, top_k=10, batch=10**9, mem_budget_mb=10**6)
        for bm, mb in [(budget_mb, 4000), (0.01, 4000), (10**6, 1), (10**6, 7)]:
            got = c2_query(qs, cindex, top_k=10, batch=mb, mem_budget_mb=bm)
            for a, g in zip(ref, got):
                assert a.dtype == g.dtype and np.array_equal(a, g), (bm, mb)
    print("c2_query batching self-test passed")
