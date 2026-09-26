"""Second-stage (stacked) features: a candidate's resemblance to its entity's
OTHER, confident candidates.

diag_matcher.py measured 38.8% of the matcher's gap as 'stopped early': true
matches ranked above every rejected false positive of their own entity, yet
with absolute p ~0.3-0.5, so any global rule that accepts them also accepts
same-scored false positives of other entities. The pairwise matcher scores
(S1, candidate) alone and cannot see that such a candidate is a near copy of
the entity's confident candidates -- which it should be, since an entity's
S2/S3 records are corrupted copies of one business.

Stage 1 is the unchanged pair model, giving p per candidate. For every
candidate c of entity e, the anchors are e's top TOP_J candidates by stage-1 p
(c itself excluded), and the stage-2 columns are:

  tri_nm_simp    max over anchors j of nm4(c, j) * p_j, where nm4 is the name
                 4-gram Jaccard between the two CANDIDATE records (not vs S1)
  tri_nm_p       p_j of that argmax anchor (0 when the max is 0)
  tri_dg_simp    the same with the address digit-token Jaccard
  tri_dg_p       p_j of that argmax anchor
  tri_n_conf_dg  anchors with p_j > 0.5 that share an address digit token with c
  tri_rank       c's rank by stage-1 p among e's candidates (0 = best; ties
                 share the best rank)
  tri_p          c's stage-1 p
  tri_xsrc_top   c and e's top-p candidate come from different sources (S2 vs
                 S3); cross-source agreement is the strong case. 0 for the top.

Leakage: stage-1 p for TRAINING entities must be out-of-fold (train_eval.py
reuses the singleton head's OOF_K folds); in-sample p would make every
training positive's anchors look confident. Validation and test use the full
model's calibrated p, as the singleton head does.

Chains (near-identical candidates of DIFFERENT entities) are the risk: a
branch of the same chain looks like a copy of e's confident candidate. The
similarity here is only ever among e's OWN candidates, so it cannot pull in a
record e did not already retrieve, and the candidate set is unchanged (matches
stay a subset of candidates). A record claimed by two entities is still
settled downstream by disjoint.resolve_conflicts when --disjoint is on.

Cost: anchors are at most TOP_J per entity, so it is O(TOP_J * n_cand) per
entity. Each anchor set is written once into a sorted key -> slot-bitmask
table per chunk of entities, and each candidate's grams (digits) are probed
once, so the work is one searchsorted per candidate token plus TOP_J bit
extractions, never a per-pair Python loop. reference() is that loop, kept as
the oracle features() must match exactly.

TRI=1 turns the second stage on in train_eval.py and predict.py; default off,
which is exactly the previous pipeline. TRI_TOP_J overrides TOP_J (<= 32).
"""
import os

import numpy as np

from strfeatures import _GRAM_SPACE, _gather

ENABLED = os.environ.get("TRI", "0") == "1"
TOP_J = int(os.environ.get("TRI_TOP_J", 10))
CONF = 0.5                   # "confident" anchor for tri_n_conf_dg
NAMES = ["tri_nm_simp", "tri_nm_p", "tri_dg_simp", "tri_dg_p", "tri_n_conf_dg",
         "tri_rank", "tri_p", "tri_xsrc_top"]
_CHUNK = 100_000             # pairs per step (whole entities), bounds RAM


def _groups(q):
    starts = np.flatnonzero(np.r_[True, q[1:] != q[:-1]])
    ends = np.r_[starts[1:], len(q)]
    return starts, ends


def _intersections(off, ids, space, c, rows, g_rows, a_rows, a_g, a_slot, top_j):
    """|set(c[r]) & set(c[a])| for every row r against its own group's anchors.
    -> (inter (len(rows), top_j) int32, candidate set sizes)."""
    av, aln = _gather(off, ids, c[a_rows])
    key = np.repeat(a_g, aln) * space + av
    bit = np.repeat(np.left_shift(np.uint32(1), a_slot.astype(np.uint32)), aln)
    o = np.argsort(key, kind="stable")
    key, bit = key[o], bit[o]
    first = np.flatnonzero(np.r_[True, key[1:] != key[:-1]]) if len(key) else \
        np.zeros(0, np.int64)
    ukey = key[first]
    mask = np.bitwise_or.reduceat(bit, first) if len(first) else bit

    cv, cln = _gather(off, ids, c[rows])
    probe = np.repeat(g_rows, cln) * space + cv
    inter = np.zeros((len(rows), top_j), np.int32)
    if len(ukey) and len(probe):
        pos = np.minimum(np.searchsorted(ukey, probe), len(ukey) - 1)
        hit = np.flatnonzero(ukey[pos] == probe)
        # most probes miss every anchor; only hits are expanded into slots
        m = mask[pos[hit]]
        r = np.repeat(np.arange(len(rows)), cln)[hit]
        for s in range(top_j):
            b = ((m >> np.uint32(s)) & np.uint32(1)).astype(bool)
            inter[:, s] = np.bincount(r[b], minlength=len(rows))
    return inter, cln


class Sets:
    """The two per-record sets features() reads, CSR as in strfeatures.Records:
    name 4-gram codes and address digit-token ids."""
    __slots__ = ("gram_off", "gram_ids", "dig_off", "dig_ids", "dig_space")

    def __init__(self, gram_off, gram_ids, dig_off, dig_ids, dig_space):
        self.gram_off, self.gram_ids = gram_off, gram_ids
        self.dig_off, self.dig_ids = dig_off, dig_ids
        self.dig_space = int(dig_space)


def sets(recs, c=None):
    """Sets of a strfeatures.Records. c=None shares the whole corpus's arrays
    (predict.py, no copy). Otherwise only the records in c are kept, so the
    corpus can be freed: -> (Sets, c renumbered into it)."""
    space = max(len(recs.vocab), 1)
    if c is None:
        return Sets(recs.gram_off, recs.gram_ids, recs.dig_off, recs.dig_ids, space)
    u, c_local = np.unique(np.asarray(c, np.int64), return_inverse=True)
    parts = []
    for kind in ("gram", "dig"):
        off, ids = getattr(recs, kind + "_off"), getattr(recs, kind + "_ids")
        v, ln = _gather(off, ids, u)
        o = np.zeros(len(u) + 1, np.int64)
        np.cumsum(ln, out=o[1:])
        parts += [o, v.astype(np.int32)]
    return Sets(*parts, space), c_local.astype(np.int64)


def concat(pairs):
    """[(Sets, c_local), ...] from different countries -> (Sets, c_local) with
    the records stacked. Digit ids of two countries' vocabularies never meet:
    anchors and candidates are always of the same entity, hence country."""
    offs, ids = {"gram": [np.zeros(1, np.int64)], "dig": [np.zeros(1, np.int64)]}, \
        {"gram": [], "dig": []}
    cs, base, seen = [], 0, {}
    for s, c in pairs:
        if id(s) not in seen:          # the same Sets twice is stacked once
            seen[id(s)] = base
            for kind in ("gram", "dig"):
                o = getattr(s, kind + "_off")
                offs[kind].append(o[1:].astype(np.int64) + offs[kind][-1][-1])
                ids[kind].append(getattr(s, kind + "_ids"))
            base += len(s.gram_off) - 1
        cs.append(c + seen[id(s)])
    space = max([s.dig_space for s, _ in pairs] or [1])
    return (Sets(np.concatenate(offs["gram"]), np.concatenate(ids["gram"]),
                 np.concatenate(offs["dig"]), np.concatenate(ids["dig"]), space),
            np.concatenate(cs) if cs else np.zeros(0, np.int64))


def features(q, c, p, is_s3, recs, top_j=None):
    """q: entity index per pair, sorted ascending. c: record index per pair
    into recs. p: stage-1 calibrated probability per pair. is_s3: candidate
    source. recs: a Sets (sets()), or strfeatures.Records of the corpus.
    -> float32 (n_pairs, len(NAMES))."""
    top_j = TOP_J if top_j is None else top_j
    if not isinstance(recs, Sets):
        recs = sets(recs)
    if not 1 <= top_j <= 32:
        raise ValueError("top_j must be in 1..32 (anchor bitmask is uint32)")
    q = np.asarray(q, np.int64)
    c = np.asarray(c, np.int64)
    p = np.asarray(p, np.float64)
    is_s3 = np.asarray(is_s3, bool)
    n = len(q)
    out = np.zeros((n, len(NAMES)), np.float32)
    if n == 0:
        return out
    if np.any(q[1:] < q[:-1]):
        raise ValueError("q must be sorted by entity")
    starts, ends = _groups(q)
    G = len(starts)
    gid = np.repeat(np.arange(G), ends - starts)

    # Position by stage-1 p within the entity: stable, as every decide() is.
    order = np.lexsort((-p, gid))
    pos = np.empty(n, np.int64)
    pos[order] = np.arange(n) - starts[gid[order]]
    ps, go = p[order], gid[order]
    new = np.r_[True, (ps[1:] != ps[:-1]) | (go[1:] != go[:-1])]
    run = np.maximum.accumulate(np.where(new, np.arange(n), 0))
    rank = np.empty(n, np.int64)
    rank[order] = run - starts[go]

    # Anchor table per entity: slot = position among the top_j.
    a_all = np.flatnonzero(pos < top_j)
    Pa = np.zeros((G, top_j))
    Pa[gid[a_all], pos[a_all]] = p[a_all]
    valid = np.zeros((G, top_j), bool)
    valid[gid[a_all], pos[a_all]] = True
    top_s3 = np.zeros(G, bool)
    t = a_all[pos[a_all] == 0]
    top_s3[gid[t]] = is_s3[t]
    gram_len = np.diff(recs.gram_off).astype(np.int64)
    dig_len = np.diff(recs.dig_off).astype(np.int64)
    Lnm = np.zeros((G, top_j), np.int64)
    Ldg = np.zeros((G, top_j), np.int64)
    Lnm[gid[a_all], pos[a_all]] = gram_len[c[a_all]]
    Ldg[gid[a_all], pos[a_all]] = dig_len[c[a_all]]

    col = {k: i for i, k in enumerate(NAMES)}
    out[:, col["tri_rank"]] = rank
    out[:, col["tri_p"]] = p
    out[:, col["tri_xsrc_top"]] = (is_s3 != top_s3[gid]) & (pos != 0)

    # Chunks of whole entities.
    g_lo = 0
    while g_lo < G:
        g_hi = int(np.searchsorted(starts, starts[g_lo] + _CHUNK, "right"))
        g_hi = max(g_hi, g_lo + 1)
        lo, hi = starts[g_lo], ends[g_hi - 1]
        rows = np.arange(lo, hi)
        g_rows = gid[rows]
        a = a_all[(a_all >= lo) & (a_all < hi)]
        o = out[lo:hi]
        sel = valid[g_rows]
        # never c against itself
        own = pos[rows] < top_j
        sel[np.flatnonzero(own), pos[rows][own]] = False
        Pr = Pa[g_rows]

        for kind, L, space, (c_simp, c_p) in (
                ("gram", Lnm, _GRAM_SPACE, (col["tri_nm_simp"], col["tri_nm_p"])),
                ("dig", Ldg, recs.dig_space,
                 (col["tri_dg_simp"], col["tri_dg_p"]))):
            inter, cln = _intersections(getattr(recs, kind + "_off"),
                                        getattr(recs, kind + "_ids"), space, c,
                                        rows, g_rows, a, gid[a], pos[a], top_j)
            den = cln[:, None] + L[g_rows] - inter
            jac = np.zeros(den.shape)
            np.divide(inter, den, out=jac, where=den > 0)
            val = np.where(sel, jac * Pr, -1.0)
            arg = val.argmax(1)
            best = val[np.arange(len(rows)), arg]
            has = best > 0
            o[:, c_simp] = np.where(has, best, 0.0)
            o[:, c_p] = np.where(has, Pr[np.arange(len(rows)), arg], 0.0)
            if kind == "dig":
                o[:, col["tri_n_conf_dg"]] = (sel & (Pr > CONF) & (inter > 0)).sum(1)
        g_lo = g_hi
    return out


# ---- reference implementation (per-pair loop, the oracle) -------------------

def _set(off, ids, j):
    return set(ids[off[j]:off[j + 1]].tolist())


def reference(q, c, p, is_s3, recs, top_j=None):
    top_j = TOP_J if top_j is None else top_j
    q, c, p = np.asarray(q), np.asarray(c), np.asarray(p, np.float64)
    out = np.zeros((len(q), len(NAMES)), np.float32)
    for s, e in zip(*_groups(q)) if len(q) else ():
        idx = list(range(s, e))
        srt = sorted(idx, key=lambda r: -p[r])          # stable
        pos = {r: k for k, r in enumerate(srt)}
        anchors = srt[:top_j]
        for r in idx:
            rank = min(k for k, x in enumerate(srt) if p[x] == p[r])
            feats = []
            for kind in ("gram", "dig"):
                off, ids = getattr(recs, kind + "_off"), getattr(recs, kind + "_ids")
                a_set = _set(off, ids, c[r])
                best, bp = 0.0, 0.0
                for j in anchors:
                    if j == r:
                        continue
                    b = _set(off, ids, c[j])
                    u = len(a_set | b)
                    v = (len(a_set & b) / u if u else 0.0) * p[j]
                    if v > best:
                        best, bp = v, p[j]
                feats += [best, bp]
            dg = _set(recs.dig_off, recs.dig_ids, c[r])
            n_conf = sum(1 for j in anchors if j != r and p[j] > CONF
                         and dg & _set(recs.dig_off, recs.dig_ids, c[j]))
            top = srt[0]
            out[r] = feats + [n_conf, rank, p[r],
                              float(bool(is_s3[r]) != bool(is_s3[top]) and pos[r] != 0)]
    return out


# ---- self-test ---------------------------------------------------------------

_WORDS = ("sharma kumar gupta verma medical traders textiles stores agency "
          "enterprises general kirana bakery sweets hardware electricals motors "
          "pharma steel plastics foods dairy jewellers opticals mobile").split()


def _synthetic(n_ent, rng):
    """Entities with three true matches each (corrupted copies of one business)
    plus distractors. Stage-1 p is drawn so the 3rd true match is exactly as
    uncertain as the entity's best distractors: p alone cannot separate them.
    -> names, addrs, is_s3 per record; q, c, p, y per pair."""
    names, addrs, s3 = [], [], []
    q, c, p, y = [], [], [], []

    def rec(nm, ad, src3):
        names.append(nm); addrs.append(ad); s3.append(src3)
        return len(names) - 1

    def corrupt(nm):
        t = nm.split()
        i = int(rng.integers(len(t)))
        w = t[i]
        if len(w) > 4 and rng.random() < 0.5:
            w = w[:-1]                                   # drop a letter
        elif len(w) > 3:
            w = w[:3]                                    # abbreviate
        t[i] = w
        return " ".join(t)

    for e in range(n_ent):
        base = " ".join(rng.choice(_WORDS, 3, replace=False))
        num = f"{rng.integers(1, 999)} {rng.integers(100000, 999999)}"
        ad = f"{num} {rng.choice(_WORDS)} road"
        true = [rec(corrupt(base), ad, False), rec(corrupt(base), ad, True),
                rec(corrupt(corrupt(base)), ad, bool(rng.random() < 0.5))]
        tp = [rng.uniform(0.85, 0.99), rng.uniform(0.6, 0.85), rng.uniform(0.25, 0.5)]
        for r, pr in zip(true, tp):
            q.append(e); c.append(r); p.append(pr); y.append(1)
        for _ in range(12):
            nm = " ".join(rng.choice(_WORDS, 3, replace=False))
            dad = f"{rng.integers(1, 999)} {rng.integers(100000, 999999)} road"
            r = rec(nm, dad, bool(rng.random() < 0.5))
            q.append(e); c.append(r); p.append(rng.uniform(0.02, 0.5)); y.append(0)
    return (names, addrs, np.array(s3), np.array(q), np.array(c),
            np.array(p), np.array(y))


def _self_test():
    import lightgbm as lgb

    import strfeatures
    rng = np.random.default_rng(0)
    names, addrs, s3, q, c, p, y = _synthetic(600, rng)
    recs = strfeatures.precompute_records(names, addrs)
    is_s3 = s3[c]

    # 1. vectorised == loop, including top_j smaller than the entity and ties
    p_t = np.round(p, 1)
    for tj in (1, 3, 10):
        for pp in (p, p_t):
            a = features(q[:600], c[:600], pp[:600], is_s3[:600], recs, tj)
            b = reference(q[:600], c[:600], pp[:600], is_s3[:600], recs, tj)
            assert np.array_equal(a, b), (tj, np.abs(a - b).max(0))
    # tiny chunks: entity boundaries must not change a single value
    full = features(q, c, p, is_s3, recs)
    assert np.array_equal(_full(q, c, p, is_s3, recs, 37), full)
    # record subsets (train_eval frees the corpus) and two stacked "countries"
    m = q < 300
    s, cl = concat([sets(recs, c[m]), sets(recs, c[~m])])
    assert np.array_equal(features(q, cl, p, is_s3, s), full)
    # one country split into halves, as train_eval's validation does
    s1, c1 = sets(recs, c)
    s, cl = concat([(s1, c1[m]), (s1, c1[~m])])
    assert len(s.gram_off) == len(s1.gram_off)
    assert np.array_equal(features(q, np.r_[cl], p, is_s3, s), full)
    print("  features() == reference() bit for bit (top_j 1/3/10, ties, chunks)")

    # 2. the 3rd true match looks like the 1st; same-p distractors do not
    T = features(q, c, p, is_s3, recs)
    col = {k: i for i, k in enumerate(NAMES)}
    third = np.zeros(len(q), bool)
    third[2::15] = True
    band = (y == 0) & (p > 0.25)          # distractors in the 3rd match's p band
    s_third = T[third, col["tri_nm_simp"]].mean()
    s_band = T[band, col["tri_nm_simp"]].mean()
    print(f"  tri_nm_simp: 3rd true match {s_third:.3f}, same-p distractors "
          f"{s_band:.3f}")
    assert s_third > 3 * s_band
    assert T[third, col["tri_n_conf_dg"]].mean() > 1.5 > T[band, col["tri_n_conf_dg"]].mean()

    # 3. stacking lifts the 3rd match above same-p distractors on held-out
    # entities, where stage-1 p alone is a coin flip between them
    tr = q < 400
    params = dict(objective="binary", learning_rate=0.1, num_leaves=15,
                  min_data_in_leaf=20, verbose=-1, num_threads=2)
    X1 = p[:, None]
    X2 = np.hstack([X1, T])
    m1 = lgb.train(params, lgb.Dataset(X1[tr], y[tr]), 60)
    m2 = lgb.train(params, lgb.Dataset(X2[tr], y[tr]), 60)
    ev = ~tr & (third | band)

    def auc(s):
        pos, neg = s[y[ev] == 1], s[y[ev] == 0]
        return (pos[:, None] > neg[None, :]).mean()
    a1, a2 = auc(m1.predict(X1[ev])), auc(m2.predict(X2[ev]))
    print(f"  held-out AUC, 3rd match vs same-p distractors: stage 1 {a1:.3f}, "
          f"stage 2 {a2:.3f}")
    assert a1 < 0.7 and a2 > 0.95
    print("triangulate self-test PASSED")


def _full(q, c, p, is_s3, recs, chunk):
    global _CHUNK
    old, _CHUNK = _CHUNK, chunk
    try:
        return features(q, c, p, is_s3, recs)
    finally:
        _CHUNK = old


if __name__ == "__main__":
    _self_test()
