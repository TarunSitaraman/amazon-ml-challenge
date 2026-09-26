"""Hard negatives from the disjointness property.

Ground truth has zero S2/S3 record re-use (AUDIT.md): every record belongs to
at most one S1 entity. So for a record r owned by entity e', the pair (e, r) is
a certified non-match for every e != e', and the owner tells us WHY it is a
negative:

  blocker    r is a distractor (owned by nobody), or owned by an unrelated entity
  sibling    r is a gold match of one of e's nearest S1 neighbours by name
  same_name  r is a gold match of an S1 entity with e's exact normalised name
             (a chain branch: provably a different business, since S1 is
             deduplicated, so only the address can tell them apart)

Only negatives inside e's own capped candidate set are used. A pair the blocker
never retrieves is never scored at inference, and its features (rank, n_cand,
channel scores) would describe a situation the model never meets. So this does
not invent pairs: it finds which of the existing negatives are the hard ones
and reweights them to a configured mix. Two knobs reach further, still in
distribution:

  HARD_NEG_ADD=1  adds each training entity's nearest S1 neighbour as an extra
                  training entity, so both sides of a confusable pair are
                  blocked and featurised normally. That is where extra sibling
                  and same-name negatives come from.
  coverage()      reports how many of the nearest sibling's gold records are in
                  e's candidates at all. The ones that are not were lost to
                  blocking or to cap_candidates, and no reweighting reaches them.

Configuration (env, default off so train_eval.py is unchanged):
  HARD_NEG="0.5,0.25,0.25"   target share of the NEGATIVE weight for
                             blocker, sibling, same_name. Unset/off = disabled.
  HARD_NEG=natural           no reweighting (all weights 1): categorise, report
                             and, with HARD_NEG_ADD=1, add neighbour entities.
                             Separates the effect of adding from reweighting.
  HARD_NEG_K=3               name neighbours per entity that count as siblings
  HARD_NEG_ADD=0             1 = also train on each entity's nearest neighbour
  HARD_NEG_MAX_W=20          cap on any single negative's weight
"""
import os

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

import blocking
from metric import choose_k, f05

BLOCKER, SIBLING, SAME_NAME = 0, 1, 2
CATS = ["blocker", "sibling", "same_name"]
POSITIVE = -1


def config():
    """-> dict or None (disabled). Parsed once per run from the environment."""
    raw = os.environ.get("HARD_NEG", "").strip().lower()
    if raw in ("", "0", "off", "false", "no"):
        return None
    cfg = {"k": int(os.environ.get("HARD_NEG_K", 3)),
           "add": os.environ.get("HARD_NEG_ADD", "0") == "1",
           "max_w": float(os.environ.get("HARD_NEG_MAX_W", 20)), "mix": None}
    if raw == "natural":
        return cfg
    mix = np.array([float(x) for x in raw.split(",")], np.float64)
    if len(mix) != len(CATS) or (mix < 0).any() or mix.sum() <= 0:
        raise ValueError(f"HARD_NEG={raw!r}: want 3 non-negative shares "
                         f"({','.join(CATS)})")
    cfg["mix"] = mix / mix.sum()
    return cfg


def record_owner(s1_ids, c_ids, gtm):
    """Owner S1 index per corpus record, -1 for distractors.

    Zero re-use means each record appears in at most one entity's gold list, so
    a single owner per record is well defined. If that ever failed, the later
    entity would win and the counts printed below would show it.
    """
    rec, ent = [], []
    for i, eid in enumerate(s1_ids):
        g = gtm.get(eid)
        if g:
            for r in g.split(","):
                if r:
                    rec.append(r)
                    ent.append(i)
    owner = np.full(len(c_ids), -1, np.int64)
    if rec:
        pos = pc.index_in(pa.array(rec, pa.string()),
                          value_set=pa.array(c_ids, pa.string()))
        pos = pc.fill_null(pos, -1).to_numpy(zero_copy_only=False)
        ok = pos >= 0
        owner[pos[ok].astype(np.int64)] = np.asarray(ent, np.int64)[ok]
    return owner


def name_ids(s1_norm):
    """Dense id per distinct normalised name; empty names get unique ids so two
    unnamed entities are never 'same name'."""
    ids = np.asarray(pa.array(s1_norm, pa.string()).dictionary_encode()
                     .indices.to_numpy(zero_copy_only=False), np.int64)
    empty = np.flatnonzero(np.array([not n for n in s1_norm], bool))
    ids[empty] = ids.max(initial=-1) + 1 + np.arange(len(empty))
    return ids


def neighbours(index, s1_norm, query, k):
    """Top-k other S1 entities per query entity by IDF cosine on the name.

    index: blocking.build_index(s1_norm), built once per country.
    -> (len(query) x k) int64 S1 indices, -1 padded, best first. The same C2
    retrieval blocking uses, pointed at S1 itself.
    """
    query = np.asarray(query, np.int64)
    out = np.full((len(query), k), -1, np.int64)
    if not len(query):
        return out
    qi, cj, sc = blocking.c2_query([s1_norm[i] for i in query], index, top_k=k + 1)
    keep = cj != query[qi]                                  # drop self
    qi, cj, sc = qi[keep], cj[keep], sc[keep]
    # best first within each query, ties by S1 index so the result is stable
    order = np.lexsort((cj, -sc, qi))
    qi, cj = qi[order], cj[order]
    rank = np.arange(len(qi)) - np.searchsorted(qi, qi, "left")
    m = rank < k
    out[qi[m], rank[m]] = cj[m]
    return out


def categorise(y, ent, c, owner, nid, nbr):
    """Category per pair: POSITIVE, or the negative's BLOCKER/SIBLING/SAME_NAME.

    y: labels. ent: S1 index of each pair's entity. c: corpus index.
    nbr: (n_S1 x k) neighbour table, -1 where there is none (or the entity
    was not queried).
    """
    cat = np.full(len(y), BLOCKER, np.int8)
    cat[y > 0] = POSITIVE
    o = owner[c]
    neg = (y == 0) & (o >= 0)
    same = neg & (nid[np.maximum(o, 0)] == nid[ent])
    cat[same] = SAME_NAME
    rest = np.flatnonzero(neg & ~same)
    if len(rest):
        cat[rest[(nbr[ent[rest]] == o[rest, None]).any(1)]] = SIBLING
    return cat


def mix_weights(y, cat, cfg, base=None):
    """Per-pair training weights: positives 1, negatives reweighted so each
    category's share of the total negative weight is cfg["mix"].

    base: optional per-pair weights to start from, the NEG_KEEP importance
    weights (bigtrain.py). Counts are then weighted counts, i.e. of the
    negatives before subsampling, and each negative's weight is base times its
    category's multiplier, so the mix is hit on the unsampled distribution.

    The total negative weight stays equal to the negative count (before the
    cap), so the positive:negative balance the model sees is unchanged and only
    the composition of the negatives moves. A configured category with no pairs
    has its share redistributed over the others. Weights are capped at
    cfg["max_w"] so a handful of rare hard negatives cannot dominate.
    """
    w = (np.ones(len(y), np.float64) if base is None
         else np.asarray(base, np.float64).copy())
    if cfg["mix"] is None:
        return w
    neg = cat != POSITIVE
    if base is None:
        n_neg = neg.sum()
        counts = np.array([(cat == k).sum() for k in range(len(CATS))], np.float64)
    else:
        n_neg = w[neg].sum()
        counts = np.array([w[cat == k].sum() for k in range(len(CATS))])
    mix = np.where(counts > 0, cfg["mix"], 0.0)
    if n_neg == 0 or mix.sum() == 0:
        return w
    mix = mix / mix.sum()
    per = np.divide(mix * n_neg, counts, out=np.zeros_like(mix), where=counts > 0)
    per = np.minimum(per, cfg["max_w"])
    w[neg] = per[cat[neg]] if base is None else w[neg] * per[cat[neg]]
    return w


def describe(y, cat, w=None, label="training negatives"):
    neg = cat != POSITIVE
    n = max(int(neg.sum()), 1)
    parts = []
    for k, name in enumerate(CATS):
        m = cat == k
        s = f"{name} {m.sum():,} ({m.sum()/n:.1%}"
        if w is not None:
            s += f" -> {w[m].sum()/max(w[neg].sum(), 1e-9):.1%} of weight"
        parts.append(s + ")")
    print(f"  {label}: " + ", ".join(parts))


def coverage(ent, q, c, owner, nbr):
    """How many of each entity's nearest neighbour's gold records are in that
    entity's candidate set at all. ent: S1 index per entity (the split's idx);
    q, c: its candidate pairs. nbr: neighbour table, column 0 = nearest.
    -> (in_candidates, total). The rest were lost to blocking or the cap.
    """
    n1 = nbr[ent, 0]
    gold_n = np.bincount(owner[owner >= 0], minlength=len(nbr))
    total = int(gold_n[n1[n1 >= 0]].sum())
    hit = int(((owner[c] == n1[q]) & (n1[q] >= 0)).sum())
    return hit, total


# ---- A/B evaluation --------------------------------------------------------

def ab_row(label, p, qva, cva, truth, n_cal, cat, chain):
    """Adaptive-k with the product P(n=0) on the held-out half, per model.

    p: calibrated validation scores. cat: validation pair categories (their
    owners are known from ground truth, so false positives can be attributed).
    chain: bool per validation entity, name_dup > 0.
    Prints macro F0.5, pair precision and recall, the same on chain entities,
    and where the false positives went. Returns macro F0.5 and precision.
    """
    n_ent = len(truth)
    starts = np.flatnonzero(np.r_[True, qva[1:] != qva[:-1]]) if len(qva) else []
    ends = np.r_[starts[1:], len(qva)] if len(qva) else []
    acc = np.zeros(len(p), bool)
    for s, e in zip(starts, ends):
        o = s + np.argsort(-p[s:e], kind="stable")
        acc[o[:choose_k(p[o], float(np.prod(1.0 - p[s:e])))]] = True
    ev_pair = qva >= n_cal
    a = acc & ev_pair
    ok = a & (cat == POSITIVE)
    k = np.bincount(qva[a], minlength=n_ent)
    c = np.bincount(qva[ok], minlength=n_ent)
    n_true = np.array([len(t) for t in truth])
    ev = np.arange(n_cal, n_ent)
    f = f05(c[ev], n_true[ev], k[ev])
    prec = ok.sum() / max(a.sum(), 1)
    rec = c[ev].sum() / max(n_true[ev].sum(), 1)
    ch = chain[ev]
    prec_ch = c[ev][ch].sum() / max(k[ev][ch].sum(), 1)
    fp = a & (cat != POSITIVE)
    fps = ", ".join(f"{n} {(fp & (cat == i)).sum():,}" for i, n in enumerate(CATS))
    print(f"  {label:16s} macro F0.5 = {f.mean():.4f}   precision = {prec:.4f}   "
          f"recall = {rec:.4f}")
    print(f"  {'':16s} chain entities: macro F0.5 = {f[ch].mean() if ch.any() else float('nan'):.4f}   "
          f"precision = {prec_ch:.4f}   ({ch.sum():,} entities)")
    print(f"  {'':16s} false positives: {fp.sum():,} ({fps})")
    return f.mean(), prec
