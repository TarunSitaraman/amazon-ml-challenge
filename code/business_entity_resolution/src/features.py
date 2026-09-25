"""Pair features.

Two groups, both fully vectorised:
  per-channel similarity  -- agreement between independent channels is the signal
                             a single fused score destroys
  entity-level context    -- rank, margin and candidate density. These carry the
                             cardinality information the decision layer needs, and
                             a pair's score only means something relative to its
                             rivals for the same entity.
"""
import numpy as np

from blocking import CHANNELS

NAMES = (CHANNELS +
         ["n_chan", "max_chan", "sum_chan",
          "rank", "n_cand", "gap_top", "ratio_top", "is_s3", "name_dup"])


def name_dup_counts(s1_names):
    """How many OTHER S1 entities share this entity's normalised name.

    S1 is deduplicated, so a shared name means a genuinely different business --
    39.28% of S1 entities are in such a collision group (chains, franchises).
    For those the name carries no identity and only the address can discriminate.
    Telling the model which entities those are lets it learn that reweighting
    instead of having to infer it from the name features alone.

    Computed per split on that split's own S1, so it is transductive and needs
    no labels -- it works identically on the unseen country.
    """
    from collections import Counter
    c = Counter(s1_names)
    return np.array([c[n] - 1 if n else 0 for n in s1_names], np.float32)


def build(q, chan, is_s3, name_dup=None):
    """q: entity index per pair (sorted ascending). chan: (n_pairs x n_channels)."""
    n_chan = (chan > 0).sum(1).astype(np.float32)
    max_chan = chan.max(1)
    sum_chan = chan.sum(1)

    # entity groups: q arrives sorted, so boundaries are where it changes
    starts = np.flatnonzero(np.r_[True, q[1:] != q[:-1]])
    ends = np.r_[starts[1:], len(q)]
    n_cand = np.repeat((ends - starts).astype(np.float32), ends - starts)
    grp_max = np.repeat(np.maximum.reduceat(max_chan, starts), ends - starts)

    # rank within entity by max channel score (0 = best)
    rank = np.empty(len(q), np.float32)
    for s, e in zip(starts, ends):
        rank[s:e] = np.argsort(np.argsort(-max_chan[s:e]))

    gap_top = grp_max - max_chan
    ratio_top = max_chan / np.maximum(grp_max, 1e-6)
    dup = (np.zeros(len(q), np.float32) if name_dup is None
           else np.asarray(name_dup, np.float32)[q])
    return np.column_stack([chan, n_chan, max_chan, sum_chan, rank, n_cand,
                            gap_top, ratio_top, is_s3.astype(np.float32),
                            dup]).astype(np.float32)
