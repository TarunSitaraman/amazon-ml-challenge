"""Per-country transductive calibration of the entity-level decision.

France has zero training labels. On the test set the pipeline predicts no match
for 11.53% of French entities against 5.58% of US ones, and the US figure equals
the train singleton rate exactly. A wrong abstention scores exactly 0, and the
leaderboard gap (offline 0.8615 -> LB 0.844) fits France scoring ~0.75 while
US/India transfer as measured (EXPERIMENTS.md, "Leaderboard calibration").

Without labels we cannot fit France's probabilities, but we can make France's
predicted cardinality distribution look like the labelled countries': the same
singleton rate and the same mean k among entities that get a match.

Why targeting the singleton rate is defensible rather than a fudge: the test
set is ~23% denser than train in every country (RESEARCH.md section 4). The
density analysis favours reading B, "same cardinality, more distractors", over
reading A, "more matches per entity": under B the true match-count distribution,
and in particular the singleton rate, is unchanged across splits, and the US
test predictions reproducing the train singleton rate exactly is what B
predicts. So a country whose predicted singleton rate is double everyone
else's is a country whose scores are miscalibrated, not one with twice the
singletons. RESEARCH.md section 4 records A vs B as unresolved; under A this
calibration would be wrong, which is why it is a flag, default off, and why it
calibrates to the labelled countries' PREDICTED distribution on the same test
run (which already bakes in the density shift) rather than to the train
ground-truth histogram.

Two knobs, both per country, solved by bisection on that country's unlabelled
test predictions:

  s  scales P(n=0), from the product rule or the singleton head alike:
     P(n=0) <- min(1, s * P(n=0)). metric.choose_k reads P(n=0) only in the
     k=0 -> k=1 decision, so s moves the singleton rate and nothing else.
  t  scales the stopping bar: accept the next candidate iff p > t * 0.8 * F_k
     (metric.choose_k's bar_scale). It is never consulted in the k=0 decision,
     so it moves mean k among non-singletons and leaves the singleton rate
     alone.

The two are therefore fitted one after the other: s to the reference singleton
rate, then t to the reference E[k | k >= 1]. Both are monotone, so bisection
finds them. Entities with no candidates are singletons whatever s is, so a
target below that floor cannot be reached; fit() reports what it got.

The calibrated P(n=0) is passed on to disjoint.resolve_conflicts too, so
--disjoint composes. The fit is to choose_k's output, before resolve_conflicts,
and the reference is the reference countries' choose_k output before
resolve_conflicts (predict.py stores it in each partial's sidecar), so both
sides then go through the same resolve step. resolve_conflicts(redecide=True)
would re-run choose_k without the bar scale; redecide is off in predict.py.

Run this module directly for the synthetic self-test.
"""
import numpy as np

from metric import FIRST_ACCEPT_FACTOR, choose_k

HIST_BINS = 12            # k = 0..10, 11+ in the last bin, for printing only
S_RANGE = (1e-4, 1e4)     # P(n=0) scale, clamped to this
T_RANGE = (0.1, 10.0)     # stopping-bar scale, clamped to this


def k_hist(k):
    """Counts of k = 0, 1, ..., max(k): the cardinality histogram."""
    k = np.asarray(k, np.int64)
    return np.bincount(k, minlength=1).tolist() if k.size else [0]


def pool(hists):
    """Sum several cardinality histograms (lists of counts)."""
    out = [0] * max((len(h) for h in hists), default=1)
    for h in hists:
        for i, v in enumerate(h):
            out[i] += int(v)
    return out


def summary(hist):
    """entities, singleton rate, mean k, mean k among k >= 1."""
    h = np.asarray(hist, float)
    n = h.sum()
    k = np.arange(len(h))
    nz = h[1:].sum()
    return {"entities": int(n),
            "singleton_rate": float(h[0] / n) if n else 0.0,
            "mean_k": float((h * k).sum() / n) if n else 0.0,
            "mean_k_nonzero": float((h * k).sum() / nz) if nz else 0.0}


def fmt_header():
    bins = "".join(f"{('k=' + str(i)) if i < HIST_BINS - 1 else f'{i}+':>7}"
                   for i in range(HIST_BINS))
    return f"    {'':22s}{'entities':>10}{'P(k=0)':>9}{'mean k':>8}{'k|k>=1':>8}{bins}"


def fmt_row(label, hist):
    s = summary(hist)
    h = np.asarray(hist, float)
    b = np.zeros(HIST_BINS)
    b[:min(len(h), HIST_BINS)] = h[:HIST_BINS]
    b[-1] += h[HIST_BINS:].sum()
    share = b / max(h.sum(), 1)
    return (f"    {label:22s}{s['entities']:>10,}{s['singleton_rate']:>9.2%}"
            f"{s['mean_k']:>8.3f}{s['mean_k_nonzero']:>8.3f}"
            + "".join(f"{x:>7.1%}" for x in share))


def product_p_zero(q, p, n_entities):
    """prod(1 - p) per entity, the same form disjoint.resolve_conflicts uses
    (clipped, so no entity has P(n=0) exactly 0 and s can always move it)."""
    with np.errstate(divide="ignore"):
        return np.exp(np.bincount(q, np.log1p(-np.clip(p, 0, 1 - 1e-12)),
                                  n_entities))


class _Prepared:
    """Per-entity quantities that make k(s, t) a vectorised function.

    choose_k stops at the first j >= 1 with p_j <= t * bar_j, where
    bar_j = 0.8 * 1.25 * c_j / (0.25 * n_hat + j) and c_j = p_0 + ... + p_{j-1}.
    With r_j = p_j / bar_j and m_j its running minimum over 1..j,
    k(t) = 1 + #{j >= 1 : m_j > t} for an entity that accepts its first pair."""

    def __init__(self, q, p, p_zero, recall, n_entities):
        q = np.asarray(q, np.int64)
        p = np.asarray(p, float)
        o = np.lexsort((-p, q))              # q, then p descending (stable)
        q, p = q[o], p[o]
        starts = np.flatnonzero(np.r_[True, q[1:] != q[:-1]]) if q.size else \
            np.zeros(0, np.int64)
        size = np.diff(np.r_[starts, q.size])
        pos = np.arange(q.size) - np.repeat(starts, size)
        n_hat = np.maximum(np.bincount(q, p, n_entities) / max(recall, 1e-6), 1e-9)
        c = np.cumsum(p) - np.repeat(np.cumsum(p)[starts] - p[starts], size)
        c -= p                                # c_j: sum of the pairs before j
        bar = 0.8 * (1.25 * c / (0.25 * n_hat[q] + pos))
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(bar > 0, p / np.where(bar > 0, bar, 1.0),
                         np.where(p > 0, np.inf, 0.0))
        r[pos == 0] = np.inf
        # running minimum within each entity, one pass
        m = r.copy()
        for s0, n in zip(starts, size):
            if n > 1:
                np.minimum.accumulate(m[s0:s0 + n], out=m[s0:s0 + n])
        self.q_tail, self.m_tail = q[pos >= 1], m[pos >= 1]
        self.n = n_entities
        self.has = np.zeros(n_entities, bool)
        self.has[q[starts]] = True
        self.first = np.zeros(n_entities)
        self.first[q[starts]] = p[starts]
        self.p_zero = np.asarray(p_zero, float)

    def abstain(self, s):
        return ~self.has | (self.first * FIRST_ACCEPT_FACTOR
                            <= np.minimum(1.0, s * self.p_zero))

    def k(self, s, t):
        knz = 1 + np.bincount(self.q_tail, self.m_tail > t, self.n).astype(np.int64)
        return np.where(self.abstain(s), 0, knz)


def _gap_point(lo, hi):
    """A point strictly inside (lo, hi), far from both ends in relative terms,
    so float rounding in choose_k's own comparison cannot flip a decision."""
    if not np.isfinite(hi):
        return lo * 2.0 if lo > 0 else 1.0
    if lo <= 0:
        return hi / 2.0
    return float(np.sqrt(lo * hi))


def _solve(v, want, rng):
    """x such that #(v <= x) is as close to `want` as the breakpoints v allow,
    found by bisection (binary search) over the sorted breakpoints, then placed
    mid-gap. Both counts here are step functions of the scale, so this is the
    exact limit of bisecting on the scale itself, without a breakpoint ending up
    within float rounding of the answer. Tied breakpoints (isotonic output is a
    step function) move together, so the nearer side of a tie wins. Clamped to
    rng. -> (x, clamped)."""
    v = np.sort(np.asarray(v, float))
    m = int(np.clip(round(want), 0, len(v)))
    if 0 < m < len(v) and v[m - 1] == v[m]:
        lo = int(np.searchsorted(v, v[m], "left"))      # binary search
        hi = int(np.searchsorted(v, v[m], "right"))
        m = lo if abs(lo - want) <= abs(hi - want) else hi
    below = v[m - 1] if m > 0 else 0.0
    above = v[m] if m < len(v) else np.inf
    x = _gap_point(below, above)
    xc = float(np.clip(x, *rng))
    return xc, xc != x


def fit(q, p, p_zero, recall, ref_hist, n_entities):
    """Solve (s, t) so this country's choose_k cardinality matches ref_hist.

    q, p: the country's scored pairs (p after Sinkhorn if that runs), entity
    index q in [0, n_entities). p_zero: uncalibrated P(n=0) per entity (head or
    product rule). ref_hist: pooled pre-resolve cardinality histogram of the
    reference countries. Returns (s, t, before_hist, warnings)."""
    ref = summary(ref_hist)
    prep = _Prepared(q, p, p_zero, recall, n_entities)
    before = k_hist(prep.k(1.0, 1.0))
    warn = []

    # s: an entity with candidates abstains iff first * FAF <= s * P(n=0), i.e.
    # iff s >= its breakpoint first * FAF / P(n=0). Entities without
    # candidates abstain whatever s is.
    h = prep.has
    with np.errstate(divide="ignore"):
        brk = prep.first[h] * FIRST_ACCEPT_FACTOR / prep.p_zero[h]
    floor = int((~h).sum())
    want = ref["singleton_rate"] * n_entities - floor
    if want < 0:
        warn.append(f"{floor / max(n_entities, 1):.2%} of entities have no "
                    f"candidates, above the {ref['singleton_rate']:.2%} target")
    s, clamped = _solve(brk, want, S_RANGE)
    if clamped:
        warn.append(f"P(n=0) scale clamped to {s:g}")

    # t: a non-abstaining entity takes 1 + #{j >= 1 : m_j > t} candidates, so
    # sum(k) over them is n_nz + #(m_tail > t) = n_nz + len - #(m_tail <= t).
    nz = ~prep.abstain(s)
    t = 1.0
    if nz.any():
        mt = prep.m_tail[nz[prep.q_tail]]
        n_nz = int(nz.sum())
        want = len(mt) - (ref["mean_k_nonzero"] - 1.0) * n_nz
        t, clamped = _solve(mt, want, T_RANGE)
        if clamped:
            warn.append(f"stopping-bar scale clamped to {t:g}")
    got = summary(k_hist(prep.k(s, t)))
    if (abs(got["singleton_rate"] - ref["singleton_rate"]) > 0.005
            or abs(got["mean_k_nonzero"] - ref["mean_k_nonzero"]) > 0.05):
        warn.append(f"reference not reached: P(k=0) {got['singleton_rate']:.2%} "
                    f"vs {ref['singleton_rate']:.2%}, k|k>=1 "
                    f"{got['mean_k_nonzero']:.3f} vs {ref['mean_k_nonzero']:.3f}")
    return s, t, before, warn


def apply(q, p, p_zero, s, t, recall=1.0):
    """Accepted mask from choose_k with P(n=0) scaled by s and bar by t, and
    the scaled P(n=0) to hand to resolve_conflicts. q sorted by entity."""
    pz = np.minimum(1.0, s * np.asarray(p_zero, float))
    acc = np.zeros(len(q), bool)
    starts = np.flatnonzero(np.r_[True, q[1:] != q[:-1]]) if len(q) else []
    ends = np.r_[starts[1:], len(q)] if len(q) else []
    for a, b in zip(starts, ends):
        o = a + np.argsort(-p[a:b], kind="stable")
        acc[o[:choose_k(p[o], float(pz[q[a]]), recall, bar_scale=t)]] = True
    return acc, pz


# ---------------------------------------------------------------- self-test

# AUDIT.md match-count shares, 0..8 (8 stands for 8+)
_N_SHARE = np.array([5.58, 5.40, 17.00, 24.05, 21.94, 14.59, 7.47, 2.90, 1.06])


def _shard(rng, n_ent, shift, recall=0.93, n_dist=25):
    """One synthetic country: true cardinalities from the train histogram, a
    blocking miss rate, uniform-ish distractors. Raw scores for the target
    country are multiplied by `shift` < 1, standing in for a country whose
    scores the model under-rates. Returns q, raw, y, n_true."""
    n_true = rng.choice(len(_N_SHARE), n_ent, p=_N_SHARE / _N_SHARE.sum())
    q, y = [], []
    for e, n in enumerate(n_true):
        found = int(rng.binomial(n, recall))
        d = int(rng.poisson(n_dist))
        q += [e] * (found + d)
        y += [True] * found + [False] * d
    q, y = np.array(q, np.int64), np.array(y)
    raw = np.where(y, rng.beta(6, 2, len(y)), rng.beta(1, 9, len(y))) * shift
    return q, raw, y, n_true


def _macro(q, acc, y, n_true):
    c = np.bincount(q[acc & y], minlength=len(n_true))
    k = np.bincount(q[acc], minlength=len(n_true))
    from metric import f05
    return float(f05(c, n_true, k).mean())


def _self_test(seed=0):
    rng = np.random.default_rng(seed)
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("  ok    " if cond else "  FAIL  ") + msg)
        ok &= bool(cond)

    # "labelled" country: fit a binned calibration on its labels, as isotonic
    qr, rr, yr, nr = _shard(rng, 4000, 1.0)
    bins = np.minimum((rr * 40).astype(int), 39)
    rate = np.bincount(bins, yr, 40) / np.maximum(np.bincount(bins, minlength=40), 1)
    rate = np.maximum.accumulate(rate)          # monotone, like isotonic
    cal = lambda raw: np.clip(rate[np.minimum((raw * 40).astype(int), 39)], 1e-3, 0.999)
    pr = cal(rr)
    pz_r = product_p_zero(qr, pr, len(nr))
    acc_r, _ = apply(qr, pr, pz_r, 1.0, 1.0)
    ref_hist = k_hist(np.bincount(qr[acc_r], minlength=len(nr)))

    # target country: same cardinality law, scores under-rated, no labels used
    qt, rt, yt, nt = _shard(rng, 3000, 0.7)
    pt = cal(rt)
    pz_t = product_p_zero(qt, pt, len(nt))
    s, t, before, _ = fit(qt, pt, pz_t, 1.0, ref_hist, len(nt))
    acc_t0, _ = apply(qt, pt, pz_t, 1.0, 1.0)
    acc_t, _ = apply(qt, pt, pz_t, s, t)
    after = k_hist(np.bincount(qt[acc_t], minlength=len(nt)))

    print(fmt_header())
    print(fmt_row("reference", ref_hist))
    print(fmt_row("target before", before))
    print(fmt_row("target after", after))
    print(f"    s = {s:.4g}, t = {t:.4g}")
    f0, f1 = _macro(qt, acc_t0, yt, nt), _macro(qt, acc_t, yt, nt)
    print(f"    target macro F0.5 {f0:.4f} -> {f1:.4f}  "
          f"(reference {_macro(qr, acc_r, yr, nr):.4f})\n")

    # 1. the vectorised k(s, t) is choose_k, at identity and away from it
    prep = _Prepared(qt, pt, pz_t, 1.0, len(nt))
    for ss, tt in ((1.0, 1.0), (s, t), (0.3, 1.7), (4.0, 0.6)):
        a, _ = apply(qt, pt, pz_t, ss, tt)
        check(np.array_equal(prep.k(ss, tt), np.bincount(qt[a], minlength=len(nt))),
              f"vectorised k(s={ss:.3g}, t={tt:.3g}) equals choose_k per entity")
    check(np.array_equal(k_hist(prep.k(1, 1)), before), "fit's 'before' is s=t=1")

    # 2. identity scales reproduce the uncalibrated decision exactly
    acc_plain = np.zeros(len(qt), bool)
    st = np.flatnonzero(np.r_[True, qt[1:] != qt[:-1]])
    for a, b in zip(st, np.r_[st[1:], len(qt)]):
        o = a + np.argsort(-pt[a:b], kind="stable")
        acc_plain[o[:choose_k(pt[o], float(np.prod(1.0 - pt[o])))]] = True
    check(np.array_equal(acc_plain, acc_t0), "s=t=1 is bit-identical to the plain rule")

    # 3. the fit hits the reference
    sb, sa, sr = summary(before), summary(after), summary(ref_hist)
    check(sb["singleton_rate"] > sr["singleton_rate"] + 0.02,
          f"target starts over-abstaining ({sb['singleton_rate']:.2%} vs "
          f"{sr['singleton_rate']:.2%})")
    check(abs(sa["singleton_rate"] - sr["singleton_rate"]) < 0.005,
          f"singleton rate matched ({sa['singleton_rate']:.2%})")
    check(abs(sa["mean_k_nonzero"] - sr["mean_k_nonzero"]) < 0.02,
          f"mean k|k>=1 matched ({sa['mean_k_nonzero']:.3f} vs {sr['mean_k_nonzero']:.3f})")
    check(f1 > f0, "calibration raises the target's macro F0.5")

    # 4. an already-matching country is left (almost) alone
    s2, t2, _, _ = fit(qr, pr, pz_r, 1.0, ref_hist, len(nr))
    check(0.8 < s2 < 1.25 and 0.9 < t2 < 1.1,
          f"reference fitted to itself gives s~1, t~1 ({s2:.3g}, {t2:.3g})")

    print("\n" + ("PASS" if ok else "FAIL"))
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if _self_test() else 1)
