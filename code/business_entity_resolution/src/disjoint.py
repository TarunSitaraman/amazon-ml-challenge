"""Disjointness: each S2/S3 record belongs to at most one S1 entity.

AUDIT.md measured record re-use at exactly 0 over 7.6M train links, yet the
per-entity pipeline scores every (S1, record) pair independently. RESEARCH.md
section 7 picks the two tractable ways to use the constraint:

  sinkhorn_normalise -- before choose_k. Rescales every edge so a record's mass
      across all its suitors is at most 1, with the remainder going to an
      explicit null sink ("belongs to nobody here"). The sink is what keeps the
      26-40% distractor records from being forced onto some entity. Entities
      are then capped at a row budget (their expected cardinality).

  resolve_conflicts -- after choose_k. Any record still accepted by >= 2
      entities goes to whichever gains the most expected F_0.5 from it, and is
      dropped from the rest. Single pass; each step picks the owner that
      maximises total expected F over the claimants, given the current sets.
      Losers are not re-decided: an entity that loses a record does not go
      back to accept its next-best candidate.

Both work on flat pair arrays (q = S1 index, c = record index, p = calibrated
probability), the layout predict.py already uses. Pure numpy; no blocking import.
"""
import numpy as np

EPS = 1e-12


def sinkhorn_normalise(q, c, p, row_budget=None, n_iter=50, tol=1e-6):
    """Capacity-normalised probabilities, same order as the input pairs.

    Kernel is the odds p/(1-p) with a null sink of weight 1 on every record, so
    a record with one suitor keeps p exactly, and a record with several gets
    odds_i / (1 + sum odds) -- the posterior of "suitor i" when at most one
    suitor can be right.

    row_budget: per-entity cap on total mass, indexed by q (array of length
    q.max()+1). Defaults to each entity's raw sum of p, which never binds. Row
    scales only shrink (never above 1), so the normalisation can move mass to
    the null sink but never manufacture it.
    """
    q, c = np.asarray(q, np.int64), np.asarray(c, np.int64)
    p = np.clip(np.asarray(p, float), 0.0, 1.0 - 1e-9)
    if not len(p):
        return p.copy()
    odds = p / (1.0 - p)
    n_rows, n_cols = q.max() + 1, c.max() + 1
    if row_budget is None:
        row_budget = np.bincount(q, p, n_rows)
    # callers may size the budget to the whole batch, including trailing
    # entities with no candidates; only the first n_rows are used
    row_budget = np.asarray(row_budget, float)[:n_rows]
    if len(row_budget) < n_rows:
        raise ValueError(f"row_budget has {len(row_budget)} entries, need {n_rows}")
    u = np.ones(n_rows)
    out = p.copy()
    for _ in range(n_iter):
        w = odds * u[q]
        col = np.bincount(c, w, n_cols) + 1.0      # + null sink
        new = w / col[c]
        rows = np.bincount(q, new, n_rows)
        u = np.minimum(1.0, u * row_budget / np.maximum(rows, EPS))
        done = np.max(np.abs(new - out)) < tol
        out = new
        if done:
            break
    return out


def _expected_f(s, k, n_hat, p_zero):
    """Expected F_0.5 of an accepted set with probability sum s and size k."""
    if k == 0:
        return p_zero
    return 1.25 * s / (0.25 * n_hat + k)


def _exchange(q, c, p, keep, n_hat, p_zero):
    """One conflict-exchange pass over keep, in place. Returns the dropped pairs."""
    n_rows = len(n_hat)
    s = np.bincount(q[keep], p[keep], n_rows)
    k = np.bincount(q[keep], minlength=n_rows)
    dropped = []

    idx = np.flatnonzero(keep)
    claims = np.bincount(c[idx])
    contested = idx[claims[c[idx]] >= 2]
    if not len(contested):
        return dropped
    # group by record, most confident record first, deterministic
    order = np.lexsort((q[contested], -p[contested], c[contested]))
    contested = contested[order]
    starts = np.flatnonzero(np.r_[True, c[contested][1:] != c[contested][:-1]])
    groups = np.split(contested, starts[1:])
    groups.sort(key=lambda g: (-p[g].max(), c[g[0]]))

    for g in groups:
        gains = np.array([
            _expected_f(s[q[i]], k[q[i]], n_hat[q[i]], p_zero[q[i]])
            - _expected_f(s[q[i]] - p[i], k[q[i]] - 1, n_hat[q[i]], p_zero[q[i]])
            for i in g])
        best = np.flatnonzero(gains == gains.max())
        winner = g[best[np.argmin(q[g[best]])]] if gains.max() > 0 else -1
        for i in g:
            if i != winner:
                keep[i] = False
                s[q[i]] -= p[i]
                k[q[i]] -= 1
                dropped.append(i)
    return dropped


def resolve_conflicts(q, c, p, accepted, redecide=False, max_rounds=3,
                      recall=1.0):
    """Make accepted pairs disjoint: every record has at most one owner.

    q, c, p: all candidate pairs (n_hat and P(n=0) per entity come from these,
    matching predict.py's choose_k call). accepted: boolean mask of the pairs
    choose_k kept. Returns a new mask.

    A contested record goes to the claimant whose expected F_0.5 drops most
    without it; if keeping it lowers every claimant's expected F, nobody keeps
    it. Ties go to the lowest q. Contested records are handled most-confident
    first, against the sets as they stand after earlier drops.

    redecide: an entity that lost a record re-runs choose_k over its remaining
    candidates, minus records it lost and records another entity now owns, so
    it can pick up its next-best candidates. New conflicts from that go through
    another exchange; after max_rounds the last exchange stands. Off by
    default: on the synthetic self-test it is flat to slightly worse (likely
    because a loser's next-best candidates are mostly distractors). With
    redecide=False the result is a subset of accepted.

    recall: blocking recall, as in metric.choose_k. n_hat is sum(p) / recall,
    so the exchange values records with the same n_hat that choose_k used to
    accept them. Pass the value given to choose_k; 1.0 is the old behaviour.
    """
    from metric import choose_k

    q, c = np.asarray(q, np.int64), np.asarray(c, np.int64)
    p = np.asarray(p, float)
    keep = np.asarray(accepted, bool).copy()
    if not keep.any():
        return keep

    n_rows = q.max() + 1
    n_hat = np.maximum(np.bincount(q, p, n_rows) / max(recall, 1e-6), 1e-9)
    with np.errstate(divide="ignore"):
        p_zero = np.exp(np.bincount(q, np.log1p(-np.clip(p, 0, 1 - 1e-12)), n_rows))
    banned = np.zeros(len(q), bool)          # pairs an entity has lost

    for rnd in range(max_rounds + 1):
        dropped = _exchange(q, c, p, keep, n_hat, p_zero)
        if not dropped or not redecide or rnd == max_rounds:
            break
        banned[dropped] = True
        owned = np.full(c.max() + 1, -1)
        owned[c[keep]] = q[keep]
        for e in np.unique(q[dropped]):
            rows = np.flatnonzero(q == e)
            free = ~banned[rows] & ((owned[c[rows]] == -1) | (owned[c[rows]] == e))
            rows = rows[free]
            o = rows[np.argsort(-p[rows], kind="stable")]
            keep[q == e] = False
            keep[o[:choose_k(p[o], p_zero[e], recall)]] = True
    return keep


def _self_test(seed=0):
    """Synthetic shard with a disjoint ground truth and calibrated scores."""
    from metric import choose_k, macro_f05

    rng = np.random.default_rng(seed)
    n_ent, n_rec = 400, 3000
    owner = np.full(n_rec, -1)
    true_recs = rng.permutation(n_rec)[: int(n_rec * 0.65)]   # ~35% distractors
    owner[true_recs] = rng.integers(0, n_ent, len(true_recs))

    q, c = [], []
    for e in range(n_ent):
        mine = np.flatnonzero(owner == e)
        others = rng.choice(n_rec, 12, replace=False)
        cand = np.unique(np.r_[mine, others])
        q += [e] * len(cand)
        c += cand.tolist()
    q, c = np.array(q), np.array(c)
    y = owner[c] == q
    # noisy scores, then binned calibration so p means what it says (predict.py
    # does the same with isotonic regression; both steps assume calibrated p)
    raw = np.where(y, rng.beta(5, 2, len(y)), rng.beta(1, 6, len(y)))
    bins = np.minimum((raw * 20).astype(int), 19)
    rate = np.bincount(bins, y, 20) / np.maximum(np.bincount(bins, minlength=20), 1)
    p = np.clip(rate[bins], 0.01, 0.99)

    def decide(prob):
        acc = np.zeros(len(q), bool)
        for e in range(n_ent):
            rows = np.flatnonzero(q == e)
            o = rows[np.argsort(-prob[rows], kind="stable")]
            kk = choose_k(prob[o], float(np.prod(1.0 - prob[o])))
            acc[o[:kk]] = True
        return acc

    def score(acc):
        pred = {e: set(c[acc & (q == e)].tolist()) for e in range(n_ent)}
        truth = {e: set(np.flatnonzero(owner == e).tolist()) for e in range(n_ent)}
        return macro_f05(pred, truth)

    ok = True
    acc = decide(p)
    base = score(acc)
    print(f"raw choose_k           macro F0.5 {base:.4f}  "
          f"max owners/record {np.bincount(c[acc]).max()}")
    drop = resolve_conflicts(q, c, p, acc, redecide=False)
    print(f"+ resolve, drop only   macro F0.5 {score(drop):.4f}  "
          f"max owners/record {np.bincount(c[drop]).max()}")
    ok &= np.bincount(c[drop]).max() <= 1 and not (drop & ~acc).any()
    res = resolve_conflicts(q, c, p, acc, redecide=True)
    after = score(res)
    reuse = np.bincount(c[res]).max()
    print(f"+ resolve, redecide    macro F0.5 {after:.4f}  max owners/record {reuse}")
    ok &= reuse <= 1 and after >= base

    pn = sinkhorn_normalise(q, c, p)
    col = np.bincount(c, pn)
    acc2 = resolve_conflicts(q, c, pn, decide(pn))
    after2 = score(acc2)
    print(f"sinkhorn + resolve     macro F0.5 {after2:.4f}  "
          f"max record mass {col.max():.4f}  max owners/record {np.bincount(c[acc2]).max()}")
    ok &= col.max() <= 1.0 + 1e-9 and np.bincount(c[acc2]).max() <= 1 and after2 >= base

    print("\n" + ("PASS" if ok else "FAIL"))
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if _self_test() else 1)
