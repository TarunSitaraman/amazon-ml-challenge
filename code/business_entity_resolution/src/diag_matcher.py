"""Why does the MATCHER err? The counterpart of diag_misses.py.

diag_misses.py explains the links blocking never retrieves (1 - ceiling). This
explains the rest of the gap: the true matches that WERE retrieved but lost at
the decision, and the false ones accepted. It reads valstate.pkl, reproduces
train_eval.py's adaptive-k decision per entity (choose_k over the calibrated
scores, product-rule P(n=0) unless --pz head), and categorises every error.

False positives (accepted, not true), by the record's OWNER in ground truth:
  chain       gold match of ANOTHER S1 entity with the same normalised name
  sibling     gold match of a different S1 entity whose name differs
  distractor  a record no S1 entity matches
  each split by whether the two addresses share a digit token (textnorm.nums).

False negatives (true, in the candidates, not accepted), by decision:
  ranked out     an accepted false candidate outranks it
  stopped early  nothing false was accepted and it outranks every rejected
                 false candidate: the ranking was right, the stop too early
  buried         nothing false was accepted but a rejected false candidate
                 outranks it, so a later stop alone would not reach it cleanly
  plus, overlapping: low p (p < LOW_P despite being true)
and by signal (first that applies): script mismatch / empty address /
  name + address overlap / name-only / address-only / no token overlap.

Counts mislead, because an error on a 1-match entity costs far more than on an
8-match one, and a false accept on an entity with no matches zeroes it. So each
entity's matcher loss, ceiling_i - F_i, is split exactly over its errors by
Shapley value (errors of one kind in one entity are symmetric, so this is a
double sum over how many of each are fixed), and every table sums to the
measured gap. ceiling_i scores a perfect decision on the retrieved candidates,
so unretrieved links stay in n and in the blocking loss, not here.

Usage: python diag_matcher.py [--pz product|head] [--all] [--examples 3]
       [--valstate valstate.pkl] [--root data/parquet]
"""
import argparse
import math
import pathlib
import pickle
from collections import defaultdict

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

import blocking
from metric import choose_k, f05
from textnorm import DEVA, norm, nums

ROOT = "data/parquet"
LOW_P = 0.1

FP_OWNER = ["chain", "sibling", "distractor"]
FN_DECISION = ["ranked out", "stopped early", "buried"]
FN_SIGNAL = ["script mismatch", "empty address", "name + address overlap",
             "name-only overlap", "address-only overlap", "no token overlap"]


def _f(c, n, k):
    return float(f05(c, n, k))


def shapley_split(c, n, k, m, f):
    """Split ceiling - F over m false positives and f in-candidate false
    negatives of one entity. -> (per-FP share, per-FN share), with
    m * fp + f * fn == F(c+f, n, c+f) - F(c, n, k) exactly (up to rounding).

    v(a, b) is the score gain from fixing a of the FPs and b of the FNs;
    every FP is interchangeable with every other, so the Shapley value is a
    sum over coalition sizes rather than over subsets."""
    N = m + f
    if N == 0:
        return 0.0, 0.0
    base = _f(c, n, k)

    def v(a, b):
        return _f(c + b, n, k - a + b) - base

    def w(s):                             # s!(N-s-1)!/N!
        return 1.0 / (N * math.comb(N - 1, s))

    fp = sum(w(a + b) * math.comb(m - 1, a) * math.comb(f, b) * (v(a + 1, b) - v(a, b))
             for a in range(m) for b in range(f + 1)) if m else 0.0
    fn = sum(w(a + b) * math.comb(m, a) * math.comb(f - 1, b) * (v(a, b + 1) - v(a, b))
             for a in range(m + 1) for b in range(f)) if f else 0.0
    return fp, fn


def decision_category(r, k, is_true):
    """Why the true candidate at rank r (>= k) was not accepted.
    is_true: truth flags of the entity's candidates in rank order."""
    if not is_true[:k].all():
        return "ranked out"
    if is_true[k:r].all():
        return "stopped early"
    return "buried"


def signal_category(n1, a1, n2, a2):
    if bool(DEVA.search(n1 or "")) != bool(DEVA.search(n2 or "")):
        return "script mismatch"
    at1, at2 = set(norm(a1 or "").split()), set(norm(a2 or "").split())
    if not at1 or not at2:
        return "empty address"
    name = bool(set(norm(n1 or "").split()) & set(norm(n2 or "").split()))
    addr = bool(at1 & at2)
    if name and addr:
        return "name + address overlap"
    if name:
        return "name-only overlap"
    if addr:
        return "address-only overlap"
    return "no token overlap"


def owner_category(s1_name, owner_name, owner):
    if owner is None:
        return "distractor"
    return "chain" if norm(s1_name or "") == norm(owner_name or "") else "sibling"


def digit_shared(a1, a2):
    return bool(nums(a1 or "") & nums(a2 or ""))


def decide(d, ev, pz_kind):
    """Reproduce train_eval.py's per-entity adaptive-k decision for entities ev.
    -> {entity: (probs desc, cand ids desc, k)}"""
    q, p, cand = d["q"], d["p"], d["cand"]
    starts = np.flatnonzero(np.r_[True, q[1:] != q[:-1]])
    ends = np.r_[starts[1:], len(q)]
    want = set(int(i) for i in ev)
    out = {}
    for s, e in zip(starts, ends):
        i = int(q[s])
        if i not in want:
            continue
        o = np.argsort(-p[s:e], kind="stable")
        pr, cd = p[s:e][o], cand[s:e][o]
        pz = (float(d["p_zero_head"][i]) if pz_kind == "head"
              else float(np.prod(1.0 - pr)))
        out[i] = (pr, cd, choose_k(pr, pz))
    return out


def find_owners(gt, records):
    """{record id: S1 entity id whose gold contains it}, only for `records`."""
    if not records:
        return {}
    lst = pc.split_pattern(pc.fill_null(gt.column("matched_entity_ids"), ""), ",")
    flat = pc.list_flatten(lst)
    parent = pc.list_parent_indices(lst)
    hit = pc.is_in(flat, value_set=pa.array(sorted(records)))
    recs = pc.filter(flat, hit).to_pylist()
    s1 = pc.take(gt.column("source1_entity_id"), pc.filter(parent, hit)).to_pylist()
    return dict(zip(recs, s1))


def load_text(root, countries, s1_want, rec_want):
    """-> ({id: (name, address)}, {country: set of S1 entity ids in a
    same-normalised-name group})."""
    text, grouped = {}, {}
    s1_set, rec_set = pa.array(sorted(s1_want)), pa.array(sorted(rec_want))
    for ctry in countries:
        s1_tab, corpus_tab = blocking.load_shard(root, "train", ctry)
        names, _ = blocking.norm_column(s1_tab.column("business_name"))
        ids = s1_tab.column("entity_id").to_pylist()
        cnt = defaultdict(int)
        for nm in names:
            cnt[nm] += 1
        grouped[ctry] = {i for i, nm in zip(ids, names) if nm and cnt[nm] > 1}
        for tab, want in ((s1_tab, s1_set), (corpus_tab, rec_set)):
            sub = tab.filter(pc.is_in(tab.column("entity_id"), value_set=want))
            for i, n, a in zip(sub.column("entity_id").to_pylist(),
                               sub.column("business_name").to_pylist(),
                               sub.column("business_address").to_pylist()):
                text[i] = (n or "", a or "")
    return text, grouped


def collect(dec, truth, ev):
    """Every matcher error with its Shapley share.
    -> (errors, per-entity (F, ceiling)). An error is a dict with kind FP/FN."""
    errors, fc = [], {}
    for i in ev:
        T = truth[i]
        n = len(T)
        if i not in dec:
            fc[i] = (_f(0, n, 0),) * 2
            continue
        pr, cd, k = dec[i]
        is_true = np.fromiter((x in T for x in cd), bool, len(cd))
        c = int(is_true[:k].sum())
        fps = np.flatnonzero(~is_true[:k])
        fns = k + np.flatnonzero(is_true[k:])
        h = int(is_true.sum())
        fc[i] = (_f(c, n, k), _f(h, n, h))
        fp_w, fn_w = shapley_split(c, n, k, len(fps), len(fns))
        for r in fps:
            errors.append(dict(kind="FP", ent=i, rec=cd[r], p=float(pr[r]), rank=int(r),
                               k=k, n=n, w=fp_w))
        for r in fns:
            errors.append(dict(kind="FN", ent=i, rec=cd[r], p=float(pr[r]), rank=int(r),
                               k=k, n=n, w=fn_w,
                               decision=decision_category(r, k, is_true)))
    return errors, fc


def table(title, rows, key, cats, n_ev, gap, total_errors, extra=None):
    print(f"\n{title}")
    hdr = f"  {'category':34}{'count':>8}{'% errs':>8}{'lost F0.5':>11}{'% of gap':>10}"
    if extra:
        hdr += f"{extra[0]:>10}"
    print(hdr)
    tot_n = tot_w = 0
    for cat in cats:
        sel = [e for e in rows if e[key] == cat]
        w = sum(e["w"] for e in sel)
        tot_n += len(sel)
        tot_w += w
        line = (f"  {cat:34}{len(sel):>8,}{len(sel)/max(total_errors,1):>8.1%}"
                f"{w/n_ev:>11.4f}{w/max(gap,1e-12):>10.1%}")
        if extra:
            line += f"{extra[1](sel):>10}"
        print(line)
    print(f"  {'total':34}{tot_n:>8,}{tot_n/max(total_errors,1):>8.1%}"
          f"{tot_w/n_ev:>11.4f}{tot_w/max(gap,1e-12):>10.1%}")


def show_examples(title, rows, key, cats, text, n_ex, rng):
    print(f"\n{title} -- {n_ex} random examples per category")
    for cat in cats:
        sel = [e for e in rows if e[key] == cat]
        if not sel:
            continue
        print(f"\n  [{cat}]  ({len(sel):,})")
        for j in rng.choice(len(sel), min(n_ex, len(sel)), replace=False):
            e = sel[j]
            n1, a1 = text.get(e["id"], ("?", "?"))
            n2, a2 = text.get(e["rec"], ("?", "?"))
            print(f"    p={e['p']:.3f} rank {e['rank'] + 1} k={e['k']} n={e['n']}")
            print(f"      S1  {n1[:70]!r}\n          @ {a1[:80]!r}")
            print(f"      rec {n2[:70]!r}\n          @ {a2[:80]!r}")
            if e.get("owner"):
                no, ao = text.get(e["owner"], ("?", "?"))
                print(f"      gold of {no[:70]!r}\n          @ {ao[:80]!r}")


def run(d, gt, root, pz_kind="product", all_ents=False, n_ex=3, seed=0):
    ids, truth, ctry = d["ids"], d["truth"], list(d["ctry"])
    n_cal = d.get("n_cal", len(ids) // 2)
    # train_eval.py fits isotonic on the first half: score only the rest.
    ev = np.arange(0 if all_ents else n_cal, len(ids))
    if pz_kind == "head" and "p_zero_head" not in d:
        raise SystemExit("valstate.pkl has no p_zero_head; rerun train_eval.py")
    dec = decide(d, ev, pz_kind)
    errors, fc = collect(dec, truth, ev)

    F = np.array([fc[i][0] for i in ev])
    C = np.array([fc[i][1] for i in ev])
    n_ev, gap = len(ev), float((C - F).sum())
    print(f"{n_ev:,} entities ({'all' if all_ents else 'held-out half'}), "
          f"P(n=0) from {'singleton head' if pz_kind == 'head' else 'product rule'}")
    print(f"  macro F0.5 = {F.mean():.4f}   ceiling = {C.mean():.4f}   "
          f"matcher gap = {C.mean() - F.mean():.4f}   realised "
          f"{F.mean() / max(C.mean(), 1e-12):.1%} of ceiling")
    print(f"  blocking loss (1 - ceiling) = {1 - C.mean():.4f}: see diag_misses.py")

    fp = [e for e in errors if e["kind"] == "FP"]
    fn = [e for e in errors if e["kind"] == "FN"]
    w_check = sum(e["w"] for e in errors)
    assert abs(w_check - gap) < 1e-6 * max(1, n_ev), (w_check, gap)

    owners = find_owners(gt, {e["rec"] for e in fp})
    s1_want = {ids[i] for i in ev} | set(owners.values())
    text, grouped = load_text(root, sorted(set(ctry[i] for i in ev)), s1_want,
                              {e["rec"] for e in errors})
    for e in errors:
        e["id"] = ids[e["ent"]]
    for e in fp:
        e["owner"] = owners.get(e["rec"])
        n1, a1 = text.get(e["id"], ("", ""))
        oc = owner_category(n1, text.get(e["owner"], ("", ""))[0], e["owner"])
        dig = digit_shared(a1, text.get(e["rec"], ("", ""))[1])
        e["cat"] = f"{oc}, {'digit shared' if dig else 'no digit shared'}"
    for e in fn:
        n1, a1 = text.get(e["id"], ("", ""))
        n2, a2 = text.get(e["rec"], ("", ""))
        e["signal"] = signal_category(n1, a1, n2, a2)

    n_err = len(errors)
    print(f"  matcher errors: {len(fp):,} false positives, {len(fn):,} true matches "
          f"retrieved but not accepted ({len(dec):,} entities with candidates)")

    fp_cats = [f"{o}, {s}" for o in FP_OWNER for s in ("digit shared", "no digit shared")]
    on_zero = ("on n=0", lambda sel: f"{np.mean([e['n'] == 0 for e in sel]):.0%}"
               if sel else "-")
    table("FALSE POSITIVES by owner of the accepted record", fp, "cat", fp_cats,
          n_ev, gap, n_err, on_zero)
    for o in FP_OWNER:
        sel = [e for e in fp if e["cat"].startswith(o)]
        print(f"    {o:12} {len(sel):>8,}   lost F0.5 {sum(e['w'] for e in sel)/n_ev:.4f}")

    table("FALSE NEGATIVES by decision", fn, "decision", FN_DECISION, n_ev, gap, n_err,
          ("low p", lambda sel: f"{np.mean([e['p'] < LOW_P for e in sel]):.0%}"
           if sel else "-"))
    low = [e for e in fn if e["p"] < LOW_P]
    print(f"  {'of which low p (< ' + str(LOW_P) + ', overlaps)':34}{len(low):>8,}"
          f"{len(low)/max(n_err,1):>8.1%}{sum(e['w'] for e in low)/n_ev:>11.4f}"
          f"{sum(e['w'] for e in low)/max(gap,1e-12):>10.1%}")
    table("FALSE NEGATIVES by shared signal", fn, "signal", FN_SIGNAL, n_ev, gap, n_err,
          ("low p", lambda sel: f"{np.mean([e['p'] < LOW_P for e in sel]):.0%}"
           if sel else "-"))

    # Entities in a same-name group (EXPERIMENTS.md: 39.28% of S1 in India).
    in_grp = np.array([ids[i] in grouped.get(ctry[i], ()) for i in ev])
    print(f"\nsame-normalised-name groups: {in_grp.mean():.2%} of these entities, "
          f"{(C - F)[in_grp].sum()/max(gap,1e-12):.1%} of the matcher gap "
          f"(F0.5 {F[in_grp].mean() if in_grp.any() else float('nan'):.4f} in a group, "
          f"{F[~in_grp].mean() if (~in_grp).any() else float('nan'):.4f} not)")
    n_true = np.array([len(truth[i]) for i in ev])
    print("matcher gap by true n:")
    for lo, hi, lab in ((0, 0, "n=0"), (1, 1, "n=1"), (2, 3, "n=2-3"),
                        (4, 7, "n=4-7"), (8, 10**9, "n>=8")):
        m = (n_true >= lo) & (n_true <= hi)
        print(f"  {lab:6} {m.mean():>7.1%} of entities   "
              f"{(C - F)[m].sum()/max(gap,1e-12):>6.1%} of gap   "
              f"F0.5 {F[m].mean() if m.any() else float('nan'):.4f}")

    rng = np.random.default_rng(seed)
    show_examples("FALSE POSITIVES", fp, "cat", fp_cats, text, n_ex, rng)
    show_examples("FALSE NEGATIVES by decision", fn, "decision", FN_DECISION, text,
                  n_ex, rng)
    show_examples("FALSE NEGATIVES low p", [dict(e, low="low p") for e in low], "low",
                  ["low p"], text, n_ex, rng)
    show_examples("FALSE NEGATIVES by signal", fn, "signal", FN_SIGNAL, text, n_ex, rng)

    ranked = sorted(
        [(f"FP {c}", sum(e["w"] for e in fp if e["cat"] == c)) for c in fp_cats]
        + [(f"FN {c}", sum(e["w"] for e in fn if e["decision"] == c))
           for c in FN_DECISION],
        key=lambda x: -x[1])
    print("\nlargest shares of the matcher gap (FP by owner + FN by decision):")
    for lab, w in ranked[:5]:
        print(f"  {lab:40} {w/max(gap,1e-12):>6.1%}   {w/n_ev:.4f} macro F0.5")
    return errors, F, C


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pz", choices=("product", "head"), default="product")
    ap.add_argument("--all", action="store_true",
                    help="include the isotonic calibration half (in-sample)")
    ap.add_argument("--examples", type=int, default=3)
    ap.add_argument("--valstate", default="valstate.pkl")
    ap.add_argument("--root", default=ROOT)
    a = ap.parse_args()
    # valstate.pkl is written by train_eval.py in this repo: a local build
    # artifact, never downloaded or user-supplied.
    with open(a.valstate, "rb") as fh:
        d = pickle.load(fh)
    gt = pq.read_table(pathlib.Path(a.root) / "train_ground_truth.parquet",
                       columns=["source1_entity_id", "matched_entity_ids"])
    run(d, gt, a.root, a.pz, a.all, a.examples)


if __name__ == "__main__":
    main()
