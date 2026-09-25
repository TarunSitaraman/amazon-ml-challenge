"""Sweep entity-level decision rules against saved validation probabilities.

train_eval.py dumps valstate.pkl (per-candidate probabilities, candidate ids,
truth). Everything here reads that, so tuning the decision rule costs a second
instead of a six-minute blocking rebuild. Nothing in this file touches the
matcher -- it only decides how many of its ranked candidates to accept.

Usage: python tune_decision.py
"""
import pickle

import numpy as np

from metric import choose_k, f05


def load():
    # valstate.pkl is written by train_eval.py in this repo: a local build
    # artifact, never downloaded or user-supplied.
    with open("valstate.pkl", "rb") as fh:
        d = pickle.load(fh)
    q, p, cand = d["q"], d["p"], d["cand"]
    starts = np.flatnonzero(np.r_[True, q[1:] != q[:-1]])
    ends = np.r_[starts[1:], len(q)]
    per = {}
    for s, e in zip(starts, ends):
        o = np.argsort(-p[s:e])
        per[int(q[s])] = (p[s:e][o], cand[s:e][o])
    return per, d["truth"], d["ids"], np.array(d["ctry"])


def score(per, truth, n_ent, policy):
    sc = np.empty(n_ent)
    ks = np.zeros(n_ent)
    for i in range(n_ent):
        probs, ids = per.get(i, (np.empty(0), np.empty(0, object)))
        k = policy(probs)
        ks[i] = k
        sc[i] = f05(len(set(ids[:k]) & truth[i]), len(truth[i]), k)
    return sc, ks


def main():
    per, truth, ids, ctry = load()
    n = len(ids)
    n_true = np.array([len(t) for t in truth])
    print(f"{n:,} validation entities, mean true n = {n_true.mean():.3f}, "
          f"singletons {(n_true == 0).mean():.2%}\n")

    rows = []

    def run(label, policy):
        sc, ks = score(per, truth, n, policy)
        rows.append((label, sc.mean(), ks.mean()))
        print(f"  {label:34s} F0.5 = {sc.mean():.4f}   mean k = {ks.mean():.2f}")
        return sc.mean()

    print("baselines:")
    for k in (2, 3):
        run(f"fixed top-{k}", lambda pr, k=k: min(k, len(pr)))
    for t in (0.3, 0.4, 0.5, 0.6, 0.7):
        run(f"threshold {t}", lambda pr, t=t: int((pr > t).sum()))

    print("\nadaptive, blocking-recall correction on n_hat:")
    for r in (1.0, 0.9, 0.87, 0.8, 0.7, 0.6):
        run(f"adaptive recall={r}",
            lambda pr, r=r: choose_k(pr, float(np.prod(1.0 - pr)) if len(pr) else 1.0, r))

    print("\nadaptive, scaling the product-rule P(n=0):")
    print("  (the product rule assumes candidate independence; measured zero-match")
    print("   correlation across sources is 3.55x what independence predicts, so it")
    print("   underestimates singleton probability for exactly the singletons)")
    for lift in (1.0, 2.0, 3.55, 6.0, 10.0):
        run(f"adaptive p0 x{lift}",
            lambda pr, L=lift: choose_k(
                pr, min(1.0, float(np.prod(1.0 - pr)) * L) if len(pr) else 1.0, 0.87))

    best = max(rows, key=lambda r: r[1])
    print(f"\nbest: {best[0]}  F0.5 = {best[1]:.4f}  (mean k = {best[2]:.2f})")
    base = next(r[1] for r in rows if r[0] == "adaptive recall=1.0")
    print(f"gain over the shipped rule: {best[1] - base:+.4f}")

    if len(set(ctry.tolist())) > 1:
        print("\nbest rule, per country:")
        pol = dict(rows)
        for c in sorted(set(ctry.tolist())):
            m = np.flatnonzero(ctry == c)
            sc, ks = score(per, truth, n, lambda pr: choose_k(
                pr, float(np.prod(1.0 - pr)) if len(pr) else 1.0, 0.87))
            print(f"  {c:8s} n={len(m):>6,}  F0.5 = {sc[m].mean():.4f}  "
                  f"mean k = {ks[m].mean():.2f}")


if __name__ == "__main__":
    main()
