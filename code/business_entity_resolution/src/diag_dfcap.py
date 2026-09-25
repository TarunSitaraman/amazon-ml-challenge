"""Why is recall stuck? Measure reachability as a function of the df cap.

A true pair is reachable by a token channel only if it shares a token that
survives the cap. This computes, for sampled true links, the document frequency
of the RAREST shared token -- so "recall at cap C" is just the fraction of pairs
whose rarest shared token has df <= C. No retrieval needed, so it isolates
reachability from top-k truncation.
"""
import pathlib
import random
import sys
from collections import Counter

import numpy as np
import pyarrow.parquet as pq

import blocking
from textnorm import fold, norm, skeleton

ROOT = "data/parquet"
random.seed(0)


def main():
    country = sys.argv[1] if len(sys.argv) > 1 else "India"
    n_q = int(sys.argv[2]) if len(sys.argv) > 2 else 8000

    s1_tab, corpus_tab = blocking.load_shard(ROOT, "train", country)
    gt = pq.read_table(pathlib.Path(ROOT) / "train_ground_truth.parquet")
    gtm = dict(zip(gt.column("source1_entity_id").to_pylist(),
                   gt.column("matched_entity_ids").to_pylist()))

    s1_ids = s1_tab.column("entity_id").to_pylist()
    pick = random.sample(range(len(s1_ids)), n_q)
    want_s1 = {s1_ids[i] for i in pick}
    links = [(s1_ids[i], m) for i in pick
             for m in (gtm.get(s1_ids[i]) or "").split(",") if m]
    want_m = {m for _, m in links}
    print(f"{country}: {len(links):,} true links from {n_q:,} entities")

    rec = {}
    for tab in (s1_tab, corpus_tab):
        for i, n, a in zip(tab.column("entity_id").to_pylist(),
                           tab.column("business_name").to_pylist(),
                           tab.column("business_address").to_pylist()):
            if i in want_s1 or i in want_m:
                rec[i] = (norm(n), norm(a))

    # corpus document frequency
    ndf, adf, fdf, sdf = Counter(), Counter(), Counter(), Counter()
    for n, a in zip(corpus_tab.column("business_name").to_pylist(),
                    corpus_tab.column("business_address").to_pylist()):
        nn = norm(n)
        ndf.update(set(nn.split()))
        adf.update(set(norm(a).split()))
        fdf.update(set(fold(nn).split()))
        sdf.update(set(skeleton(nn).split()))

    caps = [100, 500, 2000, 10000, 50000, 10 ** 9]
    chans = {"name": (ndf, lambda r: set(r[0].split())),
             "addr": (adf, lambda r: set(r[1].split())),
             "fold": (fdf, lambda r: set(fold(r[0]).split())),
             "skel": (sdf, lambda r: set(skeleton(r[0]).split()))}

    best = {k: [] for k in chans}
    for a, b in links:
        if a not in rec or b not in rec:
            continue
        for k, (df, get) in chans.items():
            shared = get(rec[a]) & get(rec[b])
            best[k].append(min((df[t] for t in shared), default=10 ** 18))

    print(f"\nreachability = share of true links with a shared token of df <= cap")
    print(f"{'cap':>10} " + "".join(f"{k:>9}" for k in chans) + f"{'name|addr':>11}")
    arr = {k: np.array(v) for k, v in best.items()}
    for c in caps:
        row = f"{c:>10,} " if c < 10 ** 9 else f"{'no cap':>10} "
        for k in chans:
            row += f"{(arr[k] <= c).mean():>8.2%} "
        union = ((arr["name"] <= c) | (arr["addr"] <= c)).mean()
        row += f"{union:>10.2%}"
        print(row)

    allu = ((arr["name"] <= 10**9) | (arr["addr"] <= 10**9) |
            (arr["fold"] <= 10**9) | (arr["skel"] <= 10**9)).mean()
    print(f"\nunion of all four, uncapped: {allu:.2%}  "
          f"(links reachable by nothing: {1-allu:.2%})")


if __name__ == "__main__":
    main()
