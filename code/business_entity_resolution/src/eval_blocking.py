"""Measure blocking quality on a labelled train shard.

Queries a stratified sample of S1 entities against the FULL corpus. Sampling the
corpus would inflate recall (recall depends on corpus density); sampling the
queries does not.

Headline metric is the implied F_0.5 ceiling -- the macro score a perfect matcher
would get on these candidate sets. Raw recall is a misleading proxy because the
F_0.5 frontier is flat above ~95%.

Usage: python eval_blocking.py [country] [n_queries]
"""
import pathlib
import sys
import time

import numpy as np
import pyarrow.parquet as pq

import blocking
from metric import f05

ROOT = "data/parquet"


def ground_truth():
    gt = pq.read_table(pathlib.Path(ROOT) / "train_ground_truth.parquet")
    return dict(zip(gt.column("source1_entity_id").to_pylist(),
                    gt.column("matched_entity_ids").to_pylist()))


def main():
    country = sys.argv[1] if len(sys.argv) > 1 else "India"
    n_q = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
    rng = np.random.default_rng(0)

    t0 = time.time()
    s1_tab, corpus_tab = blocking.load_shard(ROOT, "train", country)
    s1_ids = s1_tab.column("entity_id").to_pylist()
    c_ids = corpus_tab.column("entity_id").to_pylist()
    print(f"{country}: {len(s1_ids):,} S1, {len(c_ids):,} corpus  ({time.time()-t0:.0f}s)")

    t0 = time.time()
    corpus = blocking.build_corpus(corpus_tab)
    print(f"corpus indexed in {time.time()-t0:.0f}s")

    pick = rng.choice(len(s1_ids), size=min(n_q, len(s1_ids)), replace=False)
    sub = s1_tab.take(pick)
    q_ids = [s1_ids[i] for i in pick]
    q_names, q_addrs = blocking.normalise(sub)

    t0 = time.time()
    qi, ci, score, chan = blocking.generate(None, None, q_names, q_addrs, corpus)
    elapsed = time.time() - t0
    print(f"retrieval {elapsed:.0f}s for {len(q_ids):,} queries "
          f"-> {len(qi):,} pairs ({len(qi)/len(q_ids):.1f}/entity)")

    # ---- score ----
    gt = ground_truth()
    truth = [set(v.split(",")) - {""} if v else set() for v in (gt[i] for i in q_ids)]
    c_id_arr = np.array(c_ids, dtype=object)

    retrieved = [set() for _ in q_ids]
    for q, c in zip(qi, c_id_arr[ci]):
        retrieved[q].add(c)

    n_true = np.array([len(t) for t in truth])
    n_hit = np.array([len(t & r) for t, r in zip(truth, retrieved)])
    n_cand = np.array([len(r) for r in retrieved])

    link_recall = n_hit.sum() / max(n_true.sum(), 1)
    matched = n_true > 0
    complete = (n_hit[matched] == n_true[matched]).mean()
    ceiling = f05(n_hit, n_true, n_hit).mean()

    print(f"\n  per-link candidate recall   {link_recall:7.2%}")
    print(f"  per-entity complete recall  {complete:7.2%}")
    print(f"  implied F_0.5 ceiling       {ceiling:7.4f}")
    print(f"  candidates/entity  mean {n_cand.mean():.1f}  "
          f"median {np.median(n_cand):.0f}  p95 {np.percentile(n_cand,95):.0f}  max {n_cand.max()}")
    print(f"  entities with 0 candidates  {(n_cand==0).mean():7.2%}")

    print("\n  channel attribution (share of true links reached):")
    for bit, tag in enumerate(blocking.CHANNELS):
        sel = (chan & bit) > 0
        r = [set() for _ in q_ids]
        for q, c in zip(qi[sel], c_id_arr[ci[sel]]):
            r[q].add(c)
        hits = sum(len(t & x) for t, x in zip(truth, r))
        print(f"    {tag:14s} {hits/max(n_true.sum(),1):7.2%}  ({sel.sum():,} pairs)")

    print(f"\n  extrapolated to {len(s1_ids):,} S1 entities: "
          f"{elapsed*len(s1_ids)/len(q_ids)/60:.0f} min")


if __name__ == "__main__":
    main()
