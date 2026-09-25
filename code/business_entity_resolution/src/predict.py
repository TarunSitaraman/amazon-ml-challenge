"""Run the full pipeline over the test set and emit the two submission files.

Format rules enforced here (from the problem statement):
  * exactly one row per test S1 entity, including entities with no candidates
  * matched_entity_ids empty for predicted singletons
  * no duplicate IDs within a list
  * only S2/S3 IDs that exist in the test set
  * matching_results MUST be a subset of candidate_pairs

Country is a verified-safe hard block, so each country is an independent job and
the pipeline has natural resume points. Entities are processed in batches to
bound peak memory.

Usage: python predict.py [--limit N]
"""
import pathlib
import pickle
import sys
import time

import numpy as np
import pyarrow.compute as pc
import pyarrow.dataset as ds

import blocking
import features
from metric import choose_k
from train_eval import cap_candidates

ROOT = "data/parquet"
OUT = pathlib.Path("output")
BATCH = 20000


def countries(split="test"):
    d = ds.dataset(f"{ROOT}/{split}_source1", format="parquet", partitioning="hive")
    return sorted(set(d.to_table(columns=["country"]).column("country").to_pylist()))


def main():
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])

    # model.pkl is written by train_eval.py in this same repo -- a local build
    # artifact, never a downloaded or user-supplied file.
    with open("model.pkl", "rb") as f:
        bundle = pickle.load(f)
    model, iso = bundle["model"], bundle["iso"]

    OUT.mkdir(exist_ok=True)
    rows_match, rows_cand = [], []
    t_start = time.time()

    for country in countries():
        s1_tab, corpus_tab = blocking.load_shard(ROOT, "test", country)
        s1_ids = s1_tab.column("entity_id").to_pylist()
        c_ids = np.array(corpus_tab.column("entity_id").to_pylist(), dtype=object)
        if limit:
            s1_ids = s1_ids[:limit]
            s1_tab = s1_tab.slice(0, limit)
        print(f"\n{country}: {len(s1_ids):,} S1, {len(c_ids):,} corpus", flush=True)

        t0 = time.time()
        corpus = blocking.build_corpus(corpus_tab, verbose=False)
        print(f"  indexed in {time.time()-t0:.0f}s", flush=True)

        seen = 0
        for lo in range(0, len(s1_ids), BATCH):
            hi = min(lo + BATCH, len(s1_ids))
            sub = s1_tab.slice(lo, hi - lo)
            names, addrs = blocking.normalise(sub)
            q, c, chan = blocking.generate(None, None, names, addrs, corpus,
                                           verbose=False)
            batch_ids = s1_ids[lo:hi]
            pred = {i: [] for i in range(len(batch_ids))}
            cand = {i: [] for i in range(len(batch_ids))}

            if len(q):
                order = np.argsort(q, kind="stable")
                q, c, chan = q[order], c[order], chan[order]
                q, c, chan = cap_candidates(q, c, chan)
                cand_ids = c_ids[c]
                is_s3 = np.fromiter((s.startswith("S3-") for s in cand_ids),
                                    bool, len(cand_ids))
                p = iso.predict(model.predict(features.build(q, chan, is_s3)))

                starts = np.flatnonzero(np.r_[True, q[1:] != q[:-1]])
                ends = np.r_[starts[1:], len(q)]
                for s, e in zip(starts, ends):
                    ent = int(q[s])
                    probs, ids = p[s:e], cand_ids[s:e]
                    o = np.argsort(-probs)
                    probs, ids = probs[o], ids[o]
                    cand[ent] = list(dict.fromkeys(ids))
                    k = choose_k(probs, float(np.prod(1.0 - probs)))
                    # dict.fromkeys preserves order and removes duplicates
                    pred[ent] = list(dict.fromkeys(ids[:k]))

            for i, eid in enumerate(batch_ids):
                rows_match.append(f"{eid}\t{','.join(pred[i])}")
                rows_cand.append(f"{eid}\t{','.join(cand[i])}")
            seen += hi - lo
            print(f"    {seen:,}/{len(s1_ids):,}  ({time.time()-t0:.0f}s)", flush=True)

    hdr_m = "source1_entity_id\tmatched_entity_ids"
    hdr_c = "source1_entity_id\tcandidate_entity_ids"
    (OUT / "matching_results.tsv").write_text("\n".join([hdr_m] + rows_match) + "\n",
                                              encoding="utf-8")
    (OUT / "candidate_pairs.tsv").write_text("\n".join([hdr_c] + rows_cand) + "\n",
                                             encoding="utf-8")
    n_pred = sum(1 for r in rows_match if r.split("\t")[1])
    print(f"\nwrote {len(rows_match):,} rows in {(time.time()-t_start)/60:.0f} min")
    print(f"  entities with >=1 match: {n_pred:,} ({n_pred/len(rows_match):.1%})")


if __name__ == "__main__":
    main()
