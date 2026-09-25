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

Disjointness (disjoint.py): every S2/S3 record belongs to at most one S1
entity. Two entities in different batches can claim the same record, because
every batch in a country searches the same corpus, so scored pairs are buffered
for the whole country and the decision is made once all its batches are done.
  --disjoint off       per-entity choose_k only (the previous behaviour)
  --disjoint resolve   choose_k, then resolve_conflicts
  --disjoint sinkhorn  sinkhorn_normalise, choose_k, then resolve_conflicts
--recall R passes the measured blocking recall to choose_k and
resolve_conflicts alike, so both use the same n_hat (default 1.0).
resolve_conflicts only drops pairs, so matches stay a subset of candidates.

--singleton on takes P(n=0) from the singleton head (singleton.py) that
train_eval.py saved in model.pkl, in place of prod(1 - p), for choose_k and
resolve_conflicts alike. train_eval.py saves a head only when it passes the
precision gate; without one this falls back to the product rule and says so.
The head reads an entity's own candidates only, so it runs per batch. Default
off, which is the previous behaviour.

Usage: python predict.py [--limit N] [--disjoint off|resolve|sinkhorn]
                         [--recall R] [--singleton on|off]
"""
import pathlib
import pickle
import sys
import time

import numpy as np
import pyarrow.compute as pc
import pyarrow.dataset as ds

import blocking
import disjoint
import features
import singleton
import strfeatures
from metric import choose_k
from textnorm import norm
from train_eval import cap_candidates, head_features

ROOT = "data/parquet"
OUT = pathlib.Path("output")
BATCH = 20000
DISJOINT_MODES = ("off", "resolve", "sinkhorn")


def countries(split="test"):
    d = ds.dataset(f"{ROOT}/{split}_source1", format="parquet", partitioning="hive")
    return sorted(set(d.to_table(columns=["country"]).column("country").to_pylist()))


def decide(q, c, p, mode, recall=1.0, p_zero=None):
    """Accepted-pair mask for one country's pairs (q sorted by entity).

    p_zero: optional P(n=0) per entity, indexed by q. None is the product rule
    prod(1 - p), computed after Sinkhorn when that runs, as before."""
    if mode == "sinkhorn":
        p = disjoint.sinkhorn_normalise(q, c, p)
    acc = np.zeros(len(q), bool)
    starts = np.flatnonzero(np.r_[True, q[1:] != q[:-1]])
    ends = np.r_[starts[1:], len(q)]
    for s, e in zip(starts, ends):
        o = s + np.argsort(-p[s:e], kind="stable")
        pz = float(np.prod(1.0 - p[o])) if p_zero is None else float(p_zero[q[s]])
        acc[o[:choose_k(p[o], pz, recall)]] = True
    if mode != "off":
        acc = disjoint.resolve_conflicts(q, c, p, acc, recall=recall, p_zero=p_zero)
    return acc, p


def main():
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
    mode = "off"
    if "--disjoint" in sys.argv:
        mode = sys.argv[sys.argv.index("--disjoint") + 1]
        if mode not in DISJOINT_MODES:
            sys.exit(f"--disjoint must be one of {', '.join(DISJOINT_MODES)}")
    recall = 1.0
    if "--recall" in sys.argv:
        recall = float(sys.argv[sys.argv.index("--recall") + 1])
        if not 0.0 < recall <= 1.0:
            sys.exit("--recall must be in (0, 1]")
    use_singleton = False
    if "--singleton" in sys.argv:
        val = sys.argv[sys.argv.index("--singleton") + 1]
        if val not in ("on", "off"):
            sys.exit("--singleton must be on or off")
        use_singleton = val == "on"

    # model.pkl is written by train_eval.py in this same repo -- a local build
    # artifact, never a downloaded or user-supplied file.
    with open("model.pkl", "rb") as f:
        bundle = pickle.load(f)
    model, iso = bundle["model"], bundle["iso"]
    head = bundle.get("singleton") if use_singleton else None
    if use_singleton and head is None:
        prec = bundle.get("singleton_precision")
        print("WARNING: --singleton on, but model.pkl has no singleton head "
              + ("(it predates the head; rerun train_eval.py)" if prec is None else
                 f"(it failed the precision gate at {prec:.3f})")
              + ". Using the product rule.", flush=True)

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
        idf_lut = {t: float(corpus["index"]["idf"][i]) for i, t in
                   enumerate(corpus["index"]["vocab"].to_pylist())
                   if corpus["index"]["idf"][i] > 0}
        s_idf = singleton.idf_from_index(corpus["index"]) if head is not None else None

        # name_dup counts over the whole country's S1, as the feature is defined
        # and as train_eval.py computes it; per batch it drops every duplicate
        # that falls in another batch.
        dup = features.name_dup_counts(
            [norm(x) for x in s1_tab.column("business_name").to_pylist()])
        seen = 0
        # Scored pairs for the whole country; entity index is global (lo + q).
        all_q, all_c, all_p = [], [], []
        p_zero = np.ones(len(s1_ids)) if head is not None else None
        for lo in range(0, len(s1_ids), BATCH):
            hi = min(lo + BATCH, len(s1_ids))
            sub = s1_tab.slice(lo, hi - lo)
            names, addrs = blocking.normalise(sub)
            q, c, chan = blocking.generate(None, None, names, addrs, corpus,
                                           verbose=False)
            batch_ids = s1_ids[lo:hi]
            cand = {i: [] for i in range(len(batch_ids))}

            if len(q):
                order = np.argsort(q, kind="stable")
                q, c, chan = q[order], c[order], chan[order]
                q, c, chan = cap_candidates(q, c, chan)
                cand_ids = c_ids[c]
                is_s3 = np.fromiter((s.startswith("S3-") for s in cand_ids),
                                    bool, len(cand_ids))
                Xs = strfeatures.build(names, addrs, q,
                                       corpus["names_arr"].take(c).to_pylist(),
                                       corpus["addrs_arr"].take(c).to_pylist(),
                                       idf_lut)
                X = np.hstack([features.build(q, chan, is_s3,
                                              dup[lo:hi]), Xs])
                p = iso.predict(model.predict(X))
                if head is not None:
                    Xe, _ = head_features(q, p, X, singleton.text_features(
                        names, addrs, s_idf))
                    p_zero[lo:hi] = head.predict_proba(Xe)

                starts = np.flatnonzero(np.r_[True, q[1:] != q[:-1]])
                ends = np.r_[starts[1:], len(q)]
                for s, e in zip(starts, ends):
                    o = s + np.argsort(-p[s:e])
                    # dict.fromkeys preserves order and removes duplicates
                    cand[int(q[s])] = list(dict.fromkeys(c_ids[c[o]]))
                all_q.append(q.astype(np.int64) + lo)
                all_c.append(c.astype(np.int64))
                all_p.append(p)

            for i, eid in enumerate(batch_ids):
                rows_cand.append(f"{eid}\t{','.join(cand[i])}")
            seen += hi - lo
            print(f"    {seen:,}/{len(s1_ids):,}  ({time.time()-t0:.0f}s)", flush=True)

        pred = [[] for _ in s1_ids]
        if all_q:
            q, c, p = (np.concatenate(all_q), np.concatenate(all_c),
                       np.concatenate(all_p))
            del all_q, all_c, all_p
            t1 = time.time()
            acc, p = decide(q, c, p, mode, recall, p_zero)
            # highest probability first within each entity, as before
            keep = np.flatnonzero(acc)
            keep = keep[np.lexsort((-p[keep], q[keep]))]
            for j in keep:
                pred[q[j]].append(c_ids[c[j]])
            shared = np.bincount(c[acc]) if acc.any() else np.zeros(1, int)
            print(f"  decided ({mode}, recall {recall}, "
                  f"P(n=0) {'head' if head is not None else 'product'}) in {time.time()-t1:.0f}s, records with "
                  f"2+ owners: {(shared >= 2).sum():,}", flush=True)
        for eid, ids in zip(s1_ids, pred):
            rows_match.append(f"{eid}\t{','.join(dict.fromkeys(ids))}")

    hdr_m = "source1_entity_id\tmatched_entity_ids"
    hdr_c = "source1_entity_id\tcandidate_entity_ids"
    # newline="\n" is load-bearing on Windows. The default text mode rewrites
    # "\n" as "\r\n", and a scorer that splits on "\n" then sees every trailing
    # ID as "S3-123\r", which matches nothing in the test set and rejects the
    # whole submission. Silent, and it would cost a submission to discover.
    for path, hdr, rows in ((OUT / "matching_results.tsv", hdr_m, rows_match),
                            (OUT / "candidate_pairs.tsv", hdr_c, rows_cand)):
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join([hdr] + rows) + "\n")
    n_pred = sum(1 for r in rows_match if r.split("\t")[1])
    print(f"\nwrote {len(rows_match):,} rows in {(time.time()-t_start)/60:.0f} min")
    print(f"  entities with >=1 match: {n_pred:,} ({n_pred/len(rows_match):.1%})")


if __name__ == "__main__":
    main()
