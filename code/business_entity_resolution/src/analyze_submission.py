"""Profile a submission, and use it to attack the open train/test density question.

AUDIT.md records an unresolved fork: the test corpus holds 23% more S2/S3
records per S1 entity than train, and the two explanations imply OPPOSITE
threshold corrections.

  A: cardinality is higher on test (mean ~4.26), distractor rate stays ~26%
  B: cardinality is unchanged (3.46), distractor rate rises to ~40%

Row counts alone cannot separate them -- both predict the same ratio. But the
same model and the same stopping rule applied to both splits can: under A the
scorer should find MORE confident candidates per test entity than per train
entity; under B it should find a similar number sitting in more noise.

This is evidence, not proof -- calibration fitted on train need not transfer
perfectly. Read it as a direction, and prefer the conservative branch when the
signal is weak, because F_0.5 punishes over-prediction twice as hard.

Usage: python analyze_submission.py [--train-mean-k 2.67]
"""
import argparse
import collections
import pathlib

import numpy as np
import pyarrow.compute as pc
import pyarrow.dataset as ds

ROOT = "data/parquet"
# India train validation, same model and rule (train_eval.py)
TRAIN_MEAN_K = 2.67
TRAIN_TRUE_MEAN_N = 3.45


def read(path, skip_header=True):
    out = {}
    with open(path, encoding="utf-8") as fh:
        if skip_header:
            fh.readline()
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            eid, _, ids = line.partition("\t")
            out[eid] = [x for x in ids.split(",") if x]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-mean-k", type=float, default=TRAIN_MEAN_K)
    a = ap.parse_args()

    out = pathlib.Path("output")
    match = read(out / "matching_results.tsv")
    cand = read(out / "candidate_pairs.tsv") if (out / "candidate_pairs.tsv").exists() else {}

    d = ds.dataset(f"{ROOT}/test_source1", format="parquet", partitioning="hive")
    t = d.to_table(columns=["entity_id", "country"])
    ctry = dict(zip(t.column("entity_id").to_pylist(), t.column("country").to_pylist()))

    by = collections.defaultdict(list)
    for eid, lst in match.items():
        by[ctry.get(eid, "?")].append(len(lst))
    allk = np.array([len(v) for v in match.values()])

    print(f"{'country':10} {'entities':>10} {'mean k':>8} {'singleton':>10} "
          f"{'S2 frac':>9} {'cands':>8}")
    for c in sorted(by) + ["ALL"]:
        ks = np.array(by[c]) if c != "ALL" else allk
        ids = ([e for e in match if ctry.get(e) == c] if c != "ALL" else list(match))
        s2 = sum(1 for e in ids for x in match[e] if x.startswith("S2-"))
        tot = sum(len(match[e]) for e in ids)
        mc = np.mean([len(cand[e]) for e in ids]) if cand else float("nan")
        print(f"{c:10} {len(ks):>10,} {ks.mean():>8.3f} {(ks==0).mean():>9.2%} "
              f"{(s2/tot if tot else 0):>9.2%} {mc:>8.1f}")

    print("\npredicted match-count distribution (all countries):")
    cnt = collections.Counter(allk.tolist())
    for k in sorted(cnt)[:12]:
        print(f"  {k:>3}: {cnt[k]:>9,} ({cnt[k]/len(allk):6.2%})")

    print(f"\n--- density fork ---")
    print(f"  train validation mean k (same model + rule): {a.train_mean_k:.3f}")
    print(f"  test mean k                                : {allk.mean():.3f}")
    delta = allk.mean() / a.train_mean_k - 1
    print(f"  relative difference                        : {delta:+.1%}")
    if delta > 0.12:
        print("  -> leans A: test entities really do have more matches.")
        print("     Raise the cardinality prior; a train-tuned threshold UNDER-predicts.")
    elif delta < -0.05:
        print("  -> leans B: same cardinality, more distractors crowding the list.")
        print("     A train-tuned threshold OVER-predicts; tighten it.")
    else:
        print("  -> inconclusive. Hold the conservative branch: under F_0.5,")
        print("     being too strict costs less than being too loose.")
    print(f"\n  (train TRUE mean n = {TRAIN_TRUE_MEAN_N}; the model under-predicts by"
          f" {1 - a.train_mean_k/TRAIN_TRUE_MEAN_N:.1%} on train, so expect the same"
          f" bias on test.)")


if __name__ == "__main__":
    main()
