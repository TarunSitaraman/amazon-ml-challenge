"""Pre-ranker cost per retrieved pair on SYNTHETIC retrieval-shaped lists.

    python tests/bench_preranker.py [n_entities] [min_per_entity] [max_per_entity]

Synthetic numbers only: strfeatures' synthetic speed-up did not hold on real
data (PR #9). The real cost is the "prerank features" / "prerank score" stages
of train_eval.py / predict.py --profile, and train_eval prints ns/pair always.
"""
import pathlib
import sys
import time

import numpy as np

sys.path[:0] = [str(pathlib.Path(__file__).resolve().parent),
                str(pathlib.Path(__file__).resolve().parents[1] / "src")]
import preranker  # noqa: E402
import train_eval  # noqa: E402
from test_preranker import make_lists  # noqa: E402


def best_of(f, reps=3):
    t = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f()
        t.append(time.perf_counter() - t0)
    return min(t)


def main():
    n_e = int(sys.argv[1]) if len(sys.argv) > 1 else 15000
    lo = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    hi = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    rng = np.random.default_rng(0)
    q, chan, y, s3 = make_lists(n_e, rng, lo, hi)
    n = len(q)
    F = preranker.build(q, chan, s3)
    rankers = {k: preranker.PreRanker(k).fit(F, y) for k in ("logreg", "lgb")}
    print(f"SYNTHETIC: {n_e:,} entities, {n:,} pairs ({n/n_e:.0f}/entity)")
    rows = [("features (build)", best_of(lambda: preranker.build(q, chan, s3)))]
    for k, r in rankers.items():
        rows.append((f"score {k}", best_of(lambda r=r: r.score(F))))
    sc = rankers["lgb"].score(F)
    rows.append(("cap_keep (loop, 60)", best_of(lambda: train_eval.cap_keep(q, sc, 60))))
    rows.append(("chan.max(1) (old score)", best_of(lambda: chan.max(1))))
    for name, t in rows:
        print(f"  {name:24s} {1e9 * t / n:8.1f} ns/pair   {n / t / 1e6:7.2f} M pairs/s")


if __name__ == "__main__":
    main()
