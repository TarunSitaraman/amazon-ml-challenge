"""Old vs new "normalise corpus" timing on synthetic names and addresses.

    python tests/bench_textnorm.py [n_records]

Old path = [norm_reference(x) for x in col.to_pylist()] per column, which is
what blocking.normalise did. New path = blocking.normalise. The repeat rate of
real raw strings is unknown here, so it is swept; the real run prints it
("distinct raw strings: ...") under build_corpus. Synthetic numbers only: judge
the real effect with --profile.
"""
import pathlib
import sys
import time
import tracemalloc

import pyarrow as pa

sys.path[:0] = [str(pathlib.Path(__file__).resolve().parent),
                str(pathlib.Path(__file__).resolve().parents[1] / "src")]
import blocking  # noqa: E402
import normcorpus  # noqa: E402
from textnorm import norm_reference  # noqa: E402


def old(tab):
    return ([norm_reference(x) for x in tab.column("business_name").to_pylist()],
            [norm_reference(x) for x in tab.column("business_address").to_pylist()])


def best_of(fn, tab, k=3):
    t = 1e9
    for _ in range(k):
        t0 = time.perf_counter()
        out = fn(tab)
        t = min(t, time.perf_counter() - t0)
    return t, out


def peak_mb(fn, tab):
    tracemalloc.start()
    out = fn(tab)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    del out
    return peak / 2**20


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
    for dup in (0.0, 0.2, 0.4, 0.6):
        xs = normcorpus.make(2 * n, seed=7, dup=dup)
        tab = pa.table({"business_name": xs[::2], "business_address": xs[1::2]})
        tab = pa.concat_tables([tab.slice(i, 25_000) for i in range(0, n, 25_000)])
        t_old, ref = best_of(old, tab)
        t_new, out = best_of(blocking.normalise, tab)
        assert out == ref
        print(f"repeat rate {dup:.0%}: old {n / t_old:>9,.0f} rec/s  "
              f"new {n / t_new:>9,.0f} rec/s  {t_old / t_new:.2f}x  "
              f"peak alloc old {peak_mb(old, tab):.0f}MB new "
              f"{peak_mb(blocking.normalise, tab):.0f}MB")


if __name__ == "__main__":
    main()
