"""Old vs new strfeatures timing on retrieval-shaped synthetic pairs.

    python tests/bench_strfeatures.py [n_entities] [per_entity]

Old path = what the callers did before: gather per-pair text from the Arrow
arrays with take().to_pylist(), then the per-pair loop. New path = build() on
records precomputed once. Precompute is timed separately: it runs once per
country, where build runs once per query batch.
"""
import pathlib
import sys
import time

import numpy as np
import pyarrow as pa

sys.path[:0] = [str(pathlib.Path(__file__).resolve().parent),
                str(pathlib.Path(__file__).resolve().parents[1] / "src")]
import strfeatures  # noqa: E402
from test_strfeatures_equivalence import make_pairs  # noqa: E402


def main():
    n_e = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    per = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    rng = np.random.default_rng(1)
    q_names, q_addrs, q_idx, c_names, c_addrs, c = make_pairs(n_e, per, rng)
    vocab = {t for s in c_names for t in s.split()}
    idf_lut = {t: float(rng.gamma(2.0, 2.0)) for t in vocab}
    n = len(c)
    print(f"{n:,} pairs, {len(c_names):,} corpus records, "
          f"{n / len(c_names):.1f} slots per record")

    names_arr, addrs_arr = pa.array(c_names), pa.array(c_addrs)
    t0 = time.perf_counter()
    old = strfeatures.build_reference(q_names, q_addrs, q_idx,
                                      names_arr.take(c).to_pylist(),
                                      addrs_arr.take(c).to_pylist(), idf_lut)
    t_old = time.perf_counter() - t0

    t0 = time.perf_counter()
    recs = strfeatures.precompute_records(names_arr, addrs_arr)
    t_pre = time.perf_counter() - t0
    t0 = time.perf_counter()
    new = strfeatures.build(recs, q_names, q_addrs, q_idx, c, idf_lut)
    t_new = time.perf_counter() - t0
    # second call: the idf array is cached on recs after the first
    t0 = time.perf_counter()
    strfeatures.build(recs, q_names, q_addrs, q_idx, c, idf_lut)
    t_new2 = time.perf_counter() - t0

    same = np.array_equal(old.view(np.uint32), new.view(np.uint32))
    print(f"bit-identical: {same}")
    print(f"old  gather+loop : {t_old:8.2f}s  {n / t_old:12,.0f} pairs/s")
    print(f"new  precompute  : {t_pre:8.2f}s  {len(c_names) / t_pre:12,.0f} records/s")
    print(f"new  build       : {t_new:8.2f}s  {n / t_new:12,.0f} pairs/s "
          f"(repeat {t_new2:.2f}s)")
    print(f"speed-up, build only          : {t_old / t_new:6.1f}x")
    print(f"speed-up, precompute included : {t_old / (t_new + t_pre):6.1f}x")
    b = recs.nbytes()
    print(f"Records: {b / 1e6:.1f} MB = {b / len(c_names):.0f} bytes/record")


if __name__ == "__main__":
    main()
