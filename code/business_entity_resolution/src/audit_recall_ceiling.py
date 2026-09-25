"""Stage 0 audit, part 3: the blocking RECALL CEILING.

A true match that no blocking channel retrieves is permanently lost, so the union
of channels caps the achievable F_0.5. This measures, per candidate channel, the
fraction of TRUE links it can reach — and crucially the fraction reachable by
NOTHING, broken out by country and by script mismatch.
"""
import pathlib
import random
from collections import Counter

import pyarrow.dataset as ds
import pyarrow.parquet as pq

from textnorm import DEVA, ngrams, norm, nums, toks

P = pathlib.Path("data/parquet")
SAMPLE = 60_000
RARE_DF = 100
random.seed(0)


def load(name):
    return ds.dataset(P / name, format="parquet", partitioning="hive").to_table(
        columns=["entity_id", "business_name", "business_address", "country"])


def main():
    gt = pq.read_table(P / "train_ground_truth.parquet")
    s1_ids = gt.column("source1_entity_id").to_pylist()
    matched = gt.column("matched_entity_ids").to_pylist()
    idx = [i for i in range(len(s1_ids)) if matched[i]]
    pick = random.sample(idx, SAMPLE)
    want_s1 = {s1_ids[i] for i in pick}
    links = [(s1_ids[i], m) for i in pick for m in matched[i].split(",") if m]
    want_m = {m for _, m in links}

    t1 = load("train_source1")
    rec = {}
    for i, n, a, c in zip(t1.column("entity_id").to_pylist(),
                          t1.column("business_name").to_pylist(),
                          t1.column("business_address").to_pylist(),
                          t1.column("country").to_pylist()):
        if i in want_s1:
            rec[i] = (n, a, c)

    df = Counter()                      # name-token document frequency in S2+S3
    for src in ("train_source2", "train_source3"):
        t = load(src)
        for i, n, a, c in zip(t.column("entity_id").to_pylist(),
                              t.column("business_name").to_pylist(),
                              t.column("business_address").to_pylist(),
                              t.column("country").to_pylist()):
            df.update(toks(n))
            if i in want_m:
                rec[i] = (n, a, c)
        del t

    chans = ["name_token", "name_rare_token", "name_4gram", "addr_number",
             "addr_token", "addr_rare_token"]
    hit = Counter()
    tot = Counter()
    miss_examples = []

    for a, b in links:
        if a not in rec or b not in rec:
            continue
        (n1, a1, c1), (n2, a2, _) = rec[a], rec[b]
        t1n, t2n = toks(n1), toks(n2)
        shared = t1n & t2n
        ta1, ta2 = toks(a1), toks(a2)
        shared_a = ta1 & ta2
        got = {
            "name_token": bool(shared),
            "name_rare_token": any(df[t] <= RARE_DF for t in shared),
            "name_4gram": bool(ngrams(n1) & ngrams(n2)),
            "addr_number": bool(nums(a1) & nums(a2)),
            "addr_token": bool(shared_a),
            "addr_rare_token": any(df[t] <= RARE_DF for t in shared_a),
        }
        src = "S2" if b.startswith("S2-") else "S3"
        scripts = DEVA.search(n1) is not None, DEVA.search(n2) is not None
        strata = ["ALL", f"src={src}", f"country={c1}",
                  "script_mismatch" if scripts[0] != scripts[1] else "same_script"]
        for s in strata:
            tot[s] += 1
            for k, v in got.items():
                hit[(s, k)] += v
            hit[(s, "UNION")] += any(got.values())
            hit[(s, "name_any")] += got["name_token"] or got["name_4gram"]
            hit[(s, "addr_any")] += got["addr_number"] or got["addr_token"]
        if not any(got.values()) and len(miss_examples) < 8:
            miss_examples.append((n1, a1, n2, a2))

    order = ["name_token", "name_rare_token", "name_4gram", "name_any",
             "addr_number", "addr_token", "addr_rare_token", "addr_any", "UNION"]
    strata = ["ALL", "src=S2", "src=S3", "country=US", "country=India",
              "same_script", "script_mismatch"]
    w = max(len(c) for c in order) + 2
    print(f"{'channel':{w}}" + "".join(f"{s:>17}" for s in strata))
    for c in order:
        row = f"{c:{w}}"
        for s in strata:
            row += f"{hit[(s,c)]/tot[s]:>16.2%} " if tot[s] else f"{'-':>17}"
        print(row)
    print("\nlinks per stratum: " + ", ".join(f"{s}={tot[s]:,}" for s in strata))

    print("\nTrue links NO channel reaches (the permanent recall loss):")
    for n1, a1, n2, a2 in miss_examples:
        print(f"  S1 {n1!r} @ {a1!r}\n  -> {n2!r} @ {a2!r}\n")


if __name__ == "__main__":
    main()
