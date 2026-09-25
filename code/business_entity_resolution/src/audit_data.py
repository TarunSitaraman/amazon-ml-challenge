"""Stage 0 data audit: measure the structural properties that decide the architecture.

Answers, from the training ground truth only:
  A. Do true matches ever cross the country field?      -> is country a safe hard block?
  B. Can one S2/S3 record match more than one S1 entity? -> is this assignment or free-for-all?
  C. Joint (n_s2, n_s3) match distribution               -> are per-source priors different?
  D. What fraction of S2/S3 records match anything?      -> how much of the corpus is distractor?
"""
import pathlib
from collections import Counter

import numpy as np
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

P = pathlib.Path("data/parquet")


def read_ids(name):
    """entity_id + country for one source (country comes from the hive partition)."""
    return ds.dataset(P / name, format="parquet", partitioning="hive") \
             .to_table(columns=["entity_id", "country"])


def explode_ground_truth():
    """-> (s1_id repeated per link, match_id) with singletons dropped."""
    gt = pq.read_table(P / "train_ground_truth.parquet")
    s1 = gt.column("source1_entity_id")
    lists = pc.split_pattern(gt.column("matched_entity_ids").combine_chunks(), ",")
    flat = lists.flatten().to_numpy(zero_copy_only=False)
    counts = pc.list_value_length(lists).to_numpy(zero_copy_only=False)
    s1_rep = np.repeat(s1.to_numpy(zero_copy_only=False), counts)
    keep = flat != ""            # singleton rows split to a single empty string
    return s1_rep[keep], flat[keep], gt.num_rows


def main():
    s1_rep, match_id, n_entities = explode_ground_truth()
    print(f"links={len(match_id):,} over {n_entities:,} S1 entities\n")

    # --- B. multiplicity: does any S2/S3 record serve more than one S1 entity? ---
    uniq, cnt = np.unique(match_id, return_counts=True)
    multi = int((cnt > 1).sum())
    print("B. S2/S3 record re-use across S1 entities")
    print(f"   distinct matched records : {len(uniq):,}")
    print(f"   used by >1 S1 entity     : {multi:,} ({multi/len(uniq):.4%})")
    if multi:
        print(f"   max re-use               : {cnt.max()}")
        print(f"   re-use histogram         : {dict(Counter(cnt[cnt > 1]).most_common(5))}")

    # --- A. country crossing ---
    ctry = {}
    for src in ("train_source1", "train_source2", "train_source3"):
        t = read_ids(src)
        ctry.update(zip(t.column("entity_id").to_pylist(), t.column("country").to_pylist()))
    left = np.array([ctry.get(i) for i in s1_rep], dtype=object)
    right = np.array([ctry.get(i) for i in match_id], dtype=object)
    cross = int((left != right).sum())
    print("\nA. country crossing among true matches")
    print(f"   cross-country links : {cross:,} / {len(match_id):,} ({cross/len(match_id):.4%})")
    if cross:
        print(f"   examples : {Counter(zip(left[left != right], right[left != right])).most_common(5)}")

    # --- C. joint per-source match counts ---
    is_s2 = np.char.startswith(match_id.astype(str), "S2-")
    order = np.argsort(s1_rep, kind="stable")
    s1_sorted, s2_sorted = s1_rep[order], is_s2[order]
    _, starts = np.unique(s1_sorted, return_index=True)
    n_s2 = np.add.reduceat(s2_sorted.astype(np.int32), starts)
    n_all = np.diff(np.append(starts, len(s1_sorted)))
    joint = Counter(zip(n_s2.tolist(), (n_all - n_s2).tolist()))
    print("\nC. joint (n_S2, n_S3) for entities with >=1 match")
    for k, v in joint.most_common(10):
        print(f"   S2={k[0]} S3={k[1]} : {v:>8,} ({v/n_entities:6.2%})")
    only2 = sum(v for (a, b), v in joint.items() if b == 0)
    only3 = sum(v for (a, b), v in joint.items() if a == 0)
    print(f"   matched in S2 only : {only2:,}   S3 only : {only3:,}")

    # --- D. corpus coverage ---
    print("\nD. how much of S2/S3 is ever matched")
    matched = set(uniq.tolist())
    for src, pre in (("train_source2", "S2"), ("train_source3", "S3")):
        ids = read_ids(src).column("entity_id").to_pylist()
        hit = sum(1 for i in ids if i in matched)
        print(f"   {pre}: {hit:,}/{len(ids):,} matched ({hit/len(ids):.2%})"
              f"  -> {1 - hit/len(ids):.2%} are pure distractors")


if __name__ == "__main__":
    main()
