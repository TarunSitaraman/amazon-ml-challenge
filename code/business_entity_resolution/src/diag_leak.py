"""Look for structure in IDs, row order and distractors that the matcher ignores.

The leaderboard top is at 0.990-0.991, above our perfect-matcher ceiling
(0.978). Text similarity alone should not get there on 39% chain entities and
Devanagari names, so this checks whether the synthetic generator left
structure a team could exploit. Nothing here has been checked before: the whole
pipeline treats IDs as opaque strings and never looks at row order.

Reads data/parquet only (train ground truth for labels). Prints; writes nothing.

  L1  ID format and range per source/split
  L2  do linked S2/S3 IDs track the S1 ID?        (Spearman on numeric IDs)
  L3  are one entity's records neighbours?       (consecutive IDs -> same S1?)
  L4  same two checks on file row position (per country partition)
  L5  do distractors sit in their own ID / row range?
  L6  record-level distractor signatures (no pair needed)
  L7  S2<->S3 twins: does a matched S2 record have an exact-name S3 twin
      of the same entity more often than a distractor has one?
  L8  if output/matching_results.tsv exists, repeat L3 on OUR test predictions,
      to see whether any train structure also holds in test

Usage (from code/business_entity_resolution):
    python src/diag_leak.py [--countries India,US] [--sample 300000]
"""
import argparse
import pathlib
import re

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

P = pathlib.Path("data/parquet")
RNG = np.random.default_rng(0)


def load(split, src, cols=("entity_id", "country")):
    d = ds.dataset(P / f"{split}_source{src}", format="parquet", partitioning="hive")
    return d.to_table(columns=list(cols))


def id_num(arr):
    """'S2-123' -> 123 (the trailing digit run). Returns (int64 array, ok mask)."""
    s = pc.replace_substring_regex(arr, r"^.*?(\d+)$", r"\1")
    ok = pc.match_substring_regex(s, r"^\d+$").to_numpy(zero_copy_only=False)
    out = np.full(len(arr), -1, np.int64)
    if ok.any():
        out[ok] = pc.cast(pc.filter(s, pa.array(ok)), pa.int64()).to_numpy()
    return out, ok


def pos_in(values, value_set):
    """Row position of each value in value_set, -1 when absent."""
    return pc.fill_null(pc.index_in(values, value_set), -1).to_numpy(zero_copy_only=False)


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def explode_gt():
    gt = pq.read_table(P / "train_ground_truth.parquet")
    lists = pc.split_pattern(gt.column("matched_entity_ids").combine_chunks(), ",")
    flat = lists.flatten()
    parent = pc.list_parent_indices(lists)
    keep = pc.not_equal(flat, "")
    s1 = pc.take(gt.column("source1_entity_id").combine_chunks(), pc.filter(parent, keep))
    return s1, pc.filter(flat, keep), gt


def neighbour_share(key, owner):
    """Sort by key; share of adjacent pairs with the same owner, and the
    random baseline for that owner distribution."""
    o = owner[np.argsort(key, kind="stable")]
    same = float((o[1:] == o[:-1]).mean())
    _, c = np.unique(owner, return_counts=True)
    p = c / c.sum()
    return same, float((p * p).sum())


def q(x):
    return " ".join(f"{v:.3f}" for v in np.quantile(x, [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--countries", default="")
    ap.add_argument("--sample", type=int, default=300000)
    a = ap.parse_args()
    only = set(a.countries.split(",")) - {""}

    print("== L1 ID format ==")
    tabs = {}
    for split in ("train", "test"):
        for src in (1, 2, 3):
            t = load(split, src)
            if only:
                t = t.filter(pc.is_in(t["country"], pa.array(sorted(only))))
            tabs[split, src] = t
            num, ok = id_num(t["entity_id"])
            ex = t["entity_id"].slice(0, 3).to_pylist()
            print(f"{split} S{src}: n={t.num_rows:,} numeric={ok.mean():.4%} "
                  f"min={num[ok].min() if ok.any() else '-'} max={num[ok].max() if ok.any() else '-'} "
                  f"unique={len(np.unique(num[ok])):,} examples={ex}")
            for c in pc.unique(t["country"]).to_pylist():
                m = pc.equal(t["country"], c).to_numpy(zero_copy_only=False) & ok
                if m.any():
                    print(f"    {c:8s} id range {num[m].min():>10,}..{num[m].max():>10,}  n={m.sum():,}")

    s1_link, rec_link, gt = explode_gt()
    s1_link_num, _ = id_num(s1_link)
    rec_link_num, _ = id_num(rec_link)
    rec_src = np.where(pc.starts_with(rec_link, "S2").to_numpy(zero_copy_only=False), 2, 3)
    s1_train = tabs["train", 1]
    keep_s1 = pc.is_in(s1_link, s1_train["entity_id"]).to_numpy(zero_copy_only=False)

    print("\n== L2 do linked record IDs track the S1 ID? ==")
    for src in (2, 3):
        m = (rec_src == src) & keep_s1
        idx = np.flatnonzero(m)
        if len(idx) > a.sample:
            idx = RNG.choice(idx, a.sample, replace=False)
        print(f"S{src}: spearman(S1 id, S{src} id) = {spearman(s1_link_num[idx], rec_link_num[idx]):+.4f}"
              f"  over {len(idx):,} links (0 = no relation, near 1 = leak)")
        # normalised rank offset: if IDs were assigned in entity order the
        # offset is concentrated; independent IDs give a triangular spread.
        s1n = np.argsort(np.argsort(s1_link_num[m])) / m.sum()
        rn = np.argsort(np.argsort(rec_link_num[m])) / m.sum()
        print(f"    rank(rec)-rank(S1) quantiles 1/10/25/50/75/90/99%: {q(rn - s1n)}")

    print("\n== L3 are one entity's records neighbours in ID order? ==")
    for src in (2, 3):
        m = (rec_src == src) & keep_s1
        owner = np.unique(s1_link.to_numpy(zero_copy_only=False)[m], return_inverse=True)[1]
        same, base = neighbour_share(rec_link_num[m], owner)
        print(f"S{src}: adjacent-ID pairs with the same S1 owner {same:.4%}  (random {base:.6%})")
        # S2 and S3 of one entity interleaved?
    m = keep_s1
    owner = np.unique(s1_link.to_numpy(zero_copy_only=False)[m], return_inverse=True)[1]
    same, base = neighbour_share(rec_link_num[m] * 4 + rec_src[m], owner)
    print(f"S2+S3 merged by numeric id: same owner {same:.4%} (random {base:.6%})")

    print("\n== L4 / L5 row position and distractor placement (train, per country) ==")
    for src in (2, 3):
        t = tabs["train", src]
        num, _ = id_num(t["entity_id"])
        matched = pc.is_in(t["entity_id"], rec_link).to_numpy(zero_copy_only=False)
        for c in pc.unique(t["country"]).to_pylist():
            cm = pc.equal(t["country"], c).to_numpy(zero_copy_only=False)
            pos = np.arange(cm.sum()) / cm.sum()             # row position in partition
            idr = np.argsort(np.argsort(num[cm])) / cm.sum()  # rank of numeric id
            mm = matched[cm]
            print(f"S{src} {c:8s} distractors {(~mm).mean():.2%}")
            print(f"    row pos  matched  q: {q(pos[mm])}")
            print(f"    row pos  distract q: {q(pos[~mm])}")
            print(f"    id rank  matched  q: {q(idr[mm])}")
            print(f"    id rank  distract q: {q(idr[~mm])}")
            print(f"    corr(row pos, id rank) = {np.corrcoef(pos, idr)[0, 1]:+.4f}")
            # Row-order neighbour test for this country.
            ids_c = t["entity_id"].filter(pa.array(cm))
            at = pos_in(rec_link, ids_c)
            hit = at >= 0
            if hit.sum() > 1:
                owner = np.unique(s1_link.to_numpy(zero_copy_only=False)[hit], return_inverse=True)[1]
                same, base = neighbour_share(at[hit], owner)
                print(f"    adjacent ROWS with same S1 owner {same:.4%} (random {base:.6%})")
            # S1 row position vs linked record row position
            s1c = s1_train.filter(pc.equal(s1_train["country"], c))
            s1pos = pos_in(s1_link, s1c["entity_id"])
            both = hit & (s1pos >= 0)
            if both.sum() > 10:
                idx = np.flatnonzero(both)
                if len(idx) > a.sample:
                    idx = RNG.choice(idx, a.sample, replace=False)
                print(f"    spearman(S1 row, S{src} row) = {spearman(s1pos[idx], at[idx]):+.4f}")

    print("\n== L6 record-level distractor signatures (train) ==")
    for src in (2, 3):
        t = load("train", src, ("entity_id", "business_name", "business_address", "country"))
        if only:
            t = t.filter(pc.is_in(t["country"], pa.array(sorted(only))))
        n = t.num_rows
        idx = np.sort(RNG.choice(n, min(n, a.sample), replace=False))
        t = t.take(pa.array(idx))
        matched = pc.is_in(t["entity_id"], rec_link).to_numpy(zero_copy_only=False)
        name = t["business_name"].to_pylist()
        addr = [x or "" for x in t["business_address"].to_pylist()]
        feats = {
            "name chars": np.array([len(x) for x in name]),
            "name tokens": np.array([len(x.split()) for x in name]),
            "addr chars": np.array([len(x) for x in addr]),
            "addr empty": np.array([x == "" for x in addr]),
            "devanagari": np.array([bool(re.search("[ऀ-ॿ]", x)) for x in name]),
            "junk <<,--,##": np.array([bool(re.search(r"<<|>>|--|##|\*\*", x)) for x in name]),
            "all caps": np.array([x.isupper() for x in name]),
            "all lower": np.array([x.islower() for x in name]),
            "digit in name": np.array([bool(re.search(r"\d", x)) for x in name]),
            "leading/trailing ws": np.array([x != x.strip() for x in name]),
            "double space": np.array(["  " in x for x in name]),
            "postcode-like": np.array([bool(re.search(r"\b\d{5,6}\b", x)) for x in addr]),
        }
        print(f"S{src}: {len(idx):,} sampled records, {(~matched).mean():.2%} distractors")
        print(f"    {'feature':22s} {'matched':>9s} {'distract':>9s}")
        for k, v in feats.items():
            v = v.astype(np.float64)
            print(f"    {k:22s} {v[matched].mean():9.4f} {v[~matched].mean():9.4f}")
        # exact name present in S1?
        s1n = pc.utf8_lower(load("train", 1, ("business_name",))["business_name"])
        inn = pc.is_in(pc.utf8_lower(t["business_name"]), s1n).to_numpy(zero_copy_only=False)
        print(f"    {'name == some S1 name':22s} {inn[matched].mean():9.4f} {inn[~matched].mean():9.4f}")

    print("\n== L7 S2<->S3 exact-name twins ==")
    s2 = load("train", 2, ("entity_id", "business_name", "country"))
    s3 = load("train", 3, ("entity_id", "business_name", "country"))
    k2 = pc.binary_join_element_wise(s2["country"], pc.utf8_lower(s2["business_name"]), "|")
    k3 = pc.binary_join_element_wise(s3["country"], pc.utf8_lower(s3["business_name"]), "|")
    tw = pc.is_in(k2, k3).to_numpy(zero_copy_only=False)
    m2 = pc.is_in(s2["entity_id"], rec_link).to_numpy(zero_copy_only=False)
    print(f"S2 record has an exact-name S3 record: matched {tw[m2].mean():.4f}  "
          f"distractor {tw[~m2].mean():.4f}")

    out = pathlib.Path("output/matching_results.tsv")
    if out.exists():
        print("\n== L8 our TEST predictions: are predicted records ID-neighbours? ==")
        s1s, recs = [], []
        with open(out, encoding="utf-8") as fh:
            next(fh)
            for line in fh:
                e, ids = line.rstrip("\n").split("\t")
                for r in filter(None, ids.split(",")):
                    s1s.append(e)
                    recs.append(r)
        recs = pa.array(recs)
        num, _ = id_num(recs)
        src = np.where(pc.starts_with(recs, "S2").to_numpy(zero_copy_only=False), 2, 3)
        s1n, _ = id_num(pa.array(s1s))
        owner = np.unique(np.array(s1s, object), return_inverse=True)[1]
        for s in (2, 3):
            m = src == s
            same, base = neighbour_share(num[m], owner[m])
            idx = np.flatnonzero(m)
            if len(idx) > a.sample:
                idx = RNG.choice(idx, a.sample, replace=False)
            print(f"S{s}: adjacent predicted IDs, same S1 {same:.4%} (random {base:.6%}); "
                  f"spearman(S1 id, S{s} id) {spearman(s1n[idx], num[idx]):+.4f}")
    else:
        print("\n(L8 skipped: no output/matching_results.tsv)")


if __name__ == "__main__":
    main()
