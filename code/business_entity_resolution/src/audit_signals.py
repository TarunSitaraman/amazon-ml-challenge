"""Stage 0 audit, part 2: how separable are true matches from random records?

Answers:
  E. How much of the problem is trivially solvable (exact / normalised name equality)?
  F. Chains: how many S1 entities share a normalised name? (S1 is deduplicated, so
     same-name S1 entities are DIFFERENT businesses -> the false-merge minefield.)
  I. Script asymmetry: is Devanagari confined to S2/S3, making transliteration
     a one-directional problem?
  J. Generic tokens that carry no identity signal.
"""
import pathlib
import random
import re
import unicodedata
from collections import Counter

import numpy as np
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

P = pathlib.Path("data/parquet")
SAMPLE = 100_000
NEG_POOL = 300_000
DEVA = re.compile(r"[ऀ-ॿ]")
NONALNUM = re.compile(r"[^0-9a-z]+")
random.seed(0)


def norm(s):
    s = unicodedata.normalize("NFKD", s or "").casefold()
    s = "".join(c for c in s if not unicodedata.combining(c))
    return NONALNUM.sub(" ", s).strip()


def toks(s):
    return set(norm(s).split())


def jac(a, b):
    return len(a & b) / len(a | b) if (a or b) else 1.0


def nums(s):
    return {t for t in toks(s) if any(c.isdigit() for c in t)}


def load(name, cols=("entity_id", "business_name", "business_address")):
    return ds.dataset(P / name, format="parquet", partitioning="hive").to_table(
        columns=list(cols) + ["country"])


def index_by_id(table, wanted=None):
    ids = table.column("entity_id").to_pylist()
    nm = table.column("business_name").to_pylist()
    ad = table.column("business_address").to_pylist()
    ct = table.column("country").to_pylist()
    if wanted is None:
        return {i: (n, a, c) for i, n, a, c in zip(ids, nm, ad, ct)}
    return {i: (n, a, c) for i, n, a, c in zip(ids, nm, ad, ct) if i in wanted}


def pair_stats(label, pairs):
    """pairs: list of ((name,addr,country) left, right)"""
    if not pairs:
        return
    raw_eq = nm_eq = addr_num_hit = deva_mix = 0
    njac, ajac, contain = [], [], 0
    for (n1, a1, _), (n2, a2, _) in pairs:
        raw_eq += n1 == n2
        t1, t2 = toks(n1), toks(n2)
        nm_eq += norm(n1) == norm(n2)
        j = jac(t1, t2)
        njac.append(j)
        if t1 and t2 and (t1 <= t2 or t2 <= t1):
            contain += 1
        ajac.append(jac(toks(a1), toks(a2)))
        x, y = nums(a1), nums(a2)
        addr_num_hit += bool(x & y)
        deva_mix += bool(DEVA.search(n1)) != bool(DEVA.search(n2))
    n = len(pairs)
    print(f"  {label:26s} n={n:,}")
    print(f"    name raw equal      {raw_eq/n:7.2%}   name normalised equal {nm_eq/n:7.2%}")
    print(f"    name token jaccard  mean {np.mean(njac):.3f}  median {np.median(njac):.3f}"
          f"  ==0 {np.mean(np.array(njac) == 0):6.2%}  >=0.5 {np.mean(np.array(njac) >= .5):6.2%}")
    print(f"    one name ⊆ other    {contain/n:7.2%}")
    print(f"    addr token jaccard  mean {np.mean(ajac):.3f}  shares a number {addr_num_hit/n:7.2%}")
    print(f"    script differs      {deva_mix/n:7.2%}")


def main():
    # ---------- sample true pairs ----------
    gt = pq.read_table(P / "train_ground_truth.parquet")
    s1_ids = gt.column("source1_entity_id").to_pylist()
    matched = gt.column("matched_entity_ids").to_pylist()
    idx = [i for i in range(len(s1_ids)) if matched[i]]
    pick = random.sample(idx, min(SAMPLE, len(idx)))
    want_s1 = {s1_ids[i] for i in pick}
    links = [(s1_ids[i], m) for i in pick for m in matched[i].split(",") if m]
    want_m = {m for _, m in links}
    print(f"sampled {len(want_s1):,} S1 entities -> {len(links):,} true links\n")

    t1 = load("train_source1")
    m1 = index_by_id(t1, want_s1)
    m23, pool = {}, []
    for src in ("train_source2", "train_source3"):
        t = load(src)
        m23.update(index_by_id(t, want_m))
        allids = t.column("entity_id").to_pylist()
        sel = random.sample(range(len(allids)), min(NEG_POOL, len(allids)))
        nm, ad, ct = (t.column(c).to_pylist() for c in
                      ("business_name", "business_address", "country"))
        pool += [(nm[i], ad[i], ct[i]) for i in sel]

    by_ctry = {}
    for rec in pool:
        by_ctry.setdefault(rec[2], []).append(rec)

    print("E. true matches vs random same-country negatives")
    for tag, pref in (("S2", "S2-"), ("S3", "S3-")):
        pr = [(m1[a], m23[b]) for a, b in links
              if b.startswith(pref) and a in m1 and b in m23]
        pair_stats(f"TRUE {tag}", pr)
    neg = []
    for a in list(want_s1)[:20000]:
        if a in m1:
            c = m1[a][2]
            if by_ctry.get(c):
                neg.append((m1[a], random.choice(by_ctry[c])))
    pair_stats("RANDOM same-country", neg)

    # ---------- F. chains in S1 ----------
    print("\nF. chains: S1 entities sharing a normalised business_name")
    names = t1.column("business_name").to_pylist()
    cn = Counter(norm(x) for x in names)
    dup = {k: v for k, v in cn.items() if v > 1 and k}
    shared = sum(dup.values())
    print(f"   distinct normalised names {len(cn):,} over {len(names):,} S1 entities")
    print(f"   names used by >1 entity   {len(dup):,}")
    print(f"   entities in a name-collision group {shared:,} ({shared/len(names):.2%})")
    print(f"   worst offenders {Counter(dup).most_common(8)}")

    # ---------- I. script asymmetry ----------
    print("\nI. Devanagari presence by source (name field)")
    for src in ("train_source1", "train_source2", "train_source3"):
        t = t1 if src == "train_source1" else load(src, ("entity_id", "business_name",
                                                         "business_address"))
        nm = t.column("business_name").to_pylist()
        ct = t.column("country").to_pylist()
        d = sum(1 for x in nm if DEVA.search(x))
        ind = sum(1 for c in ct if c == "India")
        print(f"   {src:16s} devanagari {d:>9,} / {len(nm):,} ({d/len(nm):6.2%})"
              f"   = {d/max(ind,1):6.2%} of its India rows")
        if src == "train_source1":
            del t

    # ---------- J. generic tokens ----------
    print("\nJ. most frequent S1 name tokens (identity-poor, IDF-suppressed)")
    tc = Counter()
    for x in names:
        tc.update(norm(x).split())
    for tok, c in tc.most_common(25):
        print(f"   {tok:18s} {c:>8,} ({c/len(names):6.2%})")


if __name__ == "__main__":
    main()
