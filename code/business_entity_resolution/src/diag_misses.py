"""Why do we miss the links we miss?

Recall plateaus at ~93.5% and more retrieval depth does not move it
(EXPERIMENTS.md #6), so the remaining links are ones our channels cannot reach
at ANY depth. This categorises them, because the category decides the strategy:

  abbreviation      -> the corruption grammar belongs in the NORMALISER and
                       blocking keys, not only in the matcher. Fixable, large.
  cross-script      -> transliteration needs work. Fixable, bounded.
  address-only      -> the name is unusable but the address is shared; a
                       stronger address channel recovers these.
  no shared signal  -> total rewrite / DBA name with a different address.
                       A hard ceiling: no orthographic method reaches these.

Abbreviation is detected heuristically, without the mined grammar, so this can
run before the miner: one token being a prefix of the other (>=2 chars), or one
being the initials of the other's words.

Usage: python diag_misses.py [country] [n_entities]
"""
import pathlib
import random
import sys
from collections import Counter

import pyarrow.parquet as pq

import blocking
from textnorm import DEVA, ngrams, norm, nums

ROOT = "data/parquet"
random.seed(0)


def abbrev_like(a, b):
    """Could a and b be an abbreviation pair? Deliberately permissive."""
    if not a or not b or a == b:
        return False
    s, t = (a, b) if len(a) < len(b) else (b, a)
    if len(s) < 2:
        return False
    return t.startswith(s) or (len(s) <= 5 and s == "".join(w[0] for w in t.split()))


def pair_abbrev(t1, t2):
    """Any cross-token abbreviation relation between two token sets."""
    only1, only2 = t1 - t2, t2 - t1
    return any(abbrev_like(a, b) for a in only1 for b in only2)


def main():
    country = sys.argv[1] if len(sys.argv) > 1 else "India"
    n_ent = int(sys.argv[2]) if len(sys.argv) > 2 else 4000

    s1_tab, corpus_tab = blocking.load_shard(ROOT, "train", country)
    s1_ids = s1_tab.column("entity_id").to_pylist()
    gt = pq.read_table(pathlib.Path(ROOT) / "train_ground_truth.parquet")
    gtm = dict(zip(gt.column("source1_entity_id").to_pylist(),
                   gt.column("matched_entity_ids").to_pylist()))

    pick = random.sample(range(len(s1_ids)), n_ent)
    want_s1 = {s1_ids[i] for i in pick}
    links = [(s1_ids[i], m) for i in pick
             for m in (gtm.get(s1_ids[i]) or "").split(",") if m]
    want_m = {m for _, m in links}
    print(f"{country}: {len(links):,} true links from {n_ent:,} entities")

    rec = {}
    for tab in (s1_tab, corpus_tab):
        for i, n, a in zip(tab.column("entity_id").to_pylist(),
                           tab.column("business_name").to_pylist(),
                           tab.column("business_address").to_pylist()):
            if i in want_s1 or i in want_m:
                rec[i] = (n, a)

    # Retrieve with the CURRENT production settings.
    corpus = blocking.build_corpus(corpus_tab, verbose=False)
    c_ids = corpus_tab.column("entity_id").to_pylist()
    sub = s1_tab.take(pick)
    names, addrs = blocking.normalise(sub)
    q, c, _ = blocking.generate(None, None, names, addrs, corpus, verbose=False)
    got = [set() for _ in pick]
    for qi, ci in zip(q, c):
        got[qi].add(c_ids[ci])
    retrieved = {}
    for j, i in enumerate(pick):
        retrieved[s1_ids[i]] = got[j]

    cats = Counter()
    examples = {}
    n_miss = n_hit = 0
    for a, b in links:
        if a not in rec or b not in rec:
            continue
        if b in retrieved.get(a, ()):
            n_hit += 1
            continue
        n_miss += 1
        n1, a1 = rec[a]
        n2, a2 = rec[b]
        n1n, n2n = norm(n1), norm(n2)
        t1, t2 = set(n1n.split()), set(n2n.split())
        at1, at2 = set(norm(a1).split()), set(norm(a2).split())
        cross = bool(DEVA.search(n1)) != bool(DEVA.search(n2))
        shares_name = bool(t1 & t2)
        shares_gram = bool(ngrams(n1n, 4) & ngrams(n2n, 4))
        shares_addr = bool(at1 & at2)
        shares_dig = bool(nums(a1) & nums(a2))

        if cross:
            k = "cross-script"
        elif not shares_name and pair_abbrev(t1, t2):
            k = "abbreviation"
        elif shares_addr or shares_dig:
            k = "address-only (name unusable)"
        elif shares_name or shares_gram:
            k = "shared signal but ranked out"
        else:
            k = "no shared signal (hard ceiling)"
        cats[k] += 1
        examples.setdefault(k, (n1, a1, n2, a2))

    tot = n_hit + n_miss
    print(f"\nretrieved {n_hit:,}/{tot:,} = {n_hit/max(tot,1):.2%}   "
          f"missed {n_miss:,} = {n_miss/max(tot,1):.2%}\n")
    print(f"{'why the miss':34}{'count':>9}{'of misses':>11}{'of ALL links':>14}")
    for k, v in cats.most_common():
        print(f"{k:34}{v:>9,}{v/max(n_miss,1):>10.1%}{v/max(tot,1):>13.2%}")

    print("\none example of each:")
    for k, (n1, a1, n2, a2) in examples.items():
        print(f"\n  [{k}]")
        print(f"    S1 {n1!r}\n       @ {a1[:70]!r}")
        print(f"    -> {n2!r}\n       @ {a2[:70]!r}")

    fixable = cats["abbreviation"] + cats["cross-script"] + cats["address-only (name unusable)"]
    print(f"\nreachable in principle by better keys: {fixable:,} "
          f"({fixable/max(tot,1):.2%} of all links)")
    print(f"hard ceiling: {cats['no shared signal (hard ceiling)']/max(tot,1):.2%} of all links")


if __name__ == "__main__":
    main()
