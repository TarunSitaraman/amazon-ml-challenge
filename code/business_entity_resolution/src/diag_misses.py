"""Why are the unreachable links unreachable?

Recall plateaus at ~93.5% and more retrieval depth does not move it
(EXPERIMENTS.md #6), so the remaining links share no usable key with their
query at ANY depth. This categorises them, because the category decides the
strategy:

  abbreviation      -> the corruption grammar belongs in the NORMALISER and
                       blocking keys, not only in the matcher. Fixable, large.
  cross-script      -> transliteration needs work. Fixable, bounded.
  address-only      -> name unusable but address shared; a stronger address
                       channel recovers these.
  no shared signal  -> total rewrite with a different address. Hard ceiling:
                       no orthographic method reaches these.

Reachability is computed from document frequencies with Counters rather than by
building the inverted indices, because build_index's np.unique over ~43M
postings spikes several GB and this machine does not have them spare.

"Reachable" here means a shared key survives the df cap -- the same condition
blocking needs. It is an upper bound on what retrieval can find, so anything
unreachable here is unreachable at any top-k.

Abbreviation is detected heuristically (one token a prefix of the other, or the
initials of it), so this runs before the grammar has been mined.

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
DF_CAP = blocking.DF_CAP
ADDR_DF_CAP = blocking.ADDR_DF_CAP


def abbrev_like(a, b):
    """Could a and b be an abbreviation pair? Deliberately permissive."""
    if not a or not b or a == b:
        return False
    s, t = (a, b) if len(a) < len(b) else (b, a)
    if len(s) < 2:
        return False
    return t.startswith(s) or (len(s) <= 5 and s == "".join(w[0] for w in t.split()))


def pair_abbrev(t1, t2):
    only1, only2 = t1 - t2, t2 - t1
    if not only1 or not only2:
        return False
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
    print(f"{country}: {len(links):,} true links from {n_ent:,} entities", flush=True)

    rec = {}
    for tab in (s1_tab, corpus_tab):
        for i, n, a in zip(tab.column("entity_id").to_pylist(),
                           tab.column("business_name").to_pylist(),
                           tab.column("business_address").to_pylist()):
            if i in want_s1 or i in want_m:
                rec[i] = (n, a)
    del s1_tab

    # Corpus document frequencies, streamed through Counters.
    ndf, adf, gdf = Counter(), Counter(), Counter()
    names = corpus_tab.column("business_name").to_pylist()
    addrs = corpus_tab.column("business_address").to_pylist()
    del corpus_tab
    for n, a in zip(names, addrs):
        nn = norm(n)
        ndf.update(set(nn.split()))
        adf.update(set(norm(a).split()))
        gdf.update(ngrams(nn, 4))
    del names, addrs
    print(f"df tables: {len(ndf):,} name tokens, {len(adf):,} address tokens, "
          f"{len(gdf):,} 4-grams", flush=True)

    cats = Counter()
    examples = {}
    n_reach = n_un = 0
    for a, b in links:
        if a not in rec or b not in rec:
            continue
        n1, a1 = rec[a]
        n2, a2 = rec[b]
        n1n, n2n = norm(n1), norm(n2)
        t1, t2 = set(n1n.split()), set(n2n.split())
        at1, at2 = set(norm(a1).split()), set(norm(a2).split())
        g1, g2 = ngrams(n1n, 4), ngrams(n2n, 4)

        # a key is usable only if it survives the df cap blocking applies
        name_ok = any(ndf[t] <= DF_CAP for t in t1 & t2)
        gram_ok = any(gdf[g] <= DF_CAP for g in g1 & g2)
        addr_ok = any(adf[t] <= ADDR_DF_CAP for t in at1 & at2)
        if name_ok or gram_ok or addr_ok:
            n_reach += 1
            continue

        n_un += 1
        cross = bool(DEVA.search(n1)) != bool(DEVA.search(n2))
        if cross:
            k = "cross-script"
        elif pair_abbrev(t1, t2):
            k = "abbreviation"
        elif (at1 & at2) or (nums(a1) & nums(a2)):
            k = "address shared but too common"
        elif (t1 & t2) or (g1 & g2):
            k = "name shared but too common"
        else:
            k = "no shared signal (hard ceiling)"
        cats[k] += 1
        examples.setdefault(k, (n1, a1, n2, a2))

    tot = n_reach + n_un
    print(f"\nreachable {n_reach:,}/{tot:,} = {n_reach/max(tot,1):.2%}   "
          f"UNREACHABLE {n_un:,} = {n_un/max(tot,1):.2%}\n", flush=True)
    print(f"{'why unreachable':36}{'count':>8}{'of these':>10}{'of ALL links':>14}")
    for k, v in cats.most_common():
        print(f"{k:36}{v:>8,}{v/max(n_un,1):>9.1%}{v/max(tot,1):>13.2%}")

    print("\none example of each:")
    for k, (n1, a1, n2, a2) in examples.items():
        print(f"\n  [{k}]")
        print(f"    S1 {n1!r}\n       @ {a1[:70]!r}")
        print(f"    -> {n2!r}\n       @ {a2[:70]!r}")

    fix = cats["abbreviation"] + cats["cross-script"]
    cap = cats["address shared but too common"] + cats["name shared but too common"]
    print(f"\nrecoverable by better KEYS (grammar/translit): {fix/max(tot,1):.2%} of links")
    print(f"recoverable by a smarter CAP (shared but common): {cap/max(tot,1):.2%}")
    print(f"hard ceiling (no shared signal at all): "
          f"{cats['no shared signal (hard ceiling)']/max(tot,1):.2%}")


if __name__ == "__main__":
    main()
