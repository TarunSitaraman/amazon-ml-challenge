"""Mine the generator's corruption grammar from the aligned training pairs.

RESEARCH.md section 6: the 7.6M (S1 entity, S2/S3 record) links in
train_ground_truth are a supervised corpus of exactly how this generator
corrupts text. This learns the tables instead of hand-writing them, and every
entry carries its support count so a reviewer can check it came from the
provided files and not from a downloaded gazetteer.

S1 is the clean side (100% Latin, deduplicated), so each pair is read as
S1 token -> S2/S3 token. Tokens are aligned per field by Hungarian matching on
normalised edit similarity with an explicit NULL option (drop / insert), after
exact matches are taken first. A pair is mined only when its alignment is
confident: at least MIN_CONF of the tokens (of the shorter side, and half of
S1's) matched at high similarity. That
biases every table toward mild corruptions, on purpose: a table polluted by
alignment errors is worse than one that misses the heavy tail. Drop rates are
reported over all pairs too, because the confidence gate is itself a filter on
drops.

Emits one JSON with:
  abbreviations        S1 token -> S2 token (abbreviation or expansion), with
                       P(s2 | s1) and P(s1 | s2), per source and per country
  acronyms             multi-token -> initials ("uttar pradesh" -> "up")
  region_codes         the address abbreviations that sit in the tail of the
                       address (heuristic: short form <= 3 letters, long form's
                       mean relative position >= 0.6), i.e. state/region codes
  forbidden            pairs of variable tokens the generator never maps onto
                       each other despite many chances (Inc <-> Corp); the
                       precision half, with a rule-of-three upper bound
  junk_affixes         raw punctuation affixes the generator adds ("<<", "--")
  junk_tokens          alphanumeric tokens inserted into S2/S3 names
  token_drop_by_idf    drop rate binned by S1 token IDF
  numeral_mutation     how often a digit token changes or disappears
  permutations         histogram of reorder types against a uniform-random
                       baseline, and which tokens get moved
  source_signatures    per-source corruption mix and a Jensen-Shannon
                       divergence between S2 and S3 per family

Usage (from code/business_entity_resolution, src/ on PYTHONPATH):
    python src/mine_corruption.py --sample 200000          # subset first
    python src/mine_corruption.py --workers 8              # all 7.6M pairs
    python src/mine_corruption.py --self-test              # synthetic check
"""
import argparse
import datetime
import itertools
import json
import math
import multiprocessing as mp
import pathlib
import random
import re
import sys
import time
from collections import Counter, defaultdict
from functools import lru_cache

import numpy as np
from scipy.optimize import linear_sum_assignment

from textnorm import norm

FIELDS = ("name", "address")
SOURCES = ("S2", "S3")
MIN_ALIGN = 0.5       # below this, two tokens may not be aligned at all
HIGH_SIM = 0.75       # "matched at high similarity" for the confidence gate
MIN_CONF = 0.70       # share of tokens that must match at HIGH_SIM (confident())
ABBREV_SIM = 0.9      # score given to a subsequence abbreviation
MAX_RESIDUAL = 25     # longer residuals are not aligned (pathological rows)
_BIG = 1e6
_NULL_COST = (1 - MIN_ALIGN) / 2 + 1e-9   # any allowed pair beats two NULLs
_PUNCT_TOKEN = re.compile(r"^[^\w]+$")
_LEAD = re.compile(r"^[^\w]+")
_TRAIL = re.compile(r"[^\w]+$")


# ---- token similarity ------------------------------------------------------

@lru_cache(maxsize=1 << 20)
def lev(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _is_subseq(short: str, long: str) -> bool:
    it = iter(long)
    return all(c in it for c in short)


def is_abbrev(short: str, long: str) -> bool:
    """`short` abbreviates `long`: same first letter, strictly shorter, an
    in-order subsequence (corp/corporation, rd/road, pvt/private, tx/texas)."""
    return (2 <= len(short) < len(long) and short[0] == long[0]
            and short.isalpha() and long.isalpha() and _is_subseq(short, long))


@lru_cache(maxsize=1 << 20)
def sim(a: str, b: str):
    """-> (score, kind) for S1 token a aligned to S2/S3 token b."""
    if a == b:
        return 1.0, "identity"
    s = 1 - lev(a, b) / max(len(a), len(b))
    if s < 0.8:
        if is_abbrev(b, a):
            return ABBREV_SIM, "abbrev"
        if is_abbrev(a, b):
            return ABBREV_SIM, "expansion"
    return s, "typo"


# ---- alignment -------------------------------------------------------------

def align(a, b):
    """Align token lists a (S1) and b (S2/S3).

    -> (pairs, drop_a, ins_b): pairs is [(i, j, score, kind)], drop_a and
    ins_b the indices aligned to NULL. Exact matches are taken first (in order,
    as a multiset); the residual goes to Hungarian with a NULL row/column."""
    pos = defaultdict(list)
    for j, t in enumerate(b):
        pos[t].append(j)
    pairs, ra = [], []
    used = set()
    for i, t in enumerate(a):
        if pos.get(t):
            j = pos[t].pop(0)
            pairs.append((i, j, 1.0, "identity"))
            used.add(j)
        else:
            ra.append(i)
    rb = [j for j in range(len(b)) if j not in used]
    if not ra or not rb or len(ra) > MAX_RESIDUAL or len(rb) > MAX_RESIDUAL:
        return pairs, ra, rb
    n, m = len(ra), len(rb)
    S = [[sim(a[i], b[j]) for j in rb] for i in ra]
    if n == 1 or m == 1:                       # no assignment problem to solve
        best = max(((S[x][y][0], x, y) for x in range(n) for y in range(m)))
        if best[0] >= MIN_ALIGN:
            _, x, y = best
            pairs.append((ra[x], rb[y]) + S[x][y])
            return pairs, [i for k, i in enumerate(ra) if k != x], \
                [j for k, j in enumerate(rb) if k != y]
        return pairs, ra, rb
    C = np.full((n + m, m + n), _BIG)
    for x in range(n):
        for y in range(m):
            if S[x][y][0] >= MIN_ALIGN:
                C[x, y] = 1 - S[x][y][0]
        C[x, m + x] = _NULL_COST
    for y in range(m):
        C[n + y, y] = _NULL_COST
    C[n:, m:] = 0
    rows, cols = linear_sum_assignment(C)
    got_a, got_b = set(), set()
    for x, y in zip(rows, cols):
        if x < n and y < m and C[x, y] < _BIG:
            pairs.append((ra[x], rb[y]) + S[x][y])
            got_a.add(x)
            got_b.add(y)
    return (pairs, [i for k, i in enumerate(ra) if k not in got_a],
            [j for k, j in enumerate(rb) if k not in got_b])


def find_acronyms(a, b, drop_a, ins_b):
    """Unaligned token = initials of a contiguous run of unaligned tokens on the
    other side. -> [(long_phrase, short, direction, covered_a, covered_b)]."""
    out = []
    da, ib = set(drop_a), set(ins_b)

    def runs(toks, free):
        for s in range(len(toks)):
            for e in range(s + 2, min(len(toks), s + 5) + 1):
                if all(k in free for k in range(s, e)):
                    yield s, e

    for j in sorted(ib):
        t = b[j]
        if t.isalpha() and 2 <= len(t) <= 5:
            for s, e in runs(a, da):
                if e - s == len(t) and "".join(x[0] for x in a[s:e]) == t:
                    out.append((" ".join(a[s:e]), t, "contract", range(s, e), [j]))
                    da -= set(range(s, e)); ib.discard(j)
                    break
    for i in sorted(da):
        t = a[i]
        if t.isalpha() and 2 <= len(t) <= 5:
            for s, e in runs(b, ib):
                if e - s == len(t) and "".join(x[0] for x in b[s:e]) == t:
                    out.append((" ".join(b[s:e]), t, "expand", [i], range(s, e)))
                    da.discard(i); ib -= set(range(s, e))
                    break
    return out


# ---- permutations ----------------------------------------------------------

def perm_class(p):
    """Classify a rank sequence (S1 ranks in S2 order)."""
    k = len(p)
    if k < 2:
        return "n<2"
    ident = list(range(k))
    if p == ident:
        return "none"
    if k == 2:
        return "swap2"
    if p == [k - 1] + ident[:-1]:
        return "last_to_front"
    if p == ident[1:] + [0]:
        return "first_to_end"
    if p == ident[::-1]:
        return "reverse"
    diff = [x for x in range(k) if p[x] != x]
    if len(diff) == 2 and diff[1] == diff[0] + 1:
        return "adjacent_swap"
    for x in range(k):          # one token moved, the rest kept in order
        rest = p[:x] + p[x + 1:]
        if rest == sorted(rest):
            return "single_move"
    return "other"


def _moved_token(p):
    """S1 rank of the one token whose removal leaves the order monotone."""
    for x in range(len(p)):
        rest = p[:x] + p[x + 1:]
        if rest == sorted(rest):
            return p[x]
    return None


@lru_cache(maxsize=None)
def random_baseline(k):
    """Class probabilities under a uniform random reordering of k tokens,
    conditional on the order having changed."""
    c = Counter(perm_class(list(p)) for p in itertools.permutations(range(k)))
    c.pop("none", None)
    tot = sum(c.values())
    return {cls: v / tot for cls, v in c.items()}


# ---- statistics ------------------------------------------------------------

class Stats:
    """Mergeable counters for one source. Keys are (field, ...) tuples."""
    NAMES = ("pairs", "conf", "s1_tok", "s2_tok", "sub", "swap", "acro", "drop",
             "ins", "seen_all", "drop_all", "abbrev_pos", "ops", "perm", "moved",
             "moved_pos", "final", "num", "affix", "affix_s1", "df", "country_sub")

    def __init__(self):
        for n in self.NAMES:
            setattr(self, n, Counter())

    def merge(self, other):
        for n in self.NAMES:
            getattr(self, n).update(getattr(other, n))
        return self


def _affixes(raw):
    raw = (raw or "").strip()
    out = set()
    m = _LEAD.match(raw)
    if m and m.group().strip():
        out.add(("prefix", m.group().strip()))
    m = _TRAIL.search(raw)
    if m and m.group().strip() and m.start() > 0:
        out.add(("suffix", m.group().strip()))
    for t in raw.split()[1:-1]:
        if _PUNCT_TOKEN.match(t):
            out.add(("infix", t))
    return out


def confident(high, n_a, n_b):
    """MIN_CONF of the SHORTER side's tokens matched at high similarity, and at
    least half of S1's. Measured against S1 alone, dropping one token of a
    three-token name (2/3) would fail the gate and every short-name drop would
    be invisible; the half-of-S1 floor stops a one-token record that matches one
    word of a long name from counting as confident."""
    return high >= 1 and high / min(n_a, n_b) >= MIN_CONF and high / n_a >= 0.5


def mine_field(st, f, s1_raw, s2_raw, country, first_s1):
    """Accumulate one field of one pair into st."""
    st.pairs[f] += 1
    a1, a2 = _affixes(s1_raw), _affixes(s2_raw)
    for x in a1:
        st.affix_s1[(f,) + x] += 1
    for x in a2 - a1:
        st.affix[(f,) + x] += 1

    a, b = norm(s1_raw).split(), norm(s2_raw).split()
    if first_s1:
        for t in set(a):
            st.df[(f, t)] += 1
        st.df[(f, None)] += 1               # document count
    if not a:
        return
    pairs, drop_a, ins_b = align(a, b)
    acr = find_acronyms(a, b, drop_a, ins_b)
    cov_a = {i for x in acr for i in x[3]}
    cov_b = {j for x in acr for j in x[4]}
    drop_a = [i for i in drop_a if i not in cov_a]
    ins_b = [j for j in ins_b if j not in cov_b]

    for t in a:
        st.seen_all[(f, t)] += 1
    for i in drop_a:
        st.drop_all[(f, a[i])] += 1

    high = sum(1 for p in pairs if p[2] >= HIGH_SIM) + len(cov_a)
    if not confident(high, len(a), len(b)):
        return
    st.conf[f] += 1
    for t in a:
        st.s1_tok[(f, t)] += 1
    for t in b:
        st.s2_tok[(f, t)] += 1
    st.final[(f, a[-1])] += 1
    for i, j, s, kind in pairs:
        st.ops[(f, kind)] += 1
        if any(c.isdigit() for c in a[i]):
            st.num[(f, "identity" if kind == "identity" else "changed")] += 1
        if kind != "identity":
            st.sub[(f, a[i], b[j])] += 1
            st.country_sub[(f, a[i], b[j], country)] += 1
            if kind in ("abbrev", "expansion"):
                st.abbrev_pos[(f, a[i], b[j])] += i / max(len(a) - 1, 1)
    for long, short, d, ca, cb in acr:
        st.ops[(f, "acronym")] += 1
        st.acro[(f, long, short, d)] += 1
    for i in drop_a:
        st.ops[(f, "drop")] += 1
        st.drop[(f, a[i])] += 1
        if any(c.isdigit() for c in a[i]):
            st.num[(f, "dropped")] += 1
    for j in ins_b:
        st.ops[(f, "insert")] += 1
        st.ins[(f, b[j])] += 1
    # Dropped-and-inserted in the same record is where a substitution between
    # DISSIMILAR tokens would show up (sim() will never align inc with corp).
    for i in drop_a:
        for j in ins_b:
            st.swap[(f, a[i], b[j])] += 1

    order = sorted((j, i) for i, j, _, _ in pairs)
    ranks = {i: r for r, i in enumerate(sorted(i for _, i in order))}
    p = [ranks[i] for _, i in order]
    cls = perm_class(p)
    st.perm[(f, len(p), cls)] += 1
    if cls in ("last_to_front", "first_to_end", "single_move", "swap2"):
        r = _moved_token(p) if cls != "swap2" else len(p) - 1
        if r is not None:
            i = sorted(i for _, i in order)[r]
            st.moved[(f, cls, a[i])] += 1
            st.moved_pos[(f, cls, "last" if i == len(a) - 1 else
                          "first" if i == 0 else "middle")] += 1


def mine_records(records):
    """records: iterable of (source, country, s1_name, s1_addr, c_name, c_addr,
    first_s1). -> {source: Stats}."""
    out = {s: Stats() for s in SOURCES}
    for src, ctry, n1, a1, n2, a2, first in records:
        st = out[src]
        mine_field(st, "name", n1, n2, ctry, first)
        mine_field(st, "address", a1, a2, ctry, first)
    return out


def _mine_chunk(chunk):
    return mine_records(chunk)


# ---- tables ----------------------------------------------------------------

def _r(x, d=6):
    return round(float(x), d)


def js_divergence(p: Counter, q: Counter) -> float:
    """Jensen-Shannon divergence in bits between two count distributions."""
    keys = set(p) | set(q)
    sp, sq = sum(p.values()), sum(q.values())
    if not sp or not sq:
        return None
    js = 0.0
    for k in keys:
        a, b = p.get(k, 0) / sp, q.get(k, 0) / sq
        m = (a + b) / 2
        if a:
            js += 0.5 * a * math.log2(a / m)
        if b:
            js += 0.5 * b * math.log2(b / m)
    return js


def _none_or_round(x, d=5):
    return None if x is None else _r(x, d)


def _by_source(per_src, name, key):
    return {s: getattr(per_src[s], name).get(key, 0) for s in SOURCES}


def build_tables(per_src, min_support=5, min_trials=1000, forbid_vocab=300,
                 top=500):
    tot = Stats()
    for s in SOURCES:
        tot.merge(per_src[s])
    out = {}

    # abbreviations / expansions, per field, both conditional rates
    abbr = {f: [] for f in FIELDS}
    by_ctry = defaultdict(dict)
    for (f, a, b, c), n in tot.country_sub.items():
        by_ctry[(f, a, b)][c] = n
    for (f, a, b), n in tot.sub.items():
        _, kind = sim(a, b)
        if kind not in ("abbrev", "expansion") or n < min_support:
            continue
        abbr[f].append({
            "s1": a, "s2": b, "kind": kind, "count": n,
            "p_s2_given_s1": _r(n / tot.s1_tok[(f, a)]),
            "p_s1_given_s2": _r(n / tot.s2_tok[(f, b)]),
            "n_s1": tot.s1_tok[(f, a)], "n_s2": tot.s2_tok[(f, b)],
            "by_source": _by_source(per_src, "sub", (f, a, b)),
            "by_country": by_ctry[(f, a, b)],
            "mean_rel_pos": _r(tot.abbrev_pos[(f, a, b)] / n, 3)})
    for f in FIELDS:
        abbr[f].sort(key=lambda e: -e["count"])
    out["abbreviations"] = abbr

    out["typos"] = {f: sorted(
        ({"s1": a, "s2": b, "count": n,
          "p_s2_given_s1": _r(n / tot.s1_tok[(ff, a)])}
         for (ff, a, b), n in tot.sub.items()
         if ff == f and n >= min_support and sim(a, b)[1] == "typo"),
        key=lambda e: -e["count"])[:top] for f in FIELDS}

    acr = {f: [] for f in FIELDS}
    for (f, long, short, d), n in tot.acro.items():
        if n >= min_support:
            acr[f].append({"long": long, "short": short, "direction": d, "count": n,
                           "by_source": _by_source(per_src, "acro", (f, long, short, d))})
    for f in FIELDS:
        acr[f].sort(key=lambda e: -e["count"])
    out["acronyms"] = acr

    # State/region codes: no gazetteer, so identify them by shape and place.
    region = [dict(e, long=e["s1"] if e["kind"] == "abbrev" else e["s2"],
                   short=e["s2"] if e["kind"] == "abbrev" else e["s1"])
              for e in abbr["address"]
              if min(len(e["s1"]), len(e["s2"])) <= 3 and e["mean_rel_pos"] >= 0.6]
    region += [dict(e, heuristic="acronym") for e in acr["address"]
               if len(e["short"]) <= 3]
    out["region_codes"] = {
        "heuristic": "address abbreviation with a <=3 letter short form whose "
                     "long form sits in the last 40% of the S1 address, plus "
                     "address acronyms of <=3 letters",
        "entries": region}

    # FORBIDDEN: variable tokens that never map onto each other.
    forb = {}
    for f in FIELDS:
        events = Counter()
        changed, target = Counter(), Counter()
        # Typos are one-off spellings, not variable vocabulary: they still count
        # as observations below, but do not put a token in the vocabulary.
        for (ff, a, b), n in itertools.chain(
                ((k, n) for k, n in tot.sub.items() if sim(k[1], k[2])[1] != "typo"),
                tot.swap.items()):
            if ff != f or not (a.isalpha() and b.isalpha()):
                continue
            events[a] += n
            events[b] += n
            changed[a] += n
            target[b] += n
        total = sum(changed.values()) or 1
        vocab = [t for t, n in events.most_common(forbid_vocab) if n >= min_support]
        rows = []
        for a in vocab:
            n_a = tot.s1_tok[(f, a)]
            if n_a < min_trials:
                continue
            for b in vocab:
                if b == a or target[b] < min_support:
                    continue
                obs = tot.sub.get((f, a, b), 0) + tot.swap.get((f, a, b), 0)
                if obs:
                    continue
                s, _ = sim(a, b)
                rows.append({
                    "s1": a, "s2": b, "observed": 0, "n_trials": n_a,
                    "p_upper95": _r(3 / n_a),              # rule of three
                    "expected_if_independent": _r(changed[a] * target[b] / total, 2),
                    "s1_changed": changed[a], "s2_as_target": target[b],
                    "char_sim": _r(s, 3),
                    "trap": bool(s >= MIN_ALIGN or a[0] == b[0]),
                    "reverse_observed": tot.sub.get((f, b, a), 0)
                    + tot.swap.get((f, b, a), 0)})
        rows.sort(key=lambda e: (not e["trap"], -e["expected_if_independent"],
                                 -e["n_trials"]))
        forb[f] = rows
    out["forbidden"] = {
        "note": "observed == 0 over n_trials S1 occurrences in confident pairs; "
                "p_upper95 is the rule-of-three bound on P(s2 | s1). trap = a "
                "similarity metric would call the pair close. Vocabulary is the "
                "tokens seen in >= min_support substitution events.",
        **forb}

    # junk affixes (raw text, all pairs)
    junk = []
    for (f, where, aff), n in tot.affix.items():
        if n >= min_support:
            junk.append({"field": f, "position": where, "affix": aff, "count_added": n,
                         "rate_per_record": _r(n / tot.pairs[f]),
                         "count_in_s1": tot.affix_s1.get((f, where, aff), 0),
                         "by_source": {s: per_src[s].affix.get((f, where, aff), 0)
                                       for s in SOURCES}})
    junk.sort(key=lambda e: -e["count_added"])
    out["junk_affixes"] = junk[:top]

    out["junk_tokens"] = {f: sorted(
        ({"token": t, "count_inserted": n,
          "rate_per_pair": _r(n / max(tot.conf[f], 1)),
          "s1_occurrences": tot.s1_tok.get((f, t), 0),
          "by_source": _by_source(per_src, "ins", (f, t))}
         for (ff, t), n in tot.ins.items() if ff == f and n >= min_support),
        key=lambda e: -e["count_inserted"])[:top] for f in FIELDS}

    # token drop vs IDF
    drop = {}
    for f in FIELDS:
        n_doc = tot.df.get((f, None), 0)
        res = {}
        for label, seen, dropped in (("confident_pairs", tot.s1_tok, tot.drop),
                                     ("all_pairs", tot.seen_all, tot.drop_all)):
            bins = defaultdict(lambda: [0, 0])
            for (ff, t), n in seen.items():
                if ff != f or t is None:
                    continue
                df = tot.df.get((f, t), 0)
                idf = math.log((n_doc + 1) / (df + 1)) if n_doc else 0.0
                bb = bins[int(idf)]
                bb[0] += n
                bb[1] += dropped.get((f, t), 0)
            res[label] = [{"idf_bin": f"[{k},{k + 1})", "n_tokens": v[0],
                           "n_dropped": v[1], "drop_rate": _r(v[1] / v[0])}
                          for k, v in sorted(bins.items())]
        res["n_s1_documents"] = n_doc
        drop[f] = res
    out["token_drop_by_idf"] = drop

    out["numeral_mutation"] = {f: {
        k: tot.num.get((f, k), 0) for k in ("identity", "changed", "dropped")}
        for f in FIELDS}
    for f in FIELDS:
        d = out["numeral_mutation"][f]
        n = sum(d.values())
        d["n"] = n
        d["change_rate"] = _r(d["changed"] / n) if n else None
        d["drop_rate"] = _r(d["dropped"] / n) if n else None

    out["permutations"] = {f: _perm_tables(tot, per_src, f, min_support, top)
                           for f in FIELDS}
    out["source_signatures"] = _signatures(per_src)
    return out


def _perm_tables(tot, per_src, f, min_support, top):
    hist = defaultdict(Counter)
    for (ff, k, cls), n in tot.perm.items():
        if ff == f:
            hist[k][cls] += n
    by_k = []
    for k in sorted(hist):
        c = hist[k]
        moved = sum(v for cls, v in c.items() if cls not in ("none", "n<2"))
        row = {"n_tokens": k, "total": sum(c.values()), "permuted": moved,
               "histogram": dict(c)}
        if 2 <= k <= 7 and moved:
            base = random_baseline(k)
            obs = Counter({cls: v for cls, v in c.items() if cls != "none"})
            row["random_baseline_share"] = {cls: _r(p, 4) for cls, p in base.items()}
            row["observed_share"] = {cls: _r(v / moved, 4) for cls, v in obs.items()}
            row["js_vs_random_bits"] = _none_or_round(js_divergence(
                obs, Counter({cls: p * moved for cls, p in base.items()})), 4)
        by_k.append(row)
    moved = sorted(({"class": cls, "token": t, "count": n,
                     "s1_final_rate": _r(tot.final.get((f, t), 0)
                                         / max(tot.s1_tok.get((f, t), 0), 1), 3)}
                    for (ff, cls, t), n in tot.moved.items()
                    if ff == f and n >= min_support), key=lambda e: -e["count"])
    return {"by_length": by_k,
            "moved_from": {f"{cls}:{w}": n for (ff, cls, w), n in tot.moved_pos.items()
                           if ff == f},
            "moved_tokens": moved[:top],
            "note": "s1_final_rate = share of confident pairs where the token is the "
                    "last S1 token; a legal form moved to the front shows as "
                    "last_to_front with high s1_final_rate"}


def _signatures(per_src):
    fam = {}
    for s in SOURCES:
        st = per_src[s]
        fam[s] = {
            "ops": Counter({f"{f}:{k}": n for (f, k), n in st.ops.items()}),
            "perm": Counter({f"{f}:{c}": n for (f, _, c), n in st.perm.items()}),
            "affix": Counter({f"{f}:{w}:{a}": n for (f, w, a), n in st.affix.items()}),
            "insert": Counter({f"{f}:{t}": n for (f, t), n in st.ins.items()}),
            "abbrev": Counter({f"{f}:{a}>{b}": n for (f, a, b), n in st.sub.items()
                               if sim(a, b)[1] in ("abbrev", "expansion")}),
        }
        # The share of pairs WITHOUT an affix or insert is part of the
        # signature: S3 adding junk four times as often is the signal.
        for f in FIELDS:
            fam[s]["affix"][f"{f}:__none__"] = max(st.pairs[f] - sum(
                n for (ff, _, _), n in st.affix.items() if ff == f), 0)
            fam[s]["insert"][f"{f}:__none__"] = max(st.conf[f] - sum(
                n for (ff, _), n in st.ins.items() if ff == f), 0)
    out = {"divergence_js_bits": {
        k: _none_or_round(js_divergence(fam["S2"][k], fam["S3"][k])) for k in fam["S2"]}}
    for s in SOURCES:
        st = per_src[s]
        rates = {}
        for f in FIELDS:
            ops = Counter({k: n for (ff, k), n in st.ops.items() if ff == f})
            n_ops = sum(v for k, v in ops.items() if k != "insert") or 1
            perm_n = sum(n for (ff, _, c), n in st.perm.items() if ff == f)
            perm_moved = sum(n for (ff, _, c), n in st.perm.items()
                             if ff == f and c not in ("none", "n<2"))
            aff = sum(n for (ff, _, _), n in st.affix.items() if ff == f)
            rates[f] = {"pairs": st.pairs[f], "confident": st.conf[f],
                        "confident_rate": _r(st.conf[f] / max(st.pairs[f], 1)),
                        "op_share": {k: _r(v / n_ops) for k, v in sorted(ops.items())
                                     if k != "insert"},
                        "inserts_per_pair": _r(ops["insert"] / max(st.conf[f], 1)),
                        "reorder_rate": _r(perm_moved / max(perm_n, 1)),
                        "junk_affix_per_pair": _r(aff / max(st.pairs[f], 1))}
        out[s] = rates
    return out


# ---- data loading ----------------------------------------------------------

def load_records(root, sample, seed, countries=None, chunk=50000):
    """Yield (country, n_pairs_in_country, chunk_iter, n_total) per country
    shard, reusing the pipeline's ground-truth exploder and shard loader. The
    joined shard stays in Arrow; only one chunk at a time becomes Python."""
    import pyarrow as pa
    import audit_data
    import blocking

    audit_data.P = pathlib.Path(root)
    s1_rep, mid, _ = audit_data.explode_ground_truth()
    n_total = len(mid)
    if sample and sample < n_total:
        idx = np.sort(np.random.default_rng(seed).choice(n_total, sample, replace=False))
        s1_rep, mid = s1_rep[idx], mid[idx]
    pairs = pa.table({"s1": pa.array(s1_rep, pa.string()),
                      "mid": pa.array(mid, pa.string())})
    del s1_rep, mid
    have = sorted(p.name.split("=", 1)[1]
                  for p in (pathlib.Path(root) / "train_source1").glob("country=*"))
    for ctry in (countries or have):
        s1_tab, corpus = blocking.load_shard(root, "train", ctry)
        s1_tab = s1_tab.rename_columns(["s1", "n1", "a1"])
        corpus = corpus.rename_columns(["mid", "n2", "a2"])
        j = pairs.join(s1_tab, "s1", join_type="inner") \
                 .join(corpus, "mid", join_type="inner").sort_by("s1")
        del s1_tab, corpus
        yield ctry, j.num_rows, _chunks(j, ctry, chunk), n_total


def _chunks(j, ctry, size):
    prev = None                 # sorted by s1, so "first" is a change of id
    for off in range(0, j.num_rows, size):
        t = j.slice(off, size)
        s1 = t.column("s1").to_pylist()
        src = ["S2" if m.startswith("S2-") else "S3" for m in t.column("mid").to_pylist()]
        first = [s1[k] != (s1[k - 1] if k else prev) for k in range(len(s1))]
        prev = s1[-1]
        yield list(zip(src, [ctry] * len(s1), t.column("n1").to_pylist(),
                       t.column("a1").to_pylist(), t.column("n2").to_pylist(),
                       t.column("a2").to_pylist(), first))


def run(args):
    t0 = time.time()
    per_src = {s: Stats() for s in SOURCES}
    n_total, n_used, done = 0, 0, []
    pool = mp.Pool(args.workers) if args.workers > 1 else None
    for ctry, n, chunks, n_total in load_records(args.root, args.sample, args.seed,
                                                 args.countries, args.chunk):
        n_used += n
        parts = pool.imap_unordered(_mine_chunk, chunks) if pool else map(_mine_chunk, chunks)
        for part in parts:
            for s in SOURCES:
                per_src[s].merge(part[s])
        done.append({"country": ctry, "pairs": n})
        print(f"{ctry}: {n:,} pairs, {time.time() - t0:.0f}s", flush=True)
    if pool:
        pool.close()
    tables = build_tables(per_src, args.min_support, args.min_trials,
                          args.forbid_vocab, args.top)
    tables = {"provenance": provenance(args, n_total, n_used, done, per_src), **tables}
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(tables, indent=1, ensure_ascii=False, allow_nan=False))
    print(f"wrote {out} in {time.time() - t0:.0f}s")
    return tables


def provenance(args, n_total, n_used, countries, per_src):
    return {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "script": "src/mine_corruption.py",
        "inputs": ["train_ground_truth", "train_source1", "train_source2",
                   "train_source3"],
        "external_data": "none; every entry is counted from the inputs above and "
                         "carries its support count",
        "aligned_pairs_total": n_total, "aligned_pairs_used": n_used,
        "sample": args.sample, "seed": args.seed, "countries": countries,
        "confident_pairs": {s: {f: per_src[s].conf[f] for f in FIELDS}
                            for s in SOURCES},
        "params": {"min_align_sim": MIN_ALIGN, "high_sim": HIGH_SIM,
                   "min_confident_share": MIN_CONF, "min_support": args.min_support,
                   "min_trials_forbidden": args.min_trials,
                   "forbidden_vocab": args.forbid_vocab},
        "bias": "only pairs with >= min_confident_share of S1 tokens matched at "
                "high similarity are mined, so heavy corruptions are under-"
                "represented; token_drop_by_idf.all_pairs is the unfiltered view",
    }


# ---- self-test -------------------------------------------------------------

_ABBR = {"corporation": "corp", "limited": "ltd", "private": "pvt",
         "road": "rd", "street": "st", "incorporated": "inc", "company": "co",
         "texas": "tx", "haryana": "hr"}
_LEGAL = ["corporation", "limited", "incorporated", "company"]
_COMMON = ["global", "sharma", "traders", "services", "india", "systems", "foods"]


def synth_pairs(n, seed=0):
    """Synthetic aligned pairs with known corruptions. S2 and S3 differ: S3
    abbreviates less, adds more junk and moves the legal form to the front."""
    rng = random.Random(seed)
    rare = [f"zq{k:04d}x" for k in range(3000)]
    recs = []
    for k in range(n):
        src = "S2" if k % 2 else "S3"
        name = [rng.choice(rare), rng.choice(_COMMON), rng.choice(_LEGAL)]
        addr = [str(rng.randint(1, 999)), rng.choice(rare), rng.choice(["road", "street"]),
                rng.choice(rare), rng.choice(["texas", "haryana"]), str(rng.randint(10000, 99999))]
        n1, a1 = " ".join(name), " ".join(addr)
        p_abbr = 0.6 if src == "S2" else 0.3
        cn = [(_ABBR[t] if t in _ABBR and rng.random() < p_abbr else t) for t in name]
        ca = [(_ABBR[t] if t in _ABBR and rng.random() < p_abbr else t) for t in addr]
        if rng.random() < 0.15:                    # drop the RARE token
            cn = cn[1:] + ([] if rng.random() < 0.5 else [])
        elif rng.random() < 0.02:                  # rarely drop the common one
            cn = [cn[0]] + cn[2:]
        if src == "S3" and rng.random() < 0.3 and len(cn) == 3:
            cn = [cn[-1]] + cn[:-1]                # legal form to the front
        if rng.random() < 0.05:                    # a light typo
            t = cn[0]
            cn[0] = t[:-1] + ("a" if t[-1] != "a" else "e")
        c_name = " ".join(cn)
        junk_p = 0.05 if src == "S2" else 0.2
        if rng.random() < junk_p:
            c_name = rng.choice(["<<", "--"]) + " " + c_name
        recs.append((src, "Synth", n1, a1, c_name, " ".join(ca), True))
    return recs


def self_test(n=20000):
    per_src = mine_records(synth_pairs(n))
    t = build_tables(per_src, min_support=5, min_trials=200)
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + msg)
        ok &= bool(cond)

    ab = {(e["s1"], e["s2"]): e for e in t["abbreviations"]["name"]}
    e = ab.get(("corporation", "corp"))
    check(e and abs(e["p_s2_given_s1"] - 0.45) < 0.05,
          f"corporation->corp found at ~0.45 (got {e and e['p_s2_given_s1']})")
    check(e and e["p_s1_given_s2"] == 1.0, "corp always came from corporation")
    check(e and e["by_source"]["S2"] > 1.5 * e["by_source"]["S3"],
          "S2 abbreviates more than S3")
    aa = {(e["s1"], e["s2"]) for e in t["abbreviations"]["address"]}
    check({("road", "rd"), ("texas", "tx"), ("haryana", "hr")} <= aa,
          "address abbreviations road->rd, texas->tx, haryana->hr")
    reg = {(e["long"], e["short"]) for e in t["region_codes"]["entries"]}
    check({("texas", "tx"), ("haryana", "hr")} <= reg and ("road", "rd") not in reg,
          "region codes are texas/haryana, not road")
    fb = {(e["s1"], e["s2"]) for e in t["forbidden"]["name"]}
    check(("incorporated", "corp") in fb and ("corporation", "inc") in fb,
          "incorporated->corp and corporation->inc are forbidden")
    check(not fb & set(ab), "no observed abbreviation is also forbidden")
    check(not any(r["s2"] == "corporatioa" for r in t["forbidden"]["name"]),
          "typos do not enter the forbidden vocabulary")
    jk = {e["affix"] for e in t["junk_affixes"] if e["position"] == "prefix"}
    check({"<<", "--"} <= jk, "junk prefixes << and -- found")
    dr = [r for r in t["token_drop_by_idf"]["name"]["confident_pairs"]
          if r["n_tokens"] >= 1000]
    hi, lo = dr[-1]["drop_rate"], dr[0]["drop_rate"]
    check(hi > 3 * max(lo, 1e-3), f"rare tokens drop more (idf-high {hi}, idf-low {lo})")
    pm = t["permutations"]["name"]["by_length"]
    k3 = next(r for r in pm if r["n_tokens"] == 3)
    check(k3["histogram"].get("last_to_front", 0) > 0.8 * k3["permuted"],
          "reorders are last_to_front, not random")
    mv = t["permutations"]["name"]["moved_tokens"]
    check(mv and mv[0]["s1_final_rate"] == 1.0, "moved token is the S1-final legal form")
    dv = t["source_signatures"]["divergence_js_bits"]
    check(dv["perm"] > 0.01 and dv["affix"] > 0.01, f"S2/S3 diverge ({dv})")
    num = t["numeral_mutation"]["address"]
    check(num["changed"] == 0 and num["dropped"] == 0, "numerals never mutate")
    json.dumps(t, allow_nan=False)
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default="data/parquet")
    ap.add_argument("--out", default="data/corruption_grammar.json")
    ap.add_argument("--sample", type=int, default=0,
                    help="mine a random subset of N aligned pairs (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--countries", type=lambda s: s.split(","), default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=50000)
    ap.add_argument("--min-support", type=int, default=5)
    ap.add_argument("--min-trials", type=int, default=1000,
                    help="S1 occurrences needed before an absence counts as forbidden")
    ap.add_argument("--forbid-vocab", type=int, default=300)
    ap.add_argument("--top", type=int, default=500)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        sys.exit(0 if self_test() else 1)
    run(args)


if __name__ == "__main__":
    main()
