"""Learn Devanagari -> Latin transliteration from the aligned training pairs.

S1 is 100% Latin and S2/S3 carry the Devanagari, so every (S1 entity, S2/S3
record) link whose record is Devanagari says how this generator's Latin
spellings come out in Devanagari. 0.96% of India's true links are unreachable
only because the two scripts share no key after the rule transliteration
(diag_misses.py); this learns the mapping instead of hand-writing more rules,
and nothing comes from outside the provided files.

Word pairs. Each Devanagari record's words are aligned to its S1 entity's
tokens by Hungarian matching on the similarity of the rule transliteration
(folded, and as a consonant skeleton, so प्राइवेट/praaivet meets private at
"prvt"), with a floor below which a word is left unaligned.

What is learned, and applied in this order by textnorm.TranslitModel:
  lexicon   whole word -> Latin token, for words seen at least --min-support
            times whose most common spelling has at least --min-share of them:
            the loanwords (प्राइवेट -> private, लिमिटेड -> limited) and the
            frequent names (शर्मा -> sharma).
  units     grapheme unit (consonant + matra / virama, vowel, anusvara, ...)
            -> Latin string, in context: the kind and place of the next unit
            and whether the previous one ends in a vowel. That is where schwa
            deletion beyond word-final (कमला -> kamla) and anusvara place
            (अंबानी -> ambani) are learned. Units are aligned to substrings by
            hard EM over distinct word pairs (each counted once, so the
            loanwords the lexicon already covers do not dominate).
  opts      rule options for units the tables never saw, chosen by match rate
            on the training words: medial schwa deletion, homorganic anusvara,
            and long vowels written single (ा -> a).

The report compares, on held-out entities (split by S1 name), the exact token
match rate of: the current rules, the best rule options, units, and units +
lexicon; on all held-out words and on words never seen in training. Exact token
equality is what the name-token blocking channel needs.

Usage (from code/business_entity_resolution, src/ on PYTHONPATH):
    python src/mine_translit.py                     # India, 1M sampled pairs
    python src/mine_translit.py --sample 0          # every aligned pair
    python src/mine_translit.py --self-test         # synthetic check
"""
import argparse
import datetime
import itertools
import json
import math
import pathlib
import random
import sys
import time
import zlib
from collections import Counter, defaultdict
from functools import lru_cache

import numpy as np
from scipy.optimize import linear_sum_assignment

import textnorm
from textnorm import (DEVA, TranslitModel, deva_units, fold, norm, rule_unit,
                      set_translit_model, skeleton, translit_rules, unit_keys,
                      _schwa_kept)

MIN_WORD_SIM = 0.5      # below this a Devanagari word is left unaligned
MIN_REC_SHARE = 0.5     # share of a record's Devanagari words that must align
MAX_SUB = 5             # longest Latin substring one unit may produce
EM_ITERS = 4
PRIOR_W = 2.0           # prior strength: log-prior = -PRIOR_W * edit distance
ALL_OPTS = [dict(zip(("schwa", "nasal", "short"), bits))
            for bits in itertools.product((False, True), repeat=3)]


# ---- word pairs -------------------------------------------------------------

def deva_words(s):
    return textnorm._DEVA_WORD.findall(s or "")


def _lev(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _ratio(a, b):
    if not a or not b:
        return 0.0
    return 1 - _lev(a, b) / max(len(a), len(b))


@lru_cache(maxsize=1 << 20)
def word_sim(rom, lat):
    """Similarity of a romanised word and a Latin token, blind to the vowel
    and aspiration choices a transliteration gets wrong."""
    f = _ratio(fold(rom), fold(lat))
    sk = _ratio(skeleton(rom).replace(" ", ""), skeleton(lat).replace(" ", ""))
    return max(f, sk)


@lru_cache(maxsize=1 << 20)
def _rom(w):
    return norm(translit_rules(w)).replace(" ", "")


def align_words(s1_raw, s2_raw):
    """-> [(deva_word, latin_token)] for one field of one pair, or [] when
    the record is not Devanagari against Latin or aligns too poorly."""
    if not s2_raw or not s1_raw or DEVA.search(s1_raw):
        return []
    dw = deva_words(s2_raw)
    lt = norm(s1_raw).split()
    if not dw or not lt:
        return []
    S = np.array([[word_sim(_rom(w), t) for t in lt] for w in dw])
    rows, cols = linear_sum_assignment(-S)
    got = [(dw[r], lt[c]) for r, c in zip(rows, cols) if S[r, c] >= MIN_WORD_SIM]
    if len(got) < MIN_REC_SHARE * len(dw):
        return []
    return got


def extract(records):
    """records: iterable of (s1_name, s1_addr, s2_name, s2_addr). -> Counter of
    (deva_word, latin_token) and the number of Devanagari records seen."""
    pairs, n_deva = Counter(), 0
    for n1, a1, n2, a2 in records:
        if not (DEVA.search(n2 or "") or DEVA.search(a2 or "")):
            continue
        n_deva += 1
        pairs.update(align_words(n1, n2))
        pairs.update(align_words(a1, a2))
    return pairs, n_deva


# ---- unit alignment (hard EM) --------------------------------------------------

def _variants(unit):
    """What the rules could make of one unit under any options; the prior."""
    key, kind, cons = unit
    outs = {rule_unit([unit], 0, [True], o) for o in ALL_OPTS}
    if kind == "C":
        outs.add(textnorm._CONS1[cons])
    if kind in ("N", "X"):
        outs |= {"m", ""}
    return tuple(sorted(outs))


@lru_cache(maxsize=1 << 20)
def _prior(unit, sub):
    return -PRIOR_W * min(_lev(sub, v) for v in _variants(unit))


def _viterbi(units, lat, score):
    n, m = len(units), len(lat)
    NEG = -math.inf
    dp = [[NEG] * (m + 1) for _ in range(n + 1)]
    bp = [[0] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    for i in range(1, n + 1):
        u = units[i - 1]
        row, prev = dp[i], dp[i - 1]
        for j in range(m + 1):
            best, arg = NEG, 0
            for ln in range(min(MAX_SUB, j) + 1):
                p = prev[j - ln]
                if p == NEG:
                    continue
                v = p + score(u, lat[j - ln:j])
                if v > best:
                    best, arg = v, ln
            row[j], bp[i][j] = best, arg
    if dp[n][m] == NEG:
        return None
    subs, j = [], m
    for i in range(n, 0, -1):
        ln = bp[i][j]
        subs.append(lat[j - ln:j])
        j -= ln
    return subs[::-1]


_VOWELS = set("aeiou")


def _split_cv(sub):
    """"kha" -> ("kh", "a"): the consonant spelling and the vowel after it."""
    k = 0
    while k < len(sub) and sub[k] not in _VOWELS:
        k += 1
    return sub[:k], sub[k:]


def align_units(word_pairs, iters=EM_ITERS):
    """Hard EM over distinct word pairs (each counted once, so the loanwords
    the lexicon covers do not dominate). -> [(units, [substring per unit])]."""
    data = []
    for (w, lat) in word_pairs:
        units = tuple(deva_units(w))
        if units and all(k != "U" for _, k, _ in units) and len(lat) <= 4 * len(units):
            data.append((units, lat))
    score = _prior
    for _ in range(iters):
        counts = defaultdict(Counter)
        for units, lat in data:
            subs = _viterbi(units, lat, score)
            if subs is not None:
                for u, sb in zip(units, subs):
                    counts[u][sb] += 1
        tot = {u: sum(c.values()) for u, c in counts.items()}

        def score(u, sub, counts=counts, tot=tot):
            c = counts.get(u)
            if c is None:
                return _prior(u, sub)
            return math.log((c.get(sub, 0) + math.exp(_prior(u, sub))) / (tot[u] + 1.0))
    out = []
    for units, lat in data:
        subs = _viterbi(units, lat, score)
        if subs is not None:
            out.append((units, subs))
    return out


def learn_tables(word_pairs, opts, min_count=5, min_ctx=10):
    """-> textnorm.TranslitModel tables: each decision's majority over the
    aligned units, kept where it has enough support and differs from what the
    rules with `opts` would do anyway."""
    votes = {k: defaultdict(Counter) for k in ("cons", "vowel", "nasal", "schwa", "other")}
    for units, subs in align_units(word_pairs):
        keep = _schwa_kept(units, opts)
        for i, ((key, kind, cons), sb) in enumerate(zip(units, subs)):
            k = unit_keys(units, i, keep)
            if kind in ("C", "M", "H"):
                c, v = _split_cv(sb)
                if not c:
                    continue                      # the consonant was aligned away
                votes["cons"][cons][c] += 1
                if kind == "C" and v in ("", "a"):
                    votes["schwa"][k][int(v == "a")] += 1
                elif kind == "M" and v:
                    votes["vowel"][k][v] += 1
            elif kind == "V":
                votes["vowel"][k][sb] += 1
            elif kind == "N":
                votes["nasal"][k][sb] += 1
            else:
                votes["other"][k][sb] += 1
    tables = {}
    for name, vs in votes.items():
        need = min_ctx if name == "schwa" else min_count
        t = {}
        for k, c in vs.items():
            if sum(c.values()) >= need:
                t[k] = c.most_common(1)[0][0]
        tables[name] = t
    # Consonant spellings equal to the rule add nothing; drop them so the file
    # shows what was learned. (The other tables are kept whole: their rule
    # fallback depends on the options and the context.)
    for k, v in list(tables["cons"].items()):
        if v == textnorm._CONS1[k]:
            del tables["cons"][k]
    return {k: dict(sorted(v.items(), key=lambda kv: str(kv[0])))
            for k, v in tables.items()}


def learn_lexicon(word_pairs, model, min_support=5, min_share=0.6):
    """Whole-word spellings the unit model would not produce by itself."""
    by_word = defaultdict(Counter)
    for (w, lat), c in word_pairs.items():
        by_word[w][lat] += c
    lex = {}
    for w, c in by_word.items():
        n = sum(c.values())
        lat, k = c.most_common(1)[0]
        if n >= min_support and k >= min_share * n and model.word_units(w) != lat:
            lex[w] = lat
    return lex


def match_rate(word_pairs, fn):
    """Token-weighted share of (deva, latin) pairs where fn(deva) == latin."""
    hit = tot = 0
    for (w, lat), c in word_pairs.items():
        tot += c
        hit += c * (norm(fn(w)).replace(" ", "") == lat)
    return hit / tot if tot else float("nan")


def fit(train_pairs, min_support=5, min_share=0.6, min_count=5, min_ctx=10):
    """-> (TranslitModel, the options' training match rates)."""
    by_opts = [(match_rate(train_pairs, TranslitModel(opts=o).word_units), i)
               for i, o in enumerate(ALL_OPTS)]
    opts = ALL_OPTS[max(by_opts)[1]]
    tables = learn_tables(train_pairs, opts, min_count, min_ctx)
    m = TranslitModel(tables=tables, opts=opts)
    m.lexicon = learn_lexicon(train_pairs, m, min_support, min_share)
    return m, {json.dumps(ALL_OPTS[i]): round(r, 5) for r, i in by_opts}


def report(test_pairs, train_pairs, model):
    """Held-out exact-token match rates for each variant, all words and OOV."""
    seen = {w for w, _ in train_pairs}
    oov = Counter({k: c for k, c in test_pairs.items() if k[0] not in seen})
    rules_best = TranslitModel(opts=model.opts)
    units_only = TranslitModel(tables=model.tables, opts=model.opts)
    variants = [("current rules (translit_rules)", translit_rules),
                (f"rules, best options {model.opts}", rules_best.word_units),
                ("learned tables", units_only.word_units),
                ("learned tables + lexicon", model.word)]
    rows = []
    for name, fn in variants:
        rows.append({"variant": name,
                     "all": _r(match_rate(test_pairs, fn)),
                     "oov": _r(match_rate(oov, fn)),
                     "all_fold": _r(match_rate_fold(test_pairs, fn))})
    return {"pairs_test": sum(test_pairs.values()),
            "distinct_test": len(test_pairs),
            "oov_share": _r(sum(oov.values()) / max(sum(test_pairs.values()), 1)),
            "rows": rows}


def match_rate_fold(word_pairs, fn):
    hit = tot = 0
    for (w, lat), c in word_pairs.items():
        tot += c
        hit += c * (fold(norm(fn(w)).replace(" ", "")) == fold(lat))
    return hit / tot if tot else float("nan")


def record_reach(records, fn, common):
    """Share of Devanagari-vs-Latin records whose names share at least one
    exact token outside the `common` set: the name-token blocking condition."""
    hit = tot = 0
    prev = set_translit_model(None)
    try:
        for n1, _, n2, _ in records:
            if not n2 or not DEVA.search(n2) or not n1 or DEVA.search(n1):
                continue
            tot += 1
            t1 = set(norm(n1).split()) - common
            t2 = set(norm(textnorm._DEVA_WORD.sub(lambda m: " " + fn(m.group()) + " ",
                                                  n2)).split())
            hit += bool(t1 & t2)
    finally:
        set_translit_model(prev)
    return hit / tot if tot else float("nan")


def _r(x, d=5):
    return None if x != x else round(float(x), d)


def print_report(rep, reach=None):
    print(f"\nheld-out word pairs: {rep['pairs_test']:,} "
          f"({rep['distinct_test']:,} distinct), unseen in training: "
          f"{rep['oov_share']:.1%}")
    print(f"{'variant':52}{'exact':>8}{'exact OOV':>11}{'fold':>8}")
    def pct(v, w):
        # a rate is None when its denominator is empty -- e.g. no held-out word
        # is unseen in training, which is exactly what the real data shows
        return f"{v:>{w}.2%}" if v is not None else f"{'n/a':>{w}}"

    for r in rep["rows"]:
        print(f"{r['variant'][:51]:52}{pct(r['all'], 8)}{pct(r['oov'], 11)}"
              f"{pct(r['all_fold'], 8)}")
    if reach:
        print("\nDevanagari records sharing a non-common exact name token with S1:")
        for k, v in reach.items():
            print(f"  {k:30}{v:>8.2%}")


# ---- real data -----------------------------------------------------------------

def _split(n1):
    return zlib.crc32((n1 or "").encode("utf-8")) % 5 == 0      # 20% held out


def load(args):
    import mine_corruption
    tr, te = [], []
    for ctry, n, chunks, _ in mine_corruption.load_records(
            args.root, args.sample, args.seed, args.countries, args.chunk):
        k = 0
        for chunk in chunks:
            for _, _, n1, a1, n2, a2, _ in chunk:
                if DEVA.search(n2 or "") or DEVA.search(a2 or ""):
                    (te if _split(n1) else tr).append((n1, a1, n2, a2))
                    k += 1
        print(f"{ctry}: {n:,} pairs, {k:,} with Devanagari", flush=True)
    return tr, te


def common_tokens(records, top=50):
    c = Counter(t for n1, _, _, _ in records for t in set(norm(n1).split()))
    return {t for t, _ in c.most_common(top)}


def run(args, tr=None, te=None):
    t0 = time.time()
    if tr is None:
        tr, te = load(args)
    set_translit_model(None)
    train_pairs, n_tr = extract(tr)
    test_pairs, n_te = extract(te)
    print(f"aligned {sum(train_pairs.values()):,} train / "
          f"{sum(test_pairs.values()):,} held-out word pairs from {n_tr:,} / "
          f"{n_te:,} Devanagari records ({time.time() - t0:.0f}s)", flush=True)
    model, opts_rates = fit(train_pairs, args.min_support, args.min_share,
                            args.min_count, args.min_ctx)
    rep = report(test_pairs, train_pairs, model)
    common = common_tokens(te)
    reach = {"current rules": record_reach(te, translit_rules, common),
             "learned": record_reach(te, model.word, common)}
    print_report(rep, reach)
    # The report is from the split; the model written is refit on every pair.
    model, opts_rates = fit(train_pairs + test_pairs, args.min_support,
                            args.min_share, args.min_count, args.min_ctx)
    out = {"provenance": {
               "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "script": "src/mine_translit.py",
               "inputs": ["train_ground_truth", "train_source1", "train_source2",
                          "train_source3"],
               "external_data": "none; learned from the aligned training pairs",
               "sample": args.sample, "seed": args.seed, "countries": args.countries,
               "records_train": n_tr, "records_heldout": n_te,
               "word_pairs": sum(train_pairs.values()) + sum(test_pairs.values()),
               "fit": "report on an 80/20 split by S1 name; model refit on all pairs",
               "params": {"min_word_sim": MIN_WORD_SIM, "min_record_share": MIN_REC_SHARE,
                          "min_support": args.min_support, "min_share": args.min_share,
                          "min_count": args.min_count, "min_ctx": args.min_ctx}},
           "report": {**rep, "record_reach": {k: _r(v) for k, v in reach.items()},
                      "rule_options_train": opts_rates},
           "opts": model.opts, "lexicon": dict(sorted(model.lexicon.items())),
           "tables": model.tables}
    if args.out:
        p = pathlib.Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=1, ensure_ascii=False, allow_nan=False),
                     encoding="utf-8")  # the default is cp1252 on Windows
        print(f"\nwrote {p}: {len(model.lexicon):,} lexicon words, "
              f"{sum(map(len, model.tables.values())):,} table entries, {time.time() - t0:.0f}s")
    return out, model


# ---- synthetic -----------------------------------------------------------------
# A test fixture only, never read by the model: Latin words and their Devanagari,
# plus a romaniser that spells random Devanagari words the way Indian names are
# commonly written in Latin (ā -> a, medial schwa dropped, m before labials).
# It shares assumptions with the rule options, plus two conventions no rule has
# (ष -> s, ौ -> mostly ou) for the tables to find. Its numbers show the method
# works end to end; the real-data report is the one that counts.

LOANWORDS = {
    "private": "प्राइवेट", "limited": "लिमिटेड", "traders": "ट्रेडर्स",
    "enterprises": "एंटरप्राइजेज", "industries": "इंडस्ट्रीज", "services": "सर्विसेज",
    "solutions": "सॉल्यूशंस", "technologies": "टेक्नोलॉजीज", "company": "कंपनी",
    "store": "स्टोर", "medical": "मेडिकल", "hospital": "हॉस्पिटल",
    "electronics": "इलेक्ट्रॉनिक्स", "motors": "मोटर्स", "textiles": "टेक्सटाइल्स",
    "foods": "फूड्स", "jewellers": "ज्वेलर्स", "constructions": "कंस्ट्रक्शंस",
    "logistics": "लॉजिस्टिक्स", "pharma": "फार्मा", "agency": "एजेंसी",
    "international": "इंटरनेशनल", "global": "ग्लोबल", "india": "इंडिया",
    "royal": "रॉयल", "new": "न्यू", "star": "स्टार",
}
NAMES = {
    "sharma": "शर्मा", "gupta": "गुप्ता", "agarwal": "अग्रवाल", "singh": "सिंह",
    "kumar": "कुमार", "lakshmi": "लक्ष्मी", "ganesh": "गणेश", "krishna": "कृष्णा",
    "shri": "श्री", "balaji": "बालाजी", "jain": "जैन", "patel": "पटेल",
    "reddy": "रेड्डी", "verma": "वर्मा", "yadav": "यादव", "mishra": "मिश्रा",
    "sai": "साई", "durga": "दुर्गा", "shiv": "शिव", "om": "ओम",
    "mahalaxmi": "महालक्ष्मी", "bharat": "भारत", "hindustan": "हिंदुस्तान",
}
_S_CONS = {"क": ["k"], "ख": ["kh"], "ग": ["g"], "घ": ["gh"], "च": ["ch"],
           "छ": ["chh", "ch"], "ज": ["j"], "झ": ["jh"], "ट": ["t"], "ठ": ["th"],
           "ड": ["d"], "ढ": ["dh"], "ण": ["n"], "त": ["t"], "थ": ["th"], "द": ["d"],
           "ध": ["dh"], "न": ["n"], "प": ["p"], "फ": ["ph", "f"], "ब": ["b"],
           "भ": ["bh"], "म": ["m"], "य": ["y"], "र": ["r"], "ल": ["l"],
           "व": ["v", "w"], "श": ["sh"], "ष": ["s"], "स": ["s"], "ह": ["h"]}
_S_MATRA = {"ा": [("a", .85), ("aa", .15)], "ि": [("i", 1)], "ी": [("i", .6), ("ee", .4)],
            "ु": [("u", 1)], "ू": [("u", .7), ("oo", .3)], "े": [("e", 1)],
            "ै": [("ai", 1)], "ो": [("o", 1)], "ौ": [("ou", .6), ("au", .4)]}
_S_VOW = {"अ": "a", "आ": "a", "इ": "i", "उ": "u", "ए": "e", "ओ": "o"}
_CLUSTERS = ["क्ष", "न्द", "स्त", "र्म", "न्त", "प्र", "त्र", "श्व", "द्र", "ण्ड"]


def _pick(rng, opts):
    x, acc = rng.random(), 0.0
    for v, p in opts:
        acc += p
        if x < acc:
            return v
    return opts[-1][0]


def _synth_word(rng):
    """-> (devanagari, latin) for one random name-like word."""
    cons = list(_S_CONS)
    parts = []                                  # (deva, kind, latin choices)
    if rng.random() < 0.1:
        v = rng.choice(list(_S_VOW))
        parts.append((v, "V", _S_VOW[v]))
    for _ in range(rng.randint(2, 4)):
        if rng.random() < 0.15:
            cl = rng.choice(_CLUSTERS)
            c1, c2 = cl[0], cl[2]
            parts.append((c1 + "्", "H", rng.choice(_S_CONS[c1])))
            c = c2
        else:
            c = rng.choice(cons)
        lat = rng.choice(_S_CONS[c])
        if rng.random() < 0.55:
            m = rng.choice(list(_S_MATRA))
            parts.append((c + m, "M", lat + _pick(rng, _S_MATRA[m])))
        else:
            parts.append((c, "C", lat))
        if rng.random() < 0.08:
            parts.append(("ं", "N", None))
    # schwa: final always goes, medial goes in V C _ C V (right to left)
    n = len(parts)
    keep = [k == "C" for _, k, _ in parts]
    if parts[-1][1] == "C":
        keep[-1] = False
    for i in range(n - 2, 0, -1):
        if parts[i][1] != "C":
            continue
        pv = parts[i - 1][1] in ("M", "V") or (parts[i - 1][1] == "C" and keep[i - 1])
        nv = parts[i + 1][1] == "M" or (parts[i + 1][1] == "C" and keep[i + 1])
        if pv and nv and rng.random() < 0.9:
            keep[i] = False
    deva, lat = [], []
    for i, (d, k, l) in enumerate(parts):
        deva.append(d)
        if k == "N":
            nxt = parts[i + 1][0][0] if i + 1 < n else ""
            lat.append("m" if nxt in "पफबभम" and nxt else "n")
        else:
            lat.append(l + ("a" if k == "C" and keep[i] else ""))
    return "".join(deva), "".join(lat)


def synth_records(n, seed=0, n_vocab=4000, oov_share=0.2):
    """-> (train_records, test_records, test_word_pairs_truth). Test records
    also use a slice of the vocabulary no training record contains."""
    rng = random.Random(seed)
    vocab, seen = [], set()
    while len(vocab) < n_vocab:
        d, l = _synth_word(rng)
        if d not in seen and l not in seen:
            seen |= {d, l}
            vocab.append((l, d))
    names = list(NAMES.items())
    n_train_vocab = int(n_vocab * (1 - oov_share))    # the rest is test-only
    legal = [("private limited", "प्राइवेट लिमिटेड"), ("pvt ltd", "प्रा लि"),
             ("", "")]
    loans = [k for k in LOANWORDS if k not in ("private", "limited")]
    tr, te, truth = [], [], Counter()
    for k in range(n):
        test = k % 5 == 0
        words = []
        for _ in range(rng.randint(1, 2)):
            if rng.random() < 0.3:
                words.append(rng.choice(names))
            else:
                i = rng.randrange(n_vocab if test else n_train_vocab)
                words.append(vocab[i])
        if rng.random() < 0.5:
            w = rng.choice(loans)
            words.append((w, LOANWORDS[w]))
        lat_legal, deva_legal = rng.choice(legal)
        n1 = " ".join(l for l, _ in words) + (" " + lat_legal if lat_legal else "")
        dw = [d for _, d in words]
        if len(dw) > 1 and rng.random() < 0.08:          # drop a word
            j = rng.randrange(len(dw))
            dw.pop(j)
            words = words[:j] + words[j + 1:]
        if len(dw) > 1 and rng.random() < 0.05:          # reorder
            dw = dw[1:] + dw[:1]
        n2 = " ".join(dw) + (" " + deva_legal if deva_legal else "")
        if rng.random() < 0.1:
            n2 = "M/s " + n2                              # Latin prefix
        a1 = f"{rng.randint(1, 999)} {rng.choice(vocab)[0]} road"
        a2 = f"{rng.randint(1, 999)} {rng.choice(vocab)[1]} रोड"
        rec = (n1, a1, n2, a2)
        (te if test else tr).append(rec)
        if test:
            truth.update((d, l) for l, d in words)
            if deva_legal == "प्राइवेट लिमिटेड":
                truth.update([("प्राइवेट", "private"), ("लिमिटेड", "limited")])
    return tr, te, truth


def self_test(n=30000):
    tr, te, truth = synth_records(n)
    args = argparse.Namespace(min_support=5, min_share=0.6, min_count=5, min_ctx=10,
                              out=None, sample=0, seed=0, countries=None)
    out, model = run(args, tr, te)
    split_model, _ = fit(extract(tr)[0])
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + msg)
        ok &= bool(cond)

    rows = {r["variant"].split(",")[0]: r for r in out["report"]["rows"]}
    base = rows["current rules (translit_rules)"]["all"]
    best = rows["learned tables + lexicon"]["all"]
    check(best > base + 0.2, f"learned beats current rules ({best:.3f} vs {base:.3f})")
    check(rows["learned tables"]["oov"] > rows["current rules (translit_rules)"]["oov"] + 0.2,
          "units generalise to words never seen in training")
    truth_rate = match_rate(truth, split_model.word)
    check(truth_rate > best - 0.05,
          f"match on the generator's true word pairs ({truth_rate:.3f}) agrees with "
          f"the aligned pairs ({best:.3f})")
    for w, lat in (("प्राइवेट", "private"), ("लिमिटेड", "limited"),
                   ("शर्मा", "sharma")):
        check(model.word(w) == lat, f"{w} -> {model.word(w)} (want {lat})")
    t = model.tables
    check(t["cons"].get("ष") == "s" and t["vowel"].get("ौ|") == "ou",
          "tables learn the conventions no rule has (ष -> s, ौ -> ou)")
    check(rows["learned tables"]["oov"] > rows["rules"]["oov"] + 0.02,
          "tables beat the best rules on unseen words")
    check(model.opts.get("schwa") and model.opts.get("short"),
          f"rule options chosen: {model.opts}")
    reach = out["report"]["record_reach"]
    check(reach["learned"] > reach["current rules"],
          f"more records reach their S1 name ({reach})")
    json.dumps(out, allow_nan=False)
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default="data/parquet")
    ap.add_argument("--out", default=textnorm.TRANSLIT_PATH)
    ap.add_argument("--sample", type=int, default=1000000,
                    help="aligned pairs to sample before the country filter (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--countries", type=lambda s: s.split(","), default=["India"])
    ap.add_argument("--chunk", type=int, default=50000)
    ap.add_argument("--min-support", type=int, default=5)
    ap.add_argument("--min-share", type=float, default=0.6)
    ap.add_argument("--min-count", type=int, default=5)
    ap.add_argument("--min-ctx", type=int, default=10)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        sys.exit(0 if self_test() else 1)
    run(args)


if __name__ == "__main__":
    main()
