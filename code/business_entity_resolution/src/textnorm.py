"""Shared text normalisation. Deliberately conservative: it folds case, accents and
punctuation but never strips legal suffixes or stems, because those carry identity
(see audit finding F — 39% of S1 entities share a name with a DIFFERENT business)."""
import hashlib
import json
import os
import pathlib
import re
import sys
import unicodedata

DEVA = re.compile(r"[ऀ-ॿ]")
_NONALNUM = re.compile(r"[^0-9a-z]+")

# Devanagari -> Latin. This is alphabet knowledge (the same kind as knowing "St"
# abbreviates "Street"), derived from the Unicode chart, not an external dataset.
# S1 is 100% Latin and S2/S3 carry Devanagari, so the mapping is only ever needed
# in this direction. Without it ~17% of the India corpus normalises to an empty
# name and is invisible to every name channel.
_CONS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng", "च": "ch", "छ": "chh",
    "ज": "j", "झ": "jh", "ञ": "ny", "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh",
    "ण": "n", "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n", "प": "p",
    "फ": "ph", "ब": "b", "भ": "bh", "म": "m", "य": "y", "र": "r", "ल": "l",
    "व": "v", "श": "sh", "ष": "sh", "स": "s", "ह": "h", "ळ": "l", "ऴ": "l",
    "क़": "k", "ख़": "kh", "ग़": "g", "ज़": "z", "ड़": "r", "ढ़": "rh", "फ़": "f",
}
_VOW = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo", "ऋ": "ri",
    "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au", "ऍ": "e", "ऑ": "o",
}
_MATRA = {
    "ा": "aa", "ि": "i", "ी": "ee", "ु": "u", "ू": "oo", "ृ": "ri", "े": "e",
    "ै": "ai", "ो": "o", "ौ": "au", "ॅ": "e", "ॉ": "o",
}
_DIGIT = {chr(0x0966 + i): str(i) for i in range(10)}
_VIRAMA, _ANUSVARA, _CHANDRA, _VISARGA, _NUKTA = "्", "ं", "ँ", "ः", "़"


def translit(s: str) -> str:
    """Devanagari -> Latin. Uses the model mine_translit.py learned from the
    aligned training pairs when TRANSLIT_MODEL is present, else translit_rules.
    Only ever called on strings that contain Devanagari."""
    m = _ACTIVE if _ACTIVE is not _UNSET else translit_model()
    return translit_rules(s) if m is None else m.translit(s)


def translit_rules(s: str) -> str:
    """Devanagari -> Latin, syllable by syllable.

    A consonant carries an inherent 'a' which a matra replaces and a virama
    removes; that rule is what makes the output resemble how the same name is
    spelled in the Latin records. This is the fallback when no learned model is
    present, and it is kept exactly as it was so an old model.pkl stays valid.
    """
    out = []
    pending = False          # a consonant is awaiting its inherent vowel
    for ch in s:
        if ch == _NUKTA:
            continue
        if ch in _CONS:
            if pending:
                out.append("a")
            out.append(_CONS[ch])
            pending = True
        elif ch in _MATRA:
            out.append(_MATRA[ch])
            pending = False
        elif ch == _VIRAMA:
            pending = False
        elif ch in _VOW:
            if pending:
                out.append("a")
                pending = False
            out.append(_VOW[ch])
        elif ch in (_ANUSVARA, _CHANDRA):
            if pending:
                out.append("a")
                pending = False
            out.append("n")
        elif ch == _VISARGA:
            if pending:
                out.append("a")
                pending = False
            out.append("h")
        else:
            # Schwa deletion: Hindi drops the word-final inherent vowel, so
            # राम is "ram", not "raama". Getting this wrong costs every token.
            if pending:
                if not ch.isspace():
                    out.append("a")
                pending = False
            out.append(_DIGIT.get(ch, ch))
    return "".join(out)


# ---- learned transliteration ------------------------------------------------
# mine_translit.py learns two things from the (S1 Latin, S2/S3 Devanagari)
# training pairs: a lexicon of whole words (loanwords like प्राइवेट -> private,
# and frequent names) and a table of what each grapheme unit becomes in context,
# which is where schwa deletion and anusvara place come from. A word misses the
# lexicon, then the unit table, then falls to the rules below with the options
# the miner chose. Nothing here is consulted without the model file.

# Relative to the working directory (code/business_entity_resolution), like
# the corruption grammar. TRANSLIT_MODEL overrides it.
TRANSLIT_PATH = os.environ.get("TRANSLIT_MODEL", "data/translit_model.json")

_ZW = "‌‍"
# Precomposed nukta letters (U+0958..U+095F) and consonant + U+093C are the same
# letter; the rules above see neither form (they skip the nukta, and NFKD later
# strips it from the precomposed one, leaving a bare Devanagari consonant that
# becomes a space). Here both are read as the nukta consonant.
_NUKTA_OF = {unicodedata.normalize("NFD", chr(c))[0]: chr(c) for c in range(0x958, 0x960)}
_NUKTA_LAT = {"क़": "q", "ख़": "kh", "ग़": "g", "ज़": "z",
              "ड़": "r", "ढ़": "rh", "फ़": "f", "य़": "y"}
_CONS1 = {k: v for k, v in _CONS.items() if len(k) == 1}
_CONS1.update(_NUKTA_LAT)
_CONS1["ऩ"] = "n"                     # ऩ
_CONS1["ऱ"] = "r"                     # ऱ
_CONS1["ळ"] = "l"                     # ळ
_CONS1["ऴ"] = "l"                     # ऴ
_VOW_SHORT = {"आ": "a", "ई": "i", "ऊ": "u"}
_MATRA_SHORT = {"ा": "a", "ी": "i", "ू": "u"}
_LABIAL = set("पफबभम")
# Place of articulation, for the unit context: anusvara takes the place of the
# consonant after it (अंबानी -> ambani), which a context-free table cannot see.
_PLACE = {}
for _p, _cs in (("k", "कखगघङक़ख़ग़"), ("c", "चछजझञज़"),
                ("t", "टठडढणड़ढ़"), ("d", "तथदधनऩ"),
                ("p", "पफबभमफ़"), ("y", "यरलवळऴऱय़"), ("s", "शषसह")):
    for _c in _cs:
        _PLACE[_c] = _p
_DEVA_WORD = re.compile("[ऀ-ॣॱ-ॿ" + _ZW + "]+")


def deva_units(word):
    """Grapheme units of one run of Devanagari letters -> [(key, kind, cons)].

    kind: C bare consonant (carries the inherent vowel), M consonant + matra,
    H consonant + virama, V independent vowel, N anusvara/chandrabindu,
    X visarga, U anything else (kept as is)."""
    out, i, n = [], 0, len(word)
    while i < n:
        ch = word[i]
        i += 1
        if ch in _ZW or ch == _NUKTA:
            continue
        if ch in _NUKTA_OF and i < n and word[i] == _NUKTA:
            ch = _NUKTA_OF[ch]
            i += 1
        if ch in _CONS1:
            while i < n and word[i] in _ZW:
                i += 1
            if i < n and word[i] in _MATRA:
                out.append((ch + word[i], "M", ch))
                i += 1
            elif i < n and word[i] == _VIRAMA:
                out.append((ch + _VIRAMA, "H", ch))
                i += 1
            else:
                out.append((ch, "C", ch))
        elif ch in _VOW:
            out.append((ch, "V", ""))
        elif ch in (_ANUSVARA, _CHANDRA):
            out.append((ch, "N", ""))
        elif ch == _VISARGA:
            out.append((ch, "X", ""))
        else:
            out.append((ch, "U", ""))
    return out


def unit_keys(units, i, keep):
    """-> the learned-table key of unit i, from the input alone.

    Each decision a unit makes has its own key: a consonant's spelling, a
    vowel's (by whether it ends the word: ी is often "i" there and "ee"
    inside), a nasal's (by the place of the consonant after it), and whether a
    bare consonant keeps its inherent vowel (by whether the previous unit ends
    in a vowel, the kind of the next unit, whether that one ends the word, and
    what the rules decided)."""
    key, kind, cons = units[i]
    last = len(units) - 1
    if kind == "C":
        pk = units[i - 1][1] if i else "^"
        prev = "^" if not i else {"C": "v", "M": "v", "V": "v", "H": "h"}.get(pk, "n")
        nxt = "$" if i == last else units[i + 1][1] + ("$" if i + 1 == last else "")
        return f"{prev}|{nxt}|{int(keep[i])}"
    if kind in ("M", "V"):
        return f"{key[-1]}|{'$' if i == last else ''}"
    if kind == "N":
        nxt = "$" if i == last else (_PLACE.get(units[i + 1][2]) or "v")
        return f"{key}|{nxt}"
    return key


def _schwa_kept(units, opts):
    """Which bare consonants keep their inherent vowel. Word-final schwa always
    goes; with opts["schwa"] a medial one goes too in the context V C _ C V
    (कमला -> kamla, सरकार -> sarkar, but कमल -> kamal), scanning right to
    left so two deletions never make a three-consonant cluster."""
    keep = [k == "C" for _, k, _ in units]
    if units and units[-1][1] == "C":
        keep[-1] = False
    if not opts.get("schwa"):
        return keep
    last = len(units) - 1
    for i in range(last - 1, 0, -1):
        if units[i][1] != "C":
            continue
        pk = units[i - 1][1]
        nk = units[i + 1][1]
        prev_vowel = pk in ("M", "V") or (pk == "C" and keep[i - 1])
        next_vowel = nk == "M" or (nk == "C" and keep[i + 1])
        if prev_vowel and next_vowel:
            keep[i] = False
    return keep


def rule_unit(units, i, keep, opts):
    """What unit i becomes under the rules with the given options."""
    key, kind, cons = units[i]
    short = opts.get("short")
    if kind in ("C", "M", "H"):
        c = _CONS1[cons]
        if kind == "C":
            return c + ("a" if keep[i] else "")
        if kind == "H":
            return c
        m = key[-1]
        return c + ((_MATRA_SHORT.get(m) if short else None) or _MATRA[m])
    if kind == "V":
        return (_VOW_SHORT.get(key) if short else None) or _VOW[key]
    if kind == "N":
        if (opts.get("nasal") and i + 1 < len(units)
                and units[i + 1][2] in _LABIAL):
            return "m"
        return "n"
    if kind == "X":
        return "h"
    return key


class TranslitModel:
    """The learned transliteration. `lexicon` maps a whole Devanagari word to
    its Latin spelling. `tables` holds the per-unit decisions keyed by
    unit_keys: "cons" (consonant -> spelling), "vowel", "nasal", "other"
    (-> spelling) and "schwa" (-> 1 keep / 0 drop). `opts` are the rule
    options for whatever the tables do not cover."""

    def __init__(self, lexicon=None, tables=None, opts=None, sha256=None):
        self.lexicon = dict(lexicon or {})
        self.tables = {k: dict((tables or {}).get(k, {}))
                       for k in ("cons", "vowel", "nasal", "schwa", "other")}
        self.opts = dict(opts or {})
        self.sha256 = sha256
        self._cache = {}

    @classmethod
    def from_json(cls, d, sha256=None):
        return cls(d["lexicon"], d["tables"], d["opts"], sha256)

    def word(self, w):
        out = self._cache.get(w)
        if out is None:
            out = self.lexicon.get(w)
            if out is None:
                out = self.word_units(w)
            if len(self._cache) < 1 << 18:
                self._cache[w] = out
        return out

    def word_units(self, w):
        units = deva_units(w)
        keep = _schwa_kept(units, self.opts)
        T, opts, out = self.tables, self.opts, []
        for i, (key, kind, cons) in enumerate(units):
            k = unit_keys(units, i, keep)
            if kind in ("C", "M", "H"):
                c = T["cons"].get(cons, _CONS1[cons])
                if kind == "C":
                    out.append(c + ("a" if T["schwa"].get(k, keep[i]) else ""))
                elif kind == "H":
                    out.append(c)
                else:
                    m = key[-1]
                    out.append(c + T["vowel"].get(
                        k, (_MATRA_SHORT.get(m) if opts.get("short") else None)
                        or _MATRA[m]))
            elif kind == "V":
                out.append(T["vowel"].get(k, rule_unit(units, i, keep, opts)))
            elif kind == "N":
                out.append(T["nasal"].get(k, rule_unit(units, i, keep, opts)))
            else:
                out.append(T["other"].get(k, rule_unit(units, i, keep, opts)))
        return "".join(out)

    def translit(self, s):
        """Each run of Devanagari letters is one word; everything between runs
        passes through, with Devanagari digits made ASCII as the rules do."""
        out, pos = [], 0
        for m in _DEVA_WORD.finditer(s):
            out.append(s[pos:m.start()])
            out.append(self.word(m.group()))
            pos = m.end()
        out.append(s[pos:])
        return "".join(out).translate(_DIGIT_TR)


_DIGIT_TR = str.maketrans(_DIGIT)
_MODEL = {}
_UNSET = object()
_ACTIVE = _UNSET            # the model translit() uses; resolved on first use


def translit_model(path=None, verbose=True):
    """-> TranslitModel from mine_translit.py's JSON, read once per path, or
    None when the file is absent (translit_rules is then used, unchanged). A
    file that is present but broken raises, like the corruption grammar.
    Without `path` this is TRANSLIT_PATH, and it becomes what translit() uses."""
    global _ACTIVE
    p = pathlib.Path(path or TRANSLIT_PATH)
    key = str(p.resolve())
    if key not in _MODEL:
        m = None
        if p.is_file():
            raw = p.read_bytes()
            try:
                m = TranslitModel.from_json(json.loads(raw), hashlib.sha256(raw).hexdigest())
            except (ValueError, KeyError, TypeError) as ex:
                raise ValueError(f"{p} is not a valid transliteration model ({ex}); "
                                 f"rerun mine_translit.py or remove the file") from ex
        elif verbose:
            print(f"note: no transliteration model at {p}; using the rule "
                  f"transliteration", file=sys.stderr, flush=True)
        _MODEL[key] = m
    if path is None:
        _ACTIVE = _MODEL[key]
    return _MODEL[key]


def set_translit_model(model):
    """Make translit() use `model` (a TranslitModel, or None for the rules).
    For tests and for the miner's own evaluation; returns the previous one."""
    global _ACTIVE
    prev, _ACTIVE = _ACTIVE, model
    return None if prev is _UNSET else prev


def _merge_initials(s: str) -> str:
    """Rejoin runs of single letters left behind by punctuation stripping.

    "S.A.R.L." becomes "s a r l" while "Sarl" becomes "sarl", which drops name
    token Jaccard from 1.0 to 0.38 on what is the same legal form. Punctuated
    acronyms are common in French legal forms (S.A.R.L., S.A.S., S.C.I.) and in
    US ones (L.L.C., P.C.), so this costs real recall in two of three countries.
    Runs of two or more are merged; a lone initial is left alone.
    """
    out, run = [], []
    for t in s.split():
        if len(t) == 1 and t.isalpha():
            run.append(t)
            continue
        if run:
            out.append("".join(run) if len(run) > 1 else run[0])
            run = []
        out.append(t)
    if run:
        out.append("".join(run) if len(run) > 1 else run[0])
    return " ".join(out)


# NFKD leaves these ligatures whole, so "Sœur" became "s ur" against "soeur".
_LIGATURES = str.maketrans({"œ": "oe", "æ": "ae"})
# A single letter glued to its neighbour by "/" (M/s, C/O, S/O) or a possessive
# 's left a stray initial that _merge_initials then fused with the next acronym:
# "M/s A.B.C. Traders" became "msabc traders" and lost "abc". Join them first.
_SLASH_PAIR = re.compile(r"\b[a-z](?:/[a-z])+\b")
_POSSESSIVE = re.compile(r"(?<=[a-z0-9])['\u2019\u02bc]s\b")


# One translate() pass does both the combining-mark strip and the ligature
# fold. Built from unicodedata.combining over every code point, so it drops
# exactly the characters the old per-character generator dropped; the two maps
# cannot interact because no ligature is a combining mark and "oe"/"ae" are not.
_STRIP = {cp: None for cp in range(0x110000) if unicodedata.combining(chr(cp))}
_STRIP.update({ord("œ"): "oe", ord("æ"): "ae"})
# On [0-9a-z]+ tokens joined by single spaces, a run of two or more single-letter
# tokens is exactly what _merge_initials joins; a lone initial never matches.
_INITIALS = re.compile(r"\b[a-z](?: [a-z]\b)+")


def _join_slash(m):
    return m.group().replace("/", "")


def _join_initials(m):
    return m.group().replace(" ", "")


def norm(s: str) -> str:
    """Fold case, accents and punctuation to [0-9a-z] tokens joined by single
    spaces. norm_reference is the plain statement of the same function; this
    one skips steps that provably cannot change the string (see
    tests/test_textnorm_equivalence.py)."""
    if not s:
        return ""
    if s.isascii():
        # NFKD is the identity on ASCII, casefold equals lower, and there are no
        # combining marks, ligatures, Devanagari or curly apostrophes to handle.
        s = s.lower()
        if "/" in s:
            s = _SLASH_PAIR.sub(_join_slash, s)
        if "'" in s:
            s = _POSSESSIVE.sub("s", s)
    else:
        if DEVA.search(s):
            s = translit(s)
        s = unicodedata.normalize("NFKD", s).casefold().translate(_STRIP)
        if "/" in s:
            s = _SLASH_PAIR.sub(_join_slash, s)
        if "'" in s or "\u2019" in s or "\u02bc" in s:
            s = _POSSESSIVE.sub("s", s)
    s = _NONALNUM.sub(" ", s).strip()
    if " " in s:
        s = _INITIALS.sub(_join_initials, s)
    return s


def norm_reference(s: str) -> str:
    """The original norm(), kept as the oracle norm() is tested against."""
    s = s or ""
    if DEVA.search(s):
        s = translit(s)
    s = unicodedata.normalize("NFKD", s).casefold()
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = _POSSESSIVE.sub("s", _SLASH_PAIR.sub(lambda m: m.group().replace("/", ""),
                                               s.translate(_LIGATURES)))
    return _merge_initials(_NONALNUM.sub(" ", s).strip())


# Transliteration is never exact ("Sharma"/"Sarma"/"Shrma"), so a folded view
# collapses the distinctions that vary between spellings. Applied to BOTH sides
# so a Latin record and a romanised one meet in the middle.
_FOLD = [("aa", "a"), ("ee", "i"), ("ii", "i"), ("oo", "u"), ("uu", "u"),
         ("ph", "f"), ("bh", "b"), ("gh", "g"), ("jh", "j"), ("kh", "k"),
         ("dh", "d"), ("th", "t"), ("chh", "ch"), ("sh", "s"), ("ck", "k"),
         ("w", "v"), ("y", "i"), ("z", "s")]


def fold(s: str) -> str:
    """Aggressive orthographic collapse for fuzzy cross-spelling keys."""
    for a, b in _FOLD:
        s = s.replace(a, b)
    out = [c for i, c in enumerate(s) if i == 0 or c != s[i - 1]]   # collapse doubles
    return "".join(out)


_VOWELS = str.maketrans("", "", "aeiou")


def skeleton(s: str) -> str:
    """Consonant skeleton: drop vowels entirely.

    Medial schwa deletion is context-dependent and a table cannot get it right
    ("modarn" vs "modern"), but both collapse to "mdrn". Cheap insurance against
    every vowel-level transliteration disagreement.
    """
    return " ".join(w for w in (t.translate(_VOWELS) for t in fold(s).split()) if len(w) > 1)


def toks(s: str) -> set:
    return set(norm(s).split())


def nums(s: str) -> set:
    """Tokens containing a digit — house numbers, PIN/ZIP, plot numbers."""
    return {t for t in toks(s) if any(c.isdigit() for c in t)}


def ngrams(s: str, n: int = 4) -> set:
    t = norm(s).replace(" ", "")
    return {t[i:i + n] for i in range(len(t) - n + 1)}
