"""Shared text normalisation. Deliberately conservative: it folds case, accents and
punctuation but never strips legal suffixes or stems, because those carry identity
(see audit finding F — 39% of S1 entities share a name with a DIFFERENT business)."""
import re
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
    """Devanagari -> Latin, syllable by syllable.

    A consonant carries an inherent 'a' which a matra replaces and a virama
    removes; that rule is what makes the output resemble how the same name is
    spelled in the Latin records.
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
_SLASH_PAIR = re.compile(r"\b([a-z])/([a-z])\b")
_POSSESSIVE = re.compile(r"(?<=[a-z0-9])['\u2019\u02bc]s\b")


def norm(s: str) -> str:
    s = s or ""
    if DEVA.search(s):
        s = translit(s)
    s = unicodedata.normalize("NFKD", s).casefold()
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = _POSSESSIVE.sub("s", _SLASH_PAIR.sub(r"\1\2", s.translate(_LIGATURES)))
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
