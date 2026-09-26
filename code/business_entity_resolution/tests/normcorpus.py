"""Synthetic raw names and addresses for textnorm tests and benchmarks.

Covers what norm() branches on: pure ASCII, accented Latin, ligatures,
Devanagari (with matras, virama, nukta, digits), punctuated acronyms, possessives
with all three apostrophes, M/s C/O S/O forms, empty and None, digit-heavy
addresses, and a slice of random code points from every Unicode plane so the
casefold/NFKD/combining-mark edge cases (İ, ß, ﬁ, fullwidth, superscripts) are
hit too. `dup` controls how often a string repeats an earlier one, because the
production path dedupes and a synthetic corpus cannot know the real rate.
"""
import random

_LAT = "abcdefghijklmnopqrstuvwxyz"
_ACC = "àáâäãåāçčćéèêëēėęíìîïīñńóòôöõøōœæßšśûüùúūýÿžźżÀÉÎÖÜÇŒÆİıǰ"
_DEVA_CONS = "कखगघचछजझटठडढणतथदधनपफबभमयरलवशषसहळ"
_DEVA_MATRA = "ािीुूृेैोौॅॉ"
_DEVA_MISC = "्ंँः़अआइईउऊएऐओऔ०१२३४५६७८९\u0958\u095c\u0964\u0965\u200c\u200d"
_LEGAL = ["Pvt Ltd", "Private Limited", "LLC", "L.L.C.", "Inc.", "S.A.R.L.", "SARL",
          "S.A.S.", "S.C.I.", "P.C.", "& Co.", "GmbH", "Corp", "LLP", "EURL", "SA"]
_PREFIX = ["M/s", "M/S", "m/s", "C/O", "c/o", "S/O", "D/o", "W/O", "Dr.", "Mr.", "Shri"]
_APOS = ["'s", "’s", "ʼs", "'S", "s'", "'"]
_STREET = ["Street", "St.", "Road", "Rd", "Avenue", "Ave.", "Rue", "Boulevard",
           "Marg", "Nagar", "Sector", "Chowk", "Lane", "Apt", "Suite", "#", "No."]
_PUNCT = " .,-/&()'\"#:;!?+*@\t\n  --  ..//"


def _word(r, alpha=_LAT, lo=2, hi=10):
    w = "".join(r.choice(alpha) for _ in range(r.randint(lo, hi)))
    return w.capitalize() if r.random() < 0.6 else (w.upper() if r.random() < 0.3 else w)


def _deva_word(r):
    out = []
    for _ in range(r.randint(1, 4)):
        out.append(r.choice(_DEVA_CONS))
        x = r.random()
        if x < 0.5:
            out.append(r.choice(_DEVA_MATRA))
        elif x < 0.65:
            out.append(r.choice(_DEVA_MISC))
    return "".join(out)


def _acronym(r):
    letters = [r.choice(_LAT).upper() for _ in range(r.randint(1, 5))]
    sep = r.choice([".", ". ", " ", "-", "/", ""])
    return sep.join(letters) + (sep if r.random() < 0.5 else "")


def _random_unicode(r):
    out = []
    for _ in range(r.randint(1, 12)):
        plane = r.random()
        if plane < 0.5:
            cp = r.randint(0x80, 0x2FFF)          # Latin ext, Greek, combining, IPA...
        elif plane < 0.8:
            cp = r.randint(0x3000, 0xFFFF)        # CJK, fullwidth, ligatures (FBxx)
        else:
            cp = r.randint(0x10000, 0x10FFFF)
        if 0xD800 <= cp <= 0xDFFF:
            cp = 0x41
        out.append(chr(cp))
        if r.random() < 0.3:
            out.append(r.choice(_LAT + " ./'"))
    return "".join(out)


def _name(r):
    kind = r.random()
    parts = []
    if r.random() < 0.15:
        parts.append(r.choice(_PREFIX))
    if kind < 0.55:
        parts += [_word(r) for _ in range(r.randint(1, 4))]
    elif kind < 0.7:
        parts += [_word(r, _LAT + _ACC) for _ in range(r.randint(1, 4))]
    elif kind < 0.82:
        parts += [_deva_word(r) for _ in range(r.randint(1, 4))]
        if r.random() < 0.3:
            parts.append(_word(r))
    elif kind < 0.92:
        parts += [_acronym(r), _word(r)]
    else:
        parts.append(_random_unicode(r))
    if r.random() < 0.2:
        parts[r.randrange(len(parts))] += r.choice(_APOS)
    if r.random() < 0.4:
        parts.append(r.choice(_LEGAL))
    return _join(r, parts)


def _address(r):
    parts = []
    if r.random() < 0.1:
        parts.append(r.choice(_PREFIX) + " " + _word(r))
    parts.append(str(r.randint(1, 99999)) + r.choice(["", "A", "/2", "-B", ", "]))
    deva = r.random() < 0.15
    for _ in range(r.randint(1, 6)):
        x = r.random()
        if x < 0.25:
            parts.append(r.choice(["", "#", "No. ", "Plot "]) + str(r.randint(0, 9999)))
        elif deva:
            parts.append(_deva_word(r))
        elif x < 0.85:
            parts.append(_word(r, _LAT + (_ACC if r.random() < 0.2 else "")))
        else:
            parts.append(r.choice(_STREET))
    parts.append(r.choice(["", str(r.randint(100000, 999999)), "75001", "IN", "France"]))
    if r.random() < 0.03:
        parts.append(_random_unicode(r))
    return _join(r, parts)


def _join(r, parts):
    return "".join(p + r.choice(_PUNCT[:3] if r.random() < 0.8 else _PUNCT)
                   for p in parts).strip(" " if r.random() < 0.9 else "")


def _special(r):
    """Hand-picked edge cases, drawn at a low rate."""
    return r.choice([
        None, "", " ", "...", "'", "/", "a", "A B", "S.A.R.L.", "s. a. r. l.",
        "M/s A.B.C. Traders", "C/O Ram's Shop", "Sœur Æsop", "ﬁne ﬂour", "İstanbul",
        "Straße", "ǰ", "Ａ．Ｂ．Ｃ", "x²y³", "½", "a/b/c/d e/f", "O'Brien's", "L'Oréal",
        "Macy’s", "Joeʼs", "राम", "श्री गणेश ट्रेडर्स", "१२३ मार्ग", "क़ ख़ ग़ ज़", "\u0958\u095c\u093e \u095e", "क\u093c\u093e", "क\u094d\u200dष", "स\u00a0म\u2003",
        "́́", "é", "a b c d 1 e f", "1 2 3", "a1 b c", "ab c d ef g",
        "A.B. C.D.", "Dr.  J. K.  Rowling", "\t\n", "---a---b---",
    ])


def make(n, seed=0, dup=0.0):
    """n raw strings, half names and half addresses. With probability `dup` a
    string repeats one already drawn."""
    r = random.Random(seed)
    out = []
    for i in range(n):
        if out and r.random() < dup:
            out.append(out[r.randrange(len(out))])
        elif r.random() < 0.02:
            out.append(_special(r))
        else:
            out.append(_name(r) if i % 2 == 0 else _address(r))
    return out
