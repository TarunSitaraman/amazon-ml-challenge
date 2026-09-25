"""Shared text normalisation. Deliberately conservative: it folds case, accents and
punctuation but never strips legal suffixes or stems, because those carry identity
(see audit finding F — 39% of S1 entities share a name with a DIFFERENT business)."""
import re
import unicodedata

DEVA = re.compile(r"[ऀ-ॿ]")
_NONALNUM = re.compile(r"[^0-9a-z]+")


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").casefold()
    s = "".join(c for c in s if not unicodedata.combining(c))
    return _NONALNUM.sub(" ", s).strip()


def toks(s: str) -> set:
    return set(norm(s).split())


def nums(s: str) -> set:
    """Tokens containing a digit — house numbers, PIN/ZIP, plot numbers."""
    return {t for t in toks(s) if any(c.isdigit() for c in t)}


def ngrams(s: str, n: int = 4) -> set:
    t = norm(s).replace(" ", "")
    return {t[i:i + n] for i in range(len(t) - n + 1)}
