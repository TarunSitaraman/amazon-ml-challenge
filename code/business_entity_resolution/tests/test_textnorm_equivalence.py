"""textnorm.norm and blocking.norm_column must equal norm_reference (the
original norm) for every input, bit for bit. strfeatures.precompute_records
rejects anything that is not norm() output, and every index key is built from
it, so any drift would silently change retrieval."""
import pathlib
import random
import sys

import pyarrow as pa
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import blocking  # noqa: E402
import normcorpus  # noqa: E402
import textnorm  # noqa: E402
from textnorm import norm, norm_reference  # noqa: E402


def _mismatches(xs):
    return [(x, norm(x), norm_reference(x)) for x in xs if norm(x) != norm_reference(x)]


@pytest.mark.parametrize("seed", [0, 1])
def test_norm_matches_reference_on_corpus(seed):
    xs = normcorpus.make(120_000, seed=seed)
    assert _mismatches(xs) == []


def test_corpus_covers_every_branch():
    xs = [x for x in normcorpus.make(120_000, seed=0) if x]
    has = lambda f: sum(map(f, xs))  # noqa: E731
    assert has(str.isascii) > 30_000
    assert has(lambda x: bool(textnorm.DEVA.search(x))) > 10_000
    assert has(lambda x: not x.isascii() and not textnorm.DEVA.search(x)) > 10_000
    assert has(lambda x: "/" in x) > 5_000
    assert has(lambda x: "'" in x or "’" in x or "ʼ" in x) > 5_000
    runs = lambda x: textnorm._INITIALS.search(textnorm._NONALNUM.sub(" ", x.lower()))  # noqa: E731
    assert has(lambda x: bool(runs(x))) > 5_000
    assert has(lambda x: sum(c.isdigit() for c in x) > 5) > 10_000


def test_norm_matches_reference_on_every_code_point():
    # Each character alone and between letters, so casefold/NFKD expansions,
    # combining marks, apostrophe look-alikes and \b around non-ASCII word
    # characters are all exercised.
    cps = [chr(c) for c in range(0x110000) if not 0xD800 <= c <= 0xDFFF]
    assert _mismatches(cps) == []
    assert _mismatches([f"A{c}'s b/{c}" for c in cps]) == []


def test_initials_regex_matches_merge_initials():
    r = random.Random(0)
    toks = list("abcxyz0123") + ["ab", "a1", "1a", "abc", "12"]
    for _ in range(50_000):
        s = " ".join(r.choice(toks) for _ in range(r.randint(1, 9)))
        assert textnorm._INITIALS.sub(textnorm._join_initials, s) == textnorm._merge_initials(s)


def _chunked(xs, size=7_000):
    return pa.chunked_array([xs[i:i + size] for i in range(0, len(xs), size)] or [[]],
                            pa.string())


@pytest.mark.parametrize("dup", [0.0, 0.5, 0.95])
def test_norm_column_matches_reference(dup):
    xs = normcorpus.make(40_000, seed=3, dup=dup)
    xs += [None, "", None]
    out, n_distinct = blocking.norm_column(_chunked(xs))
    assert out == [norm_reference(x) for x in xs]
    assert n_distinct == len({x or "" for x in xs})


def test_norm_column_empty():
    assert blocking.norm_column(pa.chunked_array([], pa.string())) == ([], 0)
    assert blocking.norm_column(pa.chunked_array([[None]], pa.string())) == ([""], 1)


def test_normalise_table():
    xs = normcorpus.make(10_000, seed=4, dup=0.5)
    tab = pa.table({"business_name": xs[::2], "business_address": xs[1::2]})
    stats = {}
    names, addrs = blocking.normalise(tab, stats)
    assert names == [norm_reference(x) for x in xs[::2]]
    assert addrs == [norm_reference(x) for x in xs[1::2]]
    assert set(stats) == {"names", "addrs"}
