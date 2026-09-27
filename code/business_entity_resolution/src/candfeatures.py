"""Candidate-record features: properties of the S2/S3 record alone, no pair.

Real matches are corrupted copies of an S1 record; distractors (26% of S2/S3,
matching nothing) look clean and complete. Measured on train (diag_leak.py L6/L7),
matched vs distractor:

  name tokens 3.30 vs 4.04 | name chars 23.8 vs 28.6 | all-caps (S2) 20.5% vs 14.5%
  all-lower 6.8% vs 2.7%   | double space 11.6% vs 9.4%
  exact-name record in the other source: 24.9% vs 9.9%

Distractor false positives are 21.8% of the matcher's gap (diag_matcher.py), and
no pair feature can see this: it is a property of the candidate, not of how well
it agrees with the entity.

Everything is computed from the RAW name, before norm(): casing, spacing and
punctuation are exactly what normalisation erases. One vectorised Arrow pass
per corpus in blocking.build_corpus; per pair it is a row gather (take()).

Address empty is already a pair column (strfeatures ad_empty_c) and is not
repeated here.

Columns are appended after strfeatures.NAMES so every existing column keeps its
position. CAND_FEATS=0 drops them, for an A/B on the same code; either way the
model must be retrained with the same setting (predict.py checks the count).
"""
import os

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

ENABLED = os.environ.get("CAND_FEATS", "1") != "0"

ALL_NAMES = ["cand_name_tokens", "cand_name_chars", "cand_all_caps",
             "cand_all_lower", "cand_double_space", "cand_junk_affix",
             "cand_xsrc_twin"]
NAMES = ALL_NAMES if ENABLED else []
_CHUNK = 500_000

# A leading run of punctuation/symbols ("<<Acme", "-- Acme"), or a trailing run
# holding anything but periods ("Acme >>", "Acme ##", "Ltd.)"). A bare trailing
# period is an abbreviation ("Corp.", "Pvt. Ltd."), not junk -- the same rule as
# mine_corruption._affixes. Interior symbols ("A & B") are left alone: without
# the S1 side to compare against they are ordinary name text.
# RE2's \s is ASCII only; \p{Z} adds U+00A0 and the other Unicode spaces
# that Python's str.strip() and str.isspace() count.
_PREFIX = r"^[\s\p{Z}]*[\pP\pS]"
_SUFFIX = r"[\pP\pS][\s\p{Z}]*$"
_PERIOD_TAIL = r"(^|[^\pP\pS\s\p{Z}])[.\s\p{Z}]*$"


def _np(a):
    return a.to_numpy(zero_copy_only=False)


def _n_tokens(names):
    """len(x.split()) per name: Arrow splits on each Unicode whitespace
    character, so runs of spaces leave empty pieces, which are not counted."""
    parts = pc.utf8_split_whitespace(names)
    keep = _np(pc.not_equal(pc.utf8_length(pc.list_flatten(parts)), 0))
    return np.bincount(_np(pc.list_parent_indices(parts))[keep],
                       minlength=len(names))


def _flags(names):
    """names: Arrow string array, nulls filled. -> (n x 6) float32."""
    junk = pc.or_(pc.match_substring_regex(names, _PREFIX),
                  pc.and_(pc.match_substring_regex(names, _SUFFIX),
                          pc.invert(pc.match_substring_regex(names, _PERIOD_TAIL))))
    cols = [_n_tokens(names),
            pc.utf8_length(names),
            pc.utf8_is_upper(names),                     # == str.isupper()
            pc.utf8_is_lower(names),                     # == str.islower()
            pc.match_substring(names, "  "),
            junk]
    return np.column_stack([np.asarray(c if isinstance(c, np.ndarray) else _np(c),
                                       np.float32) for c in cols])


def cross_source_twin(names, is_s3):
    """1 where the record's lowercased raw name is also the lowercased raw name
    of some record in the OTHER source (S2 <-> S3), as diag_leak.py L7 defines
    it. The corpus is one country, so this is within country. Empty names are
    never twins.

    Arrow is_in builds its hash set in C++ over the other source's keys; no
    Python object per record.
    """
    key = pc.utf8_lower(names)
    s3 = pa.array(is_s3, pa.bool_())
    k2, k3 = key.filter(pc.invert(s3)), key.filter(s3)
    out = np.zeros(len(key), np.float32)
    s3n = np.asarray(is_s3, bool)
    out[~s3n] = _np(pc.is_in(k2, value_set=pc.unique(k3)))
    out[s3n] = _np(pc.is_in(k3, value_set=pc.unique(k2)))
    out[_np(pc.equal(key, ""))] = 0
    return out


def record_matrix(corpus_tab):
    """corpus_tab: the country's S2 + S3 table (blocking.load_shard), with
    entity_id and business_name. -> (n_corpus x len(NAMES)) float32."""
    n = corpus_tab.num_rows
    if not ENABLED:
        return np.zeros((n, 0), np.float32)
    names = pc.fill_null(corpus_tab.column("business_name"), "").combine_chunks()
    is_s3 = _np(pc.starts_with(corpus_tab.column("entity_id"), "S3-")).astype(bool)
    out = np.empty((n, len(NAMES)), np.float32)
    # Chunked: the split lists behind the token count cost ~80 B per record.
    for lo in range(0, n, _CHUNK):
        out[lo:lo + _CHUNK, :-1] = _flags(names.slice(lo, _CHUNK))
    out[:, -1] = cross_source_twin(names, is_s3)
    return out


def build(cand, c):
    """cand: record_matrix of the corpus. c: corpus index per pair."""
    return cand[c]


def _reference(names, ids):
    """Plain-Python oracle: the diag_leak.py definitions record by record."""
    import re
    import unicodedata

    def junk(x):
        x = x.strip()
        i = 0
        while i < len(x) and (unicodedata.category(x[i])[0] in "PS" or x[i].isspace()):
            i += 1
        if x[:i].strip():
            return True
        e = len(x)
        while e > i and (unicodedata.category(x[e - 1])[0] in "PS" or x[e - 1].isspace()):
            e -= 1
        tail = x[e:].strip()
        return bool(tail and "".join(tail.split()).strip(".") and e > 0)

    names = [x or "" for x in names]
    lo = [x.lower() for x in names]
    src = [i.startswith("S3-") for i in ids]
    by = {False: set(), True: set()}
    for k, s in zip(lo, src):
        by[s].add(k)
    rows = [[len(x.split()), len(x), x.isupper(), x.islower(), "  " in x, junk(x),
             bool(k) and k in by[not s]]
            for x, k, s in zip(names, lo, src)]
    return np.array(rows, np.float32).reshape(len(names), len(ALL_NAMES))


def _self_test(n=200_000, seed=0):
    import time
    rng = np.random.default_rng(seed)
    words = ["acme", "Global", "TRADERS", "pvt", "Ltd.", "&", "Co.", "India",
             "श्री", "गणेश", "Müller", "GmbH", "e.K.", "Inc", "S.A."]
    junk = ["<<", ">>", "--", "##", "**", "...", ".", ""]

    def name():
        t = list(rng.choice(words, rng.integers(0, 6)))
        s = (" " if rng.random() < 0.9 else "  ").join(t)
        r = rng.random()
        s = s.upper() if r < 0.2 else s.lower() if r < 0.3 else s
        if rng.random() < 0.1:
            s = rng.choice(junk) + " " + s
        if rng.random() < 0.1:
            s = s + " " + rng.choice(junk)
        if rng.random() < 0.02:
            s = " " + s + " "
        return s

    base = [name() for _ in range(n // 4)]
    names = [base[i] if rng.random() < 0.3 else name()
             for i in rng.integers(0, len(base), n)]
    names[0], names[1] = None, ""
    ids = [f"S{rng.choice([2, 3])}-{i}" for i in range(n)]
    tab = pa.table({"entity_id": ids, "business_name": names})

    fast = record_matrix(tab)
    ref = _reference(names, ids)
    bad = np.flatnonzero((fast != ref).any(1))
    for i in bad[:10]:
        print(repr(names[i]), ids[i], fast[i], ref[i])
    assert not len(bad), f"{len(bad)} records differ from the reference"
    for k, j in zip(ALL_NAMES, fast.mean(0)):
        print(f"  {k:18s} mean {j:.4f}")

    tabs = {m: pa.table({"entity_id": ids * m, "business_name": names * m})
            for m in (1, 5)}
    for m, t in tabs.items():
        t0 = time.perf_counter()
        record_matrix(t)
        dt = time.perf_counter() - t0
        print(f"  record_matrix: {t.num_rows:,} records in {dt:.2f}s "
              f"({t.num_rows / dt / 1e6:.2f}M records/s)")
    X = fast
    c = rng.integers(0, n, 5_000_000)
    t0 = time.perf_counter()
    build(X, c)
    dt = time.perf_counter() - t0
    print(f"  build (gather): {len(c):,} pairs in {dt:.3f}s "
          f"({len(c) / dt / 1e6:.1f}M pairs/s)")
    print("candfeatures self-test OK")


if __name__ == "__main__":
    _self_test()
