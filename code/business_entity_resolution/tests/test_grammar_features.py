"""strfeatures' corruption-grammar columns: behaviour on hand-made pairs, the
absent-file path, bit identity with build_reference on retrieval-shaped pairs,
and the tables mine_corruption.py actually writes."""
import json
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import strfeatures as sf  # noqa: E402
from test_strfeatures_equivalence import _assert_identical, _both, make_pairs  # noqa: E402
from textnorm import norm  # noqa: E402

COL = {k: i for i, k in enumerate(sf.NAMES)}
G = sf.GRAMMAR_NAMES

GRAMMAR = {
    "abbreviations": {"name": [
        {"s1": "corporation", "s2": "corp", "kind": "abbrev", "p_s2_given_s1": 0.45},
        {"s1": "private", "s2": "pvt", "kind": "abbrev", "p_s2_given_s1": 0.6},
        {"s1": "limited", "s2": "ltd", "kind": "abbrev", "p_s2_given_s1": 0.7}],
        "address": []},
    "region_codes": {"entries": [
        {"s1": "haryana", "s2": "hr", "kind": "abbrev", "long": "haryana", "short": "hr"},
        {"long": "uttar pradesh", "short": "up", "direction": "contract",
         "heuristic": "acronym"}]},
    "forbidden": {"note": "...",
                  "name": [{"s1": "incorporated", "s2": "corp"},
                           {"s1": "corporation", "s2": "inc"}],
                  "address": [{"s1": "road", "s2": "street"}]},
}


def pairs(qs, cs, grammar):
    """One query per (query, candidate) row. -> feature rows, by build()."""
    qn, qa = [norm(x) for x, _ in qs], [norm(x) for _, x in qs]
    cn, ca = [norm(x) for x, _ in cs], [norm(x) for _, x in cs]
    recs = sf.precompute_records(cn, ca)
    X = sf.build(recs, qn, qa, np.arange(len(qs)), np.arange(len(cs)), None, grammar)
    ref = sf.build_reference(qn, qa, np.arange(len(qs)), cn, ca, None, grammar)
    _assert_identical(ref, X)
    return X


def test_abbreviation_raises_adjusted_jaccard():
    g = sf.Grammar.from_json(GRAMMAR)
    X = pairs([("Acme Corporation", ""), ("Acme Corporation", ""),
               ("Acme Corporation", "")],
              [("Acme Corp", ""), ("Acme Inc", ""), ("Acme Corporation", "")], g)
    abbr, other, same = X
    assert abbr[COL["nm_jac"]] == pytest.approx(1 / 3)          # corp/corporation unrelated
    assert abbr[COL["abbrev_adjusted_jaccard"]] == 1.0           # ... until now
    assert abbr[COL["abbrev_matched_tokens"]] == pytest.approx(0.45)
    assert other[COL["abbrev_adjusted_jaccard"]] == other[COL["nm_jac"]]
    # a token both sides share is never re-explained by the grammar
    assert same[COL["abbrev_matched_tokens"]] == 0
    assert same[COL["abbrev_adjusted_jaccard"]] == 1.0


def test_two_abbreviations_weights_add():
    g = sf.Grammar.from_json(GRAMMAR)
    X = pairs([("Sharma Private Limited", "")], [("Sharma Pvt Ltd", "")], g)[0]
    assert X[COL["abbrev_matched_tokens"]] == pytest.approx(0.6 + 0.7)
    assert X[COL["abbrev_adjusted_jaccard"]] == 1.0


def test_forbidden_pair_is_flagged():
    g = sf.Grammar.from_json(GRAMMAR)
    X = pairs([("Acme Corporation", ""), ("Acme Incorporated", ""),
               ("Acme Incorporated Corp", ""), ("Acme", "5 Mill Road")],
              [("Acme Inc", ""), ("Acme Corp", ""), ("Acme Corp", ""),
               ("Acme", "5 Mill Street")], g)
    assert X[0, COL["forbidden_hits"]] == 1                      # corporation -> inc
    assert X[1, COL["forbidden_hits"]] == 1                      # incorporated -> corp
    assert X[2, COL["forbidden_hits"]] == 0                      # corp is shared
    assert X[3, COL["forbidden_hits"]] == 1                      # address table too
    # directional: the generator's S1 -> S2 map, not its reverse
    rev = pairs([("Acme Corp", "")], [("Acme Incorporated", "")], g)[0]
    assert rev[COL["forbidden_hits"]] == 0


def test_region_codes_single_and_multi_token():
    g = sf.Grammar.from_json(GRAMMAR)
    X = pairs([("A", "12 MG Road, Haryana"), ("A", "12 MG Road, Uttar Pradesh"),
               ("A", "12 MG Road, Uttar"), ("A", "12 MG Road, Haryana")],
              [("A", "12 MG Road HR"), ("A", "12 MG Road UP"),
               ("A", "12 MG Road UP"), ("A", "12 MG Road")], g)
    assert list(X[:, COL["region_code_matched"]]) == [1, 1, 0, 0]
    # region codes are address evidence; the name columns do not move
    assert not X[:, COL["abbrev_matched_tokens"]].any()


def test_absent_or_broken_grammar_is_zero_not_a_crash(tmp_path, capsys):
    assert sf.load_grammar(tmp_path / "missing.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert sf.load_grammar(bad) is None
    X = pairs([("Acme Corporation", "")], [("Acme Corp", "")], None)
    assert not X[:, [COL[k] for k in G]].any()


def test_load_grammar_reads_file_once(tmp_path):
    p = tmp_path / "g.json"
    p.write_text(json.dumps(GRAMMAR))
    g = sf.load_grammar(p)
    assert len(g) == 8 and len(g.sha256) == 64
    p.write_text("{}")
    assert sf.load_grammar(p) is g                               # cached by path


def test_entries_outside_the_corpus_vocab_never_fire():
    g = sf.Grammar.from_json(GRAMMAR)
    recs = sf.precompute_records(["acme ltd"], [""])
    t = sf._grammar_tables(recs, g)
    assert int((t["name"].kind == sf._ABBREV).sum()) == 1        # only limited -> ltd
    assert len(t["address"].kind) == 0


def _random_grammar(data, rng, n=400):
    """Entries drawn from the differing tokens of sampled pairs, so every kind,
    multi-token acronyms in both directions included, fires on many pairs."""
    q_names, q_addrs, q_idx, c_names, c_addrs, c = data
    ab, fb, fa, rg = [], [], [], []
    for i in rng.choice(len(c), n, replace=False):
        e, j = q_idx[i], c[i]
        for qs, cs, out in ((q_names[e], c_names[j], "name"),
                            (q_addrs[e], c_addrs[j], "address")):
            qt, ct = set(qs.split()), set(cs.split())
            qx, cx = sorted(qt - ct), sorted(ct - qt)
            if not qx or not cx:
                continue
            a, b = qx[rng.integers(len(qx))], cx[rng.integers(len(cx))]
            if out == "name":
                (ab if rng.random() < 0.5 else fb).append(
                    {"s1": a, "s2": b, "p_s2_given_s1": float(rng.random())})
            elif rng.random() < 0.3:
                fa.append({"s1": a, "s2": b})
            elif len(qx) >= 2 and rng.random() < 0.5:
                rg.append({"long": " ".join(qx[:2]), "short": b, "direction": "contract"})
            elif len(cx) >= 2 and rng.random() < 0.5:
                rg.append({"long": " ".join(cx[:2]), "short": a, "direction": "expand"})
            else:
                rg.append({"s1": a, "s2": b})
    return sf.Grammar.from_json({"abbreviations": {"name": ab},
                                 "region_codes": {"entries": rg},
                                 "forbidden": {"name": fb, "address": fa}})


@pytest.mark.parametrize("chunks", [False, True])
def test_bit_identical_with_grammar(monkeypatch, chunks):
    if chunks:
        monkeypatch.setattr(sf, "_CHUNK", 97)
        monkeypatch.setattr(sf, "_PAIR_CHUNK", 1013)
    rng = np.random.default_rng(7)
    data = make_pairs(1200 if chunks else 2500, 24, rng)
    g = _random_grammar(data, rng)
    old, new = _both(*data, None, g)
    _assert_identical(old, new)
    # the test only means something if every column fires on many pairs
    for k in G:
        assert (old[:, COL[k]] > 0).sum() >= 100, k
    multi = [x for x in g.tables["address"] if len(x[0]) > 1 or len(x[1]) > 1]
    assert multi and sum(bool(sf._fired(multi, set(data[1][e].split()),
                                        set(data[4][j].split())))
                         for e, j in zip(data[2], data[5])) >= 20
    assert (old[:, COL["abbrev_adjusted_jaccard"]] > old[:, COL["nm_jac"]]).any()


def test_consumes_what_the_miner_writes():
    import mine_corruption as mc
    tables = mc.build_tables(mc.mine_records(mc.synth_pairs(8000)),
                             min_support=5, min_trials=200)
    g = sf.Grammar.from_json(json.loads(json.dumps(tables)))
    X = pairs([("Sharma Global Corporation", "5 Station Road, Haryana"),
               ("Sharma Global Incorporated", "5 Station Road, Texas")],
              [("Sharma Global Corp", "5 Station Rd HR"),
               ("Sharma Global Corp", "5 Station Rd TX")], g)
    assert X[0, COL["abbrev_adjusted_jaccard"]] == 1.0
    assert X[0, COL["forbidden_hits"]] == 0
    assert X[1, COL["forbidden_hits"]] >= 1
    assert X[0, COL["region_code_matched"]] >= 1
    assert X[1, COL["region_code_matched"]] >= 1
