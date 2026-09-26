"""The learned Devanagari transliteration (textnorm.TranslitModel, written by
mine_translit.py): norm() is unchanged on everything that is not Devanagari,
the fast path still equals the oracle with a model installed, the model file
round-trips, an absent file falls back to the rules, and the miner's synthetic
self-test passes."""
import argparse
import json
import pathlib
import re
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import mine_translit as mt  # noqa: E402
import normcorpus  # noqa: E402
import textnorm  # noqa: E402
from textnorm import DEVA, TranslitModel, norm, norm_reference  # noqa: E402

NORM_FORM = re.compile(r"(?:[0-9a-z]+(?: [0-9a-z]+)*)?")


@pytest.fixture(scope="module")
def model():
    tr, _, _ = mt.synth_records(8000, seed=1)
    m, _ = mt.fit(mt.extract(tr)[0])
    return m


@pytest.fixture
def use(model):
    """Install a model for one test and restore whatever was there."""
    installed = []

    def _use(m):
        installed.append(textnorm.set_translit_model(m))
    yield _use
    if installed:
        textnorm.set_translit_model(installed[0])


def _corpus():
    xs = normcorpus.make(120_000, seed=0) + normcorpus.make(120_000, seed=1)
    cps = [chr(c) for c in range(0x110000) if not 0xD800 <= c <= 0xDFFF]
    return xs + cps + [f"A{c}'s b/{c}" for c in cps[:0x3000]]


def test_non_devanagari_output_unchanged(model, use):
    xs = _corpus()
    use(None)
    before = [norm(x) for x in xs]
    use(model)
    after = [norm(x) for x in xs]
    plain = [i for i, x in enumerate(xs) if not DEVA.search(x or "")]
    deva = [i for i, x in enumerate(xs) if DEVA.search(x or "")]
    assert len(plain) > 1_000_000 and len(deva) > 20_000
    assert [i for i in plain if before[i] != after[i]] == []
    # ... and the model is live on the rest
    assert sum(before[i] != after[i] for i in deva) > 0.3 * len(deva)


def test_norm_matches_reference_with_model(model, use):
    use(model)
    xs = normcorpus.make(60_000, seed=2)
    assert [(x, norm(x), norm_reference(x)) for x in xs
            if norm(x) != norm_reference(x)] == []
    # strfeatures.precompute_records accepts only this shape
    assert all(NORM_FORM.fullmatch(norm(x)) for x in xs)


def test_rules_options_on_hand_examples(use):
    use(TranslitModel(opts={"schwa": True, "nasal": True, "short": True}))
    cases = {"कमला": "kamla", "कमल": "kamal", "सरकार": "sarkar", "अंबानी": "ambani",
             "शर्मा": "sharma", "राम.": "ram", "भारत १२३": "bharat 123",
             "क़ाज़ी": "qazi", "क़ाज़ी": "qazi",
             "M/s शर्मा Traders": "ms sharma traders", "क्‍ष": "ksh"}
    assert {k: norm(k) for k in cases} == cases


def test_rules_unchanged_without_model(use):
    use(None)
    # the pre-existing transliteration, quirks included (राम. kept its "a")
    assert norm("शर्मा") == "sharmaa"
    assert norm("राम.") == "raama"
    assert norm("कमला") == "kamalaa"


def test_lexicon_then_tables_then_rules(use):
    m = TranslitModel(lexicon={"प्राइवेट": "private"},
                      tables={"cons": {"ष": "s"}, "vowel": {"ी|$": "i"},
                              "nasal": {"ं|s": "n"}, "schwa": {"^|C$|1": 0}},
                      opts={})
    use(m)
    assert norm("प्राइवेट") == "private"
    assert norm("वर्षा") == "varsaa"       # ष from the table, ा from the rules
    assert norm("लक्ष्मी") == "laksmi"      # final ी from the table
    assert norm("कमल") == "kamal"           # no schwa entry fits: rules


def test_absent_and_broken_files(tmp_path):
    assert textnorm.translit_model(tmp_path / "none.json", verbose=False) is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(ValueError, match="not a valid transliteration model"):
        textnorm.translit_model(bad)


def test_model_file_round_trip(tmp_path, use):
    tr, te, _ = mt.synth_records(6000, seed=3)
    out = tmp_path / "translit_model.json"
    args = argparse.Namespace(min_support=5, min_share=0.6, min_count=5, min_ctx=10,
                              out=str(out), sample=0, seed=0, countries=None)
    written, fitted = mt.run(args, tr, te)
    loaded = textnorm.translit_model(out)
    assert loaded.sha256 and loaded.lexicon == fitted.lexicon
    assert loaded.tables == fitted.tables and loaded.opts == fitted.opts
    assert json.loads(out.read_text(encoding="utf-8"))["provenance"]["external_data"].startswith("none")
    words = {w for _, _, n2, _ in te for w in mt.deva_words(n2)}
    assert all(loaded.word(w) == fitted.word(w) for w in words)


def test_self_test_passes():
    assert mt.self_test(20000)
