"""predict.py --calibrate: per-country cardinality calibration (calibrate.py).

Builds a three-country synthetic parquet tree and runs predict.main with a stub
model whose scores for France are deflated, standing in for a country the
matcher under-rates, so it needs no data/ and no model.pkl.
"""
import pathlib
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import calibrate  # noqa: E402
import features  # noqa: E402
import resume  # noqa: E402
import strfeatures  # noqa: E402
from metric import choose_k  # noqa: E402

COUNTRIES = ("France", "India", "US")
N_ENT = 120
MATCH_N = (0, 1, 2, 2, 3, 3, 3, 4, 4, 5)       # true cardinality draw
_NM = len(features.NAMES) + strfeatures.NAMES.index("nm_jac")
_AD = len(features.NAMES) + strfeatures.NAMES.index("ad_jac")
DEFLATE = {"France": 0.85}


def _word(rng):
    return "".join(rng.choice(list("bcdfgklmnprstvz")) + rng.choice(list("aeiou"))
                   for _ in range(3)).capitalize()


def _rows(seed):
    """country -> source -> [(id, name, address)]: S1 entities, 0-5 true
    copies each split across S2/S3 (one name token sometimes dropped), plus
    distractors that reuse a token of some entity's name."""
    rng = np.random.default_rng(seed)
    out, n2, n3, n1 = {}, 0, 0, 0
    for ctry in COUNTRIES:
        by = {1: [], 2: [], 3: []}
        for _ in range(N_ENT):
            n1 += 1
            name = " ".join(_word(rng) for _ in range(3))
            addr = f"{rng.integers(1, 999)} {_word(rng)} Road {rng.integers(10000, 99999)}"
            by[1].append((f"S1-{n1}", name, addr))
            for _ in range(int(rng.choice(MATCH_N))):
                toks = name.split()
                if rng.random() < 0.4:
                    toks.pop(int(rng.integers(len(toks))))
                if rng.random() < 0.5:
                    n2 += 1
                    by[2].append((f"S2-{n2}", " ".join(toks), addr))
                else:
                    n3 += 1
                    by[3].append((f"S3-{n3}", " ".join(toks), addr))
            for _ in range(2):          # distractors sharing one name token
                tok = name.split()[int(rng.integers(3))]
                d = (f"{tok} {_word(rng)} {_word(rng)}",
                     f"{rng.integers(1, 999)} {_word(rng)} Street {rng.integers(10000, 99999)}")
                if rng.random() < 0.5:
                    n2 += 1
                    by[2].append((f"S2-{n2}", *d))
                else:
                    n3 += 1
                    by[3].append((f"S3-{n3}", *d))
        out[ctry] = by
    return out


class _Model:
    country = None

    def predict(self, X):
        raw = 0.5 * X[:, _NM] + 0.5 * X[:, _AD]
        return raw * DEFLATE.get(self.country, 1.0)


class _Iso:
    def predict(self, x):
        return np.clip(x ** 4, 1e-3, 0.99)


@pytest.fixture
def run(tmp_path, monkeypatch):
    """-> run(out_dir, *flags) -> (countries loaded, printed text)."""
    import blocking
    import predict
    root = tmp_path / "parquet"
    rows = _rows(0)
    for src in (1, 2, 3):
        flat = [(*r, c) for c, by in rows.items() for r in by[src]]
        t = pa.table({k: [r[i] for r in flat] for i, k in enumerate(
            ("entity_id", "business_name", "business_address", "country"))})
        pq.write_to_dataset(t, str(root / f"test_source{src}"), partition_cols=["country"])
    (tmp_path / "model.pkl").write_bytes(b"stub")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(predict, "ROOT", str(root))
    monkeypatch.setattr(predict, "BATCH", 50)
    model = _Model()
    monkeypatch.setattr(predict.pickle, "load", lambda f: {"model": model, "iso": _Iso()})
    real_load = blocking.load_shard

    def go(out, *flags, capsys=None):
        loaded = []

        def load(r, split, country):
            loaded.append(country)
            model.country = country
            return real_load(r, split, country)
        monkeypatch.setattr(predict, "OUT", out)
        monkeypatch.setattr(blocking, "load_shard", load)
        monkeypatch.setattr(sys, "argv", ["predict.py", *flags])
        predict.main()
        return loaded
    return go


def _partial_rows(out, country):
    tsv, side = resume.paths(out, country)
    import json
    n = json.loads(side.read_text())["rows"]
    return tsv.read_bytes().split(b"\n")[:n]


def _cand_rows(out, country):
    tsv, _ = resume.paths(out, country)
    n = len(_partial_rows(out, country))
    return tsv.read_bytes().split(b"\n")[n:2 * n]


def _hist(rows):
    return calibrate.k_hist([0 if r.endswith(b"\t") else r.count(b",") + 1
                             for r in rows])


def test_self_test():
    assert calibrate._self_test()


def test_calibrates_france_only(run, tmp_path, capsys):
    base, cal = tmp_path / "base", tmp_path / "cal"
    assert run(base) == list(COUNTRIES)
    capsys.readouterr()
    # references are decided first, the calibrated country last
    assert run(cal, "--calibrate", "France") == ["India", "US", "France"]
    text = capsys.readouterr().out
    print(text)
    for tag in ("reference", "before calibration", "after calibration",
                "P(n=0) scale s ="):
        assert tag in text

    for c in ("India", "US"):                     # never touched
        assert _partial_rows(cal, c) == _partial_rows(base, c)
        assert "calibrate" not in resume.json.loads(
            resume.paths(cal, c)[1].read_text())["config"]
    before = calibrate.summary(_hist(_partial_rows(base, "France")))
    after = calibrate.summary(_hist(_partial_rows(cal, "France")))
    ref = calibrate.summary(calibrate.pool(
        [_hist(_partial_rows(base, c)) for c in ("India", "US")]))
    # entities without candidates abstain whatever the scale: that is a floor
    floor = sum(r.endswith(b"\t") for r in _cand_rows(cal, "France")) / N_ENT
    assert before["singleton_rate"] > max(ref["singleton_rate"], floor) + 0.05
    assert abs(after["singleton_rate"] - max(ref["singleton_rate"], floor)) < 0.02
    if floor > ref["singleton_rate"]:
        assert "no candidates, above the" in text
    assert abs(after["mean_k_nonzero"] - ref["mean_k_nonzero"]) < 0.1

    # the final files keep sorted country order and one row per entity
    m = (cal / "matching_results.tsv").read_bytes().split(b"\n")
    assert len(m) == 1 + 3 * N_ENT + 1
    assert b"".join(m[1:1 + N_ENT]) == b"".join(_partial_rows(cal, "France"))


def test_resume_never_mixes_calibrated_and_plain(run, tmp_path):
    out = tmp_path / "out"
    assert run(out) == list(COUNTRIES)
    plain = _partial_rows(out, "France")
    # references reused, France recomputed under the flag
    assert run(out, "--calibrate", "France") == ["France"]
    assert _partial_rows(out, "France") != plain
    assert run(out, "--calibrate", "France") == []
    # a different reference set is a different config for France only
    assert run(out, "--calibrate", "France", "--calibrate-ref", "US") == ["France"]
    # flag off: France goes back to the plain rows, references still reused
    assert run(out) == ["France"]
    assert _partial_rows(out, "France") == plain


def test_composes_with_disjoint(run, tmp_path, capsys):
    out = tmp_path / "out"
    run(out, "--calibrate", "France", "--disjoint", "resolve")
    text = capsys.readouterr().out
    assert "after resolve" in text and "after calibration" in text
    rows = _partial_rows(out, "France") + _partial_rows(out, "India") + \
        _partial_rows(out, "US")
    ids = [x for r in rows for x in r.split(b"\t")[1].split(b",") if x]
    assert len(ids) == len(set(ids))              # still disjoint


@pytest.mark.parametrize("flags", [
    ("--calibrate", "Germany"),
    ("--calibrate", "France", "--calibrate-ref", "France"),
    ("--calibrate-ref", "US"),
    ("--calibrate", "France,India,US"),
    ("--calibrate",),
])
def test_bad_flags_exit(run, tmp_path, flags):
    with pytest.raises(SystemExit):
        run(tmp_path / "out", *flags)


def test_bar_scale_one_is_exact():
    rng = np.random.default_rng(1)
    for _ in range(200):
        p = np.sort(rng.random(rng.integers(1, 15)))[::-1]
        pz = float(rng.random())
        assert choose_k(p, pz, 0.9) == choose_k(p, pz, 0.9, bar_scale=1.0)
