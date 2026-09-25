"""predict.py resume: crash after one country, resume, and get the same bytes.

Builds a two-country synthetic parquet tree in a temp dir and runs
predict.main with a stub model, so it needs no data/ and no model.pkl.
"""
import pathlib
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import features  # noqa: E402
import resume  # noqa: E402

CHAIN = "Sharma Medical Store"
ROWS = {  # country -> source -> (id, name, address)
    "India": {
        1: [("S1-1", CHAIN, "12 MG Road 560001"), ("S1-2", CHAIN, "99 Park Street 560002"),
            ("S1-3", CHAIN, "7 Hill Road 560003"), ("S1-4", "", ""),
            ("S1-5", CHAIN, "3 Lake View 560004")],
        2: [("S2-1", CHAIN, "12 MG Road 560001"), ("S2-2", CHAIN, "99 Park Street 560002"),
            ("S2-3", "Kumar Traders", "5 Fort Road 400001")],
        3: [("S3-1", CHAIN, "12 MG Road 560001"), ("S3-2", "Gupta Stores", "7 Hill Road 560003")],
    },
    "Mexico": {
        1: [("S1-6", "Tacos El Gordo", "Av Reforma 10"), ("S1-7", "Farmacia Guadalajara", "Calle 5 22"),
            ("S1-8", "Tacos El Gordo", "Insurgentes 400")],
        2: [("S2-4", "Tacos El Gordo", "Av Reforma 10"), ("S2-5", "Farmacia Guadalajara", "Calle 5 22")],
        3: [("S3-3", "Tacos El Gordo", "Insurgentes 400"), ("S3-4", "Panaderia Rosa", "Juarez 3")],
    },
}


class _Model:
    def predict(self, X):
        return X[:, features.NAMES.index("max_chan")]


class _Iso:
    def predict(self, x):
        return np.clip(x, 0.0, 1.0)


@pytest.fixture
def run(tmp_path, monkeypatch):
    """-> run(out_dir, *flags, fail=None). fail names a country whose shard load
    raises, standing in for a crash mid-run. Returns the countries loaded."""
    import blocking
    import predict
    root = tmp_path / "parquet"
    for src in (1, 2, 3):
        rows = [(*r, c) for c, by in ROWS.items() for r in by[src]]
        t = pa.table({k: [r[i] for r in rows] for i, k in enumerate(
            ("entity_id", "business_name", "business_address", "country"))})
        pq.write_to_dataset(t, str(root / f"test_source{src}"), partition_cols=["country"])
    (tmp_path / "model.pkl").write_bytes(b"stub")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(predict, "ROOT", str(root))
    monkeypatch.setattr(predict, "BATCH", 2)
    monkeypatch.setattr(predict.pickle, "load", lambda f: {"model": _Model(), "iso": _Iso()})
    real_load = blocking.load_shard

    def go(out, *flags, fail=None):
        loaded = []

        def load(r, split, country):
            if country == fail:
                raise MemoryError("simulated crash")
            loaded.append(country)
            return real_load(r, split, country)
        monkeypatch.setattr(predict, "OUT", out)
        monkeypatch.setattr(blocking, "load_shard", load)
        monkeypatch.setattr(sys, "argv", ["predict.py", *flags])
        predict.main()
        return loaded
    return go


def _read(out):
    return [(out / n).read_bytes() for n in ("matching_results.tsv", "candidate_pairs.tsv")]


def test_crash_then_resume_is_byte_identical(run, tmp_path):
    single = tmp_path / "single"
    assert run(single) == ["India", "Mexico"]
    want = _read(single)
    assert all(b"\r" not in w for w in want)
    assert want[0].count(b"\n") == want[1].count(b"\n") == 1 + 8
    assert b"S1-6\tS2-4" in want[0]            # the second country has matches

    out = tmp_path / "resumed"
    with pytest.raises(MemoryError):
        run(out, fail="Mexico")
    assert resume.check(out, "India", 5, _config(out, "India")) is None
    assert not (out / "partial_Mexico.json").exists()
    assert not (out / "matching_results.tsv").exists()

    assert run(out) == ["Mexico"]              # India is not recomputed
    assert _read(out) == want

    assert run(out) == []                      # everything done: concat only
    assert _read(out) == want


def _config(out, country):
    import json
    return json.loads((out / f"partial_{country}.json").read_text())["config"]


def test_bad_or_stale_partials_are_recomputed(run, tmp_path):
    out = tmp_path / "out"
    run(out)
    want = _read(out)

    tsv = out / "partial_India.tsv"
    tsv.write_bytes(tsv.read_bytes()[:-10])    # truncated
    assert run(out) == ["India"]
    assert _read(out) == want

    (out / "partial_Mexico.json").unlink()     # TSV written, crash before sidecar
    assert run(out) == ["Mexico"]
    assert _read(out) == want

    assert run(out, "--recall", "0.9") == ["India", "Mexico"]   # config changed
    assert run(out, "--recall", "0.9") == []


def test_fresh_recomputes_everything(run, tmp_path):
    out = tmp_path / "out"
    run(out)
    want = _read(out)
    assert run(out, "--fresh") == ["India", "Mexico"]
    assert _read(out) == want
    assert run(out, "--resume") == []


def test_limit_changes_expected_rows(run, tmp_path):
    out = tmp_path / "out"
    run(out, "--limit", "2")
    assert _read(out)[0].count(b"\n") == 1 + 2 + 2
    assert run(out) == ["India", "Mexico"]     # no limit: more rows expected


def test_write_and_check_roundtrip(tmp_path):
    cfg = {"a": 1}
    resume.write(tmp_path, "U.S.", ["S1-1\tS2-1", "S1-2\t"], ["S1-1\tS2-1,S3-1", "S1-2\t"], cfg)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["partial_U.S..json", "partial_U.S..tsv"]
    assert resume.check(tmp_path, "U.S.", 2, cfg) is None
    assert resume.check(tmp_path, "U.S.", 3, cfg) == "sidecar says 2 rows, country has 3"
    assert resume.check(tmp_path, "U.S.", 2, {"a": 2}) == "run config changed since it was written"
    m, c = tmp_path / "m.tsv", tmp_path / "c.tsv"
    assert resume.concat(tmp_path, ["U.S."], m, c, "h1", "h2") == (2, 1)
    assert m.read_bytes() == b"h1\nS1-1\tS2-1\nS1-2\t\n"
    assert c.read_bytes() == b"h2\nS1-1\tS2-1,S3-1\nS1-2\t\n"
    resume.clear(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["c.tsv", "m.tsv"]
    with pytest.raises(ValueError):
        resume.write(tmp_path, "X", ["a"], [], cfg)
