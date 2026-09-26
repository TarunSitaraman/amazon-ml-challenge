"""Regression tests for silent correctness bugs: no crash, just wrong output.

The pipeline tests build a tiny synthetic parquet tree in a temp dir and run
predict.main / train_eval.prepare_country against it with a stub model, so
they need no data/ and no trained model.pkl.
"""
import pathlib
import pickle
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import features  # noqa: E402
import strfeatures  # noqa: E402
from textnorm import norm  # noqa: E402


# ---- textnorm ---------------------------------------------------------------

@pytest.mark.parametrize("a, b", [
    ("Sœur Marie", "Soeur Marie"),          # French ligature dropped by NFKD
    ("Cœur de Lion", "Coeur de Lion"),
    ("Joe's B & B", "Joes B&B"),            # possessive s fused with initials
    ("McDonald’s", "McDonalds"),
    ("M/s. A.B.C. Traders", "M/s ABC Traders"),
    ("S.A.R.L.", "Sarl"),                   # the original fix still holds
    ("A/B/C Ltd", "A.B.C. Ltd"),            # a slash run longer than a pair
])
def test_norm_equivalent_spellings(a, b):
    assert norm(a) == norm(b)


@pytest.mark.parametrize("text, token", [
    ("M/s. A.B.C. Traders", "abc"),         # was "msabc traders"
    ("C/O A.K. Sharma", "ak"),              # was "coak sharma"
])
def test_norm_prefix_does_not_swallow_acronym(text, token):
    assert token in norm(text).split()


# ---- features.py ------------------------------------------------------------

def test_rank_ties_share_rank_regardless_of_order():
    # Two C1/C5-style exact hits (score 1.0) and one weaker C2 hit. The tied
    # pair is equally good, so neither may be ranked below the other.
    chan = np.zeros((3, 5), np.float32)
    chan[:, 0] = [1.0, 0.4, 1.0]
    q = np.zeros(3, np.int64)
    col = features.NAMES.index("rank")
    X = features.build(q, chan, np.zeros(3, bool))
    assert X[:, col].tolist() == [0.0, 2.0, 0.0]
    perm = [2, 1, 0]
    Xp = features.build(q, chan[perm], np.zeros(3, bool))
    assert Xp[:, col].tolist() == X[perm, col].tolist()


# ---- strfeatures.py ---------------------------------------------------------

def test_identical_records_have_full_digit_overlap():
    n, a = norm("Studio 54"), norm("12 Main St")
    recs = strfeatures.precompute_records([n], [a])
    X = strfeatures.build(recs, [n], [a], np.array([0]), np.array([0]), None, None)[0]
    col = {k: i for i, k in enumerate(strfeatures.NAMES)}
    assert X[col["dg_jac"]] == 1.0            # was 0.5: "54" only on the query side
    assert X[col["dg_shared"]] == 2.0


# ---- pipeline fixture -------------------------------------------------------

CHAIN = "Sharma Medical Store"
S1 = [  # (id, name, address): four branches of one chain, one empty record
    ("S1-1", CHAIN, "12 MG Road 560001"),
    ("S1-2", CHAIN, "99 Park Street 560002"),
    ("S1-3", CHAIN, "7 Hill Road 560003"),
    ("S1-4", "", ""),
    ("S1-5", CHAIN, "3 Lake View 560004"),
]
S2 = [("S2-1", CHAIN, "12 MG Road 560001"), ("S2-2", CHAIN, "99 Park Street 560002"),
      ("S2-3", "Kumar Traders", "5 Fort Road 400001")]
S3 = [("S3-1", CHAIN, "12 MG Road 560001"), ("S3-2", "Gupta Stores", "7 Hill Road 560003"),
      ("S3-3", "Verma Textiles", "8 Mall Road 110001")]


def _write(root, split):
    for src, rows in ((1, S1), (2, S2), (3, S3)):
        t = pa.table({"entity_id": [r[0] for r in rows],
                      "business_name": [r[1] for r in rows],
                      "business_address": [r[2] for r in rows],
                      "country": ["India"] * len(rows)})
        pq.write_to_dataset(t, str(root / f"{split}_source{src}"),
                            partition_cols=["country"])


class _Model:
    """max_chan as the score; records every feature matrix it is given."""
    def __init__(self):
        self.X = []

    def predict(self, X):
        self.X.append(X)
        return X[:, features.NAMES.index("max_chan")]


class _Iso:
    def predict(self, x):
        return np.clip(x, 0.0, 1.0)


DUP = features.NAMES.index("name_dup")


def test_predict_submission_format_and_whole_country_name_dup(tmp_path, monkeypatch):
    import predict
    _write(tmp_path / "parquet", "test")
    (tmp_path / "model.pkl").write_bytes(b"")
    model = _Model()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(predict, "ROOT", str(tmp_path / "parquet"))
    monkeypatch.setattr(predict, "OUT", tmp_path / "output")
    monkeypatch.setattr(predict, "BATCH", 2)      # chain straddles batches
    monkeypatch.setattr(predict.pickle, "load",
                        lambda f: {"model": model, "iso": _Iso()})
    monkeypatch.setattr(sys, "argv", ["predict.py"])
    predict.main()

    def read(name):
        raw = (tmp_path / "output" / name).read_bytes()
        assert b"\r" not in raw
        lines = raw.decode("utf-8").split("\n")
        assert lines[-1] == ""
        return {eid: [x for x in ids.split(",") if x] or []
                for eid, ids in (ln.split("\t") for ln in lines[1:-1])}

    match, cand = read("matching_results.tsv"), read("candidate_pairs.tsv")
    ids1 = [r[0] for r in S1]
    valid = {r[0] for r in S2 + S3}
    for rows in (match, cand):
        assert sorted(rows) == sorted(ids1)       # one row each, incl. S1-4
        for lst in rows.values():
            assert len(lst) == len(set(lst)) and set(lst) <= valid
    assert cand["S1-4"] == [] and match["S1-4"] == []
    for eid in ids1:
        assert set(match[eid]) <= set(cand[eid])

    # every branch of the chain has 3 others in the country; per batch of 2
    # the feature saw at most 1
    X = np.vstack(model.X)
    assert len(X) and set(X[:, DUP].tolist()) == {3.0}


def test_train_eval_name_dup_counts_whole_country(tmp_path, monkeypatch):
    import train_eval
    _write(tmp_path / "parquet", "train")
    monkeypatch.setattr(train_eval, "ROOT", str(tmp_path / "parquet"))
    tr, va = train_eval.prepare_country("India", 2, 2, {}, np.random.default_rng(0))
    X = np.vstack([tr[0], va[0]])
    assert len(X) and set(X[:, DUP].tolist()) == {3.0}


# ---- tune_decision.py -------------------------------------------------------

def test_tune_decision_scores_only_held_out_half(tmp_path, monkeypatch, capsys):
    import tune_decision
    # 4 entities, one candidate each. The first half (the isotonic calibration
    # half in train_eval.py) is all wrong, the held-out half all right.
    state = {"q": np.arange(4), "p": np.full(4, 0.99),
             "cand": np.array(["S2-9", "S2-9", "S2-1", "S2-2"], object),
             "truth": [{"S2-0"}, {"S2-0"}, {"S2-1"}, {"S2-2"}],
             "ids": ["a", "b", "c", "d"], "ctry": ["India"] * 4}
    with open(tmp_path / "valstate.pkl", "wb") as fh:
        pickle.dump(state, fh)
    monkeypatch.chdir(tmp_path)
    tune_decision.main()
    out = capsys.readouterr().out
    assert out.startswith("2 validation entities")
    assert "fixed top-2                        F0.5 = 1.0000" in out
