"""diag_matcher.py on a synthetic valstate and parquet (no data/ needed):
every category is hit by construction, and the Shapley shares sum to the gap."""
import itertools
import pathlib
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import diag_matcher as dm  # noqa: E402
from metric import f05  # noqa: E402

S1 = [("E0", "Acme Traders", "12 MG Road Pune"),
      ("E1", "ACME traders.", "7 Station Road"),        # same normalised name as E0
      ("E2", "Bolt Motors", "3 Ring Road"),
      ("E3", "Crown Hotel", "Lake View"),
      ("E4", "Delta Foods", "Market Yard"),             # no matches at all
      ("E5", "Delta Food Co", "Old Town")]
S2 = [("S2-R1", "Acme Trdrs", "45 Nehru Nagar"),
      ("S2-R2", "Acme", "12 Park Street"),
      ("S2-R3", "Bolt Motors", "3 Ring Road"),
      ("S2-R4", "Bolt Motor Works", "")]
S3 = [("S3-R5", "Hotel Crown", "Lake View"),
      ("S3-R6", "Crown Bakery", "Hill Road"),
      ("S3-R8", "Crown Hotel", "Lake View Road"),
      ("S3-R9", "Bolt", "9 Ring Road"),
      ("S3-R10", "Delta Food Co", "Old Town")]
GOLD = {"E0": "S2-R1", "E1": "S2-R2", "E2": "S2-R3,S2-R4", "E3": "S3-R5,S3-R8",
        "E4": None, "E5": "S3-R10"}
# (entity index, record, calibrated p)
PAIRS = [(0, "S2-R2", 0.95), (0, "S2-R1", 0.05),
         (1, "S2-R2", 0.90),
         (2, "S2-R3", 0.90), (2, "S2-R4", 0.20), (2, "S3-R9", 0.10),
         (3, "S3-R8", 0.90), (3, "S3-R6", 0.30), (3, "S3-R5", 0.25),
         (4, "S3-R10", 0.95),
         (5, "S3-R10", 0.90)]


def _fixture(root):
    for src, rows in ((1, S1), (2, S2), (3, S3)):
        t = pa.table({"entity_id": [r[0] for r in rows],
                      "business_name": [r[1] for r in rows],
                      "business_address": [r[2] for r in rows],
                      "country": ["India"] * len(rows)})
        pq.write_to_dataset(t, str(root / f"train_source{src}"),
                            partition_cols=["country"])
    gt = pa.table({"source1_entity_id": list(GOLD),
                   "matched_entity_ids": list(GOLD.values())})
    pq.write_table(gt, root / "train_ground_truth.parquet")
    ids = [r[0] for r in S1]
    d = {"q": np.array([x[0] for x in PAIRS]),
         "cand": np.array([x[1] for x in PAIRS], object),
         "p": np.array([x[2] for x in PAIRS]),
         "truth": [set((GOLD[i] or "").split(",")) - {""} for i in ids],
         "ids": ids, "ctry": ["India"] * len(ids),
         "p_zero_head": np.full(len(ids), 0.05), "n_cal": 0}
    return d, gt


def test_shapley_split_sums_to_the_entity_gap():
    for c, n, k, f in itertools.product(range(4), range(7), range(8), range(4)):
        m = k - c                                  # every non-correct accept is an FP
        if m < 0 or c + f > n:
            continue
        fp, fn = dm.shapley_split(c, n, k, m, f)
        gap = float(f05(c + f, n, c + f) - f05(c, n, k))
        assert m * fp + f * fn == pytest.approx(gap, abs=1e-12)
        assert fp >= -1e-12 and fn >= -1e-12


def test_shapley_splits_a_zero_marginal_fairly():
    # c=0, n=1, one FP and one FN: fixing the FP alone gains nothing, but the
    # FP still costs half the fix (1.25/1.25 - 1.25/2.25 when both are fixed)
    fp, fn = dm.shapley_split(0, 1, 1, 1, 1)
    assert fp > 0 and fn > fp
    assert fp + fn == pytest.approx(1.0)


def test_decision_category():
    t = np.array([True, False, True, False, True])
    assert dm.decision_category(2, 2, t) == "ranked out"
    t = np.array([True, True, True, False, True])
    assert dm.decision_category(1, 1, t) == "stopped early"
    assert dm.decision_category(2, 1, t) == "stopped early"
    assert dm.decision_category(4, 1, t) == "buried"


def test_signal_category():
    assert dm.signal_category("क्राउन होटल", "Lake", "Crown Hotel", "Lake") == \
        "script mismatch"
    assert dm.signal_category("Crown", "", "Crown", "x") == "empty address"
    assert dm.signal_category("Crown Hotel", "Lake View", "Crown", "Lake") == \
        "name + address overlap"
    assert dm.signal_category("Crown", "Hill Road", "Crown", "Lake View") == \
        "name-only overlap"
    assert dm.signal_category("Crown", "Hill Road", "Hotel", "Lake Road") == \
        "address-only overlap"
    assert dm.signal_category("Crown", "Hill", "Hotel", "Lake") == "no token overlap"


def test_end_to_end_categories_and_gap(tmp_path, capsys):
    d, gt = _fixture(tmp_path)
    errors, F, C = dm.run(d, gt, str(tmp_path))
    got = {(e["kind"], e["id"], e["rec"]): e for e in errors}
    assert set(got) == {("FP", "E0", "S2-R2"), ("FN", "E0", "S2-R1"),
                        ("FN", "E2", "S2-R4"), ("FN", "E3", "S3-R5"),
                        ("FP", "E4", "S3-R10")}

    fp = got["FP", "E0", "S2-R2"]
    assert fp["cat"] == "chain, digit shared" and fp["owner"] == "E1"
    fp = got["FP", "E4", "S3-R10"]
    assert fp["cat"] == "sibling, no digit shared" and fp["n"] == 0

    fn = got["FN", "E0", "S2-R1"]
    assert fn["decision"] == "ranked out" and fn["p"] < dm.LOW_P
    assert fn["signal"] == "name-only overlap"
    fn = got["FN", "E2", "S2-R4"]
    assert fn["decision"] == "stopped early" and fn["signal"] == "empty address"
    fn = got["FN", "E3", "S3-R5"]
    assert fn["decision"] == "buried" and fn["signal"] == "name + address overlap"

    # the shares account for exactly the measured matcher gap
    assert sum(e["w"] for e in errors) == pytest.approx((C - F).sum())
    # a false accept on a no-match entity costs the whole entity
    assert got["FP", "E4", "S3-R10"]["w"] == pytest.approx(1.0)

    out = capsys.readouterr().out
    for cat in ("chain, digit shared", "sibling, no digit shared", "ranked out",
                "stopped early", "buried", "low p", "same-normalised-name groups: 33.33%"):
        assert cat in out
    assert "gold of 'ACME traders.'" in out


def test_head_p_zero_and_distractor(tmp_path):
    d, gt = _fixture(tmp_path)
    # a record nobody owns, accepted with certainty on E5
    d["q"] = np.r_[d["q"], 5]
    d["cand"] = np.r_[d["cand"], np.array(["S3-R6"], object)]
    d["p"] = np.r_[d["p"], 0.99]
    order = np.argsort(d["q"], kind="stable")
    d = dict(d, q=d["q"][order], cand=d["cand"][order], p=d["p"][order])
    errors, _, _ = dm.run(d, gt, str(tmp_path), pz_kind="head")
    fp = [e for e in errors if e["kind"] == "FP" and e["id"] == "E5"]
    assert [e["cat"] for e in fp] == ["distractor, no digit shared"]
    assert fp[0]["owner"] is None
