"""mine_corruption: synthetic self-test, alignment units, and the data path run
end to end on a tiny parquet tree (no data/ needed)."""
import json
import pathlib
import sys

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import mine_corruption as mc  # noqa: E402


def test_self_test_passes():
    assert mc.self_test(8000)


def test_align_null_option_and_kinds():
    pairs, drop, ins = mc.align(["sharma", "global", "corporation"],
                                ["corp", "global", "xyz"])
    kinds = {(i, j): k for i, j, _, k in pairs}
    assert kinds[(1, 1)] == "identity"
    assert kinds[(2, 0)] == "abbrev"
    assert drop == [0] and ins == [2]          # sharma/xyz too far apart to align


def test_acronym_multi_token():
    a, b = "12 mg road uttar pradesh".split(), "12 mg rd up".split()
    pairs, drop, ins = mc.align(a, b)
    acr = mc.find_acronyms(a, b, drop, ins)
    assert [(x[0], x[1]) for x in acr] == [("uttar pradesh", "up")]


def test_perm_classes():
    assert mc.perm_class([0, 1, 2]) == "none"
    assert mc.perm_class([2, 0, 1]) == "last_to_front"
    assert mc.perm_class([0, 2, 1, 3]) == "adjacent_swap"
    assert mc.perm_class([1, 3, 0, 2]) == "other"
    assert abs(sum(mc.random_baseline(4).values()) - 1) < 1e-9


def _write(root):
    S1 = [("S1-1", "Sharma Global Corporation", "12 MG Road Haryana 122001", "India"),
          ("S1-2", "Acme Incorporated", "5 Main Street Texas 75001", "USA")]
    S2 = [("S2-1", "<< Sharma Global Corp", "12 MG Rd HR 122001", "India"),
          ("S2-9", "Unrelated Distractor", "1 Nowhere", "India")]
    S3 = [("S3-1", "Corporation Sharma Global", "12 MG Road Haryana 122001", "India"),
          ("S3-2", "Acme Inc", "5 Main St TX 75001", "USA")]
    for src, rows in ((1, S1), (2, S2), (3, S3)):
        t = pa.table({k: [r[i] for r in rows] for i, k in enumerate(
            ("entity_id", "business_name", "business_address", "country"))})
        pq.write_to_dataset(t, str(root / f"train_source{src}"), partition_cols=["country"])
    pq.write_table(pa.table({"source1_entity_id": ["S1-1", "S1-2"],
                             "matched_entity_ids": ["S2-1,S3-1", "S3-2"]}),
                   root / "train_ground_truth.parquet")


def test_end_to_end_on_parquet_tree(tmp_path):
    root = tmp_path / "parquet"
    _write(root)
    out = tmp_path / "grammar.json"
    mc.main(["--root", str(root), "--out", str(out), "--min-support", "1",
             "--min-trials", "1"])
    t = json.loads(out.read_text())
    assert t["provenance"]["aligned_pairs_used"] == 3
    assert t["provenance"]["external_data"].startswith("none")
    ab = {(e["s1"], e["s2"]): e for e in t["abbreviations"]["name"]}
    assert ab[("corporation", "corp")]["by_source"] == {"S2": 1, "S3": 0}
    assert ab[("incorporated", "inc")]["by_country"] == {"USA": 1}
    addr = {(e["s1"], e["s2"]) for e in t["abbreviations"]["address"]}
    assert {("road", "rd"), ("haryana", "hr"), ("texas", "tx")} <= addr
    assert any(e["affix"] == "<<" and e["position"] == "prefix"
               for e in t["junk_affixes"])
    k3 = next(r for r in t["permutations"]["name"]["by_length"] if r["n_tokens"] == 3)
    assert k3["histogram"].get("last_to_front") == 1


def test_sample_flag(tmp_path):
    root = tmp_path / "parquet"
    _write(root)
    out = tmp_path / "g.json"
    mc.main(["--root", str(root), "--out", str(out), "--sample", "1"])
    assert json.loads(out.read_text())["provenance"]["aligned_pairs_used"] == 1
