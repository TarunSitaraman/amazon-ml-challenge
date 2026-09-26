"""Local check of the documented submission rules.

This is NOT a replacement for the organisers' utils/validate_submission.py --
that one is authoritative and should be run as well. This exists because the
copy shipped in our data drop is 0 bytes, and a format rejection costs a
submission from a budget of five per day.

Rules checked (problem statement, "Output Format" and "Constraints"):
  1. exactly one row per test S1 entity; none missing, none extra, no duplicates
  2. matched_entity_ids may be empty (singleton)
  3. no duplicate IDs within a single list
  4. only S2-/S3- IDs, and only ones that exist in the test set
  5. matching_results must be a subset of candidate_pairs

Usage: python validate.py [--matching output/matching_results.tsv]
                          [--candidate output/candidate_pairs.tsv]
"""
import argparse
import pathlib
import sys

import pyarrow.dataset as ds

ROOT = "data/parquet"


def check_line_endings(path):
    """Text-mode reads strip \\r silently, so this must look at raw bytes.

    A CRLF file makes every trailing ID on a line "S3-123\\r", which matches
    nothing in the test set if the scorer splits on "\\n". It is invisible to
    every other check here.
    """
    with open(path, "rb") as fh:
        head = fh.read(1 << 20)
    if b"\r\n" in head:
        return [f"{path.name}: CRLF line endings -- trailing IDs will carry a "
                f"stray \\r and may fail to match. Write with newline='\\n'."]
    return []


def load_tsv(path, expect_header):
    rows = {}
    issues = check_line_endings(path)
    with open(path, encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n")
        if header != expect_header:
            issues.append(f"{path.name}: header is {header!r}, expected {expect_header!r}")
        for ln, line in enumerate(fh, start=2):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                issues.append(f"{path.name}:{ln}: expected 2 tab-separated fields, got {len(parts)}")
                continue
            eid, ids = parts
            lst = [x for x in ids.split(",") if x] if ids else []
            if eid in rows:
                issues.append(f"{path.name}:{ln}: duplicate source1_entity_id {eid}")
            if len(set(lst)) != len(lst):
                issues.append(f"{path.name}:{ln}: duplicate IDs within the list for {eid}")
            rows[eid] = lst
    return rows, issues


def test_ids():
    out = {}
    for src in (1, 2, 3):
        d = ds.dataset(f"{ROOT}/test_source{src}", format="parquet", partitioning="hive")
        out[src] = set(d.to_table(columns=["entity_id"]).column("entity_id").to_pylist())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matching", default="output/matching_results.tsv", type=pathlib.Path)
    ap.add_argument("--candidate", default="output/candidate_pairs.tsv", type=pathlib.Path)
    a = ap.parse_args()

    issues = []
    match, e = load_tsv(a.matching, "source1_entity_id\tmatched_entity_ids")
    issues += e

    # candidate_pairs is streamed, never held: at CAND_CAP 150 it is ~2.3GB and
    # ~150M ids, and holding it as dict-of-lists ran this machine out of memory.
    # Only the per-entity subset check needs it, and that works one row at a time.
    n_cand_rows = n_cand_ids = not_sub = 0
    have_cand = a.candidate.exists()
    if have_cand:
        issues += check_line_endings(a.candidate)
        seen = set()
        with open(a.candidate, encoding="utf-8") as fh:
            hdr = fh.readline().rstrip("\n")
            if hdr != "source1_entity_id\tcandidate_entity_ids":
                issues.append(f"{a.candidate.name}: header is {hdr!r}")
            for ln, line in enumerate(fh, start=2):
                eid, _, ids_s = line.rstrip("\n").partition("\t")
                lst = ids_s.split(",") if ids_s else []
                n_cand_rows += 1
                n_cand_ids += len(lst)
                s = set(lst)
                if len(s) != len(lst):
                    issues.append(f"{a.candidate.name}:{ln}: duplicate IDs for {eid}")
                if eid in seen:
                    issues.append(f"{a.candidate.name}:{ln}: duplicate row {eid}")
                seen.add(eid)
                if not set(match.get(eid, ())) <= s:
                    not_sub += 1
        del seen

    ids = test_ids()
    # membership against both sets; a union would build a third ~10M-id copy
    class _Valid:
        def __contains__(self, x):
            return x in ids[2] or x in ids[3]
    valid = _Valid()

    missing = ids[1] - match.keys()
    extra = match.keys() - ids[1]
    if missing:
        issues.append(f"matching_results: {len(missing):,} test S1 entities missing "
                      f"(e.g. {sorted(missing)[:3]})")
    if extra:
        issues.append(f"matching_results: {len(extra):,} rows are not test S1 entities "
                      f"(e.g. {sorted(extra)[:3]})")

    bad_src = bad_exist = 0
    for eid, lst in match.items():
        for x in lst:
            if not (x.startswith("S2-") or x.startswith("S3-")):
                bad_src += 1
            elif x not in valid:
                bad_exist += 1
    if bad_src:
        issues.append(f"matching_results: {bad_src:,} IDs are not S2-/S3- prefixed")
    if bad_exist:
        issues.append(f"matching_results: {bad_exist:,} IDs do not exist in the test set")

    if have_cand:
        if n_cand_rows != len(match):
            issues.append(f"candidate_pairs has {n_cand_rows:,} rows, "
                          f"matching_results {len(match):,}")
        if not_sub:
            issues.append(f"{not_sub:,} entities predict a match that is not in "
                          f"candidate_pairs (pipeline bug -- the validator warns on this)")

    n = len(match)
    withm = sum(1 for v in match.values() if v)
    tot = sum(len(v) for v in match.values())
    print(f"rows                 {n:,}")
    print(f"with >=1 match       {withm:,} ({withm/max(n,1):.2%})")
    print(f"predicted singletons {n-withm:,} ({(n-withm)/max(n,1):.2%})")
    print(f"mean matches/entity  {tot/max(n,1):.3f}")
    if have_cand:
        print(f"mean candidates      {n_cand_ids/max(n_cand_rows,1):.1f}")

    if issues:
        print(f"\nFAIL -- {len(issues)} issue(s):")
        for i, m in enumerate(issues, 1):
            print(f"  {i}. {m}")
        return 1
    print("\nPASS -- safe to submit (still run the organisers' validator too)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
