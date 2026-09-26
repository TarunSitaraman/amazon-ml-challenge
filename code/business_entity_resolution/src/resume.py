"""Per-country partial outputs, so a crashed predict.py run can resume.

A full run takes hours, and holding every row until the end means a crash at
the 90-minute mark loses everything. predict.py instead writes each country's
rows to output/partial_<country>.tsv as soon as the country finishes, then
frees them. The partial holds the country's n matching rows followed by its n
candidate rows, one per S1 entity each, without headers.

A partial counts as complete only if its sidecar partial_<country>.json exists
and agrees with it: the sidecar is written after the TSV, and both are written
to a temp file and renamed, so a crash mid-write never leaves a sidecar next
to a truncated TSV. The sidecar also records the row count, byte size and
sha256 of the TSV, and the run config (flags, blocking knobs, the model.pkl and
pipeline source hashes, the country's parquet files). A partial is reused only
when the row count matches the country's S1 count and the config matches this
run, so changing a flag, the code, the data or the model recomputes rather than
mixing two runs into one submission.

concat() streams the partials into the two final files in bytes, so the output
is byte-identical to a run that never stopped.
"""
import hashlib
import json
import os
import pathlib
import re

# Bumped when the partial layout changes, so old partials are never reused.
FORMAT = 1


def _safe(country):
    """Filename stem for a country. Names that need escaping get a hash suffix,
    so two countries never share a partial ("A B" vs "A_B")."""
    safe = re.sub(r"[^\w.-]", "_", country)
    if safe != country:
        safe += "_" + hashlib.sha256(country.encode("utf-8")).hexdigest()[:8]
    return safe


def paths(out, country):
    stem = f"partial_{_safe(country)}"
    return pathlib.Path(out) / f"{stem}.tsv", pathlib.Path(out) / f"{stem}.json"


def file_sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _replace(tmp, path):
    with open(tmp, "rb+") as fh:
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def write(out, country, rows_match, rows_cand, config, cardinality=None):
    """Write one country's rows, then its sidecar. Rows carry no newline.

    cardinality: optional histogram of accepted pairs per entity before
    resolve_conflicts, kept in the sidecar (outside config) so --calibrate can
    use a finished country as a reference without recomputing it."""
    if len(rows_match) != len(rows_cand):
        raise ValueError(f"{country}: {len(rows_match)} matching rows but "
                         f"{len(rows_cand)} candidate rows")
    tsv, side = paths(out, country)
    tmp = pathlib.Path(f"{tsv}.tmp")
    # newline="\n" for the reason given in predict.py: Windows text mode would
    # write "\r\n" and every trailing ID would read as "S3-123\r".
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        for r in rows_match:
            fh.write(r + "\n")
        for r in rows_cand:
            fh.write(r + "\n")
    _replace(tmp, tsv)
    meta = {"format": FORMAT, "country": country, "rows": len(rows_match),
            "bytes": tsv.stat().st_size, "sha256": file_sha256(tsv),
            "config": config}
    if cardinality is not None:
        meta["cardinality"] = [int(x) for x in cardinality]
    tmp = pathlib.Path(f"{side}.tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(meta, fh, indent=1, sort_keys=True)
    _replace(tmp, side)


def check(out, country, n_rows, config):
    """None if the country's partial is complete and reusable, else the reason."""
    tsv, side = paths(out, country)
    if not side.exists():
        return "no partial"
    try:
        meta = json.loads(side.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return f"unreadable sidecar ({e})"
    if meta.get("format") != FORMAT or meta.get("country") != country:
        return "sidecar is for another format or country"
    if meta.get("rows") != n_rows:
        return f"sidecar says {meta.get('rows')} rows, country has {n_rows}"
    if meta.get("config") != config:
        return "run config changed since it was written"
    if not tsv.exists() or tsv.stat().st_size != meta.get("bytes"):
        return "TSV missing or wrong size"
    h, lines = hashlib.sha256(), 0
    with open(tsv, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
            lines += block.count(b"\n")
    if h.hexdigest() != meta.get("sha256"):
        return "TSV checksum mismatch"
    if lines != 2 * n_rows:
        return f"TSV has {lines} lines, expected {2 * n_rows}"
    return None


def cardinality(out, country):
    """The pre-resolve cardinality histogram of a complete partial. Sidecars
    written before it was recorded fall back to counting the matching rows,
    which are after resolve_conflicts (the same thing with --disjoint off)."""
    tsv, side = paths(out, country)
    meta = json.loads(side.read_text(encoding="utf-8"))
    if "cardinality" in meta:
        return meta["cardinality"]
    counts = []
    with open(tsv, "rb") as fh:
        for _, line in zip(range(meta["rows"]), fh):
            ids = line.rstrip(b"\n").split(b"\t", 1)[1]
            counts.append(ids.count(b",") + 1 if ids else 0)
    hist = [0] * (max(counts, default=0) + 1)
    for k in counts:
        hist[k] += 1
    return hist


def concat(out, expected, match_path, cand_path, hdr_m, hdr_c):
    """Stream the partials into the two final files, in the order of expected,
    a dict country -> (rows, config). Each sidecar must still match it.

    Returns (rows, rows with at least one match)."""
    tmp_m, tmp_c = pathlib.Path(f"{match_path}.tmp"), pathlib.Path(f"{cand_path}.tmp")
    n_rows = n_pred = 0
    try:
        with open(tmp_m, "wb") as fm, open(tmp_c, "wb") as fc:
            fm.write(hdr_m.encode("utf-8") + b"\n")
            fc.write(hdr_c.encode("utf-8") + b"\n")
            for country, (n, config) in expected.items():
                tsv, side = paths(out, country)
                meta = json.loads(side.read_text(encoding="utf-8"))
                if meta.get("rows") != n or meta.get("config") != config:
                    raise RuntimeError(f"{side} changed since it was checked")
                seen = 0
                with open(tsv, "rb") as fh:
                    for line in fh:
                        if seen < n:
                            fm.write(line)
                            if not line.endswith(b"\t\n"):
                                n_pred += 1
                        else:
                            fc.write(line)
                        seen += 1
                if seen != 2 * n:
                    raise RuntimeError(f"{tsv}: {seen} lines, expected {2 * n}")
                n_rows += n
    except BaseException:
        # no half-written final files left behind
        tmp_m.unlink(missing_ok=True)
        tmp_c.unlink(missing_ok=True)
        raise
    _replace(tmp_m, match_path)
    _replace(tmp_c, cand_path)
    return n_rows, n_pred


def clear(out):
    """Delete every partial and sidecar, for --fresh."""
    for pattern in ("partial_*.tsv", "partial_*.json", "partial_*.tmp"):
        for p in pathlib.Path(out).glob(pattern):
            p.unlink()
