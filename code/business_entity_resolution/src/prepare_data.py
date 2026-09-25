"""Convert the raw challenge TSVs to country-partitioned Parquet.

Parquet keeps the strings dictionary-encoded and lets us load a single country
shard at a time, which is what makes 24M records fit in 16GB of RAM.

Usage:
    python prepare_data.py --raw-dir <dir with the .tsv files> --out-dir data/parquet
"""
import argparse
import pathlib

import pyarrow as pa
import pyarrow.csv as pv
import pyarrow.parquet as pq

# quote_char=False: the files are pure tab-delimited, and business names contain
# bare double quotes that would otherwise swallow the rest of a line.
PARSE = pv.ParseOptions(delimiter="\t", quote_char=False)
READ = pv.ReadOptions(block_size=1 << 26)
# Everything stays a string; normalisation belongs in the feature stage, not here.
SOURCE_COLS = {"entity_id": pa.string(), "business_name": pa.string(),
               "business_address": pa.string(), "country": pa.string()}


def convert_sources(raw_dir: pathlib.Path, out_dir: pathlib.Path) -> None:
    for split in ("train", "test"):
        for s in (1, 2, 3):
            src = raw_dir / f"{split}_source{s}.tsv"
            table = pv.read_csv(src, parse_options=PARSE, read_options=READ,
                                convert_options=pv.ConvertOptions(column_types=SOURCE_COLS))
            dest = out_dir / f"{split}_source{s}"
            pq.write_to_dataset(table, dest, partition_cols=["country"],
                                compression="zstd", existing_data_behavior="delete_matching")
            print(f"{src.name:26s} -> {dest.name:18s} {table.num_rows:>9,} rows")


def convert_ground_truth(raw_dir: pathlib.Path, out_dir: pathlib.Path) -> None:
    src = raw_dir / "train_ground_truth.tsv"
    # matched_entity_ids is empty for singletons; keep it as "" rather than null
    # so downstream splits never have to special-case None.
    table = pv.read_csv(src, parse_options=PARSE, read_options=READ,
                        convert_options=pv.ConvertOptions(
                            column_types={"source1_entity_id": pa.string(),
                                          "matched_entity_ids": pa.string()},
                            null_values=[], strings_can_be_null=False))
    dest = out_dir / "train_ground_truth.parquet"
    pq.write_table(table, dest, compression="zstd")
    print(f"{src.name:26s} -> {dest.name:18s} {table.num_rows:>9,} rows")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True, type=pathlib.Path)
    ap.add_argument("--out-dir", default=pathlib.Path("data/parquet"), type=pathlib.Path)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    convert_sources(args.raw_dir, args.out_dir)
    convert_ground_truth(args.raw_dir, args.out_dir)


if __name__ == "__main__":
    main()
