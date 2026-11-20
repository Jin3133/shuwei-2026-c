#!/usr/bin/env python3
"""Create a copy of an EgoDemo delivery with paths relative to the common data root."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def rewrite_csv(path: Path, columns: list[str]):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f); rows = list(reader); fields = list(reader.fieldnames or [])
    for row in rows:
        for column in columns:
            value = row.get(column, "")
            if value and not value.startswith("Raw_Data/"):
                row[column] = "Raw_Data/" + value
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists(): shutil.rmtree(args.output)
    shutil.copytree(args.input, args.output)
    rewrite_csv(args.output / "records.csv", ["source_path"])
    rewrite_csv(args.output / "audit.csv", ["source_locator"])
    path = args.output / "frames.parquet"
    table = pq.read_table(path)
    idx = table.schema.get_field_index("source_locator")
    values = table.column(idx)
    prefixed = pc.binary_join_element_wise(pa.scalar("Raw_Data/"), values, "")
    table = table.set_column(idx, table.schema.field(idx), prefixed)
    pq.write_table(table, path)
    print(f"Harmonized delivery written to {args.output}")


if __name__ == "__main__":
    main()
