#!/usr/bin/env python3
"""Merge validated team deliveries without changing their semantics."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def write_csv(path, rows, columns):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--deliveries", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"inputs": [str(x.resolve()) for x in args.deliveries], "tables": {}}
    for table, key in (("records", "episode_uuid"), ("segments", "segment_id"), ("audit", "issue_id")):
        all_rows, columns = [], None
        for folder in args.deliveries:
            rows, cols = read_csv(folder / f"{table}.csv")
            if columns is None: columns = cols
            if set(cols) != set(columns): raise SystemExit(f"Column mismatch in {folder}/{table}.csv")
            all_rows.extend(rows)
        if table != "audit":
            duplicate = [k for k, n in Counter(r[key] for r in all_rows).items() if k and n > 1]
            if duplicate: raise SystemExit(f"Duplicate {key} in merged {table}: {duplicate[:5]}")
        write_csv(args.output / f"{table}.csv", all_rows, columns or [])
        manifest["tables"][table] = len(all_rows)

    frame_csv = all((d / "frames.csv").is_file() for d in args.deliveries)
    frame_parquet = all((d / "frames.parquet").is_file() for d in args.deliveries)
    if frame_csv:
        rows, columns = [], None
        for folder in args.deliveries:
            part, cols = read_csv(folder / "frames.csv")
            if columns is None: columns = cols
            if set(cols) != set(columns): raise SystemExit(f"Column mismatch in {folder}/frames.csv")
            rows.extend(part)
        duplicate = [k for k, n in Counter(r["frame_id"] for r in rows).items() if k and n > 1]
        if duplicate: raise SystemExit(f"Duplicate frame_id: {duplicate[:5]}")
        write_csv(args.output / "frames.csv", rows, columns or [])
        manifest["tables"]["frames"] = len(rows)
        manifest["frame_format"] = "csv-development"
    elif frame_parquet:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError: raise SystemExit("Install pyarrow before merging frames.parquet")
        parts = [pq.read_table(d / "frames.parquet") for d in args.deliveries]
        table = pa.concat_tables(parts, promote_options="default")
        ids = table.column("frame_id").to_pylist()
        if len(ids) != len(set(ids)): raise SystemExit("Duplicate frame_id in merged frames")
        pq.write_table(table, args.output / "frames.parquet")
        manifest["tables"]["frames"] = table.num_rows
        manifest["frame_format"] = "parquet"
    else:
        raise SystemExit("Deliveries must all contain the same frame format")
    (args.output / "merge_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    run_info = {
        "dataset": "EgoDemo+EgoPro+EgoStandard",
        "code_version": "team-merge-1.0",
        "source_deliveries": manifest["inputs"],
        "record_count_output": manifest["tables"].get("records", 0),
        "frame_count_output": manifest["tables"].get("frames", 0),
        "segment_count_output": manifest["tables"].get("segments", 0),
        "audit_issue_count": manifest["tables"].get("audit", 0),
        "frame_format": manifest["frame_format"],
        "notes": ["Inputs were individually validated before concatenation.", "EgoDemo paths were harmonized in a derived copy; original delivery was not modified."],
    }
    (args.output / "run_info.json").write_text(json.dumps(run_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
