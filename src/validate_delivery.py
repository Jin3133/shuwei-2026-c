#!/usr/bin/env python3
"""Validate one preprocessing delivery against the frozen interface and official index."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def add(issues, severity, check, table, key, message):
    issues.append({"severity": severity, "check": check, "table": table, "key": key, "message": message})


def number(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def boolean(value):
    return str(value).strip().lower() in {"true", "false"}


def load_frames(delivery: Path, issues):
    csv_path = delivery / "frames.csv"
    parquet_path = delivery / "frames.parquet"
    if csv_path.is_file():
        rows, columns = read_csv(csv_path)
        add(issues, "warning", "development_format", "frames", "", "frames.csv accepted for development; final delivery must use frames.parquet")
        return rows, columns, "csv"
    if parquet_path.is_file():
        try:
            import pyarrow.parquet as pq
        except ImportError:
            add(issues, "error", "missing_reader", "frames", "", "frames.parquet exists but pyarrow is not installed")
            return [], [], "parquet"
        table = pq.read_table(parquet_path)
        return table.to_pylist(), table.column_names, "parquet"
    add(issues, "error", "missing_file", "frames", "", "Missing frames.parquet (or frames.csv during development)")
    return [], [], "missing"


def validate(args):
    delivery = args.delivery.resolve()
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    official_rows, _ = read_csv(args.official_records)
    official = {r["episode_uuid"]: r for r in official_rows}
    issues = []
    tables = {}

    for name in ("records", "segments", "audit"):
        path = delivery / f"{name}.csv"
        if not path.is_file():
            add(issues, "error", "missing_file", name, "", f"Missing {path.name}")
            tables[name] = ([], [])
        else:
            tables[name] = read_csv(path)
    frames, frame_columns, frame_format = load_frames(delivery, issues)
    tables["frames"] = (frames, frame_columns)

    run_info = delivery / "run_info.json"
    if not run_info.is_file():
        add(issues, "error", "missing_file", "run_info", "", "Missing run_info.json")
    else:
        try:
            json.loads(run_info.read_text(encoding="utf-8"))
        except Exception as exc:
            add(issues, "error", "invalid_json", "run_info", "", str(exc))

    for name, (rows, columns) in tables.items():
        required = schema["required_columns"][name]
        for col in sorted(set(required) - set(columns)):
            add(issues, "error", "missing_column", name, "", col)
        key_cols = schema.get("keys", {}).get(name, [])
        if key_cols and all(k in columns for k in key_cols):
            keys = [tuple(row.get(k, "") for k in key_cols) for row in rows]
            for key, count in Counter(keys).items():
                if not all(key):
                    add(issues, "error", "empty_key", name, str(key), "Primary key contains an empty value")
                elif count > 1:
                    add(issues, "error", "duplicate_key", name, str(key), f"Occurs {count} times")

    records = tables["records"][0]
    delivered_uuids = set()
    for row in records:
        uuid = row.get("episode_uuid", "")
        delivered_uuids.add(uuid)
        ref = official.get(uuid)
        if not ref:
            add(issues, "error", "unknown_episode", "records", uuid, "Not found in official_records.csv")
            continue
        for col in ("dataset", "sample", "environment_id", "scene_id", "operator_id", "split"):
            if row.get(col, "") != ref.get(col, ""):
                add(issues, "error", "official_mismatch", "records", uuid, f"{col}: delivery={row.get(col)!r}, official={ref.get(col)!r}")
        for col in ("n_frames", "n_segments", "duplicate_timestamp_count", "nonmonotonic_count", "large_gap_count"):
            if col in row and (number(row[col]) is None or number(row[col]) < 0):
                add(issues, "error", "invalid_nonnegative", "records", uuid, f"{col}={row[col]!r}")
        for col in ("left_bad_ratio", "right_bad_ratio"):
            if row.get(col, "") != "" and (number(row[col]) is None or not 0 <= number(row[col]) <= 1):
                add(issues, "error", "invalid_ratio", "records", uuid, f"{col}={row[col]!r}")
        if row.get("usable", "") and not boolean(row["usable"]):
            add(issues, "error", "invalid_boolean", "records", uuid, f"usable={row['usable']!r}")

    frame_count = Counter()
    last_time = {}
    for row in frames:
        uuid = str(row.get("episode_uuid", ""))
        frame_count[uuid] += 1
        if uuid not in delivered_uuids:
            add(issues, "error", "orphan_foreign_key", "frames", str(row.get("frame_id", "")), uuid)
        ref = official.get(uuid)
        if ref:
            for col in ("dataset", "sample", "split"):
                if str(row.get(col, "")) != ref[col]:
                    add(issues, "error", "official_mismatch", "frames", str(row.get("frame_id", "")), col)
        t = number(row.get("timestamp_s"))
        if t is None:
            add(issues, "error", "invalid_timestamp", "frames", str(row.get("frame_id", "")), repr(row.get("timestamp_s")))
        elif uuid in last_time and t < last_time[uuid]:
            add(issues, "error", "nonmonotonic_time", "frames", str(row.get("frame_id", "")), f"{t} < {last_time[uuid]}")
        if t is not None:
            last_time[uuid] = t
        for col in ("left_quality", "right_quality"):
            if row.get(col) not in schema["allowed"]["quality"]:
                add(issues, "error", "invalid_enum", "frames", str(row.get("frame_id", "")), f"{col}={row.get(col)!r}")
        for col in ("left_fit_ok", "right_fit_ok", "left_ref_ok", "right_ref_ok"):
            if not boolean(row.get(col, "")):
                add(issues, "error", "invalid_boolean", "frames", str(row.get("frame_id", "")), f"{col}={row.get(col)!r}")

    for row in records:
        uuid = row.get("episode_uuid", "")
        expected = number(row.get("n_frames"))
        if expected is not None and int(expected) != frame_count[uuid]:
            add(issues, "error", "frame_count_mismatch", "records", uuid, f"records={int(expected)}, frames={frame_count[uuid]}")

    segment_count = Counter()
    for row in tables["segments"][0]:
        sid, uuid = row.get("segment_id", ""), row.get("episode_uuid", "")
        segment_count[uuid] += 1
        if uuid not in delivered_uuids:
            add(issues, "error", "orphan_foreign_key", "segments", sid, uuid)
        start, end, duration = number(row.get("start_s")), number(row.get("end_s")), number(row.get("duration_s"))
        if start is None or end is None or duration is None or not start < end:
            add(issues, "error", "invalid_interval", "segments", sid, f"start={start}, end={end}, duration={duration}")
        elif abs((end - start) - duration) > 1e-6:
            add(issues, "error", "duration_mismatch", "segments", sid, f"end-start={end-start}, duration={duration}")
        for col in ("good_ratio", "bad_ratio"):
            if row.get(col, "") != "" and (number(row[col]) is None or not 0 <= number(row[col]) <= 1):
                add(issues, "error", "invalid_ratio", "segments", sid, f"{col}={row[col]!r}")

    for row in records:
        uuid = row.get("episode_uuid", "")
        expected = number(row.get("n_segments"))
        if expected is not None and int(expected) != segment_count[uuid]:
            add(issues, "error", "segment_count_mismatch", "records", uuid, f"records={int(expected)}, segments={segment_count[uuid]}")

    summary = {
        "delivery": str(delivery), "schema_version": schema["schema_version"], "frame_format": frame_format,
        "counts": {name: len(rows) for name, (rows, _) in tables.items()},
        "errors": sum(x["severity"] == "error" for x in issues),
        "warnings": sum(x["severity"] == "warning" for x in issues),
        "status": "pass" if not any(x["severity"] == "error" for x in issues) else "fail",
        "issues": issues,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "pass" else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--delivery", type=Path, required=True)
    p.add_argument("--schema", type=Path, required=True)
    p.add_argument("--official-records", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return validate(p.parse_args())


if __name__ == "__main__":
    sys.exit(main())
