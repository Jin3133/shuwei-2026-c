#!/usr/bin/env python3
"""Audit the official C-problem indices and emit merge-ready reference files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable


EXPECTED_RECORDS = {"EgoDemo": 31, "EgoPro": 14, "EgoStandard": 24}
EXPECTED_SPLIT_RECORDS = {"train": 55, "validation": 7, "test": 7}
EXPECTED_SPLIT_ENVIRONMENTS = {"train": 16, "validation": 4, "test": 4}
EXPECTED_CASE_JOBS = {"S1": 8, "S2": 12, "S3": 16}
EXPECTED_SEGMENTS = 349
EXPECTED_ELIGIBLE_SEGMENTS = 343
ALLOWED_SPLITS = {"train", "validation", "test"}


@dataclass
class Issue:
    issue_id: str
    severity: str
    issue_type: str
    source_file: str
    record_key: str
    message: str
    action: str = "unresolved"


class Audit:
    def __init__(self) -> None:
        self.issues: list[Issue] = []

    def add(
        self,
        severity: str,
        issue_type: str,
        source_file: str,
        record_key: str,
        message: str,
        action: str = "unresolved",
    ) -> None:
        self.issues.append(
            Issue(
                issue_id=f"ISSUE_{len(self.issues) + 1:04d}",
                severity=severity,
                issue_type=issue_type,
                source_file=source_file,
                record_key=record_key,
                message=message,
                action=action,
            )
        )

    def expect(
        self,
        condition: bool,
        issue_type: str,
        source_file: str,
        record_key: str,
        message: str,
        severity: str = "error",
    ) -> None:
        if not condition:
            self.add(severity, issue_type, source_file, record_key, message)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_float(value: str, audit: Audit, file_name: str, key: str, field: str) -> float:
    try:
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("non-finite")
        return result
    except (TypeError, ValueError):
        audit.add("error", "invalid_number", file_name, key, f"{field}={value!r} is invalid")
        return math.nan


def parse_int(value: str, audit: Audit, file_name: str, key: str, field: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        audit.add("error", "invalid_integer", file_name, key, f"{field}={value!r} is invalid")
        return -1


def check_unique(rows: list[dict[str, str]], key: str, audit: Audit, file_name: str) -> None:
    counts = Counter(row.get(key, "") for row in rows)
    for value, count in counts.items():
        if not value:
            audit.add("error", "missing_key", file_name, "", f"Missing {key}")
        elif count > 1:
            audit.add("error", "duplicate_key", file_name, value, f"{key} occurs {count} times")


def has_cycle(nodes: set[str], predecessors: dict[str, set[str]]) -> bool:
    state: dict[str, int] = {node: 0 for node in nodes}

    def visit(node: str) -> bool:
        if state[node] == 1:
            return True
        if state[node] == 2:
            return False
        state[node] = 1
        for parent in predecessors.get(node, set()):
            if parent in nodes and visit(parent):
                return True
        state[node] = 2
        return False

    return any(visit(node) for node in nodes if state[node] == 0)


def field_dictionary() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    def add(table: str, name: str, dtype: str, nullable: bool, source: str, definition: str, unit: str = "") -> None:
        rows.append(
            {
                "table_name": table,
                "field_name": name,
                "data_type": dtype,
                "nullable": str(nullable).lower(),
                "source_type": source,
                "definition": definition,
                "unit": unit,
                "version": "1.0",
            }
        )

    for name, dtype, nullable, source, definition, unit in [
        ("episode_uuid", "string", False, "official/raw", "Internal record primary key", ""),
        ("dataset", "string", False, "official", "EgoDemo, EgoPro, or EgoStandard", ""),
        ("sample", "string", False, "official/path", "Original record directory name", ""),
        ("source_path", "string", False, "derived", "Path relative to the data root", ""),
        ("environment_id", "string", False, "official", "Environment identifier", ""),
        ("scene_id", "string", True, "official", "Scene identifier", ""),
        ("operator_id", "string", True, "official", "Operator identifier", ""),
        ("split", "string", False, "official", "train, validation, or test", ""),
        ("n_frames", "int64", False, "derived", "Number of rows in the unified frame table", ""),
        ("duration_s", "float64", False, "derived", "Last timestamp minus first timestamp", "s"),
        ("actual_fps", "float64", True, "derived", "Inverse median timestamp difference", "Hz"),
        ("n_segments", "int64", False, "official/derived", "Number of mapped segments", ""),
        ("left_bad_ratio", "float64", True, "derived", "Left-hand explicit bad-observation ratio", "fraction"),
        ("right_bad_ratio", "float64", True, "derived", "Right-hand explicit bad-observation ratio", "fraction"),
        ("duplicate_timestamp_count", "int64", False, "derived", "Duplicate timestamp count", ""),
        ("nonmonotonic_count", "int64", False, "derived", "Non-monotonic timestamp count", ""),
        ("large_gap_count", "int64", False, "derived", "Large timestamp gap count", ""),
        ("duplicate_status", "string", False, "derived", "unique, duplicate_uuid, or metadata_conflict", ""),
        ("usable", "bool", False, "derived", "Usable for at least one modelling task", ""),
        ("exclusion_reason", "string", True, "derived", "Reason for exclusion or restricted use", ""),
    ]:
        add("records", name, dtype, nullable, source, definition, unit)

    for name, dtype, nullable, source, definition, unit in [
        ("frame_id", "string", False, "derived", "Unique frame-level output identifier", ""),
        ("episode_uuid", "string", False, "official/raw", "Foreign key to records", ""),
        ("dataset", "string", False, "official", "Data source", ""),
        ("sample", "string", False, "official/path", "Original directory name", ""),
        ("split", "string", False, "official", "Environment-level split", ""),
        ("timestamp_s", "float64", False, "raw/normalized", "Seconds from common record origin", "s"),
        ("timestamp_raw", "string", True, "raw", "Reversible raw timestamp representation", ""),
        ("source_frame_index", "int64", True, "raw", "Original frame or message index", ""),
        ("source_locator", "string", False, "derived", "Source file and row/frame locator", ""),
        ("left_palm_x", "float32", True, "derived", "Left palm representative x", "m"),
        ("left_palm_y", "float32", True, "derived", "Left palm representative y", "m"),
        ("left_palm_z", "float32", True, "derived", "Left palm representative z", "m"),
        ("right_palm_x", "float32", True, "derived", "Right palm representative x", "m"),
        ("right_palm_y", "float32", True, "derived", "Right palm representative y", "m"),
        ("right_palm_z", "float32", True, "derived", "Right palm representative z", "m"),
        ("left_raw_quality_flag", "bool", True, "raw", "Official left-hand quality issue flag", ""),
        ("right_raw_quality_flag", "bool", True, "raw", "Official right-hand quality issue flag", ""),
        ("left_bad_reasons", "list<string>", True, "raw/normalized", "Left-hand quality reasons", ""),
        ("right_bad_reasons", "list<string>", True, "raw/normalized", "Right-hand quality reasons", ""),
        ("left_quality", "string", False, "derived", "good, bad, or missing", ""),
        ("right_quality", "string", False, "derived", "good, bad, or missing", ""),
        ("left_fit_ok", "bool", False, "derived", "Left hand is eligible for specified model fitting", ""),
        ("right_fit_ok", "bool", False, "derived", "Right hand is eligible for specified model fitting", ""),
        ("left_ref_ok", "bool", False, "derived", "Left hand may be a masked-reference state", ""),
        ("right_ref_ok", "bool", False, "derived", "Right hand may be a masked-reference state", ""),
        ("segment_id", "string", True, "official/derived", "Covering official work-unit identifier", ""),
        ("continuity_block_id", "string", False, "derived", "Physical time-continuity block", ""),
    ]:
        add("frames", name, dtype, nullable, source, definition, unit)

    for name, dtype, nullable, source, definition, unit in [
        ("segment_id", "string", False, "official", "Official unit_id or traceable temporary identifier", ""),
        ("episode_uuid", "string", False, "official", "Foreign key to records", ""),
        ("dataset", "string", False, "official", "Data source", ""),
        ("sample", "string", False, "official", "Original record directory", ""),
        ("split", "string", False, "official", "Environment-level split", ""),
        ("source_segment_index", "int64", True, "official/raw", "Zero-based segment index in source record", ""),
        ("family_id", "string", True, "official", "P, A, or L sampling family", ""),
        ("start_s", "float64", False, "official/raw", "Segment start", "s"),
        ("end_s", "float64", False, "official/raw", "Segment end", "s"),
        ("duration_s", "float64", False, "official/derived", "end_s minus start_s", "s"),
        ("description", "string", True, "official/raw", "Offline-only semantic description", ""),
        ("eligible", "bool", True, "official", "Eligible for common Q3 sampling", ""),
        ("n_frames", "int64", False, "derived", "Mapped frame count", ""),
        ("good_ratio", "float64", True, "derived", "Usable hand-observation ratio", "fraction"),
        ("bad_ratio", "float64", True, "derived", "Explicit quality-issue ratio", "fraction"),
        ("mapping_status", "string", False, "derived", "matched, unmatched, or ambiguous", ""),
    ]:
        add("segments", name, dtype, nullable, source, definition, unit)
    return rows


def audit_indices(root: Path, output: Path) -> dict:
    constraints = root / "Restocking_Tasks_and_Constraints"
    raw_root = root / "Raw_Data"
    files = {
        "record_split": constraints / "record_split.csv",
        "segments": constraints / "source_segment_index.csv",
        "tasks": constraints / "tasks.csv",
        "conditions": constraints / "common_conditions.json",
        "families": constraints / "job_families.csv",
        "zones": constraints / "service_zones.csv",
    }
    for name, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name}: {path}")

    audit = Audit()
    records = read_csv(files["record_split"])
    segments = read_csv(files["segments"])
    tasks = read_csv(files["tasks"])
    families = read_csv(files["families"])
    zones = read_csv(files["zones"])
    with files["conditions"].open("r", encoding="utf-8") as stream:
        conditions = json.load(stream)

    check_unique(records, "episode_uuid", audit, "record_split.csv")
    audit.expect(len(records) == 69, "unexpected_record_count", "record_split.csv", "", f"Expected 69 rows, got {len(records)}")
    record_by_uuid = {row["episode_uuid"]: row for row in records if row.get("episode_uuid")}

    for row in records:
        key = row.get("episode_uuid", "")
        audit.expect(row.get("dataset") in EXPECTED_RECORDS, "invalid_dataset", "record_split.csv", key, f"Invalid dataset {row.get('dataset')!r}")
        audit.expect(row.get("split") in ALLOWED_SPLITS, "invalid_split", "record_split.csv", key, f"Invalid split {row.get('split')!r}")

    dataset_counts = Counter(row["dataset"] for row in records)
    split_record_counts = Counter(row["split"] for row in records)
    for name, expected in EXPECTED_RECORDS.items():
        audit.expect(dataset_counts[name] == expected, "dataset_count_mismatch", "record_split.csv", name, f"Expected {expected}, got {dataset_counts[name]}")
    for name, expected in EXPECTED_SPLIT_RECORDS.items():
        audit.expect(split_record_counts[name] == expected, "split_record_count_mismatch", "record_split.csv", name, f"Expected {expected}, got {split_record_counts[name]}")

    environment_splits: dict[str, set[str]] = defaultdict(set)
    for row in records:
        environment_splits[row["environment_id"]].add(row["split"])
    for environment, splits in environment_splits.items():
        audit.expect(len(splits) == 1, "environment_crosses_splits", "record_split.csv", environment, f"Environment appears in {sorted(splits)}")
    split_environment_counts = Counter()
    for environment, splits in environment_splits.items():
        split_environment_counts[next(iter(splits))] += 1
    for name, expected in EXPECTED_SPLIT_ENVIRONMENTS.items():
        audit.expect(split_environment_counts[name] == expected, "split_environment_count_mismatch", "record_split.csv", name, f"Expected {expected}, got {split_environment_counts[name]}")

    check_unique(segments, "unit_id", audit, "source_segment_index.csv")
    audit.expect(len(segments) == EXPECTED_SEGMENTS, "segment_count_mismatch", "source_segment_index.csv", "", f"Expected {EXPECTED_SEGMENTS}, got {len(segments)}")
    eligible_count = 0
    segment_samples: dict[str, set[str]] = defaultdict(set)
    for row in segments:
        key = row.get("unit_id", "")
        episode_uuid = row.get("episode_uuid", "")
        record = record_by_uuid.get(episode_uuid)
        audit.expect(record is not None, "segment_record_missing", "source_segment_index.csv", key, f"Unknown episode_uuid {episode_uuid}")
        start = parse_float(row.get("start_s", ""), audit, "source_segment_index.csv", key, "start_s")
        end = parse_float(row.get("end_s", ""), audit, "source_segment_index.csv", key, "end_s")
        duration = parse_float(row.get("duration_s", ""), audit, "source_segment_index.csv", key, "duration_s")
        audit.expect(math.isfinite(start) and math.isfinite(end) and end > start, "invalid_segment_bounds", "source_segment_index.csv", key, f"start={start}, end={end}")
        if all(map(math.isfinite, [start, end, duration])):
            audit.expect(abs((end - start) - duration) <= 1e-6, "duration_mismatch", "source_segment_index.csv", key, f"end-start={end-start}, duration={duration}")
        eligible = row.get("eligible") == "1"
        eligible_count += int(eligible)
        if eligible:
            audit.expect(duration >= 1.2 - 1e-9, "eligible_too_short", "source_segment_index.csv", key, f"Eligible duration is {duration}")
        if record:
            for field in ["dataset", "environment_id", "split"]:
                audit.expect(row.get(field) == record.get(field), "segment_record_conflict", "source_segment_index.csv", key, f"{field}: segment={row.get(field)!r}, record={record.get(field)!r}")
        audit.expect(row.get("family_id") in {"P", "A", "L"}, "invalid_family", "source_segment_index.csv", key, f"Invalid family {row.get('family_id')!r}")
        segment_samples[row.get("dataset", "")].add(row.get("sample", ""))
    audit.expect(eligible_count == EXPECTED_ELIGIBLE_SEGMENTS, "eligible_count_mismatch", "source_segment_index.csv", "", f"Expected {EXPECTED_ELIGIBLE_SEGMENTS}, got {eligible_count}")

    family_ids = {row["family_id"] for row in families}
    zone_ids = {row["zone_id"] for row in zones}
    check_unique(tasks, "job_id", audit, "tasks.csv")
    task_by_id = {row["job_id"]: row for row in tasks}
    case_counts = Counter(row["case_id"] for row in tasks)
    case_predecessors: dict[str, dict[str, set[str]]] = defaultdict(dict)
    for row in tasks:
        key = row["job_id"]
        case = row["case_id"]
        audit.expect(row.get("family_id") in family_ids, "unknown_task_family", "tasks.csv", key, f"Unknown family {row.get('family_id')!r}")
        audit.expect(row.get("zone_id") in zone_ids, "unknown_zone", "tasks.csv", key, f"Unknown zone {row.get('zone_id')!r}")
        release = parse_float(row.get("release_s", ""), audit, "tasks.csv", key, "release_s")
        due = parse_float(row.get("due_s", ""), audit, "tasks.csv", key, "due_s")
        weight = parse_float(row.get("priority_weight", ""), audit, "tasks.csv", key, "priority_weight")
        audit.expect(release >= 0, "negative_release", "tasks.csv", key, f"release_s={release}")
        audit.expect(due >= release, "due_before_release", "tasks.csv", key, f"release_s={release}, due_s={due}")
        audit.expect(weight > 0, "invalid_weight", "tasks.csv", key, f"priority_weight={weight}")
        parents = {value for value in row.get("predecessors", "").split("|") if value}
        case_predecessors[case][key] = parents
        for parent in parents:
            parent_row = task_by_id.get(parent)
            audit.expect(parent_row is not None, "unknown_predecessor", "tasks.csv", key, f"Unknown predecessor {parent}")
            if parent_row:
                audit.expect(parent_row["case_id"] == case, "cross_case_predecessor", "tasks.csv", key, f"Predecessor {parent} is in {parent_row['case_id']}")
    for case, expected in EXPECTED_CASE_JOBS.items():
        audit.expect(case_counts[case] == expected, "case_job_count_mismatch", "tasks.csv", case, f"Expected {expected}, got {case_counts[case]}")
        nodes = {row["job_id"] for row in tasks if row["case_id"] == case}
        audit.expect(not has_cycle(nodes, case_predecessors[case]), "precedence_cycle", "tasks.csv", case, "Precedence graph contains a cycle")

    condition_cases = {row["case_id"]: row for row in conditions.get("cases", [])}
    for case, expected_jobs in EXPECTED_CASE_JOBS.items():
        condition = condition_cases.get(case)
        audit.expect(condition is not None, "missing_case_condition", "common_conditions.json", case, "Missing case condition")
        if condition:
            audit.expect(condition.get("jobs") == expected_jobs, "condition_job_mismatch", "common_conditions.json", case, f"Expected jobs={expected_jobs}, got {condition.get('jobs')}")
            due_max = max(float(row["due_s"]) for row in tasks if row["case_id"] == case)
            audit.expect(due_max <= float(condition["horizon_s"]), "due_after_horizon", "tasks.csv", case, f"max due={due_max}, horizon={condition['horizon_s']}", severity="warning")
    seeds = conditions.get("validation_seeds", [])
    audit.expect(len(seeds) == 30 and len(set(seeds)) == 30, "invalid_validation_seeds", "common_conditions.json", "validation_seeds", f"Expected 30 unique seeds, got {len(seeds)} rows and {len(set(seeds))} unique")

    raw_inventory = {}
    for dataset, expected in EXPECTED_RECORDS.items():
        folder = raw_root / dataset
        samples = sorted(path.name for path in folder.iterdir() if path.is_dir()) if folder.is_dir() else []
        raw_inventory[dataset] = samples
        audit.expect(len(samples) == expected, "raw_directory_count_mismatch", str(folder), dataset, f"Expected {expected} directories, got {len(samples)}")
        indexed = segment_samples.get(dataset, set())
        missing_from_index = sorted(set(samples) - indexed)
        missing_from_raw = sorted(indexed - set(samples))
        for sample in missing_from_index:
            audit.add("error", "raw_sample_not_in_segment_index", str(folder), sample, "Raw sample has no source_segment_index row")
        for sample in missing_from_raw:
            audit.add("error", "indexed_sample_missing_from_raw", "source_segment_index.csv", sample, "Indexed sample directory is missing")

    official_records = []
    samples_by_uuid = defaultdict(set)
    for row in segments:
        samples_by_uuid[row["episode_uuid"]].add(row["sample"])
    for row in sorted(records, key=lambda item: (item["dataset"], item["episode_uuid"])):
        samples = sorted(value for value in samples_by_uuid.get(row["episode_uuid"], set()) if value)
        official_records.append(
            {
                **row,
                "sample": samples[0] if len(samples) == 1 else "",
                "sample_mapping_status": "matched" if len(samples) == 1 else ("missing" if not samples else "ambiguous"),
            }
        )
        if len(samples) != 1:
            audit.add("error", "record_sample_mapping_failed", "source_segment_index.csv", row["episode_uuid"], f"Samples={samples}")

    official_segments = []
    for row in segments:
        official_segments.append(
            {
                "segment_id": row["unit_id"],
                "episode_uuid": row["episode_uuid"],
                "dataset": row["dataset"],
                "sample": row["sample"],
                "environment_id": row["environment_id"],
                "split": row["split"],
                "family_id": row["family_id"],
                "task": row["task"],
                "source_segment_index": row["source_segment_index"],
                "start_s": row["start_s"],
                "end_s": row["end_s"],
                "duration_s": row["duration_s"],
                "description": row["description"],
                "eligible": "true" if row["eligible"] == "1" else "false",
                "exclusion_reason": row.get("exclusion_reason", ""),
            }
        )

    output.mkdir(parents=True, exist_ok=True)
    write_csv(
        output / "official_records.csv",
        official_records,
        ["episode_uuid", "dataset", "sample", "environment_id", "scene_id", "operator_id", "split", "sample_mapping_status"],
    )
    write_csv(
        output / "official_segments.csv",
        official_segments,
        ["segment_id", "episode_uuid", "dataset", "sample", "environment_id", "split", "family_id", "task", "source_segment_index", "start_s", "end_s", "duration_s", "description", "eligible", "exclusion_reason"],
    )
    dictionary_rows = field_dictionary()
    write_csv(
        output / "field_dictionary.csv",
        dictionary_rows,
        ["table_name", "field_name", "data_type", "nullable", "source_type", "definition", "unit", "version"],
    )
    write_csv(
        output / "audit.csv",
        [asdict(issue) for issue in audit.issues],
        ["issue_id", "severity", "issue_type", "source_file", "record_key", "message", "action"],
    )

    summary = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "input_root": str(root),
        "hashes": {name: file_sha256(path) for name, path in files.items()},
        "records": {
            "total": len(records),
            "by_dataset": dict(sorted(dataset_counts.items())),
            "by_split": dict(sorted(split_record_counts.items())),
            "environments_total": len(environment_splits),
            "environments_by_split": dict(sorted(split_environment_counts.items())),
        },
        "segments": {
            "total": len(segments),
            "eligible": eligible_count,
            "by_dataset": dict(sorted(Counter(row["dataset"] for row in segments).items())),
            "by_split": dict(sorted(Counter(row["split"] for row in segments).items())),
            "eligible_by_split_family": {
                f"{split}:{family}": count
                for (split, family), count in sorted(Counter((row["split"], row["family_id"]) for row in segments if row["eligible"] == "1").items())
            },
        },
        "tasks": {
            "total": len(tasks),
            "by_case": dict(sorted(case_counts.items())),
            "families": sorted(family_ids),
            "zones": sorted(zone_ids),
            "validation_seed_count": len(seeds),
        },
        "raw_inventory": {dataset: len(samples) for dataset, samples in raw_inventory.items()},
        "issues": {
            "errors": sum(issue.severity == "error" for issue in audit.issues),
            "warnings": sum(issue.severity == "warning" for issue in audit.issues),
            "total": len(audit.issues),
        },
        "status": "pass" if not any(issue.severity == "error" for issue in audit.issues) else "fail",
    }
    with (output / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Restocking_Problem_and_Data directory")
    parser.add_argument("--output", type=Path, required=True, help="Audit output directory")
    args = parser.parse_args()
    summary = audit_indices(args.data_root.resolve(), args.output.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
