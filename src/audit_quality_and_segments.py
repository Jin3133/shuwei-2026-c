#!/usr/bin/env python3
"""Audit source quality annotations and semantic segments for all 69 records."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


BASE_QUALITY_COLUMNS = {
    "_message_index", "_log_time_ns", "_publish_time_ns", "_sequence",
    "_offset_from_first_log_s", "header.ts", "header.sender", "header.count",
    "is_bad", "problem_type.length",
}


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def close(a, b, tolerance=1e-6):
    try:
        return abs(float(a) - float(b)) <= tolerance
    except (TypeError, ValueError):
        return False


def normalize_reason(value):
    return value.strip().lower().replace("near edge", "near_edge")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    root = args.data_root.resolve()
    records, _ = read_csv(args.manifest)
    official, _ = read_csv(root / "Restocking_Tasks_and_Constraints/source_segment_index.csv")
    official_by_episode = defaultdict(list)
    for row in official:
        official_by_episode[row["episode_uuid"]].append(row)
    for rows in official_by_episode.values():
        rows.sort(key=lambda x: int(x["source_segment_index"]))

    issues = []
    record_rows = []
    reason_counts = Counter()
    semantic_rows = []
    quality_schema_counts = Counter()

    for record in records:
        episode, dataset = record["episode_uuid"], record["dataset"]
        folder = root / record["source_path"]
        source_segments = []
        n_quality = n_bad = left_bad = right_bad = 0
        quality_time_kind = "frame_index" if dataset == "EgoDemo" else "relative_seconds"

        if dataset == "EgoDemo":
            annotation = json.loads((folder / "annotation.json").read_text(encoding="utf-8"))
            for i, item in enumerate(annotation.get("subtask_annotations", [])):
                source_segments.append({
                    "index": i, "start_s": item.get("start_time"), "end_s": item.get("end_time"),
                    "description": item.get("description", ""),
                })
            quality = json.loads((folder / "bad_frame_ratio.json").read_text(encoding="utf-8"))
            n_quality = int(quality.get("total_frames", 0))
            details = quality.get("hand_bad_frame_details", [])
            n_bad = len(details)
            for item in details:
                for side in ("left", "right"):
                    value = item.get(f"{side}_bad_reason")
                    if value:
                        reason = normalize_reason(value)
                        reason_counts[(dataset, side, reason)] += 1
                        if side == "left": left_bad += 1
                        else: right_bad += 1
        else:
            segment_data, _ = read_csv(folder / "tables/annotation_semantic_segments.csv")
            for i, item in enumerate(segment_data):
                source_segments.append({
                    "index": i, "start_s": item.get("segment.start_time"), "end_s": item.get("segment.end_time"),
                    "description": item.get("segment.subtask_description", ""),
                })
            quality_data, quality_columns = read_csv(folder / "tables/annotation_bad_frame_pose_hand.csv")
            quality_schema_counts[(dataset, tuple(quality_columns))] += 1
            unexpected = [c for c in quality_columns if c not in BASE_QUALITY_COLUMNS and not c.startswith("problem_type[")]
            if unexpected:
                issues.append({"episode_uuid": episode, "issue": "unexpected_quality_columns", "detail": ";".join(unexpected)})
            times = []
            for item in quality_data:
                n_quality += 1
                try: times.append(float(item["_offset_from_first_log_s"]))
                except ValueError: issues.append({"episode_uuid": episode, "issue": "invalid_quality_time", "detail": item.get("_offset_from_first_log_s", "")})
                if item.get("is_bad", "").lower() != "true":
                    continue
                n_bad += 1
                reasons = [normalize_reason(v) for k, v in item.items() if k.startswith("problem_type[") and v]
                declared = int(item.get("problem_type.length") or 0)
                if declared != len(reasons):
                    issues.append({"episode_uuid": episode, "issue": "problem_length_mismatch", "detail": f"declared={declared}, found={len(reasons)}"})
                sides = set()
                for reason in reasons:
                    if ":" not in reason:
                        issues.append({"episode_uuid": episode, "issue": "invalid_problem_type", "detail": reason})
                        continue
                    side, kind = reason.split(":", 1)
                    sides.add(side)
                    reason_counts[(dataset, side, kind)] += 1
                left_bad += "left" in sides
                right_bad += "right" in sides
            if any(b < a for a, b in zip(times, times[1:])):
                issues.append({"episode_uuid": episode, "issue": "nonmonotonic_quality_time", "detail": ""})

        expected = official_by_episode[episode]
        if len(source_segments) != len(expected):
            issues.append({"episode_uuid": episode, "issue": "semantic_count_mismatch", "detail": f"source={len(source_segments)}, official={len(expected)}"})
        for i in range(max(len(source_segments), len(expected))):
            src = source_segments[i] if i < len(source_segments) else None
            ref = expected[i] if i < len(expected) else None
            status = "matched"
            detail = ""
            if src is None or ref is None:
                status, detail = "missing", "source or official segment absent"
            else:
                mismatches = []
                if not close(src["start_s"], ref["start_s"]): mismatches.append("start_s")
                if not close(src["end_s"], ref["end_s"]): mismatches.append("end_s")
                if src["description"].strip() != ref["description"].strip(): mismatches.append("description")
                if mismatches: status, detail = "mismatch", ";".join(mismatches)
            semantic_rows.append({
                "episode_uuid": episode, "dataset": dataset, "sample": record["sample"], "source_segment_index": i,
                "official_segment_id": ref["unit_id"] if ref else "", "source_start_s": src["start_s"] if src else "",
                "source_end_s": src["end_s"] if src else "", "official_start_s": ref["start_s"] if ref else "",
                "official_end_s": ref["end_s"] if ref else "", "mapping_status": status, "detail": detail,
            })
            if status != "matched":
                issues.append({"episode_uuid": episode, "issue": "semantic_mapping_mismatch", "detail": f"index={i};{detail}"})

        record_rows.append({
            "episode_uuid": episode, "dataset": dataset, "sample": record["sample"],
            "quality_time_kind": quality_time_kind, "n_quality_rows_or_frames": n_quality,
            "n_bad_rows": n_bad, "left_bad_rows": left_bad, "right_bad_rows": right_bad,
            "n_source_segments": len(source_segments), "n_official_segments": len(expected),
            "semantic_mapping_status": "matched" if len(source_segments) == len(expected) and all(
                close(a["start_s"], b["start_s"]) and close(a["end_s"], b["end_s"]) and a["description"].strip() == b["description"].strip()
                for a, b in zip(source_segments, expected)
            ) else "mismatch",
        })

    args.output.mkdir(parents=True, exist_ok=True)
    def write(name, rows, fields):
        with (args.output / name).open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    write("quality_record_summary.csv", record_rows, list(record_rows[0]))
    reason_rows = [{"dataset": d, "hand": h, "reason": r, "count": n} for (d, h, r), n in sorted(reason_counts.items())]
    write("quality_reason_counts.csv", reason_rows, ["dataset", "hand", "reason", "count"])
    write("semantic_segment_comparison.csv", semantic_rows, list(semantic_rows[0]))
    summary = {
        "records": len(record_rows), "quality_rows_or_frames": sum(int(x["n_quality_rows_or_frames"]) for x in record_rows),
        "bad_rows": sum(int(x["n_bad_rows"]) for x in record_rows), "semantic_segments": len(semantic_rows),
        "matched_semantic_segments": sum(x["mapping_status"] == "matched" for x in semantic_rows),
        "quality_schema_variants": {dataset: len({schema for (ds, schema), n in quality_schema_counts.items() if ds == dataset}) for dataset in ("EgoPro", "EgoStandard")},
        "issues": issues, "status": "pass" if not issues else "fail",
    }
    (args.output / "quality_segment_audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
