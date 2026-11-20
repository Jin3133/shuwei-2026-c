#!/usr/bin/env python3
"""Normalize all source quality annotations into one event table."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


FIELDS = [
    "episode_uuid", "dataset", "sample", "source_frame_index", "timestamp_s",
    "timestamp_raw", "left_raw_quality_flag", "right_raw_quality_flag",
    "left_bad_reasons", "right_bad_reasons", "source_locator",
]


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def reason(value):
    return value.strip().lower().replace("near edge", "near_edge")


def compact_json(values):
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    root = args.data_root.resolve()
    records = read_csv(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts, left_bad, right_bad = Counter(), Counter(), Counter()

    with args.output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for record in records:
            episode, dataset, sample = record["episode_uuid"], record["dataset"], record["sample"]
            folder = root / record["source_path"]
            if dataset == "EgoDemo":
                info = json.loads((folder / "meta/info.json").read_text(encoding="utf-8"))
                quality = json.loads((folder / "bad_frame_ratio.json").read_text(encoding="utf-8"))
                fps = float(info["fps"])
                total = int(quality["total_frames"])
                details = {int(x["frame"]): x for x in quality.get("hand_bad_frame_details", [])}
                for frame in range(total):
                    item = details.get(frame, {})
                    left = [reason(item["left_bad_reason"])] if item.get("left_bad_reason") else []
                    right = [reason(item["right_bad_reason"])] if item.get("right_bad_reason") else []
                    writer.writerow({
                        "episode_uuid": episode, "dataset": dataset, "sample": sample,
                        "source_frame_index": frame, "timestamp_s": f"{frame / fps:.9f}", "timestamp_raw": frame,
                        "left_raw_quality_flag": str(bool(left)).lower(), "right_raw_quality_flag": str(bool(right)).lower(),
                        "left_bad_reasons": compact_json(left), "right_bad_reasons": compact_json(right),
                        "source_locator": f"{record['source_path']}/bad_frame_ratio.json#frame={frame}",
                    })
                    counts[dataset] += 1; left_bad[dataset] += bool(left); right_bad[dataset] += bool(right)
            else:
                path = folder / "tables/annotation_bad_frame_pose_hand.csv"
                with path.open("r", encoding="utf-8-sig", newline="") as source:
                    for row_number, item in enumerate(csv.DictReader(source), start=2):
                        left, right = [], []
                        for key, value in item.items():
                            if not key.startswith("problem_type[") or not value:
                                continue
                            normalized = reason(value)
                            if normalized.startswith("left:"): left.append(normalized.split(":", 1)[1])
                            elif normalized.startswith("right:"): right.append(normalized.split(":", 1)[1])
                        writer.writerow({
                            "episode_uuid": episode, "dataset": dataset, "sample": sample,
                            "source_frame_index": item["_message_index"],
                            "timestamp_s": item["_offset_from_first_log_s"], "timestamp_raw": item["_log_time_ns"],
                            "left_raw_quality_flag": str(bool(left)).lower(), "right_raw_quality_flag": str(bool(right)).lower(),
                            "left_bad_reasons": compact_json(left), "right_bad_reasons": compact_json(right),
                            "source_locator": f"{record['source_path']}/tables/annotation_bad_frame_pose_hand.csv#row={row_number}",
                        })
                        counts[dataset] += 1; left_bad[dataset] += bool(left); right_bad[dataset] += bool(right)

    summary = {
        "rows": sum(counts.values()), "rows_by_dataset": dict(counts),
        "left_bad_by_dataset": dict(left_bad), "right_bad_by_dataset": dict(right_bad),
        "time_rule": {"EgoDemo": "source_frame_index / official fps", "EgoPro": "_offset_from_first_log_s", "EgoStandard": "_offset_from_first_log_s"},
        "status": "pass",
    }
    summary_path = args.output.with_name("quality_events_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
