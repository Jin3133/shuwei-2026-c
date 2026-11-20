#!/usr/bin/env python3
"""Build and validate the official episode-to-raw-directory path manifest."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


REQUIRED_FILES = {
    "EgoDemo": [
        "annotation.json",
        "bad_frame_ratio.json",
        "data/chunk-000/file-000.parquet",
        "meta/info.json",
    ],
    "EgoPro": [
        "session_metadata.csv",
        "tables/annotation_bad_frame_pose_hand.csv",
        "tables/annotation_semantic_segments.csv",
        "tables/pose_left_hand.csv",
        "tables/pose_right_hand.csv",
    ],
    "EgoStandard": [
        "session_metadata.csv",
        "tables/annotation_bad_frame_pose_hand.csv",
        "tables/annotation_semantic_segments.csv",
        "tables/pose_left_hand.csv",
        "tables/pose_right_hand.csv",
    ],
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.data_root.resolve()
    constraints = root / "Restocking_Tasks_and_Constraints"
    records = read_csv(constraints / "record_split.csv")
    segments = read_csv(constraints / "source_segment_index.csv")
    samples_by_episode: dict[str, set[str]] = defaultdict(set)
    segment_count = Counter()
    for row in segments:
        samples_by_episode[row["episode_uuid"]].add(row["sample"])
        segment_count[row["episode_uuid"]] += 1

    output_rows = []
    issues = []
    for record in records:
        episode = record["episode_uuid"]
        dataset = record["dataset"]
        samples = sorted(samples_by_episode.get(episode, set()))
        sample = samples[0] if len(samples) == 1 else ""
        relative_path = Path("Raw_Data") / dataset / sample if sample else Path("")
        raw_path = root / relative_path if sample else None
        missing = []
        if len(samples) != 1:
            issues.append({"episode_uuid": episode, "issue": "sample_mapping", "detail": repr(samples)})
        elif not raw_path.is_dir():
            issues.append({"episode_uuid": episode, "issue": "missing_directory", "detail": str(relative_path)})
        else:
            missing = [name for name in REQUIRED_FILES[dataset] if not (raw_path / name).is_file()]
            if missing:
                issues.append({"episode_uuid": episode, "issue": "missing_required_files", "detail": ";".join(missing)})
        output_rows.append(
            {
                **record,
                "sample": sample,
                "source_path": relative_path.as_posix() if sample else "",
                "n_official_segments": segment_count[episode],
                "sample_mapping_status": "matched" if len(samples) == 1 else ("missing" if not samples else "ambiguous"),
                "directory_exists": str(bool(raw_path and raw_path.is_dir())).lower(),
                "required_files_status": "complete" if not missing and raw_path and raw_path.is_dir() else "incomplete",
                "missing_required_files": ";".join(missing),
            }
        )

    args.output.mkdir(parents=True, exist_ok=True)
    columns = [
        "episode_uuid", "dataset", "sample", "source_path", "environment_id", "scene_id", "operator_id", "split",
        "n_official_segments", "sample_mapping_status", "directory_exists", "required_files_status", "missing_required_files",
    ]
    with (args.output / "record_path_manifest.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(output_rows)
    summary = {
        "records": len(output_rows),
        "by_dataset": dict(sorted(Counter(row["dataset"] for row in output_rows).items())),
        "matched_samples": sum(row["sample_mapping_status"] == "matched" for row in output_rows),
        "existing_directories": sum(row["directory_exists"] == "true" for row in output_rows),
        "complete_required_files": sum(row["required_files_status"] == "complete" for row in output_rows),
        "issues": issues,
        "status": "pass" if not issues else "fail",
    }
    (args.output / "path_manifest_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
