#!/usr/bin/env python3
"""Strictly validate a NAVSIM NavHard score CSV against a reference token set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path


SUMMARY_PREFIX = "extended_pdm_score_"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    real = [row for row in rows if not row["token"].startswith(SUMMARY_PREFIX)]
    summary = {
        row["token"].removeprefix(SUMMARY_PREFIX): float(row["score"])
        for row in rows
        if row["token"].startswith(SUMMARY_PREFIX)
    }
    stage_one = [row for row in real if row["ego_progress_stage_one"]]
    stage_two = [row for row in real if row["ego_progress_stage_two"]]
    scores = [float(row["score"]) for row in real]
    return {
        "path": str(path),
        "sha256": sha256(path),
        "row_count": len(rows),
        "real_count": len(real),
        "summary_count": len(rows) - len(real),
        "tokens": [row["token"] for row in real],
        "valid_all": all(row["valid"].strip().lower() == "true" for row in real),
        "unique_token_count": len({row["token"] for row in real}),
        "stage_one_count": len(stage_one),
        "stage_two_count": len(stage_two),
        "summary": summary,
        "finite_scores": all(math.isfinite(score) for score in scores),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--expected-scenes", type=int, default=5912)
    args = parser.parse_args()

    baseline = load(args.baseline)
    candidate = load(args.candidate)
    baseline_tokens = baseline.pop("tokens")
    candidate_tokens = candidate.pop("tokens")
    required_summary = {"stage_one", "stage_two", "combined"}
    token_order_exact = candidate_tokens == baseline_tokens
    checks = {
        "expected_scene_count": candidate["real_count"] == args.expected_scenes,
        "all_valid": candidate["valid_all"],
        "unique_tokens": candidate["unique_token_count"] == args.expected_scenes,
        "token_set_exact": set(candidate_tokens) == set(baseline_tokens),
        "stage_partition_exact": (
            candidate["stage_one_count"] + candidate["stage_two_count"]
            == args.expected_scenes
        ),
        "summary_rows_exact": set(candidate["summary"]) == required_summary,
        "finite_scores": candidate["finite_scores"],
    }
    deltas = {
        key: candidate["summary"][key] - baseline["summary"][key]
        for key in sorted(required_summary)
    }
    report = {
        "baseline": baseline,
        "candidate": candidate,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "informational": {"token_order_exact": token_order_exact},
        "summary_delta_vs_baseline": deltas,
    }
    if not report["all_checks_passed"]:
        raise RuntimeError(json.dumps(report, indent=2))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.report.with_name(args.report.name + f".tmp.{os.getpid()}")
    try:
        tmp.write_text(json.dumps(report, indent=2) + "\n")
        os.replace(tmp, args.report)
    finally:
        if tmp.exists():
            tmp.unlink()
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
