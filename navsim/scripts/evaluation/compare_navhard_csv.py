#!/usr/bin/env python3
"""Compare per-token NAVSIM CSV values while ignoring worker row ordering."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def load(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as stream:
        rows = csv.DictReader(stream)
        return {
            row["token"]: row
            for row in rows
            if not row["token"].startswith("extended_pdm_score_")
        }


def as_float(value: str) -> float | None:
    if value == "":
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    args = parser.parse_args()
    left, right = load(args.left), load(args.right)
    if set(left) != set(right):
        raise RuntimeError("token sets differ")
    tokens = sorted(left)
    fields = [field for field in next(iter(left.values())) if field not in {"", "token", "valid"}]
    result: dict[str, object] = {
        "tokens": len(tokens),
        "exact_rows": sum(left[token] == right[token] for token in tokens),
        "columns": {},
    }
    for field in fields:
        pairs = []
        for token in tokens:
            a, b = as_float(left[token][field]), as_float(right[token][field])
            if a is not None and b is not None:
                pairs.append((a, b))
        if not pairs:
            continue
        differences = [b - a for a, b in pairs]
        changed = [delta for delta in differences if delta != 0.0]
        result["columns"][field] = {
            "comparable": len(pairs),
            "changed": len(changed),
            "mean_delta_right_minus_left": sum(differences) / len(differences),
            "max_abs_delta": max(abs(delta) for delta in differences),
        }
    score_changes = []
    for token in tokens:
        a, b = as_float(left[token]["score"]), as_float(right[token]["score"])
        if a is not None and b is not None and a != b:
            score_changes.append(
                {
                    "token": token,
                    "left": a,
                    "right": b,
                    "delta": b - a,
                    "stage": "one" if left[token]["ego_progress_stage_one"] else "two",
                }
            )
    result["largest_score_changes"] = sorted(
        score_changes, key=lambda item: abs(item["delta"]), reverse=True
    )[:25]
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
