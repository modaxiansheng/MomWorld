#!/usr/bin/env python3
"""Freeze candidate-deletion rules on all unique NAVTRAIN scenes only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from scripts.training.refit_momworld_gtrs_scorer_all_navtrain import (
    combine_all_unique_navtrain,
)
from scripts.training.train_momworld_rule_scorer_from_cache import (
    _discover_manifests,
    _discover_npz,
    _load_records,
    _validate_manifests,
)
from scripts.training.tune_momworld_relu_guard_from_cache import (
    _arrays,
    _effective_mask,
    _load_rule_scorer_logits,
    _metrics,
    _reference_scores,
)


REPORT_SCHEMA = "momworld-candidate-safety-rule-all-navtrain-v1"
BENCHMARKS = (
    "navsim_v1_navtest",
    "navsim_v2_navtest",
    "navsim_v2_navhard",
)
MINIMUM_FIELDS = (
    "no_collision_min",
    "drivable_area_min",
    "ttc_min",
    "progress_min",
    "direction_min",
    "lane_min",
    "traffic_light_min",
)
RELEVANT_SOURCE_FILES = (
    "navsim/agents/momworld/momworld_config.py",
    "navsim/agents/momworld/momworld_model.py",
    "scripts/training/tune_momworld_candidate_safety_rule_all_navtrain.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_audit(repository: Path) -> Dict[str, Any]:
    commit = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    status = subprocess.check_output(
        [
            "git",
            "-C",
            str(repository),
            "status",
            "--porcelain",
            "--",
            *RELEVANT_SOURCE_FILES,
        ],
        text=True,
    ).splitlines()
    if status:
        raise RuntimeError(f"Candidate safety-rule sources are dirty: {status}")
    files = {}
    for relative in RELEVANT_SOURCE_FILES:
        path = (repository / relative).resolve()
        blob = subprocess.check_output(
            ["git", "-C", str(repository), "show", f"{commit}:{relative}"]
        )
        disk_sha = _sha256(path)
        if disk_sha != hashlib.sha256(blob).hexdigest():
            raise RuntimeError(f"Source differs from committed blob: {relative}")
        files[relative] = {"path": str(path), "sha256": disk_sha}
    return {"commit": commit, "working_tree_clean": True, "files": files}


def neutral_rule() -> Dict[str, float]:
    return {
        **{name: 0.0 for name in MINIMUM_FIELDS},
        "collision_risk_max": 1.000001,
        "kinematic_max": 1e6,
        "momentum_max": 1e6,
        "fallback_risk_slack": 0.0,
    }


def apply_safety_rule(
    features: np.ndarray,
    finite: np.ndarray,
    parameters: Mapping[str, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NumPy twin of the online candidate deletion/fallback rule."""

    features = np.asarray(features, dtype=np.float32)
    finite = np.asarray(finite, dtype=np.bool_)
    if features.ndim != 3 or features.shape[-1] < 10:
        raise ValueError("features must have shape [scenes, candidates, >=10]")
    if finite.shape != features.shape[:2]:
        raise ValueError("finite mask shape mismatch")
    minimums = np.asarray(
        [float(parameters[name]) for name in MINIMUM_FIELDS], dtype=np.float32
    )
    maxima = np.asarray(
        [
            float(parameters["collision_risk_max"]),
            float(parameters["kinematic_max"]),
            float(parameters["momentum_max"]),
        ],
        dtype=np.float32,
    )
    slack = float(parameters["fallback_risk_slack"])
    if (
        np.any(~np.isfinite(minimums))
        or np.any((minimums < 0.0) | (minimums > 1.0))
        or np.any(~np.isfinite(maxima))
        or not 0.0 <= maxima[0] <= 1.000001
        or np.any(maxima[1:] < 0.0)
        or not math.isfinite(slack)
        or slack < 0.0
    ):
        raise ValueError("Invalid safety-rule parameters")

    strict = finite.copy()
    violation = np.zeros(finite.shape, dtype=np.float32)
    for index, minimum in enumerate(minimums):
        strict &= features[..., index] >= minimum
        if minimum > 0.0:
            violation += np.maximum(minimum - features[..., index], 0.0) / max(
                float(minimum), 0.1
            )
    for index, maximum in zip((7, 8, 9), maxima):
        strict &= features[..., index] < maximum
        violation += np.maximum(features[..., index] - maximum, 0.0) / max(
            float(maximum), 0.1
        )
    finite_violation = np.where(finite, violation, np.inf)
    minimum_violation = finite_violation.min(axis=1, keepdims=True)
    fallback = finite & (violation <= minimum_violation + slack)
    has_strict = strict.any(axis=1, keepdims=True)
    has_finite = finite.any(axis=1, keepdims=True)
    eligible = np.where(
        has_strict,
        strict,
        np.where(has_finite, fallback, np.ones_like(finite)),
    )
    return eligible, strict, violation


def _select(reference_score: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    score = np.where(
        eligible,
        np.asarray(reference_score, dtype=np.float32),
        np.finfo(np.float32).min,
    )
    return score.argmax(axis=1).astype(np.int64)


def _threshold_options(arrays: Mapping[str, np.ndarray]) -> Dict[str, Tuple[float, ...]]:
    finite = np.asarray(arrays["candidate_is_finite"], dtype=np.bool_)
    features = np.asarray(arrays["features"], dtype=np.float32)
    kinematic = features[..., 8][finite]
    momentum = features[..., 9][finite]
    return {
        "no_collision_min": (0.25, 0.40, 0.55),
        "drivable_area_min": (0.25, 0.40, 0.55),
        "ttc_min": (0.25, 0.40, 0.55),
        "progress_min": (0.05, 0.10, 0.20),
        "direction_min": (0.25, 0.40, 0.55),
        "lane_min": (0.25, 0.40, 0.55),
        "traffic_light_min": (0.25, 0.40, 0.55),
        "collision_risk_max": (0.80, 0.65, 0.50, 0.35),
        "kinematic_max": tuple(
            sorted({float(np.quantile(kinematic, q)) for q in (0.50, 0.75, 0.90)})
        ),
        "momentum_max": tuple(
            sorted({float(np.quantile(momentum, q)) for q in (0.50, 0.75, 0.90)})
        ),
    }


def tune_rule(
    arrays: Mapping[str, np.ndarray],
    benchmark: str,
    v2_logits: np.ndarray,
    max_active_rules: int = 3,
) -> Dict[str, Any]:
    reference_score, reference, description = _reference_scores(
        arrays, benchmark, v2_logits
    )
    protocol = "v1" if benchmark == "navsim_v1_navtest" else "v2"
    target = np.asarray(arrays[f"target_{protocol}"], dtype=np.float32)
    target_collision = np.asarray(arrays["target_collision"], dtype=np.float32)
    rows = np.arange(len(reference))
    reference_collision = target_collision[rows, reference]
    reference_score_mean = float(target[rows, reference].mean())
    reference_collision_mean = float(reference_collision.mean())
    reference_collision_rate = float((reference_collision >= 0.5).mean())
    parameters = neutral_rule()
    options = _threshold_options(arrays)
    used: set[str] = set()
    trace: List[Dict[str, Any]] = []
    current_mean = reference_score_mean
    for step in range(max_active_rules):
        candidates = []
        for name, values in options.items():
            if name in used:
                continue
            for value in values:
                trial = {**parameters, name: float(value)}
                eligible, strict, violation = apply_safety_rule(
                    arrays["features"], arrays["candidate_is_finite"], trial
                )
                selected = _select(reference_score, eligible)
                metrics = _metrics(arrays, benchmark, selected, reference)
                fallback_rate = float((~strict.any(axis=1)).mean())
                if (
                    metrics["target_collision_rate_ge_0_5"]
                    <= reference_collision_rate + 1e-12
                    and metrics["mean_target_collision_risk"]
                    <= reference_collision_mean + 1e-12
                ):
                    key = (
                        metrics["mean_target_score"],
                        -metrics["target_collision_rate_ge_0_5"],
                        -metrics["mean_target_collision_risk"],
                        -fallback_rate,
                        -metrics["changed_vs_reference_rate"],
                    )
                    candidates.append(
                        (key, name, value, trial, metrics, fallback_rate, violation)
                    )
        if not candidates:
            break
        _, name, value, trial, metrics, fallback_rate, _ = max(
            candidates, key=lambda item: item[0]
        )
        if metrics["mean_target_score"] <= current_mean + 1e-12:
            break
        parameters = trial
        used.add(name)
        current_mean = metrics["mean_target_score"]
        trace.append(
            {
                "step": step + 1,
                "field": name,
                "threshold": float(value),
                "fallback_scene_rate": fallback_rate,
                "metrics": metrics,
            }
        )

    eligible, strict, violation = apply_safety_rule(
        arrays["features"], arrays["candidate_is_finite"], parameters
    )
    selected = _select(reference_score, eligible)
    metrics = _metrics(arrays, benchmark, selected, reference)
    active = {
        name: value
        for name, value in parameters.items()
        if value != neutral_rule()[name]
    }
    return {
        "benchmark": benchmark,
        "reference": description,
        "reference_mean_target_score": reference_score_mean,
        "parameters": parameters,
        "active_rules": active,
        "trace": trace,
        "metrics": metrics,
        "strict_safe_scene_rate": float(strict.any(axis=1).mean()),
        "fallback_scene_rate": float((~strict.any(axis=1)).mean()),
        "mean_candidates_kept": float(eligible.sum(axis=1).mean()),
        "mean_aggregate_violation": float(violation.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--v2-rule-scorer-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8192)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    repository = Path(__file__).resolve().parents[2]
    source = _source_audit(repository)
    cache_root = args.cache_root.expanduser().resolve()
    cache_audit = _validate_manifests(_discover_manifests([cache_root]), False)
    raw = _load_records(
        _discover_npz([cache_root]), str(cache_audit["checkpoint_sha256"])
    )
    records, partition = combine_all_unique_navtrain(raw)
    arrays = _arrays(records)
    v2_logits, scorer_audit = _load_rule_scorer_logits(
        args.v2_rule_scorer_state, records, cache_audit, args.batch_size
    )
    results = {
        benchmark: tune_rule(arrays, benchmark, v2_logits)
        for benchmark in BENCHMARKS
    }
    payload = {
        "schema": REPORT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_audit": source,
        "data_scope": {
            "source_split": "navtrain",
            "label_scope": "navtrain-only",
            "samples": len(records),
            "partition": partition,
            "evaluator_calls": 0,
            "benchmark_cache_reads": 0,
            "excluded_splits": ["navtest", "navhard"],
        },
        "cache_audit": cache_audit,
        "v2_rule_scorer": scorer_audit,
        "selection": "greedy maximum three distinct hard candidate rules; all-unsafe minimum-violation fallback",
        "results": results,
    }
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, output)
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": _sha256(output),
                "gains": {
                    name: result["metrics"]["score_gain"]
                    for name, result in results.items()
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
