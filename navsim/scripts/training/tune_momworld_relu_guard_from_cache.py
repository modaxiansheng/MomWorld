#!/usr/bin/env python3
"""Tune semantic ReLU guard penalties on a fresh NAVTRAIN-only reserve.

The three immutable references mirror the best completed benchmark runs:
ProtocolProxy for v1 NAVTest and v2 NAVHard, and the original RuleScorer for
v2 NAVTest.  Calibration greedily adds at most three non-negative ReLU penalty
terms while forbidding a higher target collision rate.  A fresh holdout is
read once after all three configurations are frozen.  No evaluator or
NAVTest/NAVHard cache is imported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch

from navsim.agents.momworld.momworld_model import RuleAwareTrajectoryScorer
from scripts.training.train_momworld_guarded_rule_scorer_from_cache import (
    _protocol_proxy_numpy,
    _zscore_numpy,
)
from scripts.training.train_momworld_rule_scorer_from_cache import (
    STATE_SCHEMA,
    CacheRecord,
    _discover_manifests,
    _discover_npz,
    _exclude_validation_overlap,
    _load_records,
    _validate_manifests,
)


REPORT_SCHEMA = "momworld-relu-guard-tuning-v1"
GATE_SCHEMA = "momworld-relu-guard-gate-v1"
PREVIOUS_PARTITION_SALTS = (
    "momworld-guarded-rule-scorer-effective-train-v1",
    "momworld-guarded-rule-scorer-effective-train-v2",
    "momworld-guarded-pairwise-scorer-effective-train-v3",
)
PARTITION_SALT = "momworld-relu-guard-effective-train-v4"
CALIBRATION_FRACTION = 0.80
MINIMUM_HOLDOUT_GAIN = {
    "navsim_v1_navtest": 0.0010,
    "navsim_v2_navtest": 0.0015,
    "navsim_v2_navhard": 0.0015,
}
WEIGHT_STEPS = (0.25, 0.5, 1.0, 2.0)
MAX_ACTIVE_TERMS = 3
PENALTY_NAMES = (
    "collision_risk",
    "no_collision",
    "drivable_area",
    "ttc",
    "progress",
    "direction",
    "lane",
    "traffic_light",
    "kinematic",
    "momentum",
)
RELEVANT_SOURCE_FILES = (
    "navsim/agents/momworld/momworld_model.py",
    "scripts/training/tune_momworld_relu_guard_from_cache.py",
    "scripts/training/train_momworld_rule_scorer_from_cache.py",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _token_sha256(records: Sequence[CacheRecord]) -> str:
    return hashlib.sha256(
        "\n".join(sorted(record.token for record in records)).encode("utf-8")
    ).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _source_audit(repository: Path) -> Dict[str, Any]:
    commit = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise RuntimeError("Invalid source commit")
    status = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain", "--", *RELEVANT_SOURCE_FILES],
        text=True,
    ).splitlines()
    if status:
        raise RuntimeError(f"Relevant ReLU guard sources are dirty: {status}")
    files: Dict[str, Any] = {}
    for relative in RELEVANT_SOURCE_FILES:
        path = (repository / relative).resolve()
        committed = subprocess.check_output(
            ["git", "-C", str(repository), "show", f"{commit}:{relative}"]
        )
        disk_sha = _sha256(path)
        if disk_sha != hashlib.sha256(committed).hexdigest():
            raise RuntimeError(f"Source differs from committed blob: {relative}")
        files[relative] = {"path": str(path), "sha256": disk_sha}
    return {"commit": commit, "working_tree_clean": True, "files": files}


def partition_fresh_relu_reserve(
    records: Sequence[CacheRecord],
) -> Tuple[List[CacheRecord], List[CacheRecord], Dict[str, Any]]:
    """Exclude all earlier exposed subsets, then make calibration/holdout."""

    if len(records) < 20:
        raise ValueError("Need at least twenty effective NAVTRAIN records")
    eligible = list(records)
    exposed: set[str] = set()
    for salt in PREVIOUS_PARTITION_SALTS:
        keyed = sorted(
            (
                hashlib.sha256(f"{salt}\0{record.token}".encode()).hexdigest(),
                record.token,
                record,
            )
            for record in eligible
        )
        cutoff = int(len(keyed) * 0.80)
        exposed.update(row[2].token for row in keyed[cutoff:])
        eligible = [row[2] for row in keyed[:cutoff]]
    keyed = sorted(
        (
            hashlib.sha256(f"{PARTITION_SALT}\0{record.token}".encode()).hexdigest(),
            record.token,
            record,
        )
        for record in eligible
    )
    calibration_end = int(len(keyed) * CALIBRATION_FRACTION)
    calibration = [row[2] for row in keyed[:calibration_end]]
    holdout = [row[2] for row in keyed[calibration_end:]]
    cal_tokens = {record.token for record in calibration}
    hold_tokens = {record.token for record in holdout}
    if not calibration or not holdout or cal_tokens & hold_tokens:
        raise RuntimeError("Invalid fresh ReLU calibration/holdout partition")
    if cal_tokens | hold_tokens != {record.token for record in eligible}:
        raise RuntimeError("Fresh ReLU partition coverage mismatch")
    if ({record.token for record in records} - cal_tokens - hold_tokens) != exposed:
        raise RuntimeError("Earlier exposed subset exclusion mismatch")
    return calibration, holdout, {
        "method": "nested-prior-fit-intersection-then-sha256-sort-fixed-count",
        "previous_partition_salts": list(PREVIOUS_PARTITION_SALTS),
        "salt": PARTITION_SALT,
        "source": "effective NAVTRAIN train only",
        "historical_navtrain_val_used": False,
        "previous_exposed_samples_excluded": len(exposed),
        "eligible_samples": len(eligible),
        "calibration_samples": len(calibration),
        "holdout_samples": len(holdout),
        "calibration_tokens_sha256": _token_sha256(calibration),
        "holdout_tokens_sha256": _token_sha256(holdout),
    }


def _arrays(records: Sequence[CacheRecord]) -> Dict[str, np.ndarray]:
    return {
        name: np.stack([getattr(record, name) for record in records])
        for name in (
            "features",
            "target_v1",
            "target_v2",
            "target_collision",
            "predicted_collision",
            "kinematic_penalty",
            "momentum_error",
            "normalized_base",
            "candidate_is_finite",
        )
    }


def _effective_mask(finite: np.ndarray) -> np.ndarray:
    finite = np.asarray(finite, dtype=np.bool_)
    return np.where(finite.any(axis=1, keepdims=True), finite, np.ones_like(finite))


def _sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(value, dtype=np.float32), -80.0, 80.0)
    return (1.0 / (1.0 + np.exp(-value))).astype(np.float32, copy=False)


def _load_rule_scorer_logits(
    state_path: Path,
    records: Sequence[CacheRecord],
    cache_audit: Mapping[str, Any],
    batch_size: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    state_path = state_path.expanduser().resolve()
    try:
        payload = torch.load(state_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(state_path, map_location="cpu")
    if not isinstance(payload, Mapping) or payload.get("schema") != STATE_SCHEMA:
        raise RuntimeError("Invalid frozen RuleScorer state")
    metadata = payload.get("metadata")
    state_dict = payload.get("state_dict")
    if not isinstance(metadata, Mapping) or not isinstance(state_dict, Mapping):
        raise RuntimeError("RuleScorer state is incomplete")
    state_cache = metadata.get("cache_audit")
    if not isinstance(state_cache, Mapping) or not (
        state_cache.get("source_split") == "navtrain"
        and state_cache.get("label_scope") == "navtrain-only"
        and state_cache.get("checkpoint_sha256") == cache_audit.get("checkpoint_sha256")
        and int(state_cache.get("rule_feature_dim", -1)) == int(cache_audit["rule_feature_dim"])
        and int(state_cache.get("rule_candidate_topk", -1)) == int(cache_audit["rule_candidate_topk"])
    ):
        raise RuntimeError("Frozen RuleScorer/cache provenance mismatch")
    model = RuleAwareTrajectoryScorer(
        SimpleNamespace(rule_scorer_hidden_dim=int(cache_audit["rule_scorer_hidden_dim"]))
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    features = np.stack([record.features for record in records]).astype(np.float32)
    chunks: List[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            chunks.append(model(torch.from_numpy(features[start : start + batch_size]), "v2").numpy())
    return np.concatenate(chunks), {
        "path": str(state_path),
        "sha256": _sha256(state_path),
        "schema": STATE_SCHEMA,
        "source_commit": str(metadata.get("source_commit", "")),
        "source_split": "navtrain",
        "label_scope": "navtrain-only",
        "forward_calls": len(chunks),
    }


def relu_guard_penalties(arrays: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Return semantically fixed, per-scene standardized ReLU violations."""

    features = np.asarray(arrays["features"], dtype=np.float32)
    raw = {
        "collision_risk": np.maximum(np.asarray(arrays["predicted_collision"], dtype=np.float32) - 0.50, 0.0),
        "no_collision": np.maximum(0.50 - features[..., 0], 0.0),
        "drivable_area": np.maximum(0.50 - features[..., 1], 0.0),
        "ttc": np.maximum(0.50 - features[..., 2], 0.0),
        "progress": np.maximum(0.20 - features[..., 3], 0.0),
        "direction": np.maximum(0.50 - features[..., 4], 0.0),
        "lane": np.maximum(0.50 - features[..., 5], 0.0),
        "traffic_light": np.maximum(0.50 - features[..., 6], 0.0),
        "kinematic": np.maximum(np.asarray(arrays["kinematic_penalty"], dtype=np.float32), 0.0),
        "momentum": np.maximum(np.asarray(arrays["momentum_error"], dtype=np.float32) - 0.10, 0.0),
    }
    return {name: _zscore_numpy(value) for name, value in raw.items()}


def _reference_scores(
    arrays: Mapping[str, np.ndarray],
    benchmark: str,
    v2_logits: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, str]:
    base = np.asarray(arrays["normalized_base"], dtype=np.float32)
    finite = _effective_mask(arrays["candidate_is_finite"])
    if benchmark == "navsim_v1_navtest":
        score = base + 1.6 * _zscore_numpy(_protocol_proxy_numpy(arrays["features"], "v1"))
        eligible = finite
        description = "ProtocolProxy: normalized_base + 1.6*zscore(v1_proxy)"
    elif benchmark == "navsim_v2_navhard":
        score = base + 3.2 * _zscore_numpy(_protocol_proxy_numpy(arrays["features"], "v2"))
        eligible = finite
        description = "ProtocolProxy: normalized_base + 3.2*zscore(v2_proxy)"
    elif benchmark == "navsim_v2_navtest":
        score = (
            base
            + _sigmoid(v2_logits)
            - 2.0 * np.asarray(arrays["predicted_collision"], dtype=np.float32)
            - 0.1 * np.asarray(arrays["momentum_error"], dtype=np.float32)
        )
        safe = (np.asarray(arrays["predicted_collision"]) < 0.8) & finite
        eligible = np.where(safe.any(axis=1, keepdims=True), safe, finite)
        description = "RuleScorer v2: base + sigmoid(logit) - 2*collision - 0.1*momentum; threshold=0.8"
    else:
        raise ValueError(f"Unsupported benchmark: {benchmark}")
    score = np.nan_to_num(score, nan=-1e9, posinf=1e9, neginf=-1e9)
    # Preserve the reference selector's hard eligibility mask when adding
    # ReLU terms.  In particular, the v2 RuleScorer candidate must never
    # reintroduce a collision-filtered trajectory merely because its penalty
    # is small.
    score = np.where(eligible, score, np.float32(-1e9))
    selected = score.argmax(axis=1)
    return score, selected.astype(np.int64), description


def _select(
    reference_score: np.ndarray,
    finite: np.ndarray,
    penalties: Mapping[str, np.ndarray],
    weights: Mapping[str, float],
) -> np.ndarray:
    score = np.asarray(reference_score, dtype=np.float32).copy()
    for name, weight in weights.items():
        if float(weight) != 0.0:
            score -= float(weight) * np.asarray(penalties[name], dtype=np.float32)
    effective = _effective_mask(finite)
    return np.where(effective, score, np.finfo(np.float32).min).argmax(axis=1).astype(np.int64)


def _metrics(
    arrays: Mapping[str, np.ndarray],
    benchmark: str,
    selected: np.ndarray,
    reference: np.ndarray,
) -> Dict[str, Any]:
    protocol = "v1" if benchmark == "navsim_v1_navtest" else "v2"
    rows = np.arange(len(selected))
    target = np.asarray(arrays[f"target_{protocol}"], dtype=np.float32)
    collision = np.asarray(arrays["target_collision"], dtype=np.float32)
    chosen = target[rows, selected]
    baseline = target[rows, reference]
    chosen_collision = collision[rows, selected]
    difference = chosen - baseline
    standard_error = float(difference.std(ddof=1) / math.sqrt(len(difference))) if len(difference) > 1 else 0.0
    return {
        "samples": len(selected),
        "mean_target_score": float(chosen.mean()),
        "reference_mean_target_score": float(baseline.mean()),
        "score_gain": float(difference.mean()),
        "score_gain_standard_error": standard_error,
        "score_gain_95ci_low": float(difference.mean() - 1.96 * standard_error),
        "mean_target_collision_risk": float(chosen_collision.mean()),
        "target_collision_rate_ge_0_5": float((chosen_collision >= 0.5).mean()),
        "changed_vs_reference_rate": float((selected != reference).mean()),
    }


def tune_relu_guard(
    arrays: Mapping[str, np.ndarray],
    benchmark: str,
    v2_logits: np.ndarray,
) -> Dict[str, Any]:
    penalties = relu_guard_penalties(arrays)
    reference_score, reference, description = _reference_scores(arrays, benchmark, v2_logits)
    reference_metrics = _metrics(arrays, benchmark, reference, reference)
    weights = {name: 0.0 for name in PENALTY_NAMES}
    trace: List[Dict[str, Any]] = []
    current = reference_metrics
    for step in range(MAX_ACTIVE_TERMS):
        candidates = []
        for name in PENALTY_NAMES:
            if weights[name] != 0.0:
                continue
            for weight in WEIGHT_STEPS:
                candidate_weights = {**weights, name: float(weight)}
                selected = _select(reference_score, arrays["candidate_is_finite"], penalties, candidate_weights)
                metrics = _metrics(arrays, benchmark, selected, reference)
                if metrics["target_collision_rate_ge_0_5"] <= reference_metrics["target_collision_rate_ge_0_5"] + 1e-12:
                    key = (
                        metrics["mean_target_score"],
                        -metrics["target_collision_rate_ge_0_5"],
                        -metrics["mean_target_collision_risk"],
                        -metrics["changed_vs_reference_rate"],
                        -weight,
                    )
                    candidates.append((key, name, weight, candidate_weights, metrics))
        if not candidates:
            break
        _, name, weight, candidate_weights, metrics = max(candidates, key=lambda row: row[0])
        if metrics["mean_target_score"] <= current["mean_target_score"] + 1e-12:
            break
        weights = candidate_weights
        current = metrics
        trace.append({"step": step + 1, "added_term": name, "weight": weight, "metrics": metrics})
    selected = _select(reference_score, arrays["candidate_is_finite"], penalties, weights)
    return {
        "benchmark": benchmark,
        "reference": description,
        "weights": weights,
        "active_terms": {name: weight for name, weight in weights.items() if weight != 0.0},
        "fixed_relu_thresholds": {
            "collision_risk_above": 0.50,
            "no_collision_below": 0.50,
            "drivable_area_below": 0.50,
            "ttc_below": 0.50,
            "progress_below": 0.20,
            "direction_below": 0.50,
            "lane_below": 0.50,
            "traffic_light_below": 0.50,
            "kinematic_above": 0.0,
            "momentum_above": 0.10,
        },
        "selection_trace": trace,
        "calibration": current,
        "reference_indices": reference,
        "selected_indices": selected,
    }


def evaluate_frozen_relu_guard(
    arrays: Mapping[str, np.ndarray],
    benchmark: str,
    v2_logits: np.ndarray,
    weights: Mapping[str, float],
) -> Dict[str, Any]:
    penalties = relu_guard_penalties(arrays)
    reference_score, reference, description = _reference_scores(arrays, benchmark, v2_logits)
    selected = _select(reference_score, arrays["candidate_is_finite"], penalties, weights)
    return {
        "benchmark": benchmark,
        "reference": description,
        "weights": dict(weights),
        "holdout": _metrics(arrays, benchmark, selected, reference),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, action="append", required=True)
    parser.add_argument("--scorer-state", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inference-batch-size", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.inference_batch_size < 1:
        raise ValueError("inference batch size must be positive")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "relu_guard_tuning.json"
    subset_path = output / "fixed_subset_tokens.jsonl"
    gate_path = output / "fixed_subset_gate_result.json"
    if any(path.exists() for path in (report_path, subset_path, gate_path)):
        raise FileExistsError("Refusing to overwrite ReLU guard artifacts")
    repository = Path(__file__).resolve().parents[2]
    source = _source_audit(repository)
    cache_roots = [path.expanduser().resolve() for path in args.cache_root]
    cache_audit = _validate_manifests(_discover_manifests(cache_roots), False)
    records = _load_records(_discover_npz(cache_roots), str(cache_audit["checkpoint_sha256"]))
    records, overlap_audit = _exclude_validation_overlap(records)
    calibration, holdout, partition = partition_fresh_relu_reserve(records["train"])
    val_tokens = {record.token for record in records["val"]}
    if val_tokens & {record.token for record in calibration + holdout}:
        raise RuntimeError("Historical NAVTRAIN val overlaps fresh ReLU reserve")
    _atomic_jsonl(
        subset_path,
        (
            {"token": record.token, "dataset": record.dataset, "role": "fixed_subset"}
            for record in sorted(holdout, key=lambda item: item.token)
        ),
    )
    partition["fixed_subset_artifact"] = {
        "path": str(subset_path), "sha256": _sha256(subset_path), "samples": len(holdout)
    }
    combined = calibration + holdout
    logits, scorer_audit = _load_rule_scorer_logits(
        args.scorer_state, combined, cache_audit, args.inference_batch_size
    )
    cal_arrays = _arrays(calibration)
    hold_arrays = _arrays(holdout)
    cal_logits = logits[: len(calibration)]
    hold_logits = logits[len(calibration) :]
    benchmarks = ("navsim_v1_navtest", "navsim_v2_navtest", "navsim_v2_navhard")
    calibrated: Dict[str, Any] = {}
    holdout_results: Dict[str, Any] = {}
    for benchmark in benchmarks:
        result = tune_relu_guard(cal_arrays, benchmark, cal_logits)
        result["reference_indices"] = "omitted"
        result["selected_indices"] = "omitted"
        calibrated[benchmark] = result
        holdout_results[benchmark] = evaluate_frozen_relu_guard(
            hold_arrays, benchmark, hold_logits, result["weights"]
        )
    report = {
        "schema": REPORT_SCHEMA,
        "created_at": _utc_now(),
        "source_audit": source,
        "cache_audit": {**cache_audit, "split_partition": overlap_audit},
        "scorer_audit": scorer_audit,
        "partition": partition,
        "data_scope": {
            "source_split": "navtrain",
            "label_scope": "navtrain-only",
            "historical_navtrain_val_used": False,
            "evaluator_calls": 0,
            "benchmark_cache_reads": 0,
            "excluded_splits": ["navtest", "navhard"],
        },
        "search": {
            "method": "collision-constrained greedy forward selection",
            "weight_steps": list(WEIGHT_STEPS),
            "maximum_active_terms": MAX_ACTIVE_TERMS,
            "thresholds_fixed_before_calibration": True,
        },
        "calibration": calibrated,
        "holdout_access_count": 1,
        "selection_frozen_before_holdout": True,
        "holdout": holdout_results,
        "reproduction_command": [sys.executable, *sys.argv],
    }
    _atomic_json(report_path, report)
    checks: Dict[str, Any] = {}
    allowed: List[str] = []
    for benchmark in benchmarks:
        metrics = holdout_results[benchmark]["holdout"]
        reference_collision_rate = _metrics(
            hold_arrays,
            benchmark,
            _reference_scores(hold_arrays, benchmark, hold_logits)[1],
            _reference_scores(hold_arrays, benchmark, hold_logits)[1],
        )["target_collision_rate_ge_0_5"]
        check = {
            "score_gain": metrics["score_gain"],
            "score_gain_95ci_low": metrics["score_gain_95ci_low"],
            "minimum_score_gain": MINIMUM_HOLDOUT_GAIN[benchmark],
            "score_pass": metrics["score_gain"] >= MINIMUM_HOLDOUT_GAIN[benchmark],
            "collision_rate_delta": metrics["target_collision_rate_ge_0_5"] - reference_collision_rate,
        }
        check["collision_pass"] = check["collision_rate_delta"] <= 0.0
        check["passed"] = check["score_pass"] and check["collision_pass"]
        checks[benchmark] = check
        if check["passed"]:
            allowed.append(benchmark)
    gate = {
        "schema": GATE_SCHEMA,
        "created_at": _utc_now(),
        "checks": checks,
        "allowed_benchmarks": allowed,
        "full_evaluation_allowed": bool(allowed),
        "full_evaluation_launched": False,
        "report": {"path": str(report_path), "sha256": _sha256(report_path)},
        "fixed_subset": partition["fixed_subset_artifact"],
    }
    _atomic_json(gate_path, gate)
    print(json.dumps(gate, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
