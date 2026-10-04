#!/usr/bin/env python3
"""Cross-fit a conservative V1 ProtocolProxy selector gate on NAVTRAIN only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from scripts.training.crossfit_momworld_gtrs_fusion_all_navtrain import deterministic_fold
from scripts.training.refit_momworld_gtrs_scorer_all_navtrain import (
    _sha256,
    combine_all_unique_navtrain,
)
from scripts.training.train_momworld_rule_scorer_from_cache import (
    _discover_manifests,
    _discover_npz,
    _load_records,
    _validate_manifests,
)
from scripts.training.tune_momworld_protocol_proxy_from_cache import (
    _candidate_zscore,
    _protocol_proxy,
)
from scripts.training.tune_momworld_relu_guard_from_cache import _arrays


REPORT_SCHEMA = "momworld-v1-protocol-selector-gate-all-navtrain-v1"
STATE_SCHEMA = "momworld-v1-protocol-selector-gate-state-v1"
INCUMBENT_WEIGHT = 1.6
ALTERNATIVE_WEIGHTS = (0.0, 0.4, 0.8, 1.2, 2.0, 2.4, 2.8, 3.2)
RIDGE_ALPHA = 100.0
MIN_PREDICTED_GAIN = 0.002
MAX_PREDICTED_COLLISION_DELTA = 0.0
RELEVANT_SOURCE_FILES = (
    "scripts/training/crossfit_momworld_gtrs_fusion_all_navtrain.py",
    "scripts/training/refit_momworld_gtrs_scorer_all_navtrain.py",
    "scripts/training/train_momworld_rule_scorer_from_cache.py",
    "scripts/training/train_momworld_v1_protocol_gate_all_navtrain.py",
    "scripts/training/tune_momworld_protocol_proxy_from_cache.py",
    "scripts/training/tune_momworld_relu_guard_from_cache.py",
)


def protocol_gate_features(
    rule_features: np.ndarray,
    normalized_base: np.ndarray,
    proxy_zscore: np.ndarray,
    incumbent: np.ndarray,
    proposals: np.ndarray,
    weights: np.ndarray,
    finite: np.ndarray,
) -> np.ndarray:
    """Build online-reproducible proposal/incumbent and scene summaries."""

    values = np.asarray(rule_features, dtype=np.float32)
    base = np.asarray(normalized_base, dtype=np.float32)
    proxy = np.asarray(proxy_zscore, dtype=np.float32)
    reference = np.asarray(incumbent, dtype=np.int64)
    candidates = np.asarray(proposals, dtype=np.int64)
    weight_values = np.asarray(weights, dtype=np.float32)
    valid = np.asarray(finite, dtype=np.bool_)
    if values.ndim != 3 or values.shape[-1] != 10:
        raise ValueError("rule_features must have shape [scenes,candidates,10]")
    if base.shape != values.shape[:2] or proxy.shape != base.shape or valid.shape != base.shape:
        raise ValueError("candidate array shapes differ")
    if reference.shape != (len(values),):
        raise ValueError("incumbent shape differs")
    if candidates.shape != (len(values), len(weight_values)):
        raise ValueError("proposal/weight shapes differ")
    if np.any((reference < 0) | (reference >= values.shape[1])) or np.any(
        (candidates < 0) | (candidates >= values.shape[1])
    ):
        raise ValueError("candidate index is out of range")

    rows = np.arange(len(values))
    safe_values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    masked = np.where(valid[..., None], safe_values, np.nan)
    scene_mean = np.nan_to_num(np.nanmean(masked, axis=1), nan=0.0)
    scene_std = np.nan_to_num(np.nanstd(masked, axis=1), nan=0.0)
    scene_min = np.nan_to_num(np.nanmin(masked, axis=1), nan=0.0)
    scene_max = np.nan_to_num(np.nanmax(masked, axis=1), nan=0.0)
    scene_span = scene_max - scene_min
    incumbent_features = safe_values[rows, reference]
    incumbent_base = base[rows, reference]
    incumbent_proxy = proxy[rows, reference]
    result = []
    for column, weight in enumerate(weight_values):
        selected = candidates[:, column]
        difference = safe_values[rows, selected] - incumbent_features
        block = np.concatenate(
            (
                difference,
                (base[rows, selected] - incumbent_base)[:, None],
                (proxy[rows, selected] - incumbent_proxy)[:, None],
                np.full((len(values), 1), weight, dtype=np.float32),
                difference * weight,
                incumbent_features,
                scene_mean,
                scene_std,
                scene_span,
            ),
            axis=1,
        )
        result.append(block.astype(np.float32, copy=False))
    stacked = np.stack(result, axis=1)
    if stacked.shape != (len(values), len(weight_values), 63):
        raise RuntimeError(f"unexpected gate feature shape: {stacked.shape}")
    if not np.isfinite(stacked).all():
        raise FloatingPointError("non-finite protocol gate features")
    return stacked


def fit_ridge(values: np.ndarray, target: np.ndarray) -> Dict[str, np.ndarray | float]:
    scaler = StandardScaler().fit(values)
    model = Ridge(alpha=RIDGE_ALPHA, solver="lsqr", tol=1e-4).fit(
        scaler.transform(values), target
    )
    return {
        "mean": scaler.mean_.astype(np.float64),
        "scale": scaler.scale_.astype(np.float64),
        "coefficient": np.asarray(model.coef_, dtype=np.float64),
        "intercept": float(model.intercept_),
    }


def predict_ridge(state: Mapping[str, Any], values: np.ndarray) -> np.ndarray:
    mean = np.asarray(state["mean"], dtype=np.float64)
    scale = np.asarray(state["scale"], dtype=np.float64)
    coefficient = np.asarray(state["coefficient"], dtype=np.float64)
    prediction = ((np.asarray(values, dtype=np.float64) - mean) / scale) @ coefficient
    return (prediction + float(state["intercept"])).astype(np.float32)


def select_protocol_proposal(
    gain_prediction: np.ndarray,
    collision_prediction: np.ndarray,
    proposals: np.ndarray,
    incumbent: np.ndarray,
    *,
    min_predicted_gain: float = MIN_PREDICTED_GAIN,
    max_predicted_collision_delta: float = MAX_PREDICTED_COLLISION_DELTA,
) -> np.ndarray:
    gain = np.asarray(gain_prediction, dtype=np.float32)
    collision = np.asarray(collision_prediction, dtype=np.float32)
    candidates = np.asarray(proposals, dtype=np.int64)
    reference = np.asarray(incumbent, dtype=np.int64)
    if gain.ndim != 2 or collision.shape != gain.shape or candidates.shape != gain.shape:
        raise ValueError("gate prediction/proposal shapes differ")
    if reference.shape != (len(gain),):
        raise ValueError("incumbent shape differs")
    eligible = collision <= float(max_predicted_collision_delta)
    score = np.where(eligible, gain, np.finfo(np.float32).min)
    rows = np.arange(len(gain))
    best = score.argmax(axis=1)
    switch = eligible[rows, best] & (score[rows, best] >= float(min_predicted_gain))
    return np.where(switch, candidates[rows, best], reference).astype(np.int64)


def _source_audit(repository: Path) -> Dict[str, Any]:
    commit = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain", "--", *RELEVANT_SOURCE_FILES],
        text=True,
    ).splitlines()
    if status:
        raise RuntimeError(f"V1 protocol gate sources are dirty: {status}")
    files: Dict[str, Any] = {}
    for relative in RELEVANT_SOURCE_FILES:
        path = repository / relative
        blob = subprocess.check_output(["git", "-C", str(repository), "show", f"{commit}:{relative}"])
        disk_sha = _sha256(path)
        if disk_sha != hashlib.sha256(blob).hexdigest():
            raise RuntimeError(f"source differs from committed blob: {relative}")
        files[relative] = {"path": str(path.resolve()), "sha256": disk_sha}
    return {"commit": commit, "working_tree_clean": True, "files": files}


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()


def _atomic_state(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as stream:
        torch.save(dict(payload), stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()


def _gate_metrics(
    selected: np.ndarray,
    incumbent: np.ndarray,
    target: np.ndarray,
    collision: np.ndarray,
    folds: np.ndarray,
) -> Dict[str, Any]:
    rows = np.arange(len(selected))
    delta = target[rows, selected] - target[rows, incumbent]
    collision_delta = collision[rows, selected] - collision[rows, incumbent]
    fold_gains = [float(delta[folds == fold].mean()) for fold in range(5)]
    standard_error = float(delta.std(ddof=1) / math.sqrt(len(delta)))
    return {
        "samples": len(selected),
        "gain_vs_incumbent": float(delta.mean()),
        "gain_standard_error": standard_error,
        "gain_95ci_low": float(delta.mean() - 1.96 * standard_error),
        "fold_gains": fold_gains,
        "minimum_fold_gain": min(fold_gains),
        "mean_collision_delta": float(collision_delta.mean()),
        "collision_rate_delta": float(
            (collision[rows, selected] >= 0.5).mean()
            - (collision[rows, incumbent] >= 0.5).mean()
        ),
        "switch_rate": float((selected != incumbent).mean()),
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-state", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_report.exists() or args.output_state.exists():
        raise FileExistsError("V1 protocol gate output already exists")
    repository = Path(__file__).resolve().parents[2]
    source = _source_audit(repository)
    cache_root = args.cache_root.expanduser().resolve()
    cache_audit = _validate_manifests(_discover_manifests([cache_root]), False)
    raw = _load_records(_discover_npz([cache_root]), str(cache_audit["checkpoint_sha256"]))
    records, partition = combine_all_unique_navtrain(raw)
    arrays = _arrays(records)
    features = np.asarray(arrays["features"], dtype=np.float32)
    finite = np.asarray(arrays["candidate_is_finite"], dtype=np.bool_)
    base = np.asarray(arrays["normalized_base"], dtype=np.float32)
    proxy = _candidate_zscore(_protocol_proxy(features, "v1"), 1e-4)
    rows = np.arange(len(records))
    incumbent = np.where(finite, base + INCUMBENT_WEIGHT * proxy, -1e9).argmax(axis=1)
    weights = np.asarray(ALTERNATIVE_WEIGHTS, dtype=np.float32)
    proposals = np.stack(
        [np.where(finite, base + float(weight) * proxy, -1e9).argmax(axis=1) for weight in weights],
        axis=1,
    )
    gate_features = protocol_gate_features(
        features, base, proxy, incumbent, proposals, weights, finite
    )
    target = np.asarray(arrays["target_v1"], dtype=np.float32)
    collision = np.asarray(arrays["target_collision"], dtype=np.float32)
    gain_target = target[rows[:, None], proposals] - target[rows, incumbent][:, None]
    collision_target = collision[rows[:, None], proposals] - collision[rows, incumbent][:, None]
    folds = np.asarray([deterministic_fold(record.token, 5) for record in records])
    oof_gain = np.full(gain_target.shape, np.nan, dtype=np.float32)
    oof_collision = np.full(collision_target.shape, np.nan, dtype=np.float32)
    fold_models = []
    for fold in range(5):
        train = folds != fold
        holdout = ~train
        train_values = gate_features[train].reshape(-1, gate_features.shape[-1])
        gain_state = fit_ridge(train_values, gain_target[train].reshape(-1))
        collision_state = fit_ridge(train_values, collision_target[train].reshape(-1))
        holdout_values = gate_features[holdout].reshape(-1, gate_features.shape[-1])
        oof_gain[holdout] = predict_ridge(gain_state, holdout_values).reshape(-1, len(weights))
        oof_collision[holdout] = predict_ridge(collision_state, holdout_values).reshape(-1, len(weights))
        fold_models.append({"fold": fold, "train": int(train.sum()), "holdout": int(holdout.sum())})
    if not np.isfinite(oof_gain).all() or not np.isfinite(oof_collision).all():
        raise FloatingPointError("incomplete V1 protocol gate OOF predictions")
    selected = select_protocol_proposal(oof_gain, oof_collision, proposals, incumbent)
    metrics = _gate_metrics(selected, incumbent, target, collision, folds)
    passed = bool(
        metrics["gain_vs_incumbent"] >= 5e-5
        and metrics["gain_95ci_low"] > 0.0
        and metrics["minimum_fold_gain"] >= 0.0
        and metrics["mean_collision_delta"] <= 0.0
        and metrics["collision_rate_delta"] <= 0.0
        and metrics["switch_rate"] >= 0.001
    )
    state_audit = None
    if passed:
        all_values = gate_features.reshape(-1, gate_features.shape[-1])
        gain_state = fit_ridge(all_values, gain_target.reshape(-1))
        collision_state = fit_ridge(all_values, collision_target.reshape(-1))
        args.output_state.parent.mkdir(parents=True, exist_ok=True)
        _atomic_state(
            args.output_state,
            {
                "schema": STATE_SCHEMA,
                "source_commit": source["commit"],
                "feature_dim": 63,
                "incumbent_weight": INCUMBENT_WEIGHT,
                "alternative_weights": ALTERNATIVE_WEIGHTS,
                "ridge_alpha": RIDGE_ALPHA,
                "min_predicted_gain": MIN_PREDICTED_GAIN,
                "max_predicted_collision_delta": MAX_PREDICTED_COLLISION_DELTA,
                "gain_model": gain_state,
                "collision_model": collision_state,
            },
        )
        state_audit = {
            "path": str(args.output_state.resolve()),
            "sha256": _sha256(args.output_state),
            "size": args.output_state.stat().st_size,
        }
    report = {
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
        "benchmark": "navsim_v1_navtest",
        "incumbent": f"normalized_base + {INCUMBENT_WEIGHT} * zscore(v1 ProtocolProxy)",
        "alternative_weights": list(ALTERNATIVE_WEIGHTS),
        "features": "proposal-minus-incumbent raw10/base/proxy + weight interactions + incumbent raw10 + scene mean/std/span",
        "model": {"type": "standardized ridge", "alpha": RIDGE_ALPHA, "feature_dim": 63},
        "selection": {
            "min_predicted_gain": MIN_PREDICTED_GAIN,
            "max_predicted_collision_delta": MAX_PREDICTED_COLLISION_DELTA,
        },
        "folds": {
            "count": 5,
            "assignment": "sha256(token)[0:8] modulo 5",
            "counts": [int((folds == fold).sum()) for fold in range(5)],
            "models": fold_models,
        },
        "gate": {"passed": passed, "metrics": metrics},
        "state": state_audit,
        "arguments": {key: str(value.expanduser().resolve()) for key, value in vars(args).items()},
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.output_report, report)
    print(json.dumps({"passed": passed, "metrics": metrics, "state": state_audit}, sort_keys=True))


if __name__ == "__main__":
    main()
