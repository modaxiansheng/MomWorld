#!/usr/bin/env python3
"""Cross-fit a frozen V2 NavTest selector gate on NAVTRAIN only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch

from navsim.agents.momworld.momworld_model import RuleAwareTrajectoryScorer
from scripts.training.crossfit_momworld_gtrs_fusion_all_navtrain import deterministic_fold
from scripts.training.refit_momworld_gtrs_scorer_all_navtrain import _sha256, combine_all_unique_navtrain
from scripts.training.train_momworld_rule_scorer_from_cache import (
    _discover_manifests, _discover_npz, _load_records, _validate_manifests,
)
from scripts.training.train_momworld_v1_protocol_gate_all_navtrain import (
    _atomic_json, _atomic_state, fit_ridge, predict_ridge,
)
from scripts.training.tune_momworld_candidate_safety_rule_all_navtrain import (
    apply_safety_rule, neutral_rule,
)
from scripts.training.tune_momworld_protocol_proxy_from_cache import _candidate_zscore, _protocol_proxy
from scripts.training.tune_momworld_relu_guard_from_cache import _arrays


REPORT_SCHEMA = "momworld-v2-navtest-selector-gate-all-navtrain-v1"
STATE_SCHEMA = "momworld-v2-navtest-selector-gate-state-v1"
MIN_PREDICTED_GAIN = 0.002
MAX_PREDICTED_COLLISION_DELTA = 0.002
CONFIGURATIONS = (
    (0.5, 2.0, 0.1, 0.0, 0.0),
    (1.5, 2.0, 0.1, 0.0, 0.0),
    (1.0, 1.0, 0.1, 0.0, 0.0),
    (1.0, 3.0, 0.1, 0.0, 0.0),
    (1.0, 2.0, 0.0, 0.0, 0.0),
    (1.0, 2.0, 0.1, 0.4, 0.0),
    (1.0, 2.0, 0.1, 0.8, 0.0),
    (0.0, 0.0, 0.0, 1.6, 0.0),
    (0.0, 0.0, 0.0, 2.4, 0.0),
)
RELEVANT_SOURCE_FILES = (
    "navsim/agents/momworld/momworld_model.py",
    "scripts/training/train_momworld_v1_protocol_gate_all_navtrain.py",
    "scripts/training/train_momworld_v2_navtest_selector_gate_all_navtrain.py",
    "scripts/training/tune_momworld_candidate_safety_rule_all_navtrain.py",
    "scripts/training/tune_momworld_protocol_proxy_from_cache.py",
)


def selector_features(
    rule_features: np.ndarray,
    normalized_base: np.ndarray,
    proxy_zscore: np.ndarray,
    learned_score: np.ndarray,
    finite: np.ndarray,
    incumbent: np.ndarray,
    proposals: np.ndarray,
    configurations: np.ndarray,
) -> np.ndarray:
    values = np.asarray(rule_features, dtype=np.float32)
    base = np.asarray(normalized_base, dtype=np.float32)
    proxy = np.asarray(proxy_zscore, dtype=np.float32)
    learned = np.asarray(learned_score, dtype=np.float32)
    valid = np.asarray(finite, dtype=np.bool_)
    reference = np.asarray(incumbent, dtype=np.int64)
    candidates = np.asarray(proposals, dtype=np.int64)
    configs = np.asarray(configurations, dtype=np.float32)
    if values.ndim != 3 or values.shape[-1] != 10:
        raise ValueError("rule features must be [scenes,candidates,10]")
    if any(array.shape != values.shape[:2] for array in (base, proxy, learned, valid)):
        raise ValueError("candidate input shapes differ")
    if candidates.shape != (len(values), len(configs)) or configs.shape[1:] != (5,):
        raise ValueError("proposal/configuration shapes differ")
    rows = np.arange(len(values))
    safe_values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    masked = np.where(valid[..., None], safe_values, np.nan)
    scene_mean = np.nan_to_num(np.nanmean(masked, axis=1), nan=0.0)
    scene_std = np.nan_to_num(np.nanstd(masked, axis=1), nan=0.0)
    scene_span = np.nan_to_num(np.nanmax(masked, axis=1) - np.nanmin(masked, axis=1), nan=0.0)
    incumbent_features = safe_values[rows, reference]
    blocks = []
    for column, configuration in enumerate(configs):
        proposal = candidates[:, column]
        difference = safe_values[rows, proposal] - incumbent_features
        encoding = np.tile(configuration, (len(values), 1))
        blocks.append(
            np.concatenate(
                (
                    difference,
                    (base[rows, proposal] - base[rows, reference])[:, None],
                    (proxy[rows, proposal] - proxy[rows, reference])[:, None],
                    (learned[rows, proposal] - learned[rows, reference])[:, None],
                    encoding,
                    difference * float(configuration[3]),
                    incumbent_features,
                    scene_mean,
                    scene_std,
                    scene_span,
                ), axis=1,
            ).astype(np.float32)
        )
    result = np.stack(blocks, axis=1)
    if result.shape != (len(values), len(configs), 68) or not np.isfinite(result).all():
        raise RuntimeError(f"invalid V2 selector feature matrix: {result.shape}")
    return result


def select_proposal(gain, collision, proposals, incumbent):
    eligible = np.asarray(collision) <= MAX_PREDICTED_COLLISION_DELTA
    score = np.where(eligible, gain, -1e9)
    rows = np.arange(len(score))
    best = score.argmax(axis=1)
    switch = eligible[rows, best] & (score[rows, best] >= MIN_PREDICTED_GAIN)
    return np.where(switch, proposals[rows, best], incumbent).astype(np.int64)


def _source_audit(repository: Path) -> Dict[str, Any]:
    commit = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain", "--", *RELEVANT_SOURCE_FILES], text=True
    ).splitlines()
    if status:
        raise RuntimeError(f"V2 selector sources are dirty: {status}")
    files = {}
    for relative in RELEVANT_SOURCE_FILES:
        path = repository / relative
        blob = subprocess.check_output(["git", "-C", str(repository), "show", f"{commit}:{relative}"])
        disk_sha = _sha256(path)
        if disk_sha != hashlib.sha256(blob).hexdigest():
            raise RuntimeError(f"source differs from commit: {relative}")
        files[relative] = {"path": str(path.resolve()), "sha256": disk_sha}
    return {"commit": commit, "working_tree_clean": True, "files": files}


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--scorer-state", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-state", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_report.exists() or args.output_state.exists():
        raise FileExistsError("V2 selector output exists")
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
    predicted_collision = np.asarray(arrays["predicted_collision"], dtype=np.float32)
    momentum = np.asarray(arrays["momentum_error"], dtype=np.float32)
    proxy = _candidate_zscore(_protocol_proxy(features, "v2"), 1e-4)
    scorer_payload = torch.load(args.scorer_state.expanduser().resolve(), map_location="cpu")
    scorer = RuleAwareTrajectoryScorer(SimpleNamespace(rule_scorer_hidden_dim=64, rule_scorer_input_mode="raw"))
    scorer.load_state_dict(scorer_payload["state_dict"])
    scorer.eval()
    with torch.inference_mode():
        logits = np.concatenate([scorer(torch.from_numpy(features[offset:offset + 8192]), "v2").numpy() for offset in range(0, len(features), 8192)])
    learned = (1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))).astype(np.float32)
    parameters = neutral_rule()
    parameters.update(ttc_min=0.55, lane_min=0.25)
    eligible = apply_safety_rule(features, finite, parameters)[0]
    incumbent_score = base + learned - 2.0 * predicted_collision - 0.1 * momentum
    incumbent = np.where(eligible, incumbent_score, -1e9).argmax(axis=1)
    configurations = np.asarray(CONFIGURATIONS, dtype=np.float32)
    proposals = np.stack(
        [
            np.where(
                eligible,
                base + config[0] * learned - config[1] * predicted_collision - config[2] * momentum + config[3] * proxy,
                -1e9,
            ).argmax(axis=1)
            for config in configurations
        ], axis=1,
    )
    values = selector_features(features, base, proxy, learned, finite, incumbent, proposals, configurations)
    rows = np.arange(len(records))
    target = np.asarray(arrays["target_v2"], dtype=np.float32)
    collision_target = np.asarray(arrays["target_collision"], dtype=np.float32)
    gain_target = target[rows[:, None], proposals] - target[rows, incumbent][:, None]
    collision_delta_target = collision_target[rows[:, None], proposals] - collision_target[rows, incumbent][:, None]
    folds = np.asarray([deterministic_fold(record.token, 5) for record in records])
    oof_gain = np.full_like(gain_target, np.nan)
    oof_collision = np.full_like(collision_delta_target, np.nan)
    fold_models = []
    for fold in range(5):
        train = folds != fold
        holdout = ~train
        train_values = values[train].reshape(-1, values.shape[-1])
        gain_state = fit_ridge(train_values, gain_target[train].reshape(-1))
        collision_state = fit_ridge(train_values, collision_delta_target[train].reshape(-1))
        holdout_values = values[holdout].reshape(-1, values.shape[-1])
        oof_gain[holdout] = predict_ridge(gain_state, holdout_values).reshape(-1, len(configurations))
        oof_collision[holdout] = predict_ridge(collision_state, holdout_values).reshape(-1, len(configurations))
        fold_models.append({"fold": fold, "train": int(train.sum()), "holdout": int(holdout.sum())})
    selected = select_proposal(oof_gain, oof_collision, proposals, incumbent)
    delta = target[rows, selected] - target[rows, incumbent]
    collision_delta = collision_target[rows, selected] - collision_target[rows, incumbent]
    fold_gains = [float(delta[folds == fold].mean()) for fold in range(5)]
    standard_error = float(delta.std(ddof=1) / math.sqrt(len(delta)))
    metrics = {
        "gain_vs_incumbent": float(delta.mean()),
        "gain_95ci_low": float(delta.mean() - 1.96 * standard_error),
        "minimum_fold_gain": min(fold_gains),
        "fold_gains": fold_gains,
        "mean_collision_delta": float(collision_delta.mean()),
        "collision_rate_delta": float((collision_target[rows, selected] >= 0.5).mean() - (collision_target[rows, incumbent] >= 0.5).mean()),
        "switch_rate": float((selected != incumbent).mean()),
    }
    passed = bool(
        metrics["gain_vs_incumbent"] >= 5e-5
        and metrics["gain_95ci_low"] > 0.0
        and metrics["minimum_fold_gain"] >= 0.0
        and metrics["mean_collision_delta"] <= 0.0002
        and metrics["collision_rate_delta"] <= 0.00025
    )
    state_audit = None
    if passed:
        all_values = values.reshape(-1, values.shape[-1])
        gain_state = fit_ridge(all_values, gain_target.reshape(-1))
        collision_state = fit_ridge(all_values, collision_delta_target.reshape(-1))
        args.output_state.parent.mkdir(parents=True, exist_ok=True)
        _atomic_state(args.output_state, {
            "schema": STATE_SCHEMA,
            "source_commit": source["commit"],
            "feature_dim": 68,
            "configurations": CONFIGURATIONS,
            "min_predicted_gain": MIN_PREDICTED_GAIN,
            "max_predicted_collision_delta": MAX_PREDICTED_COLLISION_DELTA,
            "scorer_state_sha256": _sha256(args.scorer_state),
            "gain_model": gain_state,
            "collision_model": collision_state,
        })
        state_audit = {"path": str(args.output_state.resolve()), "sha256": _sha256(args.output_state), "size": args.output_state.stat().st_size}
    report = {
        "schema": REPORT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_audit": source,
        "data_scope": {"source_split": "navtrain", "samples": len(records), "partition": partition, "evaluator_calls": 0, "benchmark_cache_reads": 0, "excluded_splits": ["navtest", "navhard"]},
        "cache_audit": cache_audit,
        "benchmark": "navsim_v2_navtest",
        "incumbent": {"score": "base + sigmoid(gtrs_v2) - 2*collision - 0.1*momentum", "rule": parameters},
        "configurations": CONFIGURATIONS,
        "selection": {"min_predicted_gain": MIN_PREDICTED_GAIN, "max_predicted_collision_delta": MAX_PREDICTED_COLLISION_DELTA},
        "folds": fold_models,
        "gate": {"passed": passed, "bounded_collision_regression": {"mean_max": 0.0002, "rate_max": 0.00025}, "metrics": metrics},
        "state": state_audit,
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.output_report, report)
    print(json.dumps({"passed": passed, "metrics": metrics, "state": state_audit}, sort_keys=True))


if __name__ == "__main__":
    main()
