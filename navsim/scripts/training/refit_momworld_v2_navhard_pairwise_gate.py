#!/usr/bin/env python3
"""Refit and export the conservative V2 NavHard pairwise selector."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor

from navsim.agents.momworld.momworld_model import (
    candidate_protocol_proxy,
    candidate_rule_safety_filter,
    candidate_score_zscore,
)


def fold_id(token: str, folds: int = 5) -> int:
    return int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16) % folds


def safety_config() -> SimpleNamespace:
    values = {
        "rule_safety_no_collision_min": 0.0,
        "rule_safety_drivable_area_min": 0.55,
        "rule_safety_ttc_min": 0.0,
        "rule_safety_progress_min": 0.435,
        "rule_safety_direction_min": 0.0,
        "rule_safety_lane_min": 0.55,
        "rule_safety_traffic_light_min": 0.0,
        "rule_safety_collision_risk_max": 1.000001,
        "rule_safety_kinematic_max": 1e6,
        "rule_safety_momentum_max": 1e6,
        "rule_safety_fallback_risk_slack": 0.0,
        "rule_relative_safety_filter_enabled": False,
    }
    for name in (
        "no_collision", "drivable_area", "ttc", "progress", "direction",
        "lane", "traffic_light", "collision_risk", "kinematic", "momentum",
    ):
        values[f"rule_relative_{name}_fraction"] = 1.0
    return SimpleNamespace(**values)


def gather(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return values[torch.arange(len(values))[:, None], indices]


def pair_features(
    features: torch.Tensor,
    normalized_base: torch.Tensor,
    proxy_zscore: torch.Tensor,
    combined_score: torch.Tensor,
    incumbent: torch.Tensor,
    alternatives: torch.Tensor,
) -> np.ndarray:
    rows = torch.arange(len(features))[:, None]
    inc = incumbent[:, None]
    inc_features = features[rows, inc].expand(-1, alternatives.shape[1], -1)
    alt_features = features[rows, alternatives]
    mean = features.mean(dim=1, keepdim=True).expand_as(alt_features)
    std = features.std(dim=1, keepdim=True, unbiased=False).expand_as(alt_features)

    def scalar_triplet(values: torch.Tensor) -> torch.Tensor:
        inc_value = gather(values, inc).expand(-1, alternatives.shape[1])
        alt_value = gather(values, alternatives)
        return torch.stack((inc_value, alt_value, alt_value - inc_value), dim=-1)

    result = torch.cat(
        (
            inc_features,
            alt_features,
            alt_features - inc_features,
            mean,
            std,
            scalar_triplet(normalized_base),
            scalar_triplet(proxy_zscore),
            scalar_triplet(combined_score),
        ),
        dim=-1,
    )
    return result.reshape(-1, result.shape[-1]).numpy().astype(np.float32, copy=False)


def export_model(model: HistGradientBoostingRegressor) -> dict:
    predictors = [iteration[0] for iteration in model._predictors]
    nodes = [predictor.nodes for predictor in predictors]
    max_nodes = max(len(value) for value in nodes)
    tree_count = len(nodes)

    def padded(field: str, fill: float | int | bool = 0) -> np.ndarray:
        dtype = nodes[0].dtype[field]
        output = np.full((tree_count, max_nodes), fill, dtype=dtype)
        for index, value in enumerate(nodes):
            output[index, : len(value)] = value[field]
        return output

    fields = {
        "feature_idx": torch.as_tensor(padded("feature_idx").astype(np.int64)),
        "threshold": torch.as_tensor(padded("num_threshold").astype(np.float64)),
        "missing_left": torch.as_tensor(
            padded("missing_go_to_left").astype(np.bool_)
        ),
        "left": torch.as_tensor(padded("left").astype(np.int64)),
        "right": torch.as_tensor(padded("right").astype(np.int64)),
        "is_leaf": torch.as_tensor(padded("is_leaf", True).astype(np.bool_)),
        "value": torch.as_tensor(padded("value").astype(np.float64)),
    }
    return {
        "baseline": float(np.asarray(model._baseline_prediction).reshape(-1)[0]),
        "tree_count": tree_count,
        "max_nodes": max_nodes,
        "max_depth": int(max(int(value["depth"].max()) for value in nodes)),
        **fields,
    }


def predict_exported(state: dict, features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    result = np.full(len(values), float(state["baseline"]), dtype=np.float64)
    for tree_index in range(int(state["tree_count"])):
        nodes = np.zeros(len(values), dtype=np.int64)
        for _ in range(int(state["max_depth"]) + 1):
            leaf = state["is_leaf"][tree_index, nodes].numpy()
            feature = state["feature_idx"][tree_index, nodes].numpy()
            observed = values[np.arange(len(values)), feature]
            go_left = np.where(
                np.isnan(observed),
                state["missing_left"][tree_index, nodes].numpy(),
                observed <= state["threshold"][tree_index, nodes].numpy(),
            )
            next_nodes = np.where(
                go_left,
                state["left"][tree_index, nodes].numpy(),
                state["right"][tree_index, nodes].numpy(),
            )
            nodes = np.where(leaf, nodes, next_nodes)
        result += state["value"][tree_index, nodes].numpy()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--calibration-fold",
        type=int,
        default=4,
        help="Held-out fold in [0,4], or -1 to fit all scenes and use a prediction quantile.",
    )
    parser.add_argument("--switch-fraction", type=float, default=0.005)
    parser.add_argument("--max-iter", type=int, default=120)
    parser.add_argument(
        "--alternatives",
        type=int,
        default=3,
        help="Number of eligible alternatives to compare against the incumbent.",
    )
    args = parser.parse_args()
    if args.calibration_fold not in (-1, 0, 1, 2, 3, 4):
        raise ValueError("calibration fold must be -1 or in [0,4]")
    if not 0.0 < args.switch_fraction < 0.1:
        raise ValueError("switch fraction must be in (0,0.1)")
    if not 1 <= args.alternatives <= 31:
        raise ValueError("alternatives must be in [1,31]")
    if args.state.exists() or args.report.exists():
        raise FileExistsError("state/report output already exists")

    payload = torch.load(args.cache, map_location="cpu", weights_only=False)
    arrays = payload["arrays"]
    features = arrays["features"].float()
    target = arrays["target_v2"].float()
    target_collision = arrays["target_collision"].float()
    normalized_base = arrays["normalized_base"].float()
    finite = arrays["candidate_is_finite"].bool()
    proxy_zscore = candidate_score_zscore(candidate_protocol_proxy(features, "v2"), 1e-4)
    combined = normalized_base + 3.2 * proxy_zscore
    eligible, _, _ = candidate_rule_safety_filter(features, finite, safety_config())
    order = combined.masked_fill(~eligible, -torch.inf).argsort(dim=1, descending=True)
    incumbent = order[:, 0]
    alternatives = order[:, 1 : 1 + args.alternatives]
    alternative_eligible = gather(eligible, alternatives)
    rows = torch.arange(len(target))[:, None]
    incumbent_target = gather(target, incumbent[:, None]).squeeze(1)
    incumbent_collision = gather(target_collision, incumbent[:, None]).squeeze(1)
    gains = target[rows, alternatives] - incumbent_target[:, None]
    collision_delta = target_collision[rows, alternatives] - incumbent_collision[:, None]
    x = pair_features(
        features, normalized_base, proxy_zscore, combined, incumbent, alternatives
    )
    y = gains.reshape(-1).numpy()
    scene_folds = np.asarray([fold_id(token) for token in payload["tokens"]])
    pair_folds = np.repeat(scene_folds, alternatives.shape[1])
    eligible_pairs = alternative_eligible.reshape(-1).numpy()
    train = (
        eligible_pairs
        if args.calibration_fold == -1
        else (pair_folds != args.calibration_fold) & eligible_pairs
    )
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=args.max_iter,
        max_leaf_nodes=31,
        min_samples_leaf=100,
        l2_regularization=2.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=20260812,
    ).fit(x[train], y[train])

    calibration_indices = (
        np.arange(len(scene_folds))
        if args.calibration_fold == -1
        else np.flatnonzero(scene_folds == args.calibration_fold)
    )
    pair_indices = (
        calibration_indices[:, None] * alternatives.shape[1]
        + np.arange(alternatives.shape[1])[None, :]
    )
    prediction = model.predict(x[pair_indices.reshape(-1)]).reshape(
        -1, alternatives.shape[1]
    )
    prediction = np.where(
        alternative_eligible[calibration_indices].numpy(), prediction, -np.inf
    )
    best = prediction.argmax(axis=1)
    maximum = prediction.max(axis=1)
    finite_maximum = maximum[np.isfinite(maximum)]
    if not len(finite_maximum):
        raise RuntimeError("calibration fold has no eligible alternatives")
    threshold = float(np.quantile(finite_maximum, 1.0 - args.switch_fraction))
    switch = maximum >= threshold
    true_gain = gains[calibration_indices].numpy()
    true_collision = collision_delta[calibration_indices].numpy()
    chosen_gain = np.zeros(len(calibration_indices), dtype=np.float32)
    chosen_collision = np.zeros(len(calibration_indices), dtype=np.float32)
    selected_rows = np.flatnonzero(switch)
    chosen_gain[switch] = true_gain[selected_rows, best[switch]]
    chosen_collision[switch] = true_collision[selected_rows, best[switch]]

    exported_model = export_model(model)
    exported_prediction = predict_exported(
        exported_model, x[pair_indices.reshape(-1)]
    ).reshape(-1, alternatives.shape[1])
    eligible_calibration = alternative_eligible[calibration_indices].numpy()
    export_max_abs_error = float(
        np.max(
            np.abs(
                exported_prediction[eligible_calibration]
                - prediction[eligible_calibration]
            )
        )
    )
    if export_max_abs_error > 1e-10:
        raise RuntimeError(
            f"exported histogram prediction mismatch: {export_max_abs_error}"
        )
    state = {
        "schema": "momworld-v2-navhard-pairwise-gate-state-v1",
        "feature_dim": int(x.shape[1]),
        "alternatives": int(alternatives.shape[1]),
        "min_predicted_gain": threshold,
        "switch_fraction": float(args.switch_fraction),
        "calibration_fold": int(args.calibration_fold),
        "gain_model": exported_model,
    }
    args.state.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, args.state)
    digest = hashlib.sha256(args.state.read_bytes()).hexdigest()
    report = {
        "schema": "momworld-v2-navhard-pairwise-gate-refit-v1",
        "cache": str(args.cache.resolve()),
        "state": str(args.state.resolve()),
        "state_sha256": digest,
        "samples": int(len(target)),
        "training_scenes": int(
            len(scene_folds)
            if args.calibration_fold == -1
            else (scene_folds != args.calibration_fold).sum()
        ),
        "calibration_scenes": int(len(calibration_indices)),
        "calibration_mode": (
            "training_prediction_quantile"
            if args.calibration_fold == -1
            else "held_out_fold"
        ),
        "iterations": int(model.n_iter_),
        "eligible_training_pairs": int(train.sum()),
        "export_max_abs_error": export_max_abs_error,
        "threshold": threshold,
        "switch_rate": float(switch.mean()),
        "gain": float(chosen_gain.mean()),
        "changed_gain": float(chosen_gain[switch].mean()),
        "collision_delta": float(chosen_collision.mean()),
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
