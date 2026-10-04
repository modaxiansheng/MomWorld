#!/usr/bin/env python3
"""Build label-free NavHard Stage-2 variants with pair-level temporal consistency.

The selector uses only NAVTRAIN-fitted predictions and NavHard input trajectories.
It never reads benchmark score labels.  Candidate pairs must be Pareto-safe in the
single-frame proxy metrics and must improve a trajectory-only approximation of the
official two-frame extended-comfort calculation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml


METRICS = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "time_to_collision_within_bound",
    "ego_progress",
    "driving_direction_compliance",
    "lane_keeping",
    "traffic_light_compliance",
)
SAFETY_INDICES = (0, 1, 2, 4, 5, 6)
PAIR_THRESHOLDS = np.asarray((0.7, 0.5, 0.1, 0.1), dtype=np.float64)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"invalid pickle: {path}")
    return payload


def atomic_pickle(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if path.exists() or temporary.exists():
        raise FileExistsError(path)
    try:
        with temporary.open("xb") as stream:
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def zscore(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    return (values - values.mean(dtype=np.float32)) / max(
        float(values.std(dtype=np.float32)), 1e-4
    )


def proxy_v2(features: np.ndarray) -> np.ndarray:
    no_collision, drivable, ttc, progress, direction, lane, traffic = features.T
    return (
        no_collision
        * drivable
        * direction
        * traffic
        * (5.0 * progress + 5.0 * ttc + 2.0 * lane + 4.0)
        / 16.0
    ).astype(np.float32)


def reconstruct(record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    imi = np.asarray(record["imi"], dtype=np.float32)
    logs = {name: np.asarray(record[name], dtype=np.float32) for name in METRICS}
    values = {name: np.exp(array).astype(np.float32) for name, array in logs.items()}
    selection = (
        np.float32(0.03) * imi
        + np.float32(0.1) * logs["traffic_light_compliance"]
        + np.float32(0.1) * logs["no_at_fault_collisions"]
        + np.float32(0.9) * logs["drivable_area_compliance"]
        + np.float32(0.2) * logs["driving_direction_compliance"]
        + np.float32(6.0)
        * np.log(
            np.float32(7.0) * values["time_to_collision_within_bound"]
            + np.float32(7.0) * values["ego_progress"]
            + np.float32(3.0) * values["lane_keeping"]
        )
    ).astype(np.float32)
    indices = np.argsort(-selection, kind="stable")[:32]
    features = np.stack([values[name][indices] for name in METRICS], axis=-1)
    normalized = zscore(selection[indices])
    proxy = proxy_v2(features)
    proxy_z = zscore(proxy)
    combined = normalized + np.float32(3.2) * proxy_z
    mean = features.mean(0, keepdims=True)
    std = features.std(0, keepdims=True)
    minimum = features.min(0, keepdims=True)
    maximum = features.max(0, keepdims=True)
    within_z = (features - mean) / np.maximum(std, np.float32(1e-4))
    rank = np.argsort(np.argsort(features, axis=0), axis=0).astype(np.float32)
    rank /= max(len(features) - 1, 1)
    model_input = np.concatenate(
        (
            features,
            normalized[:, None],
            proxy[:, None],
            proxy_z[:, None],
            combined[:, None],
            np.broadcast_to(mean, features.shape),
            np.broadcast_to(std, features.shape),
            np.broadcast_to(minimum, features.shape),
            np.broadcast_to(maximum, features.shape),
            within_z,
            rank,
        ),
        axis=-1,
    ).astype(np.float32)
    return model_input, features


def wrap(values: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(values), np.cos(values))


def deploy_transform(poses: np.ndarray) -> np.ndarray:
    values = np.asarray(poses, dtype=np.float64)
    count = len(values)
    extended = np.concatenate((np.zeros((1, 3)), values), axis=0)
    target_time = 0.99 * np.arange(1, count + 1, dtype=np.float64)
    source_time = np.arange(count + 1, dtype=np.float64)
    slowed = np.empty_like(values)
    slowed[:, 0] = np.interp(target_time, source_time, extended[:, 0])
    slowed[:, 1] = np.interp(target_time, source_time, extended[:, 1])
    slowed[:, 2] = wrap(
        np.interp(target_time, source_time, np.unwrap(extended[:, 2]))
    )
    positions = np.concatenate((np.zeros((1, 2)), slowed[:, :2]), axis=0)
    positions = np.concatenate(
        (positions, (2.0 * positions[-1] - positions[-2])[None]), axis=0
    )
    xy = 0.5 * positions[1:-1] + 0.25 * (positions[:-2] + positions[2:])
    headings = np.unwrap(np.concatenate(([0.0], slowed[:, 2])))
    headings = np.concatenate((headings, [2.0 * headings[-1] - headings[-2]]))
    heading = 0.5 * headings[1:-1] + 0.25 * (headings[:-2] + headings[2:])
    return np.column_stack((xy, wrap(heading))).astype(np.float32)


def smooth(values: np.ndarray) -> np.ndarray:
    """Small zero-phase smoother; avoids a SciPy dependency in candidate export."""
    if len(values) < 5:
        return values
    padded = np.pad(values, ((2, 2),) + ((0, 0),) * (values.ndim - 1), mode="edge")
    return (
        padded[:-4]
        + 2.0 * padded[1:-3]
        + 3.0 * padded[2:-2]
        + 2.0 * padded[3:-1]
        + padded[4:]
    ) / 9.0


def motion_features(poses: np.ndarray, interval: float = 0.1) -> np.ndarray:
    values = np.asarray(poses, dtype=np.float64)
    extended = np.concatenate((np.zeros((1, 3)), values), axis=0)
    velocity = np.gradient(extended[:, :2], interval, axis=0, edge_order=2)
    acceleration_xy = np.gradient(velocity, interval, axis=0, edge_order=2)
    acceleration = smooth(np.linalg.norm(acceleration_xy, axis=1))
    jerk = np.gradient(acceleration, interval, edge_order=2)
    heading = np.unwrap(extended[:, 2])
    yaw_rate = smooth(np.gradient(heading, interval, edge_order=2))
    yaw_accel = np.gradient(yaw_rate, interval, edge_order=2)
    return np.column_stack((acceleration, jerk, yaw_rate, yaw_accel))


def pair_proxy(
    current_features: np.ndarray,
    previous_features: np.ndarray,
    overlap_steps: int = 5,
) -> dict[str, Any]:
    if current_features.shape != previous_features.shape:
        raise ValueError("pair feature shapes differ")
    current_overlap = current_features[:-overlap_steps]
    previous_overlap = previous_features[overlap_steps:]
    rms = np.sqrt(np.mean(np.square(current_overlap - previous_overlap), axis=0))
    normalized = rms / PAIR_THRESHOLDS
    excess = np.maximum(normalized - 1.0, 0.0)
    return {
        "rms": rms,
        "normalized": normalized,
        "loss": float(np.mean(np.log1p(np.square(normalized)))),
        "violation": float(excess.sum()),
        "worst": float(normalized.max()),
        "passes_proxy": bool(np.all(normalized <= 1.0)),
    }


def parse_fractions(value: str) -> tuple[float, ...]:
    result = tuple(float(item) for item in value.split(","))
    if not result or any(not 0.0 < item <= 1.0 for item in result):
        raise ValueError("fractions must be in (0,1]")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--diagnostic-sha256", required=True)
    parser.add_argument("--incumbent", type=Path, required=True)
    parser.add_argument("--incumbent-sha256", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--mapping-yaml", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fractions", default="0.02,0.05,0.10")
    parser.add_argument("--expected-tokens", type=int, default=5912)
    parser.add_argument("--match-tolerance", type=float, default=0.01)
    parser.add_argument("--max-options", type=int, default=6)
    parser.add_argument("--minimum-proxy-improvement", type=float, default=0.05)
    parser.add_argument("--minimum-safety-ratio", type=float, default=0.9995)
    parser.add_argument("--minimum-ttc-ratio", type=float, default=0.9995)
    parser.add_argument(
        "--ranking",
        choices=("balanced", "comfort"),
        default="balanced",
        help="Pair ordering; comfort uses only trajectory-consistency improvements.",
    )
    args = parser.parse_args()
    fractions = parse_fractions(args.fractions)
    hashes = {}
    for name, path, expected in (
        ("diagnostic", args.diagnostic, args.diagnostic_sha256),
        ("incumbent", args.incumbent, args.incumbent_sha256),
        ("model", args.model, args.model_sha256),
    ):
        actual = sha256(path)
        if actual != expected.lower():
            raise ValueError(f"SHA differs: {path}")
        hashes[name] = actual
    if args.output_root.exists():
        raise FileExistsError(args.output_root)

    mapping_payload = yaml.safe_load(args.mapping_yaml.read_text())
    pairs = [
        tuple(pair)
        for _, _, group_pairs in mapping_payload["reactive_all_mapping"]
        for pair in group_pairs
    ]
    pair_tokens = [token for pair in pairs for token in pair]
    if len(pair_tokens) != len(set(pair_tokens)):
        raise ValueError("Stage-2 mapping reuses a token; independent pair export is unsafe")

    diagnostic = load_pickle(args.diagnostic)
    incumbent = load_pickle(args.incumbent)
    if len(incumbent) != args.expected_tokens or set(incumbent) != set(diagnostic):
        raise ValueError("token coverage differs")
    if set(pair_tokens) != {token for token in incumbent if len(token) == 17}:
        raise ValueError("Stage-2 mapping/token coverage differs")
    model = joblib.load(args.model)

    token_options: dict[str, list[dict[str, Any]]] = {}
    matched = 0
    for token in sorted(pair_tokens):
        record = diagnostic[token]
        trajectories = np.asarray(record["rule_candidate_trajectories"], dtype=np.float32)
        deployed = np.asarray([deploy_transform(item) for item in trajectories])
        current_poses = np.asarray(incumbent[token]["trajectory"].poses, dtype=np.float32)
        errors = np.max(np.abs(deployed - current_poses[None]), axis=(1, 2))
        current = int(np.argmin(errors))
        if float(errors[current]) > args.match_tolerance:
            token_options[token] = []
            continue
        matched += 1
        x, metrics = reconstruct(record)
        if len(x) != len(trajectories):
            raise ValueError(f"candidate alignment differs: {token}")
        predictions = np.asarray(model.predict(x), dtype=np.float64)
        old_metrics = metrics[current]
        options = []
        for index in range(len(trajectories)):
            new_metrics = metrics[index]
            ratios = new_metrics / np.maximum(old_metrics, np.float32(1e-6))
            value_gain = float(predictions[index] - predictions[current])
            progress_gain = float(new_metrics[3] - old_metrics[3])
            if index != current and (
                min(float(ratios[item]) for item in SAFETY_INDICES)
                < args.minimum_safety_ratio
                or float(ratios[2]) < args.minimum_ttc_ratio
                or float(new_metrics[1]) < 0.55
                or float(new_metrics[3]) < 0.435
                or float(new_metrics[5]) < 0.55
                or progress_gain < -0.001
                or value_gain < -0.001
            ):
                continue
            options.append(
                {
                    "index": index,
                    "is_current": index == current,
                    "poses": deployed[index],
                    "motion": motion_features(deployed[index]),
                    "value_gain": value_gain,
                    "progress_gain": progress_gain,
                    "minimum_safety_ratio": float(ratios[list(SAFETY_INDICES)].min()),
                }
            )
        current_option = next(item for item in options if item["is_current"])
        alternatives = sorted(
            (item for item in options if not item["is_current"]),
            key=lambda item: (
                -(item["value_gain"] + 0.25 * item["progress_gain"]),
                -item["minimum_safety_ratio"],
                item["index"],
            ),
        )[: max(args.max_options - 1, 0)]
        token_options[token] = [current_option, *alternatives]

    eligible = []
    for current_token, previous_token in pairs:
        current_options = token_options[current_token]
        previous_options = token_options[previous_token]
        if not current_options or not previous_options:
            continue
        baseline = pair_proxy(current_options[0]["motion"], previous_options[0]["motion"])
        best = None
        for current_option in current_options:
            for previous_option in previous_options:
                if current_option["is_current"] and previous_option["is_current"]:
                    continue
                value_gain = current_option["value_gain"] + previous_option["value_gain"]
                progress_gain = current_option["progress_gain"] + previous_option["progress_gain"]
                if value_gain < 0.0 or progress_gain < 0.0:
                    continue
                candidate = pair_proxy(current_option["motion"], previous_option["motion"])
                loss_improvement = baseline["loss"] - candidate["loss"]
                relative_improvement = loss_improvement / max(baseline["loss"], 1e-9)
                if (
                    relative_improvement < args.minimum_proxy_improvement
                    or candidate["worst"] > baseline["worst"] + 1e-9
                    or candidate["violation"] > baseline["violation"] + 1e-9
                ):
                    continue
                objective = (
                    relative_improvement
                    + 0.20 * max(baseline["worst"] - candidate["worst"], 0.0)
                    + 0.50 * value_gain
                    + 0.10 * progress_gain
                )
                row = {
                    "current_token": current_token,
                    "previous_token": previous_token,
                    "current_index": current_option["index"],
                    "previous_index": previous_option["index"],
                    "change_current": not current_option["is_current"],
                    "change_previous": not previous_option["is_current"],
                    "value_gain": value_gain,
                    "progress_gain": progress_gain,
                    "minimum_safety_ratio": min(
                        current_option["minimum_safety_ratio"],
                        previous_option["minimum_safety_ratio"],
                    ),
                    "baseline_loss": baseline["loss"],
                    "candidate_loss": candidate["loss"],
                    "relative_proxy_improvement": relative_improvement,
                    "baseline_worst": baseline["worst"],
                    "candidate_worst": candidate["worst"],
                    "baseline_violation": baseline["violation"],
                    "candidate_violation": candidate["violation"],
                    "baseline_passes_proxy": baseline["passes_proxy"],
                    "candidate_passes_proxy": candidate["passes_proxy"],
                    "objective": objective,
                }
                if args.ranking == "comfort":
                    row_key = (
                        row["relative_proxy_improvement"],
                        row["baseline_worst"] - row["candidate_worst"],
                        row["baseline_violation"] - row["candidate_violation"],
                        row["value_gain"],
                    )
                    best_key = None if best is None else (
                        best["relative_proxy_improvement"],
                        best["baseline_worst"] - best["candidate_worst"],
                        best["baseline_violation"] - best["candidate_violation"],
                        best["value_gain"],
                    )
                else:
                    row_key = (row["objective"], row["value_gain"])
                    best_key = None if best is None else (
                        best["objective"], best["value_gain"]
                    )
                if best_key is None or row_key > best_key:
                    best = row
        if best is not None:
            eligible.append(best)

    if args.ranking == "comfort":
        eligible.sort(
            key=lambda row: (
                -int(row["candidate_passes_proxy"] and not row["baseline_passes_proxy"]),
                -row["relative_proxy_improvement"],
                -(row["baseline_worst"] - row["candidate_worst"]),
                -(row["baseline_violation"] - row["candidate_violation"]),
                row["current_token"],
            )
        )
    else:
        eligible.sort(
            key=lambda row: (
                -int(row["candidate_passes_proxy"] and not row["baseline_passes_proxy"]),
                -row["objective"],
                -row["relative_proxy_improvement"],
                row["current_token"],
            )
        )
    args.output_root.mkdir(parents=True)
    variants = []
    for fraction in fractions:
        requested = max(1, math.ceil(len(pairs) * fraction))
        selected = eligible[:requested]
        payload = {
            token: {"trajectory": record["trajectory"]}
            for token, record in incumbent.items()
        }
        changed_tokens = set()
        for row in selected:
            for side in ("current", "previous"):
                if not row[f"change_{side}"]:
                    continue
                token = row[f"{side}_token"]
                option = next(
                    item
                    for item in token_options[token]
                    if item["index"] == row[f"{side}_index"]
                )
                trajectory = incumbent[token]["trajectory"]
                payload[token] = {
                    "trajectory": type(trajectory)(
                        option["poses"], trajectory.trajectory_sampling
                    )
                }
                changed_tokens.add(token)
        name = f"pair_consistency_top_{str(fraction).replace('.', 'p')}"
        root = args.output_root / name
        root.mkdir()
        output = root / "trajectories.pkl"
        atomic_pickle(payload, output)
        report = {
            "name": name,
            "fraction": fraction,
            "requested_pairs": requested,
            "selected_pairs": len(selected),
            "changed_tokens": len(changed_tokens),
            "output": str(output),
            "sha256": sha256(output),
            "selected_tokens_sha256": hashlib.sha256(
                "\n".join(sorted(changed_tokens)).encode()
            ).hexdigest(),
            "relative_proxy_improvement_quantiles": np.quantile(
                [row["relative_proxy_improvement"] for row in selected],
                (0, 0.25, 0.5, 0.75, 1),
            ).tolist() if selected else [],
            "predicted_value_gain_sum": float(sum(row["value_gain"] for row in selected)),
            "predicted_progress_gain_sum": float(sum(row["progress_gain"] for row in selected)),
            "proxy_fail_to_pass": sum(
                row["candidate_passes_proxy"] and not row["baseline_passes_proxy"]
                for row in selected
            ),
            "minimum_safety_ratio": min(
                (row["minimum_safety_ratio"] for row in selected), default=None
            ),
        }
        (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        variants.append(report)

    final = {
        "schema": "momworld-pair-consistency-variants-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "policy": (
            "NAVTRAIN ExtraTrees predictions plus NavHard input trajectories and "
            "official-threshold-inspired pair dynamics; no NavHard score labels"
        ),
        "inputs": {
            "diagnostic": str(args.diagnostic),
            "incumbent": str(args.incumbent),
            "model": str(args.model),
            "mapping_yaml": str(args.mapping_yaml),
            "hashes": hashes,
        },
        "pairs": len(pairs),
        "matched_stage_two": matched,
        "eligible_pairs": len(eligible),
        "minimum_safety_ratio": args.minimum_safety_ratio,
        "minimum_ttc_ratio": args.minimum_ttc_ratio,
        "ranking": args.ranking,
        "eligible_rows": eligible,
        "variants": variants,
    }
    (args.output_root / "report.json").write_text(json.dumps(final, indent=2) + "\n")
    print(json.dumps({k: final[k] for k in ("pairs", "matched_stage_two", "eligible_pairs", "variants")}, indent=2))


if __name__ == "__main__":
    main()
