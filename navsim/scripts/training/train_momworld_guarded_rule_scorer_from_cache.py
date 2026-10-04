#!/usr/bin/env python3
"""Train a conservative pairwise switcher from a fresh NAVTRAIN partition.

The frozen reference selector is the already validated protocol proxy
(v1 weight 1.6, v2 weight 3.2).  A tiny learned scorer predicts whether each
candidate improves upon that reference; a calibrated margin and component-wise
safety guard permit only conservative switches.  The effective NAVTRAIN
training set is hash-partitioned into fit/calibration/holdout after excluding
all earlier exposed subsets; NAVTEST/NAVHARD inputs are rejected by construction.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from navsim.agents.momworld.momworld_model import (
    RuleAwareTrajectoryScorer,
    candidate_protocol_proxy,
    candidate_score_zscore,
)
from navsim.agents.momworld.context_ranker import (
    effective_candidate_mask,
    pairwise_ranknet_loss,
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


REPORT_SCHEMA = "momworld-guarded-rule-scorer-training-v1"
HOLDOUT_SCHEMA = "momworld-guarded-rule-scorer-holdout-v1"
GATE_SCHEMA = "momworld-guarded-rule-scorer-gate-v1"
PREVIOUS_PARTITION_SALTS = (
    "momworld-guarded-rule-scorer-effective-train-v1",
    "momworld-guarded-rule-scorer-effective-train-v2",
)
PARTITION_SALT = "momworld-guarded-pairwise-scorer-effective-train-v3"
FIT_FRACTION = 0.80
CALIBRATION_FRACTION = 0.10
REFERENCE_PROXY_WEIGHTS = {"v1": 1.6, "v2": 3.2}
SWITCH_MARGIN_GRID = (0.0, 0.01, 0.025, 0.05, 0.10, 0.20, 0.40, 1_000_000.0)
SAFETY_TOLERANCE_GRID = (0.0, 0.02, 0.05)
KINEMATIC_TOLERANCE = 0.10
MINIMUM_HOLDOUT_GAIN = {"v1": 0.0010, "v2": 0.0015}
RELEVANT_SOURCE_FILES = (
    "navsim/agents/momworld/momworld_model.py",
    "scripts/training/train_momworld_guarded_rule_scorer_from_cache.py",
    "scripts/training/train_momworld_rule_scorer_from_cache.py",
    "scripts/training/inject_momworld_rule_scorer_checkpoint.py",
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


def _atomic_torch(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _source_audit(repository: Path) -> Dict[str, Any]:
    commit = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain", "--", *RELEVANT_SOURCE_FILES],
        text=True,
    ).splitlines()
    if status:
        raise RuntimeError(f"Relevant training sources are dirty: {status}")
    files: Dict[str, Any] = {}
    for relative in RELEVANT_SOURCE_FILES:
        path = (repository / relative).resolve()
        committed = subprocess.check_output(
            ["git", "-C", str(repository), "show", f"{commit}:{relative}"]
        )
        disk_sha = _sha256(path)
        commit_sha = hashlib.sha256(committed).hexdigest()
        if disk_sha != commit_sha:
            raise RuntimeError(f"Source differs from committed blob: {relative}")
        files[relative] = {"path": str(path), "sha256": disk_sha}
    return {"commit": commit, "working_tree_clean": True, "files": files}


def partition_fresh_effective_train(
    records: Sequence[CacheRecord],
) -> Tuple[List[CacheRecord], List[CacheRecord], List[CacheRecord], Dict[str, Any]]:
    """Make a third fresh split, excluding every previously evaluated token."""

    if len(records) < 10:
        raise ValueError("Need at least ten effective NAVTRAIN training records")
    eligible = list(records)
    previously_exposed: set[str] = set()
    # Reconstruct both earlier partitions and retain only their nested fit
    # intersection.  Thus no token previously used for model/calibration
    # selection or fixed-subset reporting is eligible in this attempt.
    for previous_salt in PREVIOUS_PARTITION_SALTS:
        previous = sorted(
            (
                hashlib.sha256(
                    f"{previous_salt}\0{record.token}".encode()
                ).hexdigest(),
                record.token,
                record,
            )
            for record in eligible
        )
        cutoff = int(len(previous) * FIT_FRACTION)
        previously_exposed.update(row[2].token for row in previous[cutoff:])
        eligible = [row[2] for row in previous[:cutoff]]
    keyed = sorted(
        (
            hashlib.sha256(f"{PARTITION_SALT}\0{record.token}".encode()).hexdigest(),
            record.token,
            record,
        )
        for record in eligible
    )
    fit_end = int(len(keyed) * FIT_FRACTION)
    calibration_end = fit_end + int(len(keyed) * CALIBRATION_FRACTION)
    fit = [row[2] for row in keyed[:fit_end]]
    calibration = [row[2] for row in keyed[fit_end:calibration_end]]
    holdout = [row[2] for row in keyed[calibration_end:]]
    token_sets = [{record.token for record in split} for split in (fit, calibration, holdout)]
    if any(token_sets[i] & token_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("Fresh NAVTRAIN fit/calibration/holdout overlap")
    if set.union(*token_sets) != {record.token for record in eligible}:
        raise RuntimeError("Fresh NAVTRAIN partition does not cover eligible train")
    excluded = {record.token for record in records} - set.union(*token_sets)
    if excluded != previously_exposed:
        raise RuntimeError("Fresh partition did not exclude the prior exposed subsets exactly")
    audit = {
        "method": "sha256-sort-fixed-count",
        "salt": PARTITION_SALT,
        "source": "effective NAVTRAIN train after excluding every NAVTRAIN val token",
        "historical_navtrain_val_used": False,
        "previous_partition_salts": list(PREVIOUS_PARTITION_SALTS),
        "previous_calibration_holdout_excluded": True,
        "previous_exposed_samples_excluded": len(previously_exposed),
        "eligible_samples": len(eligible),
        "fit_fraction": FIT_FRACTION,
        "calibration_fraction": CALIBRATION_FRACTION,
        "fit_samples": len(fit),
        "calibration_samples": len(calibration),
        "holdout_samples": len(holdout),
        "fit_tokens_sha256": _token_sha256(fit),
        "calibration_tokens_sha256": _token_sha256(calibration),
        "holdout_tokens_sha256": _token_sha256(holdout),
    }
    return fit, calibration, holdout, audit


def _dataset(records: Sequence[CacheRecord]) -> TensorDataset:
    return TensorDataset(
        torch.from_numpy(np.stack([record.features for record in records])),
        torch.from_numpy(np.stack([record.normalized_base for record in records])),
        torch.from_numpy(np.stack([record.candidate_is_finite for record in records])),
        torch.from_numpy(np.stack([record.target_collision for record in records])),
        torch.from_numpy(np.stack([record.target_v1 for record in records])),
        torch.from_numpy(np.stack([record.target_v2 for record in records])),
    )


def _reference_indices(
    features: torch.Tensor,
    normalized_base: torch.Tensor,
    finite: torch.Tensor,
    protocol: str,
) -> torch.Tensor:
    proxy = candidate_protocol_proxy(features, protocol)
    score = normalized_base.float() + REFERENCE_PROXY_WEIGHTS[protocol] * candidate_score_zscore(proxy)
    mask = effective_candidate_mask(finite)
    return score.masked_fill(~mask, torch.finfo(score.dtype).min).argmax(dim=1)


def _catastrophic_negative_loss(
    logits: torch.Tensor,
    target_collision: torch.Tensor,
    finite: torch.Tensor,
) -> torch.Tensor:
    mask = effective_candidate_mask(finite)
    safe = mask & (target_collision <= 0.1)
    unsafe = mask & (target_collision >= 0.5)
    pairs = safe[:, :, None] & unsafe[:, None, :]
    difference = logits[:, :, None] - logits[:, None, :]
    loss = F.softplus(0.25 - difference)
    return (loss * pairs.to(loss.dtype)).sum() / pairs.sum().clamp(min=1)


def _pairwise_features(
    features: torch.Tensor, reference_indices: torch.Tensor
) -> torch.Tensor:
    rows = torch.arange(features.shape[0], device=features.device)
    reference = features[rows, reference_indices][:, None, :]
    return features - reference


def _binary_improvement_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    finite: torch.Tensor,
    reference_indices: torch.Tensor,
) -> torch.Tensor:
    rows = torch.arange(logits.shape[0], device=logits.device)
    target_difference = targets - targets[rows, reference_indices][:, None]
    effective = effective_candidate_mask(finite) & (target_difference.abs() >= 0.002)
    labels = (target_difference > 0.0).to(logits.dtype)
    positives = (labels * effective).sum()
    negatives = ((1.0 - labels) * effective).sum()
    positive_weight = (negatives / positives.clamp(min=1.0)).clamp(1.0, 20.0)
    loss = F.binary_cross_entropy_with_logits(
        logits, labels, pos_weight=positive_weight.detach(), reduction="none"
    )
    gap_weight = (target_difference.abs() / 0.05).clamp(min=0.25, max=4.0).detach()
    weight = effective.to(loss.dtype) * gap_weight
    return (loss * weight).sum() / weight.sum().clamp(min=1.0)


def _delta_regression_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    finite: torch.Tensor,
    reference_indices: torch.Tensor,
) -> torch.Tensor:
    rows = torch.arange(logits.shape[0], device=logits.device)
    target_difference = targets - targets[rows, reference_indices][:, None]
    loss = F.smooth_l1_loss(logits, 10.0 * target_difference, reduction="none")
    weight = effective_candidate_mask(finite).to(loss.dtype)
    return (loss * weight).sum() / weight.sum().clamp(min=1.0)


def _loss(
    model: torch.nn.Module,
    features: torch.Tensor,
    base: torch.Tensor,
    finite: torch.Tensor,
    target_collision: torch.Tensor,
    target_v1: torch.Tensor,
    target_v2: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    totals: Dict[str, torch.Tensor] = {}
    protocol_losses: List[torch.Tensor] = []
    for protocol, targets in (("v1", target_v1), ("v2", target_v2)):
        reference = _reference_indices(features, base, finite, protocol)
        relative_features = _pairwise_features(features, reference)
        logits = model(relative_features, protocol)
        binary = _binary_improvement_loss(logits, targets, finite, reference)
        pairwise = pairwise_ranknet_loss(logits, targets, finite)
        regression = _delta_regression_loss(logits, targets, finite, reference)
        catastrophic = _catastrophic_negative_loss(logits, target_collision, finite)
        value = binary + 0.75 * pairwise + 0.25 * regression + 0.05 * catastrophic
        protocol_losses.append(value)
        totals[f"{protocol}_binary_improvement"] = binary
        totals[f"{protocol}_pairwise"] = pairwise
        totals[f"{protocol}_delta_regression"] = regression
        totals[f"{protocol}_catastrophic"] = catastrophic
    total = 0.5 * (protocol_losses[0] + protocol_losses[1])
    totals["loss"] = total
    return total, totals


def _predict(
    model: torch.nn.Module,
    records: Sequence[CacheRecord],
    device: torch.device,
    batch_size: int,
) -> Dict[str, np.ndarray]:
    output: Dict[str, List[np.ndarray]] = {"v1": [], "v2": []}
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            chunk = records[start : start + batch_size]
            features = torch.from_numpy(
                np.stack([record.features for record in chunk])
            ).to(device)
            base = torch.from_numpy(
                np.stack([record.normalized_base for record in chunk])
            ).to(device)
            finite = torch.from_numpy(
                np.stack([record.candidate_is_finite for record in chunk])
            ).to(device)
            for protocol in ("v1", "v2"):
                reference = _reference_indices(features, base, finite, protocol)
                relative = _pairwise_features(features, reference)
                output[protocol].append(
                    model(relative, protocol).float().cpu().numpy()
                )
    return {key: np.concatenate(value, axis=0) for key, value in output.items()}


def _arrays(records: Sequence[CacheRecord]) -> Dict[str, np.ndarray]:
    return {
        name: np.stack([getattr(record, name) for record in records])
        for name in (
            "features",
            "normalized_base",
            "candidate_is_finite",
            "kinematic_penalty",
            "target_collision",
            "target_v1",
            "target_v2",
        )
    }


def _zscore_numpy(value: np.ndarray, epsilon: float = 1e-4) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    mean = value.mean(axis=1, keepdims=True)
    std = value.std(axis=1, keepdims=True)
    return (value - mean) / np.maximum(std, epsilon)


def _protocol_proxy_numpy(features: np.ndarray, protocol: str) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    no_collision, drivable, ttc, progress, direction, lane, traffic = [
        np.clip(features[..., index], 0.0, 1.0) for index in range(7)
    ]
    if protocol == "v1":
        return no_collision * drivable * (5 * progress + 5 * ttc + 2) / 12
    return no_collision * drivable * direction * traffic * (
        5 * progress + 5 * ttc + 2 * lane + 4
    ) / 16


def _selection_metrics(
    arrays: Mapping[str, np.ndarray],
    logits: np.ndarray,
    protocol: str,
    switch_margin: float,
    safety_tolerance: float,
) -> Dict[str, Any]:
    base = np.asarray(arrays["normalized_base"], dtype=np.float32)
    proxy_z = _zscore_numpy(_protocol_proxy_numpy(arrays["features"], protocol))
    reference_score = base + REFERENCE_PROXY_WEIGHTS[protocol] * proxy_z
    finite = np.asarray(arrays["candidate_is_finite"], dtype=np.bool_)
    effective = np.where(finite.any(axis=1, keepdims=True), finite, np.ones_like(finite))
    reference = np.where(
        effective, reference_score, np.finfo(np.float32).min
    ).argmax(axis=1)
    rows = np.arange(len(reference))
    features = np.asarray(arrays["features"], dtype=np.float32)
    reference_features = features[rows, reference][:, None, :]
    safety_indices = (0, 1, 2) if protocol == "v1" else (0, 1, 2, 4, 5, 6)
    component_safe = (
        features[..., safety_indices]
        >= reference_features[..., safety_indices] - float(safety_tolerance)
    ).all(axis=-1)
    kinematic = np.asarray(arrays["kinematic_penalty"], dtype=np.float32)
    kinematic_safe = (
        kinematic
        <= kinematic[rows, reference][:, None] + float(KINEMATIC_TOLERANCE)
    )
    eligible = effective & component_safe & kinematic_safe
    eligible[rows, reference] = True
    challenger = np.where(
        eligible, np.asarray(logits, dtype=np.float32), np.finfo(np.float32).min
    ).argmax(axis=1)
    logit_margin = logits[rows, challenger] - logits[rows, reference]
    switch = (challenger != reference) & (logit_margin >= float(switch_margin))
    selected = np.where(switch, challenger, reference)
    target = np.asarray(arrays[f"target_{protocol}"], dtype=np.float32)[rows, selected]
    collision = np.asarray(arrays["target_collision"], dtype=np.float32)[rows, selected]
    return {
        "parameters": {
            "base_score_weight": 1.0,
            "protocol_proxy_weight": REFERENCE_PROXY_WEIGHTS[protocol],
            "pairwise_guarded": True,
            "switch_margin": float(switch_margin),
            "safety_tolerance": float(safety_tolerance),
            "kinematic_tolerance": float(KINEMATIC_TOLERANCE),
            "learned_zscore_weight": 0.0,
            "collision_weight": 0.0,
            "kinematic_weight": 0.0,
            "momentum_weight": 0.0,
            "collision_threshold": 1.000001,
        },
        "mean_target_score": float(target.mean()),
        "mean_target_collision_risk": float(collision.mean()),
        "target_collision_rate_ge_0_5": float((collision >= 0.5).mean()),
        "switch_rate": float(switch.mean()),
        "mean_switch_margin": float(logit_margin[switch].mean()) if switch.any() else 0.0,
        "selected_indices": selected,
    }


def _select_calibration(
    arrays: Mapping[str, np.ndarray], logits: Mapping[str, np.ndarray]
) -> Tuple[Dict[str, Any], Tuple[float, float, float, float]]:
    protocols: Dict[str, Any] = {}
    for protocol in ("v1", "v2"):
        reference = _selection_metrics(arrays, logits[protocol], protocol, 1_000_000.0, 0.0)
        candidates = [
            _selection_metrics(arrays, logits[protocol], protocol, margin, tolerance)
            for margin in SWITCH_MARGIN_GRID
            for tolerance in SAFETY_TOLERANCE_GRID
        ]
        feasible = [
            result
            for result in candidates
            if result["target_collision_rate_ge_0_5"]
            <= reference["target_collision_rate_ge_0_5"] + 1e-12
        ]
        if not feasible:
            raise RuntimeError("Collision-constrained grid unexpectedly excluded control")
        selected = max(
            feasible,
            key=lambda result: (
                result["mean_target_score"],
                -result["target_collision_rate_ge_0_5"],
                -result["mean_target_collision_risk"],
                -result["switch_rate"],
                result["parameters"]["switch_margin"],
            ),
        )
        selected.pop("selected_indices")
        reference.pop("selected_indices")
        selected["calibration_reference"] = reference
        selected["collision_constraint"] = "candidate_rate_ge_0_5 <= reference_rate_ge_0_5"
        protocols[protocol] = selected
    objective = (
        float(np.mean([protocols[p]["mean_target_score"] for p in ("v1", "v2")])),
        float(min(protocols[p]["mean_target_score"] for p in ("v1", "v2"))),
        -float(np.mean([protocols[p]["target_collision_rate_ge_0_5"] for p in ("v1", "v2")])),
        -float(sum(protocols[p]["switch_rate"] for p in ("v1", "v2"))),
    )
    return {"protocols": protocols, "grid_size_per_protocol": 24, "objective": list(objective)}, objective


def _evaluate_frozen_holdout(
    arrays: Mapping[str, np.ndarray],
    logits: Mapping[str, np.ndarray],
    selected: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    protocols: Dict[str, Any] = {}
    for protocol in ("v1", "v2"):
        raw = selected[protocol]["parameters"]
        candidate = _selection_metrics(
            arrays, logits[protocol], protocol,
            float(raw["switch_margin"]), float(raw["safety_tolerance"])
        )
        reference = _selection_metrics(
            arrays, logits[protocol], protocol, 1_000_000.0, 0.0
        )
        candidate.pop("selected_indices")
        reference.pop("selected_indices")
        protocols[protocol] = {
            "candidate": candidate,
            "reference_protocol_proxy": reference,
            "score_gain": candidate["mean_target_score"] - reference["mean_target_score"],
            "collision_rate_delta": candidate["target_collision_rate_ge_0_5"] - reference["target_collision_rate_ge_0_5"],
        }
    return protocols


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-parallel-device-ids", default="0,1")
    parser.add_argument("--memory-fraction", type=float, default=0.06)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260809)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.epochs, args.patience, args.batch_size) < 1 or args.num_workers < 0:
        raise ValueError("Invalid training duration, batch size, or workers")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        name: output / filename
        for name, filename in {
            "state": "rule_scorer_state.pt",
            "training": "training_report.json",
            "holdout": "holdout_report.json",
            "subset": "fixed_subset_tokens.jsonl",
            "gate": "fixed_subset_gate_result.json",
        }.items()
    }
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("Refusing to overwrite guarded scorer artifacts")

    repository = Path(__file__).resolve().parents[2]
    source = _source_audit(repository)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)

    cache_roots = [path.expanduser().resolve() for path in args.cache_root]
    manifests = _discover_manifests(cache_roots)
    cache_audit = _validate_manifests(manifests, False)
    records = _load_records(_discover_npz(cache_roots), str(cache_audit["checkpoint_sha256"]))
    records, overlap_audit = _exclude_validation_overlap(records)
    fit, calibration, holdout, partition = partition_fresh_effective_train(records["train"])
    if {record.token for record in records["val"]} & {
        record.token for record in fit + calibration + holdout
    }:
        raise RuntimeError("Historical NAVTRAIN val overlaps fresh effective-train partition")
    _atomic_jsonl(
        paths["subset"],
        (
            {"token": record.token, "dataset": record.dataset, "role": "fixed_subset"}
            for record in sorted(holdout, key=lambda item: item.token)
        ),
    )
    partition["fixed_subset_artifact"] = {
        "path": str(paths["subset"]),
        "sha256": _sha256(paths["subset"]),
        "samples": len(holdout),
    }

    device = torch.device(args.device)
    logical_ids = tuple(int(value) for value in args.data_parallel_device_ids.split(","))
    if device.type == "cuda":
        if not torch.cuda.is_available() or max(logical_ids) >= torch.cuda.device_count():
            raise RuntimeError("Requested CUDA/DataParallel devices are unavailable")
        if not 0.0 < args.memory_fraction <= (10 * 1024) / 49140:
            raise ValueError("Per-device allocator fraction exceeds 10 GiB")
        torch.cuda.set_device(device)
        for logical_id in logical_ids:
            torch.cuda.set_per_process_memory_fraction(args.memory_fraction, logical_id)
    hidden_dim = int(cache_audit["rule_scorer_hidden_dim"])
    base_model = RuleAwareTrajectoryScorer(SimpleNamespace(rule_scorer_hidden_dim=hidden_dim)).to(device)
    model: torch.nn.Module = base_model
    if len(logical_ids) > 1:
        model = torch.nn.DataParallel(base_model, device_ids=list(logical_ids), output_device=logical_ids[0])
    optimizer = torch.optim.AdamW(base_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        _dataset(fit),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    best_head_state: Dict[str, Dict[str, torch.Tensor]] = {}
    best_protocol_calibration: Dict[str, Dict[str, Any]] = {}
    best_protocol_key: Dict[str, Tuple[float, float, float, float]] = {}
    best_epoch: Dict[str, int] = {}
    stale = {"v1": 0, "v2": 0}
    history: List[Dict[str, Any]] = []
    for epoch in range(args.epochs):
        model.train()
        totals: Dict[str, float] = {}
        samples = 0
        for batch in loader:
            features, base, finite, collision, target_v1, target_v2 = [
                value.to(device, non_blocking=True) for value in batch
            ]
            optimizer.zero_grad(set_to_none=True)
            loss, losses = _loss(model, features, base, finite, collision, target_v1, target_v2)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite guarded scorer loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), 5.0)
            optimizer.step()
            samples += features.shape[0]
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach()) * features.shape[0]
        calibration_logits = _predict(model, calibration, device, args.batch_size)
        calibration_result, objective = _select_calibration(_arrays(calibration), calibration_logits)
        epoch_report = {
            "epoch": epoch,
            "train": {name: value / max(samples, 1) for name, value in totals.items()},
            "calibration": calibration_result,
        }
        history.append(epoch_report)
        print(json.dumps(epoch_report, sort_keys=True), flush=True)
        current_state = base_model.state_dict()
        for protocol in ("v1", "v2"):
            result = calibration_result["protocols"][protocol]
            key = (
                float(result["mean_target_score"]),
                -float(result["target_collision_rate_ge_0_5"]),
                -float(result["mean_target_collision_risk"]),
                -float(result["switch_rate"]),
            )
            if protocol not in best_protocol_key or key > best_protocol_key[protocol]:
                prefix = f"heads.{protocol}."
                best_protocol_key[protocol] = key
                best_protocol_calibration[protocol] = copy.deepcopy(result)
                best_head_state[protocol] = {
                    name: value.detach().cpu().clone()
                    for name, value in current_state.items()
                    if name.startswith(prefix)
                }
                best_epoch[protocol] = epoch
                stale[protocol] = 0
            else:
                stale[protocol] += 1
        if all(value >= args.patience for value in stale.values()):
            break
    if set(best_head_state) != {"v1", "v2"} or set(best_protocol_calibration) != {"v1", "v2"}:
        raise RuntimeError("Training produced no guarded scorer artifact")

    best_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in base_model.state_dict().items()
    }
    for protocol in ("v1", "v2"):
        best_state.update(best_head_state[protocol])
    base_model.load_state_dict(best_state, strict=True)
    best_objective = (
        float(np.mean([best_protocol_calibration[p]["mean_target_score"] for p in ("v1", "v2")])),
        float(min(best_protocol_calibration[p]["mean_target_score"] for p in ("v1", "v2"))),
        -float(np.mean([best_protocol_calibration[p]["target_collision_rate_ge_0_5"] for p in ("v1", "v2")])),
        -float(sum(best_protocol_calibration[p]["switch_rate"] for p in ("v1", "v2"))),
    )
    best_calibration = {
        "protocols": best_protocol_calibration,
        "grid_size_per_protocol": 24,
        "objective": list(best_objective),
        "selection": "independent best epoch per independent protocol head",
        "best_epoch": dict(best_epoch),
    }
    holdout_logits = _predict(base_model, holdout, device, args.batch_size)
    holdout_protocols = _evaluate_frozen_holdout(
        _arrays(holdout), holdout_logits, best_calibration["protocols"]
    )
    holdout_report = {
        "schema": HOLDOUT_SCHEMA,
        "created_at": _utc_now(),
        "access_count": 1,
        "selection_frozen_before_access": True,
        "samples": len(holdout),
        "tokens_sha256": _token_sha256(holdout),
        "partition": partition,
        "protocols": holdout_protocols,
        "data_scope": {
            "source_split": "navtrain",
            "label_scope": "navtrain-only",
            "historical_navtrain_val_used": False,
            "evaluator_calls": 0,
            "benchmark_cache_reads": 0,
            "excluded_splits": ["navtest", "navhard"],
        },
    }
    _atomic_json(paths["holdout"], holdout_report)

    selected_parameters = {
        protocol: best_calibration["protocols"][protocol]["parameters"]
        for protocol in ("v1", "v2")
    }
    metadata = {
        "schema": REPORT_SCHEMA,
        "created_at": _utc_now(),
        "source_commit": source["commit"],
        "source_audit": source,
        "cache_audit": {**cache_audit, "split_partition": overlap_audit},
        "data_scope": holdout_report["data_scope"],
        "partition": partition,
        "model": {"class": "RuleAwareTrajectoryScorer", "hidden_dim": hidden_dim},
        "loss": {
            "binary_reference_improvement_weight": 1.0,
            "all_pairs_ranknet_weight": 0.75,
            "delta_regression_weight": 0.25,
            "catastrophic_collision_weight": 0.05,
        },
        "reference": {
            "v1": "normalized_base + 1.6 * zscore(protocol_proxy)",
            "v2": "normalized_base + 3.2 * zscore(protocol_proxy)",
        },
        "selection_grid": {
            "switch_margin": list(SWITCH_MARGIN_GRID),
            "safety_component_tolerance": list(SAFETY_TOLERANCE_GRID),
            "kinematic_relu_penalty_tolerance": KINEMATIC_TOLERANCE,
            "combinations_per_protocol": 24,
        },
        "best_epoch": best_epoch,
        "best_objective": list(best_objective),
        "selected_calibration": best_calibration,
        "selected_parameters": selected_parameters,
        "history": history,
        "arguments": {
            name: [str(item.expanduser().resolve()) for item in value]
            if name == "cache_root"
            else str(value.expanduser().resolve())
            if isinstance(value, Path)
            else value
            for name, value in vars(args).items()
        },
        "reproduction_command": [sys.executable, *sys.argv],
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "host": socket.gethostname(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "physical_gpu_policy": "CUDA_VISIBLE_DEVICES=0,3",
        },
        "holdout_report": {"path": str(paths["holdout"]), "sha256": _sha256(paths["holdout"])},
    }
    injection_state = {
        f"agent.model._rule_scorer.{name}": tensor.clone() for name, tensor in best_state.items()
    }
    _atomic_torch(
        paths["state"],
        {
            "schema": STATE_SCHEMA,
            "state_dict": best_state,
            "injection_state_dict": injection_state,
            "metadata": copy.deepcopy(metadata),
        },
    )
    metadata["state_file"] = str(paths["state"])
    metadata["state_sha256"] = _sha256(paths["state"])
    _atomic_json(paths["training"], metadata)

    checks: Dict[str, Any] = {}
    for protocol in ("v1", "v2"):
        result = holdout_protocols[protocol]
        checks[protocol] = {
            "score_gain": result["score_gain"],
            "minimum_score_gain": MINIMUM_HOLDOUT_GAIN[protocol],
            "score_pass": result["score_gain"] >= MINIMUM_HOLDOUT_GAIN[protocol],
            "collision_rate_delta": result["collision_rate_delta"],
            "collision_pass": result["collision_rate_delta"] <= 0.0,
        }
    protocol_pass = {
        protocol: check["score_pass"] and check["collision_pass"]
        for protocol, check in checks.items()
    }
    allowed_benchmarks = []
    if protocol_pass["v1"]:
        allowed_benchmarks.append("navsim_v1_navtest")
    if protocol_pass["v2"]:
        allowed_benchmarks.extend(("navsim_v2_navtest", "navsim_v2_navhard"))
    gate_pass = all(protocol_pass.values())
    gate = {
        "schema": GATE_SCHEMA,
        "created_at": _utc_now(),
        "gate_pass": gate_pass,
        "protocol_pass": protocol_pass,
        "checks": checks,
        "training_report": {"path": str(paths["training"]), "sha256": _sha256(paths["training"])},
        "state": {"path": str(paths["state"]), "sha256": _sha256(paths["state"])},
        "holdout_report": {"path": str(paths["holdout"]), "sha256": _sha256(paths["holdout"])},
        "fixed_subset": partition["fixed_subset_artifact"],
        "full_evaluation": {
            "allowed": bool(allowed_benchmarks),
            "allowed_benchmarks": allowed_benchmarks,
            "launched": False,
            "reason": (
                "fresh NAVTRAIN fixed-subset gate passed for listed benchmarks"
                if allowed_benchmarks
                else "fresh NAVTRAIN fixed-subset gate failed for every benchmark"
            ),
        },
    }
    _atomic_json(paths["gate"], gate)
    print(json.dumps(gate, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
