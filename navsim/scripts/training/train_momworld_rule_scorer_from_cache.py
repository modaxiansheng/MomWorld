#!/usr/bin/env python3
"""Train the tiny MomWorld v1/v2 rule scorer from cached NAVTRAIN features.

The output contains both the scorer-local state dict and checkpoint-compatible
keys prefixed with ``agent.model._rule_scorer.``.  It never rewrites or embeds
the source MomWorld checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import os
import random
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from navsim.agents.momworld.momworld_model import RuleAwareTrajectoryScorer


CACHE_SCHEMA = "momworld-rule-features-v2"
STATE_SCHEMA = "momworld-rule-scorer-state-v1"
TUNING_SCHEMA = "momworld-rule-tuning-v1"


@dataclass(frozen=True)
class CacheRecord:
    path: Path
    token: str
    dataset: str
    features: np.ndarray
    target_v1: np.ndarray
    target_v2: np.ndarray
    target_collision: np.ndarray
    predicted_collision: np.ndarray
    kinematic_penalty: np.ndarray
    momentum_error: np.ndarray
    normalized_base: np.ndarray
    candidate_is_finite: np.ndarray


@dataclass(frozen=True)
class RuleSelectionParameters:
    learned_weight: float
    collision_weight: float
    kinematic_weight: float
    momentum_weight: float
    collision_threshold: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "learned_weight": float(self.learned_weight),
            "collision_weight": float(self.collision_weight),
            "kinematic_weight": float(self.kinematic_weight),
            "momentum_weight": float(self.momentum_weight),
            "collision_threshold": float(self.collision_threshold),
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _git_commit(repository: Path) -> Optional[str]:
    commit: Optional[str] = None
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        marker = repository / ".momworld_source_commit"
        if marker.is_file():
            commit = marker.read_text(encoding="utf-8").strip()
    if commit is None:
        return None
    if not re.fullmatch(r"[0-9a-fA-F]{40}", commit):
        raise RuntimeError(f"Invalid MomWorld source commit marker: {commit!r}")
    return commit.lower()


def _discover_manifests(cache_roots: Sequence[Path]) -> List[Tuple[Path, Dict[str, Any]]]:
    manifests: List[Tuple[Path, Dict[str, Any]]] = []
    seen: set[Path] = set()
    for root in cache_roots:
        candidates = [root] if root.name == "manifest.json" else root.rglob("manifest.json")
        for path in candidates:
            path = path.resolve()
            if path in seen:
                continue
            with path.open("r", encoding="utf-8") as stream:
                payload = json.load(stream)
            if payload.get("schema") == CACHE_SCHEMA:
                manifests.append((path, payload))
                seen.add(path)
    return manifests


def _validate_manifests(
    manifests: Sequence[Tuple[Path, Dict[str, Any]]], allow_incomplete: bool
) -> Dict[str, Any]:
    if not manifests:
        raise RuntimeError("No MomWorld rule-feature manifests were found")
    checkpoint_hashes = {payload.get("checkpoint_sha256") for _, payload in manifests}
    topks = {payload.get("rule_candidate_topk") for _, payload in manifests}
    feature_dims = {payload.get("rule_feature_dim") for _, payload in manifests}
    hidden_dims = {payload.get("rule_scorer_hidden_dim") for _, payload in manifests}
    source_splits = {payload.get("source_split") for _, payload in manifests}
    label_scopes = {payload.get("label_scope") for _, payload in manifests}
    source_commits = {payload.get("source_commit") for _, payload in manifests}
    if len(checkpoint_hashes) != 1 or None in checkpoint_hashes:
        raise RuntimeError(f"Caches use different source checkpoints: {checkpoint_hashes}")
    if (
        len(topks) != 1
        or None in topks
        or len(feature_dims) != 1
        or None in feature_dims
        or len(hidden_dims) != 1
        or None in hidden_dims
    ):
        raise RuntimeError("Caches disagree on candidate count or feature dimension")
    if source_splits != {"navtrain"} or label_scopes != {"navtrain-only"}:
        raise RuntimeError(
            "Rule tuning accepts only audited NAVTRAIN train/val caches; got "
            f"source_split={source_splits}, label_scope={label_scopes}"
        )
    if len(source_commits) != 1 or not re.fullmatch(
        r"[0-9a-fA-F]{40}", str(next(iter(source_commits), ""))
    ):
        raise RuntimeError(
            f"Caches do not identify one valid source commit: {source_commits}"
        )

    shard_groups: Dict[Tuple[str, int], set[int]] = {}
    dataset_sizes: Dict[str, set[int]] = {}
    expected_samples = {"train": 0, "val": 0}
    for path, payload in manifests:
        if payload.get("status") != "complete" and not allow_incomplete:
            raise RuntimeError(f"Cache shard is not complete: {path}")
        dataset = str(payload.get("dataset"))
        num_shards = int(payload.get("num_shards", 0))
        shard_id = int(payload.get("shard_id", -1))
        dataset_size = int(payload.get("dataset_size", -1))
        assigned_samples = int(payload.get("assigned_samples", -1))
        completed_samples = int(payload.get("completed_samples", -1))
        if (
            dataset not in {"train", "val"}
            or num_shards < 1
            or not 0 <= shard_id < num_shards
            or dataset_size < 1
            or assigned_samples < 0
            or completed_samples < 0
            or completed_samples > assigned_samples
        ):
            raise RuntimeError(f"Invalid shard metadata in {path}")
        if payload.get("status") == "complete" and completed_samples != assigned_samples:
            raise RuntimeError(
                f"Complete shard has {completed_samples}/{assigned_samples} samples: {path}"
            )
        dataset_sizes.setdefault(dataset, set()).add(dataset_size)
        expected_samples[dataset] += (
            assigned_samples
            if payload.get("status") == "complete"
            else completed_samples
        )
        key = (dataset, num_shards)
        if shard_id in shard_groups.setdefault(key, set()):
            raise RuntimeError(f"Duplicate cache shard {dataset} {shard_id}/{num_shards}")
        shard_groups[key].add(shard_id)
    if not allow_incomplete:
        for (dataset, num_shards), shard_ids in shard_groups.items():
            expected = set(range(num_shards))
            if shard_ids != expected:
                raise RuntimeError(
                    f"Missing {dataset} cache shards: {sorted(expected - shard_ids)}"
                )
        if not any(dataset == "train" for dataset, _ in shard_groups):
            raise RuntimeError("No train cache shards")
        if not any(dataset == "val" for dataset, _ in shard_groups):
            raise RuntimeError("No validation cache shards")
        for dataset in ("train", "val"):
            if len(dataset_sizes.get(dataset, set())) != 1:
                raise RuntimeError(f"Caches disagree on {dataset} dataset size")
            dataset_size = next(iter(dataset_sizes[dataset]))
            if expected_samples[dataset] != dataset_size:
                raise RuntimeError(
                    f"Incomplete {dataset} token coverage: "
                    f"{expected_samples[dataset]} / {dataset_size}"
                )
    return {
        "checkpoint_sha256": next(iter(checkpoint_hashes)),
        "rule_candidate_topk": next(iter(topks)),
        "rule_feature_dim": next(iter(feature_dims)),
        "rule_scorer_hidden_dim": next(iter(hidden_dims)),
        "source_split": "navtrain",
        "label_scope": "navtrain-only",
        "source_commit": str(next(iter(source_commits))).lower(),
        "expected_samples": expected_samples,
        "manifest_sha256": {
            str(path): _sha256(path) for path, _ in sorted(manifests, key=lambda item: str(item[0]))
        },
    }


def _discover_npz(cache_roots: Sequence[Path]) -> List[Path]:
    files: set[Path] = set()
    for root in cache_roots:
        if root.suffix == ".npz":
            files.add(root.resolve())
        else:
            files.update(path.resolve() for path in root.rglob("*.npz"))
    return sorted(files)


def _load_records(
    paths: Iterable[Path], expected_checkpoint_sha256: str
) -> Dict[str, List[CacheRecord]]:
    records: Dict[str, List[CacheRecord]] = {"train": [], "val": []}
    # The competition training split intentionally contains every validation
    # log.  A token may therefore occur once in each split, but it must never
    # occur twice within the same split/shard collection.
    token_locations: Dict[Tuple[str, str], Path] = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            if "schema" not in data.files:
                continue
            schema = str(data["schema"].item())
            token = str(data["token"].item())
            dataset = str(data["dataset"].item())
            checkpoint_sha256 = str(data["checkpoint_sha256"].item())
            if schema != CACHE_SCHEMA:
                continue
            if checkpoint_sha256 != expected_checkpoint_sha256:
                raise RuntimeError(
                    f"Cache file {path} came from checkpoint {checkpoint_sha256}, "
                    f"expected {expected_checkpoint_sha256}"
                )
            if dataset not in records:
                raise ValueError(f"Unsupported cached dataset {dataset!r} in {path}")
            token_key = (dataset, token)
            if token_key in token_locations:
                raise RuntimeError(
                    f"Duplicate {dataset} token {token!r}: "
                    f"{token_locations[token_key]} and {path}"
                )
            token_locations[token_key] = path
            record = CacheRecord(
                path=path,
                token=token,
                dataset=dataset,
                features=np.asarray(data["features"], dtype=np.float32).copy(),
                target_v1=np.asarray(data["target_v1"], dtype=np.float32).copy(),
                target_v2=np.asarray(data["target_v2"], dtype=np.float32).copy(),
                target_collision=np.asarray(
                    data["target_collision"], dtype=np.float32
                ).copy(),
                predicted_collision=np.asarray(
                    data["predicted_collision"], dtype=np.float32
                ).copy(),
                kinematic_penalty=np.asarray(
                    data["kinematic_penalty"], dtype=np.float32
                ).copy(),
                momentum_error=np.asarray(
                    data["momentum_error"], dtype=np.float32
                ).copy(),
                normalized_base=np.asarray(
                    data["normalized_base"], dtype=np.float32
                ).copy(),
                candidate_is_finite=np.asarray(
                    data["candidate_is_finite"], dtype=np.bool_
                ).copy(),
            )
        shapes = {
            record.features.shape[0],
            record.target_v1.shape[0],
            record.target_v2.shape[0],
            record.target_collision.shape[0],
            record.predicted_collision.shape[0],
            record.kinematic_penalty.shape[0],
            record.momentum_error.shape[0],
            record.normalized_base.shape[0],
            record.candidate_is_finite.shape[0],
        }
        if len(shapes) != 1 or record.features.ndim != 2:
            raise ValueError(f"Inconsistent candidate shapes in {path}")
        candidate_count = record.features.shape[0]
        if any(
            value.shape != (candidate_count,)
            for value in (
                record.target_v1,
                record.target_v2,
                record.target_collision,
                record.predicted_collision,
                record.kinematic_penalty,
                record.momentum_error,
                record.normalized_base,
                record.candidate_is_finite,
            )
        ):
            raise ValueError(f"Non-vector per-candidate field in {path}")
        for value in (
            record.features,
            record.target_v1,
            record.target_v2,
            record.target_collision,
            record.predicted_collision,
            record.kinematic_penalty,
            record.momentum_error,
            record.normalized_base,
        ):
            if not np.isfinite(value).all():
                raise FloatingPointError(f"Non-finite cache values in {path}")
        records[dataset].append(record)
    return records


def _exclude_validation_overlap(
    records: Mapping[str, Sequence[CacheRecord]],
) -> Tuple[Dict[str, List[CacheRecord]], Dict[str, int]]:
    """Return a strict train/val partition from competition-split caches.

    NAVSIM's competition training configuration includes the validation logs
    in ``train_logs``.  Feature-cache coverage is audited against that raw
    configuration first; this function then removes every validation token
    from the effective scorer training set so validation remains a genuine
    holdout for early stopping and rule-parameter tuning.
    """

    raw_train = list(records.get("train", ()))
    validation = list(records.get("val", ()))
    validation_tokens = {record.token for record in validation}
    effective_train = [
        record for record in raw_train if record.token not in validation_tokens
    ]
    effective_train_tokens = {record.token for record in effective_train}
    remaining_overlap = effective_train_tokens & validation_tokens
    if remaining_overlap:
        raise RuntimeError(
            "Effective scorer train/val token overlap remains after exclusion: "
            f"{sorted(remaining_overlap)[:10]}"
        )
    if not effective_train:
        raise RuntimeError(
            "Validation-overlap exclusion left no effective scorer training samples"
        )

    audit = {
        "raw_train": len(raw_train),
        "effective_train": len(effective_train),
        "val": len(validation),
        "overlap_excluded": len(raw_train) - len(effective_train),
    }
    return {"train": effective_train, "val": validation}, audit


def _stack_dataset(records: Sequence[CacheRecord]) -> TensorDataset:
    if not records:
        raise RuntimeError("Cannot train with an empty cache split")
    return TensorDataset(
        torch.from_numpy(np.stack([record.features for record in records])),
        torch.from_numpy(np.stack([record.target_v1 for record in records])),
        torch.from_numpy(np.stack([record.target_v2 for record in records])),
        torch.from_numpy(np.stack([record.target_collision for record in records])),
        torch.from_numpy(np.stack([record.predicted_collision for record in records])),
    )


def _pairwise_best_rank_loss(
    logits: torch.Tensor, targets: torch.Tensor, margin: float
) -> torch.Tensor:
    best_indices = targets.argmax(dim=1)
    batch_indices = torch.arange(logits.shape[0], device=logits.device)
    best_logits = logits[batch_indices, best_indices][:, None]
    best_targets = targets[batch_indices, best_indices][:, None]
    competitor_mask = targets < (best_targets - 1e-3)
    violations = F.relu(float(margin) - best_logits + logits)
    return (
        (violations * competitor_mask.to(violations.dtype)).sum()
        / competitor_mask.sum().clamp(min=1)
    )


def _losses(
    scorer: RuleAwareTrajectoryScorer,
    features: torch.Tensor,
    target_v1: torch.Tensor,
    target_v2: torch.Tensor,
    rank_margin: float,
    scorer_weight: float,
    rank_weight: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    logits_v1 = scorer(features, "v1")
    logits_v2 = scorer(features, "v2")
    bce_v1 = F.binary_cross_entropy_with_logits(logits_v1, target_v1)
    bce_v2 = F.binary_cross_entropy_with_logits(logits_v2, target_v2)
    scorer_loss = 0.5 * (bce_v1 + bce_v2)
    rank_loss = 0.5 * (
        _pairwise_best_rank_loss(logits_v1, target_v1, rank_margin)
        + _pairwise_best_rank_loss(logits_v2, target_v2, rank_margin)
    )
    total = float(scorer_weight) * scorer_loss + float(rank_weight) * rank_loss
    return total, {
        "loss": total,
        "scorer_loss": scorer_loss,
        "rank_loss": rank_loss,
        "bce_v1": bce_v1,
        "bce_v2": bce_v2,
        "mae_v1": (torch.sigmoid(logits_v1) - target_v1).abs().mean(),
        "mae_v2": (torch.sigmoid(logits_v2) - target_v2).abs().mean(),
    }


def _run_epoch(
    scorer: RuleAwareTrajectoryScorer,
    loader: DataLoader,
    device: torch.device,
    rank_margin: float,
    scorer_weight: float,
    rank_weight: float,
    optimizer: Optional[torch.optim.Optimizer],
) -> Dict[str, float]:
    training = optimizer is not None
    scorer.train(training)
    totals: Dict[str, float] = {}
    samples = 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for features, target_v1, target_v2, target_collision, predicted_collision in loader:
            features = features.to(device, non_blocking=True)
            target_v1 = target_v1.to(device, non_blocking=True)
            target_v2 = target_v2.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            loss, metrics = _losses(
                scorer,
                features,
                target_v1,
                target_v2,
                rank_margin,
                scorer_weight,
                rank_weight,
            )
            if training:
                loss.backward()
                optimizer.step()
            batch_size = features.shape[0]
            samples += batch_size
            for name, value in metrics.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach()) * batch_size
            collision_mae = (predicted_collision - target_collision).abs().mean()
            totals["collision_proxy_mae"] = (
                totals.get("collision_proxy_mae", 0.0)
                + float(collision_mae) * batch_size
            )
    return {name: value / max(samples, 1) for name, value in totals.items()}


def _eligible_candidate_mask(
    predicted_collision: np.ndarray,
    candidate_is_finite: np.ndarray,
    collision_threshold: float,
) -> np.ndarray:
    """Mirror MomWorld's safe/finite/all candidate fallback exactly."""
    predicted_collision = np.asarray(predicted_collision, dtype=np.float32)
    candidate_is_finite = np.asarray(candidate_is_finite, dtype=np.bool_)
    if predicted_collision.ndim != 2:
        raise ValueError("predicted_collision must have shape [samples, candidates]")
    if candidate_is_finite.shape != predicted_collision.shape:
        raise ValueError("candidate_is_finite shape does not match collision risks")
    if not np.isfinite(collision_threshold):
        raise ValueError("collision_threshold must be finite")
    safe_mask = (
        predicted_collision < float(collision_threshold)
    ) & candidate_is_finite
    has_safe_candidate = safe_mask.any(axis=1, keepdims=True)
    has_finite_candidate = candidate_is_finite.any(axis=1, keepdims=True)
    return np.where(
        has_safe_candidate,
        safe_mask,
        np.where(
            has_finite_candidate,
            candidate_is_finite,
            np.ones_like(candidate_is_finite),
        ),
    )


def _sigmoid_numpy(value: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(value, dtype=np.float32), -80.0, 80.0)
    return (1.0 / (1.0 + np.exp(-value))).astype(np.float32, copy=False)


def _select_rule_candidates(
    normalized_base: np.ndarray,
    learned_logits: np.ndarray,
    predicted_collision: np.ndarray,
    kinematic_penalty: np.ndarray,
    momentum_error: np.ndarray,
    candidate_is_finite: np.ndarray,
    parameters: RuleSelectionParameters,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return top-k indices using the same formula/mask as model inference."""
    arrays = [
        np.asarray(value, dtype=np.float32)
        for value in (
            normalized_base,
            learned_logits,
            predicted_collision,
            kinematic_penalty,
            momentum_error,
        )
    ]
    shape = arrays[0].shape
    if len(shape) != 2 or any(value.shape != shape for value in arrays[1:]):
        raise ValueError("All rule selection arrays must share shape [samples, candidates]")
    if not all(np.isfinite(value).all() for value in arrays):
        raise FloatingPointError("Rule selection inputs contain non-finite values")
    eligible_mask = _eligible_candidate_mask(
        arrays[2], candidate_is_finite, parameters.collision_threshold
    )
    combined_scores = (
        arrays[0]
        + float(parameters.learned_weight) * _sigmoid_numpy(arrays[1])
        - float(parameters.collision_weight) * arrays[2]
        - float(parameters.kinematic_weight) * arrays[3]
        - float(parameters.momentum_weight) * arrays[4]
    )
    combined_scores = np.nan_to_num(
        combined_scores, nan=-1e9, posinf=1e9, neginf=-1e9
    )
    combined_scores = np.where(
        eligible_mask, combined_scores, np.finfo(combined_scores.dtype).min
    )
    return combined_scores.argmax(axis=1).astype(np.int64), eligible_mask


def _selection_metrics(
    selected_indices: np.ndarray,
    eligible_mask: np.ndarray,
    target_score: np.ndarray,
    target_collision: np.ndarray,
    predicted_collision: np.ndarray,
    candidate_is_finite: np.ndarray,
    collision_threshold: float,
) -> Dict[str, float]:
    sample_indices = np.arange(selected_indices.shape[0])
    selected_target = target_score[sample_indices, selected_indices]
    selected_target_collision = target_collision[sample_indices, selected_indices]
    selected_predicted_collision = predicted_collision[
        sample_indices, selected_indices
    ]
    safe_mask = (
        predicted_collision < float(collision_threshold)
    ) & candidate_is_finite
    return {
        "mean_target_score": float(selected_target.mean()),
        "mean_target_collision_risk": float(selected_target_collision.mean()),
        "target_collision_rate_ge_0_5": float(
            (selected_target_collision >= 0.5).mean()
        ),
        "mean_predicted_collision_risk": float(
            selected_predicted_collision.mean()
        ),
        "selected_predicted_collision_rate_ge_threshold": float(
            (selected_predicted_collision >= float(collision_threshold)).mean()
        ),
        "safe_candidate_available_rate": float(safe_mask.any(axis=1).mean()),
        "finite_candidate_available_rate": float(
            candidate_is_finite.any(axis=1).mean()
        ),
        "mean_eligible_candidates": float(eligible_mask.sum(axis=1).mean()),
    }


def _evaluate_rule_parameters(
    arrays: Mapping[str, np.ndarray],
    learned_logits: np.ndarray,
    target_score: np.ndarray,
    parameters: RuleSelectionParameters,
) -> Dict[str, Any]:
    selected_indices, eligible_mask = _select_rule_candidates(
        arrays["normalized_base"],
        learned_logits,
        arrays["predicted_collision"],
        arrays["kinematic_penalty"],
        arrays["momentum_error"],
        arrays["candidate_is_finite"],
        parameters,
    )
    return {
        "parameters": parameters.as_dict(),
        "metrics": _selection_metrics(
            selected_indices,
            eligible_mask,
            target_score,
            arrays["target_collision"],
            arrays["predicted_collision"],
            arrays["candidate_is_finite"],
            parameters.collision_threshold,
        ),
    }


def _grid_search_rule_parameters(
    arrays: Mapping[str, np.ndarray],
    learned_logits: np.ndarray,
    target_score: np.ndarray,
    grids: Mapping[str, Sequence[float]],
) -> Tuple[Dict[str, Any], int]:
    names = (
        "learned_weight",
        "collision_weight",
        "kinematic_weight",
        "momentum_weight",
        "collision_threshold",
    )
    values = [tuple(float(value) for value in grids[name]) for name in names]
    if any(not value for value in values):
        raise ValueError("Rule tuning grids must not be empty")
    best: Optional[Dict[str, Any]] = None
    combinations = 0
    for combination in itertools.product(*values):
        parameters = RuleSelectionParameters(*combination)
        result = _evaluate_rule_parameters(
            arrays, learned_logits, target_score, parameters
        )
        combinations += 1
        metrics = result["metrics"]
        key = (
            metrics["mean_target_score"],
            -metrics["mean_target_collision_risk"],
            -metrics["mean_predicted_collision_risk"],
        )
        if best is None or key > best["_search_key"]:
            best = {**result, "_search_key": key}
    if best is None:
        raise RuntimeError("Rule tuning grid produced no candidates")
    best.pop("_search_key")
    return best, combinations


def _parse_float_grid(value: str) -> List[float]:
    grid = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        parsed = float(item)
        if not np.isfinite(parsed):
            raise ValueError(f"Non-finite rule grid value: {item!r}")
        if parsed not in grid:
            grid.append(parsed)
    if not grid:
        raise ValueError("Rule tuning grid must contain at least one value")
    return grid


def _predict_rule_logits(
    scorer: RuleAwareTrajectoryScorer,
    records: Sequence[CacheRecord],
    device: torch.device,
    batch_size: int,
) -> Dict[str, np.ndarray]:
    outputs: Dict[str, List[np.ndarray]] = {"v1": [], "v2": []}
    scorer.eval()
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            features = torch.from_numpy(
                np.stack(
                    [record.features for record in records[start : start + batch_size]]
                )
            ).to(device)
            for protocol in ("v1", "v2"):
                outputs[protocol].append(
                    scorer(features, protocol).float().cpu().numpy()
                )
    return {
        protocol: np.concatenate(chunks, axis=0)
        for protocol, chunks in outputs.items()
    }


def _stack_tuning_arrays(records: Sequence[CacheRecord]) -> Dict[str, np.ndarray]:
    return {
        name: np.stack([getattr(record, name) for record in records])
        for name in (
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


def _cpu_state_dict(module: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in module.state_dict().items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=None,
        help="Must match the cache/evaluation scorer; defaults to cache metadata",
    )
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--rank-margin", type=float, default=0.05)
    parser.add_argument("--scorer-weight", type=float, default=1.0)
    parser.add_argument("--rank-weight", type=float, default=0.25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--memory-fraction", type=float, default=0.02)
    parser.add_argument("--baseline-learned-weight", type=float, default=1.0)
    parser.add_argument("--baseline-collision-weight", type=float, default=4.0)
    parser.add_argument("--baseline-kinematic-weight", type=float, default=0.25)
    parser.add_argument("--baseline-momentum-weight", type=float, default=0.10)
    parser.add_argument("--baseline-collision-threshold", type=float, default=0.65)
    parser.add_argument("--learned-weight-grid", default="0,0.5,1,2")
    parser.add_argument("--collision-weight-grid", default="0,2,4,6")
    parser.add_argument("--kinematic-weight-grid", default="0,0.25,0.5")
    parser.add_argument("--momentum-weight-grid", default="0,0.1,0.25")
    parser.add_argument("--collision-threshold-grid", default="0.35,0.5,0.65,0.8")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cache_roots = [path.expanduser().resolve() for path in args.cache_root]
    for root in cache_roots:
        if not root.exists():
            raise FileNotFoundError(root)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "rule_scorer_state.pt"
    report_path = output_dir / "training_report.json"
    tuning_path = output_dir / "rule_tuning.json"
    if not args.overwrite and (
        state_path.exists() or report_path.exists() or tuning_path.exists()
    ):
        raise FileExistsError(
            f"Refusing to overwrite existing scorer output in {output_dir}"
        )

    manifests = _discover_manifests(cache_roots)
    cache_audit = _validate_manifests(manifests, args.allow_incomplete)
    cached_hidden_dim = int(cache_audit["rule_scorer_hidden_dim"])
    hidden_dim = cached_hidden_dim if args.hidden_dim is None else int(args.hidden_dim)
    if hidden_dim != cached_hidden_dim:
        raise ValueError(
            f"hidden-dim {hidden_dim} is incompatible with cache/eval model "
            f"dimension {cached_hidden_dim}"
        )
    records = _load_records(
        _discover_npz(cache_roots), str(cache_audit["checkpoint_sha256"])
    )
    if not records["train"] or not records["val"]:
        raise RuntimeError("Both train and val feature caches are required")
    if not args.allow_incomplete:
        for split in ("train", "val"):
            expected = int(cache_audit["expected_samples"][split])
            if len(records[split]) != expected:
                raise RuntimeError(
                    f"Discovered {len(records[split])} {split} cache files, "
                    f"expected {expected}"
                )
    records, split_partition = _exclude_validation_overlap(records)
    cache_audit["split_partition"] = copy.deepcopy(split_partition)
    print(
        json.dumps({"split_partition": split_partition}, sort_keys=True),
        flush=True,
    )
    train_dataset = _stack_dataset(records["train"])
    val_dataset = _stack_dataset(records["val"])
    feature_shape = train_dataset.tensors[0].shape[1:]
    if tuple(val_dataset.tensors[0].shape[1:]) != tuple(feature_shape):
        raise RuntimeError("Train and validation feature shapes differ")
    if int(feature_shape[-1]) != int(cache_audit["rule_feature_dim"]):
        raise RuntimeError("Feature data disagrees with cache manifests")
    if int(feature_shape[0]) != int(cache_audit["rule_candidate_topk"]):
        raise RuntimeError("Candidate data disagrees with cache manifests")

    generator = torch.Generator().manual_seed(args.seed)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": str(args.device).startswith("cuda"),
    }
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=generator, **loader_kwargs
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        if not 0.0 < args.memory_fraction <= 1.0:
            raise ValueError("memory-fraction must be in (0, 1]")
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device=device)
    scorer = RuleAwareTrajectoryScorer(
        SimpleNamespace(rule_scorer_hidden_dim=hidden_dim)
    ).to(device)
    optimizer = torch.optim.AdamW(
        scorer.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    history: List[Dict[str, Any]] = []
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val = float("inf")
    best_epoch = -1
    stale_epochs = 0
    for epoch in range(args.epochs):
        train_metrics = _run_epoch(
            scorer,
            train_loader,
            device,
            args.rank_margin,
            args.scorer_weight,
            args.rank_weight,
            optimizer,
        )
        val_metrics = _run_epoch(
            scorer,
            val_loader,
            device,
            args.rank_margin,
            args.scorer_weight,
            args.rank_weight,
            None,
        )
        epoch_record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(epoch_record)
        print(json.dumps(epoch_record, sort_keys=True), flush=True)
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_epoch = epoch
            best_state = _cpu_state_dict(scorer)
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("No finite scorer state was produced")

    # The learned heads are selected by validation loss first.  Only then are
    # the inference rule weights tuned, exclusively on the cached NAVTRAIN
    # validation subset.  No evaluator or test/hard labels are available here.
    scorer.load_state_dict(best_state)
    val_logits = _predict_rule_logits(
        scorer, records["val"], device, args.batch_size
    )
    tuning_arrays = _stack_tuning_arrays(records["val"])
    baseline_parameters = RuleSelectionParameters(
        learned_weight=args.baseline_learned_weight,
        collision_weight=args.baseline_collision_weight,
        kinematic_weight=args.baseline_kinematic_weight,
        momentum_weight=args.baseline_momentum_weight,
        collision_threshold=args.baseline_collision_threshold,
    )
    parsed_grids = {
        "learned_weight": _parse_float_grid(args.learned_weight_grid),
        "collision_weight": _parse_float_grid(args.collision_weight_grid),
        "kinematic_weight": _parse_float_grid(args.kinematic_weight_grid),
        "momentum_weight": _parse_float_grid(args.momentum_weight_grid),
        "collision_threshold": _parse_float_grid(
            args.collision_threshold_grid
        ),
    }
    for name, baseline_value in baseline_parameters.as_dict().items():
        if baseline_value not in parsed_grids[name]:
            parsed_grids[name].append(baseline_value)
        parsed_grids[name] = sorted(parsed_grids[name])
    if any(value < 0.0 for grid in parsed_grids.values() for value in grid):
        raise ValueError("Rule tuning weights and thresholds must be non-negative")

    tuning_protocols: Dict[str, Any] = {}
    total_combinations = 0
    base_rank_parameters = RuleSelectionParameters(
        learned_weight=0.0,
        collision_weight=0.0,
        kinematic_weight=0.0,
        momentum_weight=0.0,
        collision_threshold=1.000001,
    )
    for protocol in ("v1", "v2"):
        target_score = tuning_arrays[f"target_{protocol}"]
        base_rank = _evaluate_rule_parameters(
            tuning_arrays,
            val_logits[protocol],
            target_score,
            base_rank_parameters,
        )
        baseline = _evaluate_rule_parameters(
            tuning_arrays,
            val_logits[protocol],
            target_score,
            baseline_parameters,
        )
        tuned, combinations = _grid_search_rule_parameters(
            tuning_arrays,
            val_logits[protocol],
            target_score,
            parsed_grids,
        )
        total_combinations += combinations
        tuned_metrics = tuned["metrics"]
        baseline_metrics = baseline["metrics"]
        base_rank_metrics = base_rank["metrics"]
        tuning_protocols[protocol] = {
            "base_rank_only": base_rank,
            "configured_baseline": baseline,
            "tuned": tuned,
            "improvement_vs_configured_baseline": {
                "mean_target_score": tuned_metrics["mean_target_score"]
                - baseline_metrics["mean_target_score"],
                "mean_target_collision_risk": tuned_metrics[
                    "mean_target_collision_risk"
                ]
                - baseline_metrics["mean_target_collision_risk"],
                "target_collision_rate_ge_0_5": tuned_metrics[
                    "target_collision_rate_ge_0_5"
                ]
                - baseline_metrics["target_collision_rate_ge_0_5"],
            },
            "improvement_vs_base_rank_only": {
                "mean_target_score": tuned_metrics["mean_target_score"]
                - base_rank_metrics["mean_target_score"],
                "mean_target_collision_risk": tuned_metrics[
                    "mean_target_collision_risk"
                ]
                - base_rank_metrics["mean_target_collision_risk"],
                "target_collision_rate_ge_0_5": tuned_metrics[
                    "target_collision_rate_ge_0_5"
                ]
                - base_rank_metrics["target_collision_rate_ge_0_5"],
            },
            "grid_combinations": combinations,
        }

    validation_tokens_sha256 = hashlib.sha256(
        "\n".join(record.token for record in records["val"]).encode("utf-8")
    ).hexdigest()
    rule_tuning: Dict[str, Any] = {
        "schema": TUNING_SCHEMA,
        "created_at": _utc_now(),
        "data_scope": {
            "source_split": "navtrain",
            "subset": "val",
            "samples": len(records["val"]),
            "tokens_sha256": validation_tokens_sha256,
            "label_scope": "navtrain-only",
            "evaluator_calls": 0,
            "excluded_splits": ["navtest", "navhard"],
        },
        "training_partition": copy.deepcopy(split_partition),
        "selection_semantics": {
            "safe_comparison": "predicted_collision < collision_threshold",
            "fallback_order": ["safe_and_finite", "finite", "all"],
            "learned_activation": "sigmoid",
        },
        "search_grid": parsed_grids,
        "total_grid_combinations": total_combinations,
        "protocols": tuning_protocols,
    }

    repository = Path(__file__).resolve().parents[2]
    arguments = {
        key: (
            [str(path) for path in value]
            if key == "cache_root"
            else str(value)
            if isinstance(value, Path)
            else value
        )
        for key, value in vars(args).items()
    }
    metadata: Dict[str, Any] = {
        "schema": STATE_SCHEMA,
        "created_at": _utc_now(),
        "source_commit": _git_commit(repository),
        "arguments": arguments,
        "cache_audit": cache_audit,
        "raw_train": split_partition["raw_train"],
        "effective_train": split_partition["effective_train"],
        "overlap_excluded": split_partition["overlap_excluded"],
        "train_samples": len(records["train"]),
        "val_samples": len(records["val"]),
        "candidate_topk": int(feature_shape[0]),
        "feature_dim": int(feature_shape[1]),
        "hidden_dim": hidden_dim,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "history": history,
        "rule_tuning": copy.deepcopy(rule_tuning),
    }
    injection_state = {
        f"agent.model._rule_scorer.{name}": tensor
        for name, tensor in best_state.items()
    }
    _atomic_torch_save(
        state_path,
        {
            "schema": STATE_SCHEMA,
            "state_dict": best_state,
            "injection_state_dict": injection_state,
            "metadata": copy.deepcopy(metadata),
        },
    )
    metadata["state_file"] = str(state_path)
    metadata["state_sha256"] = _sha256(state_path)
    rule_tuning["scorer_state_file"] = str(state_path)
    rule_tuning["scorer_state_sha256"] = metadata["state_sha256"]
    _atomic_json(report_path, metadata)
    _atomic_json(tuning_path, rule_tuning)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
