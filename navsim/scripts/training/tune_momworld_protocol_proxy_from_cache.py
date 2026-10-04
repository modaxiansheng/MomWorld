#!/usr/bin/env python3
"""Tune isolated MomWorld rank fusion using NAVTRAIN validation caches only.

The search never imports NAVTEST/NAVHARD metric caches and never invokes an
evaluator.  Hyperparameters are selected on a deterministic token-hash
calibration partition.  The holdout partition is evaluated exactly once after
selection and is never consulted by the search loop.  In hybrid mode this is a
holdout for the new fusion weights only: the frozen first-round scorer was
already selected with the complete NAVTRAIN validation split.  Strict
``--proxy-only`` mode forbids that artifact and consumes no scorer outputs.

This utility always reuses the immutable rule-feature cache and only hybrid
mode loads a scorer state.  It does not edit or emit a model checkpoint; the
selected weights are applied through the optional rank-fusion configuration in
``MomWorldConfig``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch


CACHE_SCHEMA = "momworld-rule-features-v2"
SCORER_SCHEMA = "momworld-rule-scorer-state-v1"
OUTPUT_SCHEMA = "momworld-protocol-proxy-tuning-v2"
REQUIRED_FIELDS = (
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


@dataclass(frozen=True)
class ValidationCache:
    tokens: Tuple[str, ...]
    arrays: Mapping[str, np.ndarray]
    audit: Mapping[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _token_set_sha256(tokens: Iterable[str]) -> str:
    return hashlib.sha256(
        "\n".join(sorted(str(token) for token in tokens)).encode("utf-8")
    ).hexdigest()


def _git_commit(repository: Path) -> str:
    """Resolve a reproducible source identity without trusting the CWD."""

    try:
        commit = subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        marker = repository / ".momworld_source_commit"
        if not marker.is_file():
            raise RuntimeError("Cannot resolve MomWorld source commit")
        commit = marker.read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"[0-9a-fA-F]{40}", commit):
        raise RuntimeError(f"Invalid MomWorld source commit: {commit!r}")
    return commit.lower()


def _source_audit(extra_relative_paths: Sequence[str] = ()) -> Dict[str, Any]:
    repository = Path(__file__).resolve().parents[2]
    relative_paths = (
        "navsim/agents/momworld/momworld_config.py",
        "navsim/agents/momworld/momworld_model.py",
        "scripts/training/tune_momworld_protocol_proxy_from_cache.py",
        "scripts/evaluation/prepare_momworld_rank_fusion_manifest.py",
        "scripts/evaluation/run_momworld_rule_scorer_navtest_v1.sh",
        "scripts/evaluation/run_momworld_rule_scorer_navsim_v2.sh",
    ) + tuple(extra_relative_paths)
    paths = {name: repository / name for name in relative_paths}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing rank-fusion source files: {missing}")
    try:
        tracked_status = subprocess.check_output(
            ["git", "-C", str(repository), "status", "--porcelain", "--", *relative_paths],
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()
    except (OSError, subprocess.CalledProcessError):
        tracked_status = None
    return {
        "repository": str(repository),
        "commit": _git_commit(repository),
        "source_paths_clean": (
            None if tracked_status is None else len(tracked_status) == 0
        ),
        "source_path_status": tracked_status,
        "file_sha256": {name: _sha256(path) for name, path in paths.items()},
        "runtime": {
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
    }


def _parse_grid(value: str) -> Tuple[float, ...]:
    result: List[float] = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        parsed = float(raw)
        if not math.isfinite(parsed) or parsed < 0.0:
            raise ValueError(f"Grid values must be finite and non-negative: {raw!r}")
        if parsed not in result:
            result.append(parsed)
    if not result:
        raise ValueError("Weight grid must not be empty")
    return tuple(sorted(result))


def _validation_root(cache_root: Path) -> Path:
    cache_root = cache_root.expanduser().resolve()
    candidate = cache_root / "val"
    return candidate if candidate.is_dir() else cache_root


def _validate_manifests(val_root: Path) -> Dict[str, Any]:
    manifests = sorted(val_root.glob("shard-*/manifest.json"))
    if not manifests:
        raise RuntimeError(f"No validation cache manifests under {val_root}")
    payloads = []
    for path in manifests:
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        if not (
            payload.get("schema") == CACHE_SCHEMA
            and payload.get("dataset") == "val"
            and payload.get("status") == "complete"
            and payload.get("source_split") == "navtrain"
            and payload.get("label_scope") == "navtrain-only"
        ):
            raise RuntimeError(f"Invalid NAVTRAIN validation manifest: {path}")
        payloads.append((path, payload))

    stable_fields = (
        "checkpoint_sha256",
        "rule_candidate_topk",
        "rule_feature_dim",
        "rule_scorer_hidden_dim",
        "source_commit",
        "dataset_size",
        "num_shards",
    )
    stable: Dict[str, Any] = {}
    for name in stable_fields:
        values = {payload.get(name) for _, payload in payloads}
        if len(values) != 1 or None in values:
            raise RuntimeError(f"Validation manifests disagree on {name}: {values}")
        stable[name] = next(iter(values))

    num_shards = int(stable["num_shards"])
    shard_ids = {int(payload["shard_id"]) for _, payload in payloads}
    if shard_ids != set(range(num_shards)):
        raise RuntimeError(
            f"Incomplete validation shard set: {sorted(shard_ids)} / {num_shards}"
        )
    completed = 0
    for path, payload in payloads:
        assigned = int(payload.get("assigned_samples", -1))
        shard_completed = int(payload.get("completed_samples", -1))
        if assigned < 0 or shard_completed != assigned:
            raise RuntimeError(f"Incomplete validation cache shard: {path}")
        completed += shard_completed
    if completed != int(stable["dataset_size"]):
        raise RuntimeError(
            f"Validation coverage mismatch: {completed} / {stable['dataset_size']}"
        )
    return {
        **stable,
        "completed_samples": completed,
        "manifest_sha256": {str(path): _sha256(path) for path, _ in payloads},
        "source_split": "navtrain",
        "label_scope": "navtrain-only",
    }


def _load_validation_cache(cache_root: Path) -> ValidationCache:
    val_root = _validation_root(cache_root)
    audit = _validate_manifests(val_root)
    paths = sorted(val_root.glob("shard-*/tokens/*.npz"))
    expected = int(audit["completed_samples"])
    if len(paths) != expected:
        raise RuntimeError(f"Found {len(paths)} validation records, expected {expected}")

    rows: Dict[str, List[np.ndarray]] = {name: [] for name in REQUIRED_FIELDS}
    tokens: List[str] = []
    seen_tokens: set[str] = set()
    loaded_records_digest = hashlib.sha256()
    expected_checkpoint = str(audit["checkpoint_sha256"])
    candidate_topk = int(audit["rule_candidate_topk"])
    feature_dim = int(audit["rule_feature_dim"])
    for path in paths:
        with np.load(path, allow_pickle=False) as payload:
            if str(payload["schema"].item()) != CACHE_SCHEMA:
                raise RuntimeError(f"Unexpected cache schema in {path}")
            if str(payload["dataset"].item()) != "val":
                raise RuntimeError(f"Non-validation record in {path}")
            if str(payload["checkpoint_sha256"].item()) != expected_checkpoint:
                raise RuntimeError(f"Checkpoint provenance mismatch in {path}")
            token = str(payload["token"].item())
            if token in seen_tokens:
                raise RuntimeError(f"Duplicate validation token: {token}")
            seen_tokens.add(token)
            tokens.append(token)
            loaded_records_digest.update(token.encode("utf-8"))
            loaded_records_digest.update(b"\0")
            for name in REQUIRED_FIELDS:
                value = np.asarray(payload[name]).copy()
                if value.dtype.kind == "f" and not np.isfinite(value).all():
                    raise FloatingPointError(f"Non-finite {name} in {path}")
                rows[name].append(value)
                loaded_records_digest.update(name.encode("utf-8"))
                loaded_records_digest.update(b"\0")
                loaded_records_digest.update(value.dtype.str.encode("ascii"))
                loaded_records_digest.update(b"\0")
                loaded_records_digest.update(
                    np.asarray(value.shape, dtype=np.int64).tobytes()
                )
                loaded_records_digest.update(np.ascontiguousarray(value).tobytes())

    arrays = {name: np.stack(values) for name, values in rows.items()}
    if arrays["features"].shape != (expected, candidate_topk, feature_dim):
        raise RuntimeError(
            f"Unexpected feature cache shape: {arrays['features'].shape}"
        )
    expected_vector_shape = (expected, candidate_topk)
    for name in REQUIRED_FIELDS[1:]:
        if arrays[name].shape != expected_vector_shape:
            raise RuntimeError(f"Unexpected {name} shape: {arrays[name].shape}")
    audit = {
        **audit,
        "loaded_records_sha256": loaded_records_digest.hexdigest(),
        "loaded_tokens_sha256": _token_set_sha256(tokens),
    }
    return ValidationCache(tuple(tokens), arrays, audit)


def _load_logits(
    scorer_state_path: Path,
    cache: ValidationCache,
    batch_size: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    # Lazy import keeps strict proxy_only mode independent of the scorer module
    # as well as its state and outputs.
    from navsim.agents.momworld.momworld_model import RuleAwareTrajectoryScorer

    scorer_state_path = scorer_state_path.expanduser().resolve()
    try:
        payload = torch.load(
            scorer_state_path, map_location=torch.device("cpu"), weights_only=False
        )
    except TypeError:
        payload = torch.load(scorer_state_path, map_location=torch.device("cpu"))
    if not isinstance(payload, Mapping) or payload.get("schema") != SCORER_SCHEMA:
        raise RuntimeError(f"Invalid scorer state: {scorer_state_path}")
    metadata = payload.get("metadata")
    state_dict = payload.get("state_dict")
    if not isinstance(metadata, Mapping) or not isinstance(state_dict, Mapping):
        raise RuntimeError("Scorer state is missing metadata or state_dict")
    scorer_source_commit = str(metadata.get("source_commit", "")).lower()
    if not re.fullmatch(r"[0-9a-f]{40}", scorer_source_commit):
        raise RuntimeError("Scorer state does not identify a valid source commit")
    state_cache_audit = metadata.get("cache_audit")
    if not isinstance(state_cache_audit, Mapping):
        raise RuntimeError("Scorer state is missing its cache provenance audit")
    if not (
        state_cache_audit.get("source_split") == "navtrain"
        and state_cache_audit.get("label_scope") == "navtrain-only"
    ):
        raise RuntimeError("Scorer state is not restricted to NAVTRAIN labels")
    for name in (
        "checkpoint_sha256",
        "rule_candidate_topk",
        "rule_feature_dim",
        "rule_scorer_hidden_dim",
        "source_commit",
    ):
        if state_cache_audit.get(name) != cache.audit.get(name):
            raise RuntimeError(
                f"Scorer/cache provenance mismatch for {name}: "
                f"{state_cache_audit.get(name)!r} != {cache.audit.get(name)!r}"
            )
    state_manifest_hashes = state_cache_audit.get("manifest_sha256")
    current_manifest_hashes = cache.audit.get("manifest_sha256")
    if not (
        isinstance(state_manifest_hashes, Mapping)
        and isinstance(current_manifest_hashes, Mapping)
        and {str(value) for value in current_manifest_hashes.values()}.issubset(
            {str(value) for value in state_manifest_hashes.values()}
        )
    ):
        raise RuntimeError("Scorer state did not audit the tuner's cache manifests")
    split_partition = state_cache_audit.get("split_partition")
    if not isinstance(split_partition, Mapping):
        raise RuntimeError("Scorer state does not audit train/validation separation")
    raw_train = int(split_partition.get("raw_train", -1))
    effective_train = int(split_partition.get("effective_train", -1))
    overlap_excluded = int(split_partition.get("overlap_excluded", -1))
    validation_samples = int(split_partition.get("val", -1))
    if (
        min(raw_train, effective_train, overlap_excluded, validation_samples) < 0
        or effective_train < 1
        or raw_train != effective_train + overlap_excluded
        or validation_samples != len(cache.tokens)
        or int(metadata.get("val_samples", -1)) != len(cache.tokens)
    ):
        raise RuntimeError(
            "Scorer state has an invalid or incompatible train/validation partition"
        )
    rule_tuning = metadata.get("rule_tuning")
    rule_scope = (
        rule_tuning.get("data_scope") if isinstance(rule_tuning, Mapping) else None
    )
    ordered_validation_hash = hashlib.sha256(
        "\n".join(cache.tokens).encode("utf-8")
    ).hexdigest()
    if not (
        isinstance(rule_scope, Mapping)
        and rule_scope.get("source_split") == "navtrain"
        and rule_scope.get("label_scope") == "navtrain-only"
        and int(rule_scope.get("evaluator_calls", -1)) == 0
        and int(rule_scope.get("samples", -1)) == len(cache.tokens)
        and str(rule_scope.get("tokens_sha256", "")).lower()
        == ordered_validation_hash
        and {"navtest", "navhard"}.issubset(
            {str(value).lower() for value in rule_scope.get("excluded_splits", ())}
        )
    ):
        raise RuntimeError("Scorer state rule tuning has unaudited data provenance")
    scorer_checkpoint = (
        state_cache_audit.get("checkpoint_sha256")
    )
    if scorer_checkpoint != cache.audit["checkpoint_sha256"]:
        raise RuntimeError("Scorer and validation cache use different checkpoints")

    hidden_dim = int(cache.audit["rule_scorer_hidden_dim"])
    scorer = RuleAwareTrajectoryScorer(
        SimpleNamespace(rule_scorer_hidden_dim=hidden_dim)
    )
    scorer.load_state_dict(state_dict, strict=True)
    scorer.eval()
    features = torch.from_numpy(cache.arrays["features"].astype(np.float32))
    logits: Dict[str, List[np.ndarray]] = {"v1": [], "v2": []}
    forward_calls = 0
    with torch.inference_mode():
        for start in range(0, features.shape[0], batch_size):
            batch = features[start : start + batch_size]
            for protocol in ("v1", "v2"):
                logits[protocol].append(scorer(batch, protocol).numpy())
                forward_calls += 1
    return (
        {name: np.concatenate(values, axis=0) for name, values in logits.items()},
        {
            "path": str(scorer_state_path),
            "sha256": _sha256(scorer_state_path),
            "schema": SCORER_SCHEMA,
            "source_commit": scorer_source_commit,
            "checkpoint_sha256": scorer_checkpoint,
            "used": True,
            "state_loaded": True,
            "scorer_module_imported": True,
            "forward_calls": forward_calls,
            "scorer_labels_used": True,
            "scorer_calibration_or_holdout_influence": True,
            "training_partition": {
                "raw_train": raw_train,
                "effective_train": effective_train,
                "overlap_excluded": overlap_excluded,
                "val": validation_samples,
            },
            "rule_tuning_scope": dict(rule_scope),
        },
    )


def _candidate_zscore(value: np.ndarray, epsilon: float) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if value.ndim != 2:
        raise ValueError("Candidate values must have shape [samples, candidates]")
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("zscore epsilon must be finite and positive")
    return (value - value.mean(axis=1, keepdims=True)) / np.maximum(
        value.std(axis=1, keepdims=True), float(epsilon)
    )


def _protocol_proxy(features: np.ndarray, protocol: str) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 3 or features.shape[-1] < 7:
        raise ValueError("features must have shape [samples, candidates, >=7]")
    no_collision, drivable, ttc, progress, direction, lane, traffic_light = [
        np.clip(features[..., index], 0.0, 1.0) for index in range(7)
    ]
    if protocol == "v1":
        return no_collision * drivable * (
            5.0 * progress + 5.0 * ttc + 2.0
        ) / 12.0
    if protocol == "v2":
        return no_collision * drivable * direction * traffic_light * (
            5.0 * progress + 5.0 * ttc + 2.0 * lane + 4.0
        ) / 16.0
    raise ValueError(f"Unsupported protocol: {protocol!r}")


def _partition_mask(tokens: Sequence[str], salt: str) -> np.ndarray:
    """Return the deterministic calibration half; complement is holdout."""

    if not salt:
        raise ValueError("partition salt must not be empty")
    unsalted = salt == "unsalted-sha256-v1"
    return np.asarray(
        [
            (
                hashlib.sha256(
                    (token if unsalted else f"{salt}:{token}").encode("utf-8")
                ).digest()[0]
                & 1
            )
            == 0
            for token in tokens
        ],
        dtype=np.bool_,
    )


def _select_candidates(
    score: np.ndarray,
    predicted_collision: np.ndarray,
    candidate_is_finite: np.ndarray,
    collision_threshold: float,
) -> np.ndarray:
    if not math.isfinite(float(collision_threshold)) or collision_threshold <= 1.0:
        raise ValueError(
            "This isolated candidate requires collision_threshold > 1 to disable "
            "the uncalibrated hard filter"
        )
    safe = (predicted_collision < collision_threshold) & candidate_is_finite
    has_safe = safe.any(axis=1, keepdims=True)
    has_finite = candidate_is_finite.any(axis=1, keepdims=True)
    eligible = np.where(
        has_safe,
        safe,
        np.where(has_finite, candidate_is_finite, np.ones_like(candidate_is_finite)),
    )
    return np.where(eligible, score, np.finfo(np.float32).min).argmax(axis=1)


def _selection_margin_metrics(
    score: np.ndarray,
    candidate_is_finite: np.ndarray,
    mask: np.ndarray,
) -> Dict[str, Any]:
    """Summarize sensitivity to small cross-device float32 differences."""

    score = np.asarray(score, dtype=np.float32)
    finite = np.asarray(candidate_is_finite, dtype=np.bool_)
    eligible = np.where(
        finite.any(axis=1, keepdims=True), finite, np.ones_like(finite)
    )
    masked_score = np.where(eligible, score, np.finfo(np.float32).min)
    if masked_score.shape[1] < 2:
        margins = np.full(
            masked_score.shape[0], np.finfo(np.float32).max, dtype=np.float32
        )
    else:
        top_two = np.partition(masked_score, -2, axis=1)[:, -2:]
        margins = top_two.max(axis=1) - top_two.min(axis=1)
    margins = margins[mask]
    return {
        "samples": int(margins.size),
        "minimum": float(margins.min()),
        "p01": float(np.quantile(margins, 0.01)),
        "median": float(np.median(margins)),
        "rate_le_1e_6": float((margins <= 1e-6).mean()),
        "rate_le_1e_5": float((margins <= 1e-5).mean()),
        "note": (
            "CPU/GPU float32 kernels are not bitwise identical; near-zero gaps "
            "can change argmax"
        ),
    }


def _metrics(
    selected: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
    target_score: np.ndarray,
    target_collision: np.ndarray,
    predicted_collision: np.ndarray,
) -> Dict[str, Any]:
    rows = np.arange(selected.shape[0])
    chosen_score = target_score[rows, selected][mask]
    chosen_collision = target_collision[rows, selected][mask]
    chosen_predicted_collision = predicted_collision[rows, selected][mask]
    reference_score = target_score[rows, reference][mask]
    difference = chosen_score - reference_score
    standard_error = (
        float(difference.std(ddof=1) / math.sqrt(difference.size))
        if difference.size > 1
        else 0.0
    )
    delta = float(difference.mean())
    return {
        "samples": int(mask.sum()),
        "mean_target_score": float(chosen_score.mean()),
        "mean_target_collision_risk": float(chosen_collision.mean()),
        "target_collision_rate_ge_0_5": float((chosen_collision >= 0.5).mean()),
        "mean_predicted_collision_risk": float(chosen_predicted_collision.mean()),
        "changed_vs_base_rate": float((selected[mask] != reference[mask]).mean()),
        "paired_score_delta_vs_base": delta,
        "paired_score_delta_standard_error": standard_error,
        "paired_score_delta_95ci_low": delta - 1.96 * standard_error,
        "paired_score_delta_95ci_high": delta + 1.96 * standard_error,
        "uncertainty_note": (
            "token-IID standard error; temporal/log correlation is not modeled"
        ),
    }


def _base_metrics(
    selected: np.ndarray,
    mask: np.ndarray,
    target_score: np.ndarray,
    target_collision: np.ndarray,
    predicted_collision: np.ndarray,
) -> Dict[str, Any]:
    metrics = _metrics(
        selected,
        selected,
        mask,
        target_score,
        target_collision,
        predicted_collision,
    )
    return metrics


def _top32_oracles(
    base_selected: np.ndarray,
    holdout: np.ndarray,
    target_score: np.ndarray,
    target_collision: np.ndarray,
    predicted_collision: np.ndarray,
) -> Dict[str, Any]:
    """Report diagnostic top-32 upper bounds; never return deployable indices."""

    score_oracle = target_score.argmax(axis=1)
    collision_minimum = target_collision.min(axis=1, keepdims=True)
    minimum_collision_candidates = target_collision <= collision_minimum + 1e-7
    collision_oracle = np.where(
        minimum_collision_candidates, target_score, np.finfo(np.float32).min
    ).argmax(axis=1)
    collision_safe = target_collision < 0.5
    has_collision_safe = collision_safe.any(axis=1, keepdims=True)
    collision_safe_eligible = np.where(
        has_collision_safe, collision_safe, minimum_collision_candidates
    )
    safe_score_oracle = np.where(
        collision_safe_eligible, target_score, np.finfo(np.float32).min
    ).argmax(axis=1)
    return {
        "scope": "diagnostic NAVTRAIN holdout only; never used for selection",
        "max_target_score": _metrics(
            score_oracle,
            base_selected,
            holdout,
            target_score,
            target_collision,
            predicted_collision,
        ),
        "max_target_score_subject_to_target_collision_lt_0_5": _metrics(
            safe_score_oracle,
            base_selected,
            holdout,
            target_score,
            target_collision,
            predicted_collision,
        ),
        "min_target_collision_then_max_target_score": _metrics(
            collision_oracle,
            base_selected,
            holdout,
            target_score,
            target_collision,
            predicted_collision,
        ),
    }


def _predicted_component_diagnostics(
    features: np.ndarray,
    base_selected: np.ndarray,
    holdout: np.ndarray,
    target_score: np.ndarray,
    target_collision: np.ndarray,
    predicted_collision: np.ndarray,
) -> Dict[str, Any]:
    names = (
        "no_at_fault_collisions",
        "drivable_area_compliance",
        "time_to_collision_within_bound",
        "ego_progress",
        "driving_direction_compliance",
        "lane_keeping",
        "traffic_light_compliance",
    )
    return {
        name: _metrics(
            np.asarray(features[..., index]).argmax(axis=1),
            base_selected,
            holdout,
            target_score,
            target_collision,
            predicted_collision,
        )
        for index, name in enumerate(names)
    }


def _tune_protocol(
    protocol: str,
    arrays: Mapping[str, np.ndarray],
    logits: Optional[np.ndarray],
    calibration: np.ndarray,
    holdout: np.ndarray,
    proxy_weights: Sequence[float],
    learned_weights: Sequence[float],
    collision_threshold: float,
    epsilon: float,
    proxy_only: bool = False,
) -> Dict[str, Any]:
    target_score = np.asarray(arrays[f"target_{protocol}"], dtype=np.float32)
    base = np.asarray(arrays["normalized_base"], dtype=np.float32)
    predicted_collision = np.asarray(
        arrays["predicted_collision"], dtype=np.float32
    )
    candidate_is_finite = np.asarray(arrays["candidate_is_finite"], dtype=np.bool_)
    target_collision = np.asarray(arrays["target_collision"], dtype=np.float32)
    proxy_zscore = _candidate_zscore(
        _protocol_proxy(arrays["features"], protocol), epsilon
    )
    if proxy_only:
        if tuple(float(value) for value in learned_weights) != (0.0,):
            raise ValueError(
                "proxy-only tuning requires learned-zscore-weight-grid exactly 0"
            )
        if logits is not None:
            raise ValueError("proxy-only tuning must not receive scorer logits")
        logit_zscore = np.zeros_like(base, dtype=np.float32)
    else:
        if logits is None:
            raise ValueError("hybrid tuning requires scorer logits")
        logit_zscore = _candidate_zscore(logits, epsilon)
    base_selected = _select_candidates(
        base, predicted_collision, candidate_is_finite, collision_threshold
    )
    rows = np.arange(base.shape[0])

    candidates: List[Tuple[Tuple[float, float, float, float], Dict[str, Any]]] = []
    for proxy_weight in proxy_weights:
        for learned_weight in learned_weights:
            score = (
                base
                + float(proxy_weight) * proxy_zscore
                + float(learned_weight) * logit_zscore
            )
            selected = _select_candidates(
                score,
                predicted_collision,
                candidate_is_finite,
                collision_threshold,
            )
            selected_target = target_score[rows, selected][calibration]
            selected_collision = target_collision[rows, selected][calibration]
            # Holdout is deliberately absent from this key and loop.
            key = (
                float(selected_target.mean()),
                -float(selected_collision.mean()),
                -float(proxy_weight + learned_weight),
                -float(learned_weight),
            )
            candidates.append(
                (
                    key,
                    {
                        "protocol_proxy_weight": float(proxy_weight),
                        "learned_zscore_weight": float(learned_weight),
                        "selected": selected,
                    },
                )
            )
    if not candidates:
        raise RuntimeError("Rank-fusion grid produced no candidates")
    candidates.sort(key=lambda item: item[0], reverse=True)
    best = candidates[0][1]
    selected = best.pop("selected")
    selected_score = (
        base
        + float(best["protocol_proxy_weight"]) * proxy_zscore
        + float(best["learned_zscore_weight"]) * logit_zscore
    )
    parameters = {
        "base_score_weight": 1.0,
        "raw_learned_score_weight": 0.0,
        **best,
        "collision_weight": 0.0,
        "kinematic_weight": 0.0,
        "momentum_weight": 0.0,
        "collision_threshold": float(collision_threshold),
    }
    # This is the sole post-selection candidate read of the holdout labels.
    return {
        "parameters": parameters,
        "formula": (
            "normalized_base + protocol_proxy_weight * zscore(protocol_proxy)"
            if proxy_only
            else (
                "normalized_base + protocol_proxy_weight * zscore(protocol_proxy) "
                "+ learned_zscore_weight * zscore(protocol_mlp_logits)"
            )
        ),
        "rank_fusion_mode": "proxy_only" if proxy_only else "hybrid",
        "scorer_logits_used": not proxy_only,
        "scorer_forward_calls_during_protocol_search": 0,
        "calibration": _metrics(
            selected,
            base_selected,
            calibration,
            target_score,
            target_collision,
            predicted_collision,
        ),
        "holdout": _metrics(
            selected,
            base_selected,
            holdout,
            target_score,
            target_collision,
            predicted_collision,
        ),
        "numerical_stability": {
            "calibration": _selection_margin_metrics(
                selected_score, candidate_is_finite, calibration
            ),
            "holdout": _selection_margin_metrics(
                selected_score, candidate_is_finite, holdout
            ),
        },
        "base_rank_only": {
            "calibration": _base_metrics(
                base_selected,
                calibration,
                target_score,
                target_collision,
                predicted_collision,
            ),
            "holdout": _base_metrics(
                base_selected,
                holdout,
                target_score,
                target_collision,
                predicted_collision,
            ),
        },
        "top32_oracles": _top32_oracles(
            base_selected,
            holdout,
            target_score,
            target_collision,
            predicted_collision,
        ),
        "predicted_component_argmax_diagnostics": _predicted_component_diagnostics(
            arrays["features"],
            base_selected,
            holdout,
            target_score,
            target_collision,
            predicted_collision,
        ),
        "search": {
            "combinations": len(candidates),
            "selection_subset": "calibration",
            "primary_objective": "max_mean_target_score",
            "tie_breakers": [
                "min_mean_target_collision_risk",
                "min_total_fusion_weight",
                "min_learned_zscore_weight",
            ],
            "candidate_holdout_evaluations_after_selection": 1,
            "top_calibration_candidates": [
                {
                    "mean_target_score": float(key[0]),
                    "mean_target_collision_risk": float(-key[1]),
                    "protocol_proxy_weight": float(candidate["protocol_proxy_weight"]),
                    "learned_zscore_weight": float(candidate["learned_zscore_weight"]),
                }
                for key, candidate in candidates[:5]
            ],
        },
    }


def tune(args: argparse.Namespace) -> Dict[str, Any]:
    source_audit = _source_audit()
    if source_audit["source_paths_clean"] is not True:
        raise RuntimeError(
            "Refusing non-reproducible tuning from dirty or unauditable source files"
        )
    cache = _load_validation_cache(args.cache_root)
    calibration = _partition_mask(cache.tokens, args.partition_salt)
    holdout = ~calibration
    if int(calibration.sum()) == 0 or int(holdout.sum()) == 0:
        raise RuntimeError("Token-hash partition produced an empty subset")
    proxy_weights = _parse_grid(args.protocol_proxy_weight_grid)
    learned_grid = args.learned_zscore_weight_grid
    if learned_grid is None:
        learned_grid = (
            "0"
            if args.proxy_only
            else "0,0.025,0.05,0.1,0.2,0.4,0.8,1.6,3.2"
        )
    learned_weights = _parse_grid(learned_grid)
    if args.proxy_only:
        if args.scorer_state is not None:
            raise ValueError(
                "proxy-only mode forbids --scorer-state; no scorer artifact may be loaded"
            )
        if learned_weights != (0.0,):
            raise ValueError(
                "proxy-only mode requires --learned-zscore-weight-grid=0"
            )
        logits: Mapping[str, Optional[np.ndarray]] = {"v1": None, "v2": None}
        scorer_audit: Mapping[str, Any] = {
            "used": False,
            "state_loaded": False,
            "scorer_module_imported": False,
            "forward_calls": 0,
            "scorer_labels_used": False,
            "scorer_calibration_or_holdout_influence": False,
            "reason": (
                "strict proxy_only selection uses cached predicted protocol "
                "components and normalized base scores only"
            ),
        }
    else:
        if args.scorer_state is None:
            raise ValueError("hybrid mode requires --scorer-state")
        loaded_logits, loaded_scorer_audit = _load_logits(
            args.scorer_state, cache, int(args.inference_batch_size)
        )
        logits = loaded_logits
        scorer_audit = loaded_scorer_audit
    if (
        not math.isfinite(float(args.collision_threshold))
        or not math.isclose(
            float(args.collision_threshold), 1.000001, rel_tol=0.0, abs_tol=1e-12
        )
    ):
        raise ValueError(
            "collision-threshold must be exactly 1.000001 for this no-filter candidate"
        )

    protocols = {
        protocol: _tune_protocol(
            protocol,
            cache.arrays,
            logits[protocol],
            calibration,
            holdout,
            proxy_weights,
            learned_weights,
            float(args.collision_threshold),
            float(args.zscore_epsilon),
            bool(args.proxy_only),
        )
        for protocol in ("v1", "v2")
    }
    arguments = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    return {
        "schema": OUTPUT_SCHEMA,
        "created_at": _utc_now(),
        "source_audit": source_audit,
        "arguments": arguments,
        "data_scope": {
            "source_split": "navtrain",
            "subset": "val",
            "label_scope": "navtrain-only",
            "evaluator_calls": 0,
            "excluded_splits": ["navtest", "navhard"],
            "samples": len(cache.tokens),
            "tokens_sha256": _token_set_sha256(cache.tokens),
            "rank_fusion_mode": "proxy_only" if args.proxy_only else "hybrid",
            "scorer_state_loaded": not args.proxy_only,
            "scorer_forward_calls": int(scorer_audit["forward_calls"]),
            "scorer_labels_used": False if args.proxy_only else True,
            "navtrain_aggregate_targets_used": {
                "calibration": "weight selection",
                "holdout": "post-selection confirmation only",
            },
        },
        "partition": {
            "method": (
                "sha256(token)[0] & 1 == 0"
                if args.partition_salt == "unsalted-sha256-v1"
                else "sha256(salt + ':' + token)[0] & 1 == 0"
            ),
            "salt": args.partition_salt,
            "calibration_samples": int(calibration.sum()),
            "holdout_samples": int(holdout.sum()),
            "calibration_tokens_sha256": _token_set_sha256(
                token for token, keep in zip(cache.tokens, calibration) if keep
            ),
            "holdout_tokens_sha256": _token_set_sha256(
                token for token, keep in zip(cache.tokens, holdout) if keep
            ),
            "holdout_usage": "one confirmation after calibration-only selection",
            "holdout_reuse_policy": (
                "do not alter weights, grids, formulas, or deployment choice after "
                "reading this report"
            ),
            "independence_scope": (
                (
                    "strict proxy-only reranker; no scorer state, scorer output, "
                    "or scorer label can influence calibration or holdout selection. "
                    "The fixed upstream base checkpoint/candidate cache remains an "
                    "external pre-existing artifact"
                )
                if args.proxy_only
                else (
                    "fusion-weight holdout only; the frozen first-round scorer was "
                    "previously selected using the complete NAVTRAIN validation split"
                )
            ),
        },
        "cache_audit": cache.audit,
        "scorer_audit": scorer_audit,
        "selection_semantics": {
            "rank_fusion_mode": "proxy_only" if args.proxy_only else "hybrid",
            "zscore_axis": "candidate axis within each scene",
            "zscore_epsilon": float(args.zscore_epsilon),
            "collision_threshold": float(args.collision_threshold),
            "collision_filter": "disabled because threshold > maximum sigmoid risk",
            "finite_fallback": ["finite", "all"],
        },
        "cache_limitations": {
            "exact_target_protocol_components_cached": False,
            "available_target_fields": ["aggregate_v1", "aggregate_v2", "collision"],
            "component_diagnostics_use": "predicted component features only",
            "protocol_proxy": (
                "scene-local rank proxy only; missing comfort terms are fixed to one "
                "and v2 two-frame aggregation is not modeled"
            ),
        },
        "search_grid": {
            "protocol_proxy_weight": list(proxy_weights),
            "learned_zscore_weight": list(learned_weights),
        },
        "protocols": protocols,
    }


def _write_report(output: str, report: Mapping[str, Any]) -> None:
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if output == "-":
        print(rendered, end="")
        return
    path = Path(output).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(rendered)
            stream.flush()
            os.fsync(stream.fileno())
        # Publish atomically without replacing a concurrently created report.
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--scorer-state", type=Path)
    parser.add_argument(
        "--proxy-only",
        action="store_true",
        help=(
            "forbid scorer-state loading/forward and search only base plus the "
            "predicted protocol proxy"
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--partition-salt", default="unsalted-sha256-v1"
    )
    parser.add_argument(
        "--protocol-proxy-weight-grid",
        default="0,0.025,0.05,0.1,0.2,0.4,0.8,1.6,3.2",
    )
    parser.add_argument(
        "--learned-zscore-weight-grid",
        default=None,
    )
    parser.add_argument("--collision-threshold", type=float, default=1.000001)
    parser.add_argument("--zscore-epsilon", type=float, default=1e-4)
    parser.add_argument("--inference-batch-size", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.inference_batch_size < 1:
        raise ValueError("inference-batch-size must be positive")
    if args.output != "-":
        output_path = Path(args.output).expanduser().resolve()
        if output_path.exists():
            raise FileExistsError(output_path)
    report = tune(args)
    _write_report(args.output, report)


if __name__ == "__main__":
    main()
