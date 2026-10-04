#!/usr/bin/env python3
"""Refit the lightweight trajectory scorer on every unique NAVTRAIN record.

Architecture is unchanged: the existing 10-D candidate features feed the two
small RuleAwareTrajectoryScorer heads.  Ground-truth ranking supervision uses
ListNet, all-pairs RankNet, score calibration, and explicit collision hard
negatives.  Epoch count and hyperparameters are fixed before training; no
validation, evaluator, NAVTest, NAVHard, or benchmark metric cache is read.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from navsim.agents.momworld.context_ranker import (
    effective_candidate_mask,
    pairwise_ranknet_loss,
)
from navsim.agents.momworld.momworld_model import RuleAwareTrajectoryScorer
from scripts.training.train_momworld_rule_scorer_from_cache import (
    STATE_SCHEMA,
    CacheRecord,
    _discover_manifests,
    _discover_npz,
    _exclude_validation_overlap,
    _load_records,
    _validate_manifests,
)


REPORT_SCHEMA = "momworld-gtrs-all-navtrain-refit-v2"
SCORER_INPUT_MODES = ("raw", "raw_plus_scene_zscore")
RELEVANT_SOURCE_FILES = (
    "navsim/agents/momworld/momworld_config.py",
    "navsim/agents/momworld/momworld_model.py",
    "navsim/agents/momworld/context_ranker.py",
    "scripts/training/refit_momworld_gtrs_scorer_all_navtrain.py",
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


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
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
        raise RuntimeError(f"Relevant GTRS refit sources are dirty: {status}")
    files = {}
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


def combine_all_unique_navtrain(
    records: Mapping[str, Sequence[CacheRecord]],
) -> Tuple[List[CacheRecord], Dict[str, Any]]:
    """Use effective train plus validation exactly once per NAVTRAIN token."""

    separated, overlap = _exclude_validation_overlap(records)
    combined = list(separated["train"]) + list(separated["val"])
    tokens = [record.token for record in combined]
    if len(tokens) != len(set(tokens)):
        raise RuntimeError("All-NAVTRAIN refit contains duplicate tokens")
    if not combined:
        raise RuntimeError("All-NAVTRAIN refit is empty")
    return combined, {
        **overlap,
        "combined_unique_navtrain": len(combined),
        "combined_tokens_sha256": hashlib.sha256(
            "\n".join(sorted(tokens)).encode("utf-8")
        ).hexdigest(),
        "validation_role": "training data; no model selection or reporting",
    }


def _dataset(records: Sequence[CacheRecord]) -> TensorDataset:
    return TensorDataset(
        torch.from_numpy(np.stack([record.features for record in records])),
        torch.from_numpy(np.stack([record.target_v1 for record in records])),
        torch.from_numpy(np.stack([record.target_v2 for record in records])),
        torch.from_numpy(np.stack([record.target_collision for record in records])),
        torch.from_numpy(np.stack([record.candidate_is_finite for record in records])),
    )


def _listnet_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    finite: torch.Tensor,
    temperature: float = 0.05,
) -> torch.Tensor:
    mask = effective_candidate_mask(finite)
    floor = torch.finfo(logits.dtype).min
    prediction = F.log_softmax(logits.masked_fill(~mask, floor), dim=1)
    target = F.softmax((targets / temperature).masked_fill(~mask, floor), dim=1)
    return -(target * prediction).sum(dim=1).mean()


def _collision_hard_negative_loss(
    logits: torch.Tensor,
    target_collision: torch.Tensor,
    finite: torch.Tensor,
) -> torch.Tensor:
    mask = effective_candidate_mask(finite)
    safe = mask & (target_collision <= 0.10)
    unsafe = mask & (target_collision >= 0.50)
    pairs = safe[:, :, None] & unsafe[:, None, :]
    margin = logits[:, :, None] - logits[:, None, :]
    loss = F.softplus(0.25 - margin)
    return (loss * pairs.to(loss.dtype)).sum() / pairs.sum().clamp(min=1)


def _top1_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    finite: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Directly optimize selection of the highest-target finite candidate."""

    mask = effective_candidate_mask(finite)
    floor = torch.finfo(logits.dtype).min
    masked_targets = targets.masked_fill(~mask, floor)
    best = masked_targets.argmax(dim=1)
    masked_logits = logits.masked_fill(~mask, floor)
    loss = F.cross_entropy(masked_logits, best)
    selected = masked_logits.argmax(dim=1)
    rows = torch.arange(logits.shape[0], device=logits.device)
    accuracy = (selected == best).to(logits.dtype).mean()
    regret = (masked_targets[rows, best] - masked_targets[rows, selected]).mean()
    return loss, accuracy, regret


def gtrs_loss(
    model: torch.nn.Module,
    features: torch.Tensor,
    target_v1: torch.Tensor,
    target_v2: torch.Tensor,
    target_collision: torch.Tensor,
    finite: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    metrics: Dict[str, torch.Tensor] = {}
    protocol_losses = []
    for protocol, targets in (("v1", target_v1), ("v2", target_v2)):
        logits = model(features, protocol)
        calibration = F.binary_cross_entropy_with_logits(logits, targets)
        listnet = _listnet_loss(logits, targets, finite)
        ranknet = pairwise_ranknet_loss(logits, targets, finite)
        collision = _collision_hard_negative_loss(logits, target_collision, finite)
        top1, top1_accuracy, top1_regret = _top1_cross_entropy(
            logits, targets, finite
        )
        value = (
            0.25 * calibration
            + 0.50 * listnet
            + 0.50 * ranknet
            + 0.10 * collision
            + top1
        )
        protocol_losses.append(value)
        metrics[f"{protocol}_calibration"] = calibration
        metrics[f"{protocol}_listnet"] = listnet
        metrics[f"{protocol}_ranknet"] = ranknet
        metrics[f"{protocol}_collision_hard_negative"] = collision
        metrics[f"{protocol}_top1_cross_entropy"] = top1
        metrics[f"{protocol}_top1_accuracy"] = top1_accuracy
        metrics[f"{protocol}_top1_regret"] = top1_regret
    total = 0.5 * (protocol_losses[0] + protocol_losses[1])
    metrics["loss"] = total
    return total, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--memory-fraction", type=float, default=0.92)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument(
        "--input-mode", choices=SCORER_INPUT_MODES, default="raw"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.epochs, args.batch_size) < 1 or args.num_workers < 0:
        raise ValueError("Invalid GTRS refit duration or loader configuration")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "rule_scorer_state.pt"
    report_path = output / "training_report.json"
    if state_path.exists() or report_path.exists():
        raise FileExistsError("Refusing to overwrite GTRS refit artifacts")
    repository = Path(__file__).resolve().parents[2]
    source = _source_audit(repository)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    cache_roots = [path.expanduser().resolve() for path in args.cache_root]
    cache_audit = _validate_manifests(_discover_manifests(cache_roots), False)
    raw = _load_records(_discover_npz(cache_roots), str(cache_audit["checkpoint_sha256"]))
    all_navtrain, partition = combine_all_unique_navtrain(raw)
    cache_audit = {**cache_audit, "split_partition": partition}
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available() or not 0.0 < args.memory_fraction <= 0.95:
            raise RuntimeError("Requested CUDA device/fraction is unavailable or unsafe")
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device=device)
    hidden_dim = int(cache_audit["rule_scorer_hidden_dim"])
    model = RuleAwareTrajectoryScorer(
        SimpleNamespace(
            rule_scorer_hidden_dim=hidden_dim,
            rule_scorer_input_mode=args.input_mode,
        )
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    loader = DataLoader(
        _dataset(all_navtrain),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    history: List[Dict[str, Any]] = []
    for epoch in range(args.epochs):
        model.train()
        totals: Dict[str, float] = {}
        samples = 0
        for batch in loader:
            features, target_v1, target_v2, target_collision, finite = [
                value.to(device, non_blocking=True) for value in batch
            ]
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = gtrs_loss(
                model, features, target_v1, target_v2, target_collision, finite
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite all-NAVTRAIN GTRS loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            samples += features.shape[0]
            for name, value in metrics.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach()) * features.shape[0]
        record = {"epoch": epoch, **{name: value / samples for name, value in totals.items()}}
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    metadata: Dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "created_at": _utc_now(),
        "source_commit": source["commit"],
        "source_audit": source,
        "cache_audit": cache_audit,
        "data_scope": {
            "source_split": "navtrain",
            "label_scope": "navtrain-only",
            "samples": len(all_navtrain),
            "evaluator_calls": 0,
            "benchmark_cache_reads": 0,
            "excluded_splits": ["navtest", "navhard"],
        },
        "model": {
            "class": "RuleAwareTrajectoryScorer",
            "hidden_dim": hidden_dim,
            "input_mode": args.input_mode,
        },
        "training": {
            "method": "fixed-epoch all-NAVTRAIN GTRS refit",
            "epochs": args.epochs,
            "epoch_count_source": (
                "explicit fixed run; no validation or benchmark-based epoch selection"
            ),
            "loss": {
                "calibration": 0.25,
                "listnet": 0.5,
                "all_pairs_ranknet": 0.5,
                "collision_hard_negative": 0.1,
                "top1_cross_entropy": 1.0,
            },
            "history": history,
        },
        "arguments": {
            name: [str(item.expanduser().resolve()) for item in value]
            if name == "cache_root"
            else str(value.expanduser().resolve()) if isinstance(value, Path) else value
            for name, value in vars(args).items()
        },
        "reproduction_command": [sys.executable, *sys.argv],
    }
    injection_state = {
        f"agent.model._rule_scorer.{name}": value.clone() for name, value in state.items()
    }
    _atomic_torch(
        state_path,
        {
            "schema": STATE_SCHEMA,
            "state_dict": state,
            "injection_state_dict": injection_state,
            "metadata": copy.deepcopy(metadata),
        },
    )
    metadata["state_file"] = str(state_path)
    metadata["state_sha256"] = _sha256(state_path)
    _atomic_json(report_path, metadata)
    print(json.dumps({"state": metadata["state_sha256"], "report": _sha256(report_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
