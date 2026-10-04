#!/usr/bin/env python3
"""Cross-fit GTRS fusion weights on every NAVTRAIN scene, without test data.

Each scene is predicted exactly once by a scorer trained on the other folds.
The resulting out-of-fold logits gate whether a learned scorer can safely be
added to the frozen protocol proxy and candidate-level safety Rule.  This is a
selection gate only: the final deployable scorer remains the all-NAVTRAIN
refit produced by ``refit_momworld_gtrs_scorer_all_navtrain.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch.utils.data import DataLoader

from navsim.agents.momworld.momworld_model import RuleAwareTrajectoryScorer
from scripts.training.refit_momworld_gtrs_scorer_all_navtrain import (
    _dataset,
    _sha256,
    combine_all_unique_navtrain,
    gtrs_loss,
)
from scripts.training.train_momworld_rule_scorer_from_cache import (
    CacheRecord,
    _discover_manifests,
    _discover_npz,
    _load_records,
    _validate_manifests,
)
from scripts.training.tune_momworld_candidate_safety_rule_all_navtrain import (
    apply_safety_rule,
)
from scripts.training.tune_momworld_protocol_proxy_from_cache import (
    _candidate_zscore,
    _protocol_proxy,
)
from scripts.training.tune_momworld_relu_guard_from_cache import _arrays


REPORT_SCHEMA = "momworld-gtrs-all-navtrain-crossfit-fusion-v1"
RELEVANT_SOURCE_FILES = (
    "navsim/agents/momworld/momworld_model.py",
    "scripts/training/crossfit_momworld_gtrs_fusion_all_navtrain.py",
    "scripts/training/refit_momworld_gtrs_scorer_all_navtrain.py",
)


def deterministic_fold(token: str, folds: int) -> int:
    if folds < 2:
        raise ValueError("cross-fit requires at least two folds")
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % folds


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
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
        raise RuntimeError(f"Cross-fit sources are dirty: {status}")
    files = {}
    for relative in RELEVANT_SOURCE_FILES:
        path = repository / relative
        blob = subprocess.check_output(
            ["git", "-C", str(repository), "show", f"{commit}:{relative}"]
        )
        digest = _sha256(path)
        if digest != hashlib.sha256(blob).hexdigest():
            raise RuntimeError(f"Cross-fit source differs from commit: {relative}")
        files[relative] = {"path": str(path.resolve()), "sha256": digest}
    return {"commit": commit, "working_tree_clean": True, "files": files}


def _fit(
    records: Sequence[CacheRecord],
    *,
    hidden_dim: int,
    input_mode: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> Tuple[RuleAwareTrajectoryScorer, List[Dict[str, float]]]:
    model = RuleAwareTrajectoryScorer(
        SimpleNamespace(
            rule_scorer_hidden_dim=hidden_dim,
            rule_scorer_input_mode=input_mode,
        )
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loader = DataLoader(
        _dataset(records),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    history: List[Dict[str, float]] = []
    for epoch in range(epochs):
        totals: Dict[str, float] = {}
        samples = 0
        model.train()
        for batch in loader:
            values = [value.to(device, non_blocking=True) for value in batch]
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = gtrs_loss(model, *values)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite cross-fit GTRS loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            count = int(values[0].shape[0])
            samples += count
            for name, value in metrics.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach()) * count
        history.append(
            {"epoch": float(epoch), **{name: value / samples for name, value in totals.items()}}
        )
    return model, history


def _predict(
    model: RuleAwareTrajectoryScorer,
    records: Sequence[CacheRecord],
    protocol: str,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    features = np.stack([record.features for record in records]).astype(np.float32)
    chunks = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(features[start : start + batch_size]).to(device)
            chunks.append(model(batch, protocol).float().cpu().numpy())
    return np.concatenate(chunks)


def select_with_rule(score: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    if score.shape != eligible.shape or score.ndim != 2:
        raise ValueError("score/eligible shape mismatch")
    if not eligible.any(axis=1).all():
        raise RuntimeError("candidate Rule removed every candidate from a scene")
    return np.where(eligible, score, np.finfo(np.float32).min).argmax(axis=1)


def crossfit_metrics(
    *,
    logits: np.ndarray,
    folds: np.ndarray,
    target: np.ndarray,
    target_collision: np.ndarray,
    reference_score: np.ndarray,
    eligible: np.ndarray,
    weights: Sequence[float],
) -> Dict[str, Any]:
    learned = _candidate_zscore(logits.astype(np.float32), 1e-4)
    rows = np.arange(len(target))
    reference = select_with_rule(reference_score, eligible)
    reference_mean = float(target[rows, reference].mean())
    reference_collision = float((target_collision[rows, reference] >= 0.5).mean())
    trials = []
    for weight in weights:
        selected = select_with_rule(reference_score + float(weight) * learned, eligible)
        score = float(target[rows, selected].mean())
        collision = float((target_collision[rows, selected] >= 0.5).mean())
        fold_gains = []
        for fold in sorted(np.unique(folds).tolist()):
            mask = folds == fold
            fold_rows = rows[mask]
            fold_gains.append(
                float(
                    target[fold_rows, selected[mask]].mean()
                    - target[fold_rows, reference[mask]].mean()
                )
            )
        trials.append(
            {
                "weight": float(weight),
                "mean_target": score,
                "gain": score - reference_mean,
                "collision_rate": collision,
                "collision_rate_delta": collision - reference_collision,
                "fold_gains": fold_gains,
                "minimum_fold_gain": min(fold_gains),
            }
        )
    admissible = [
        trial for trial in trials
        if trial["gain"] > 0.0
        and trial["collision_rate_delta"] <= 0.0
        and trial["minimum_fold_gain"] >= 0.0
    ]
    selected = max(admissible, key=lambda item: item["gain"]) if admissible else None
    return {
        "reference_mean_target": reference_mean,
        "reference_collision_rate": reference_collision,
        "trials": trials,
        "selected": selected,
        "gate_passed": selected is not None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--memory-fraction", type=float, default=0.92)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--inference-batch-size", type=int, default=8192)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--input-mode", default="raw_plus_scene_zscore")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.folds < 2 or args.epochs < 1 or args.batch_size < 1:
        raise ValueError("invalid cross-fit arguments")
    repository = Path(__file__).resolve().parents[2]
    source = _source_audit(repository)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)

    root = args.cache_root.expanduser().resolve()
    cache_audit = _validate_manifests(_discover_manifests([root]), False)
    raw = _load_records(_discover_npz([root]), str(cache_audit["checkpoint_sha256"]))
    records, partition = combine_all_unique_navtrain(raw)
    folds = np.asarray(
        [deterministic_fold(record.token, args.folds) for record in records],
        dtype=np.int64,
    )
    if set(folds.tolist()) != set(range(args.folds)):
        raise RuntimeError("cross-fit produced an empty fold")
    oof = {
        "v1": np.full((len(records), 32), np.nan, dtype=np.float32),
        "v2": np.full((len(records), 32), np.nan, dtype=np.float32),
    }
    histories = []
    for fold in range(args.folds):
        train = [record for index, record in enumerate(records) if folds[index] != fold]
        hold = [record for index, record in enumerate(records) if folds[index] == fold]
        model, history = _fit(
            train,
            hidden_dim=args.hidden_dim,
            input_mode=args.input_mode,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed + fold,
            device=device,
        )
        mask = folds == fold
        for protocol in ("v1", "v2"):
            oof[protocol][mask] = _predict(
                model, hold, protocol, args.inference_batch_size, device
            )
        histories.append({"fold": fold, "train": len(train), "holdout": len(hold), "history": history})
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if not all(np.isfinite(values).all() for values in oof.values()):
        raise FloatingPointError("cross-fit logits are incomplete or non-finite")

    arrays = _arrays(records)
    features = np.asarray(arrays["features"], dtype=np.float32)
    finite = np.asarray(arrays["candidate_is_finite"], dtype=np.bool_)
    base = np.asarray(arrays["normalized_base"], dtype=np.float32)
    collision = np.asarray(arrays["target_collision"], dtype=np.float32)
    v1_reference = base + 1.6 * _candidate_zscore(_protocol_proxy(features, "v1"), 1e-4)
    v2_reference = base + 3.2 * _candidate_zscore(_protocol_proxy(features, "v2"), 1e-4)
    hard_parameters = {
        "no_collision_min": 0.0,
        "drivable_area_min": 0.55,
        "ttc_min": 0.0,
        "progress_min": 0.0,
        "direction_min": 0.0,
        "lane_min": 0.55,
        "traffic_light_min": 0.0,
        "collision_risk_max": 1.000001,
        "kinematic_max": 1e9,
        "momentum_max": 1e9,
        "fallback_risk_slack": 0.0,
    }
    hard_eligible, _, _ = apply_safety_rule(features, finite, hard_parameters)
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
        "folds": {
            "count": args.folds,
            "assignment": "sha256(token)[0:8] modulo folds",
            "counts": [int((folds == fold).sum()) for fold in range(args.folds)],
            "histories": histories,
        },
        "protocols": {
            "v1_protocol_proxy": crossfit_metrics(
                logits=oof["v1"], folds=folds,
                target=np.asarray(arrays["target_v1"], dtype=np.float32),
                target_collision=collision, reference_score=v1_reference,
                eligible=finite, weights=(0.0, 0.2, 0.4, 0.8, 1.2),
            ),
            "v2_navhard_protocol_proxy_rule": crossfit_metrics(
                logits=oof["v2"], folds=folds,
                target=np.asarray(arrays["target_v2"], dtype=np.float32),
                target_collision=collision, reference_score=v2_reference,
                eligible=hard_eligible, weights=(0.0, 0.025, 0.05, 0.1, 0.2),
            ),
        },
        "arguments": {
            key: str(value.expanduser().resolve()) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "reproduction_command": [str(Path(os.sys.executable).resolve()), *os.sys.argv],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.output, report)
    print(json.dumps({name: value["selected"] for name, value in report["protocols"].items()}, sort_keys=True))


if __name__ == "__main__":
    main()
