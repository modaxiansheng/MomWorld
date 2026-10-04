"""Deterministic, resumable prediction shards for low-memory evaluation."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


SHARD_SCHEMA = "momworld-prediction-shard-v1"
MERGE_SCHEMA = "momworld-prediction-shard-merge-v1"
SHARD_ALGORITHM = "sha256-token-modulo-v1"


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def token_digest(tokens: Iterable[str]) -> str:
    normalized = sorted(tokens)
    if len(normalized) != len(set(normalized)) or any(
        not isinstance(token, str) or not token for token in normalized
    ):
        raise ValueError("Prediction-shard tokens must be unique non-empty strings")
    return _sha256_bytes("".join(f"{len(token)}:{token}\n" for token in normalized).encode("utf-8"))


def shard_tokens(tokens: Iterable[str], shard_id: int, num_shards: int) -> list[str]:
    if num_shards < 1 or not 0 <= shard_id < num_shards:
        raise ValueError("Prediction shard requires 0 <= shard_id < num_shards")
    normalized = sorted(tokens)
    if len(normalized) != len(set(normalized)):
        raise ValueError("Full prediction token set contains duplicates")
    return [
        token
        for token in normalized
        if int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16) % num_shards
        == shard_id
    ]


def configure_two_stage_prediction_shard(
    scene_filter: Any,
    assigned_tokens: Iterable[str],
    stage_one_tokens: Iterable[str],
    synthetic_tokens: Iterable[str],
) -> Any:
    """Apply an exact mixed stage-one/synthetic shard to a two-stage scene filter.

    ``SceneFilter.tokens`` only filters original stage-one scenes.  Synthetic
    scenes have a separate explicit filter; leaving it as ``None`` asks the
    loader for the relationship closure of the selected stage-one scenes,
    which is not the same partition as hashing all stage-one and stage-two
    tokens together.
    """

    assigned_list = list(assigned_tokens)
    stage_one_list = list(stage_one_tokens)
    synthetic_list = list(synthetic_tokens)
    for label, tokens in (
        ("assigned", assigned_list),
        ("stage-one", stage_one_list),
        ("synthetic", synthetic_list),
    ):
        if len(tokens) != len(set(tokens)):
            raise ValueError(f"Two-stage prediction shard {label} tokens contain duplicates")

    assigned = set(assigned_list)
    stage_one = set(stage_one_list)
    synthetic = set(synthetic_list)
    overlap = stage_one & synthetic
    if overlap:
        raise ValueError(
            "Two-stage prediction stage-one/synthetic token universes overlap: "
            f"{sorted(overlap)[:3]}"
        )
    unknown = assigned - stage_one - synthetic
    if unknown:
        raise ValueError(
            "Two-stage prediction shard contains tokens outside the full scene universe: "
            f"{sorted(unknown)[:3]}"
        )

    scene_filter.tokens = sorted(assigned & stage_one)
    # An empty list is intentional: None would re-enable relationship-based
    # synthetic-scene expansion in filter_synthetic_scenes().
    scene_filter.synthetic_scene_tokens = sorted(assigned & synthetic)
    return scene_filter


def shard_configuration_from_environment() -> Optional[Dict[str, Any]]:
    names = (
        "MOMWORLD_PREDICTION_SHARD_ID",
        "MOMWORLD_PREDICTION_NUM_SHARDS",
        "MOMWORLD_SHARD_OUTPUT_PATH",
        "MOMWORLD_SHARD_MANIFEST_PATH",
        "MOMWORLD_SHARD_EXPERIMENT_ROOT",
        "MOMWORLD_SHARD_SPLIT",
        "MOMWORLD_SHARD_SOURCE_COMMIT",
        "MOMWORLD_SHARD_CHECKPOINT_PATH",
        "MOMWORLD_SHARD_CHECKPOINT_SHA256",
    )
    values = {name: os.environ.get(name) for name in names}
    enabled = os.environ.get("MOMWORLD_PREDICTION_ONLY", "").lower() == "true"
    if not enabled and not any(value is not None for value in values.values()):
        return None
    if not enabled or any(value is None for value in values.values()):
        missing = [name for name, value in values.items() if value is None]
        raise ValueError(
            "Prediction sharding requires MOMWORLD_PREDICTION_ONLY=true and all "
            f"shard identity variables; missing={missing}"
        )
    shard_id = int(str(values["MOMWORLD_PREDICTION_SHARD_ID"]))
    num_shards = int(str(values["MOMWORLD_PREDICTION_NUM_SHARDS"]))
    if num_shards < 1 or not 0 <= shard_id < num_shards:
        raise ValueError("Prediction shard identity is out of range")
    source_commit = str(values["MOMWORLD_SHARD_SOURCE_COMMIT"]).lower()
    checkpoint_sha256 = str(values["MOMWORLD_SHARD_CHECKPOINT_SHA256"]).lower()
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("Prediction shard source commit is invalid")
    if re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha256) is None:
        raise ValueError("Prediction shard checkpoint SHA256 is invalid")
    experiment_root = Path(str(values["MOMWORLD_SHARD_EXPERIMENT_ROOT"])).expanduser().resolve()
    output_path = Path(str(values["MOMWORLD_SHARD_OUTPUT_PATH"])).expanduser().resolve()
    manifest_path = Path(str(values["MOMWORLD_SHARD_MANIFEST_PATH"])).expanduser().resolve()
    checkpoint_path = Path(str(values["MOMWORLD_SHARD_CHECKPOINT_PATH"])).expanduser().resolve()
    if not experiment_root.is_absolute() or not checkpoint_path.is_file():
        raise ValueError("Prediction shard experiment/checkpoint path is invalid")
    if sha256_file(checkpoint_path) != checkpoint_sha256:
        raise ValueError("Prediction shard checkpoint SHA256 differs")
    for label, path in (("output", output_path), ("manifest", manifest_path)):
        try:
            path.relative_to(experiment_root)
        except ValueError as error:
            raise ValueError(f"Prediction shard {label} escapes its experiment") from error
    if output_path == manifest_path:
        raise ValueError("Prediction shard output and manifest paths collide")
    return {
        "shard_id": shard_id,
        "num_shards": num_shards,
        "output_path": output_path,
        "manifest_path": manifest_path,
        "experiment_root": experiment_root,
        "source_commit": source_commit,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha256,
        "split": str(values["MOMWORLD_SHARD_SPLIT"]),
    }


def strict_merge_prediction_batches(batches: Sequence[Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for process_batches in batches:
        if process_batches is None:
            continue
        for predictions in process_batches:
            if not isinstance(predictions, Mapping):
                raise TypeError("Prediction batch is not a mapping")
            overlap = set(merged) & set(predictions)
            if overlap:
                raise RuntimeError(f"Duplicate prediction tokens: {sorted(overlap)[:3]}")
            merged.update(predictions)
    return merged


def _atomic_pickle(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _exclusive_json(path: Path, payload: Mapping[str, Any]) -> str:
    rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != dict(payload):
            raise RuntimeError(f"Refusing mismatched immutable shard artifact: {path}")
        return "matched"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(rendered)
        stream.flush()
        os.fsync(stream.fileno())
    return "created"


def _match_existing_timestamped_manifest(
    path: Path, payload: Mapping[str, Any]
) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    existing = json.loads(path.read_text(encoding="utf-8"))
    candidate = dict(payload)
    candidate["created_at"] = existing.get("created_at")
    if existing != candidate:
        raise RuntimeError(f"Refusing mismatched resumable shard manifest: {path}")
    return existing


def load_prediction_file(path: Path, protocol: str) -> Dict[str, Any]:
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    if protocol == "v1":
        if (
            not isinstance(payload, Mapping)
            or set(payload) != {"predictions"}
            or not isinstance(payload["predictions"], list)
            or len(payload["predictions"]) != 1
            or not isinstance(payload["predictions"][0], Mapping)
        ):
            raise ValueError("NAVSIM v1 shard payload shape differs")
        return dict(payload["predictions"][0])
    if protocol == "v2" and isinstance(payload, Mapping):
        return dict(payload)
    raise ValueError("Prediction shard protocol/payload differs")


def write_prediction_shard(
    predictions: Mapping[str, Any],
    full_tokens: Iterable[str],
    protocol: str,
    split: str,
    configuration: Mapping[str, Any],
) -> Dict[str, Any]:
    if protocol not in {"v1", "v2"} or not split:
        raise ValueError("Prediction shard protocol/split differs")
    full = sorted(full_tokens)
    assigned = shard_tokens(full, int(configuration["shard_id"]), int(configuration["num_shards"]))
    if set(predictions) != set(assigned):
        missing = sorted(set(assigned) - set(predictions))[:3]
        extra = sorted(set(predictions) - set(assigned))[:3]
        raise RuntimeError(f"Shard prediction coverage differs; missing={missing}, extra={extra}")
    output_path = Path(configuration["output_path"])
    canonical = {"predictions": [dict(predictions)]} if protocol == "v1" else dict(predictions)
    if output_path.exists():
        existing = load_prediction_file(output_path, protocol)
        if set(existing) != set(assigned):
            raise RuntimeError("Existing resumable shard output has different tokens")
    else:
        _atomic_pickle(canonical, output_path)
    output_tokens = load_prediction_file(output_path, protocol)
    output_digest = token_digest(output_tokens)
    if output_digest != token_digest(assigned):
        raise RuntimeError("Persisted shard output token digest differs")
    payload = {
        "schema": SHARD_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "algorithm": SHARD_ALGORITHM,
        "protocol": protocol,
        "split": split,
        "shard_id": int(configuration["shard_id"]),
        "num_shards": int(configuration["num_shards"]),
        "source_commit": str(configuration["source_commit"]),
        "checkpoint": {
            "path": str(Path(configuration["checkpoint_path"])),
            "sha256": str(configuration["checkpoint_sha256"]),
        },
        "experiment_root": str(Path(configuration["experiment_root"])),
        "full_token_count": len(full),
        "full_tokens_sha256": token_digest(full),
        "assigned_token_count": len(assigned),
        "assigned_tokens_sha256": token_digest(assigned),
        "assigned_tokens": assigned,
        "trajectory": {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
            "size": output_path.stat().st_size,
            "token_count": len(output_tokens),
            "tokens_sha256": output_digest,
        },
    }
    manifest_path = Path(configuration["manifest_path"])
    existing = _match_existing_timestamped_manifest(manifest_path, payload)
    if existing is not None:
        return {"status": "matched", "manifest": existing}
    status = _exclusive_json(manifest_path, payload)
    return {"status": status, "manifest": payload}


def merge_prediction_shards(
    manifest_paths: Sequence[Path], output_path: Path, merge_manifest_path: Path
) -> Dict[str, Any]:
    if not manifest_paths:
        raise ValueError("No prediction shard manifests supplied")
    manifests = []
    for path in manifest_paths:
        path = path.expanduser().resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema") != SHARD_SCHEMA:
            raise ValueError(f"Invalid prediction shard manifest: {path}")
        manifests.append((path, payload))
    reference = manifests[0][1]
    identity_fields = (
        "algorithm",
        "protocol",
        "split",
        "num_shards",
        "source_commit",
        "checkpoint",
        "experiment_root",
        "full_token_count",
        "full_tokens_sha256",
    )
    for _, manifest in manifests:
        if any(manifest.get(field) != reference.get(field) for field in identity_fields):
            raise ValueError("Prediction shard identities differ")
    num_shards = int(reference["num_shards"])
    shard_ids = [int(manifest["shard_id"]) for _, manifest in manifests]
    if sorted(shard_ids) != list(range(num_shards)):
        raise ValueError("Prediction shard IDs are incomplete or duplicated")
    all_assigned: set[str] = set()
    merged: Dict[str, Any] = {}
    shard_artifacts = []
    for path, manifest in sorted(manifests, key=lambda item: int(item[1]["shard_id"])):
        assigned = manifest.get("assigned_tokens")
        if (
            not isinstance(assigned, list)
            or len(assigned) != int(manifest.get("assigned_token_count", -1))
            or token_digest(assigned) != manifest.get("assigned_tokens_sha256")
        ):
            raise ValueError("Prediction shard assigned-token evidence differs")
        overlap = all_assigned & set(assigned)
        if overlap:
            raise ValueError(f"Prediction shards overlap: {sorted(overlap)[:3]}")
        all_assigned.update(assigned)
        trajectory = manifest.get("trajectory")
        trajectory_path = Path(str(trajectory.get("path"))).expanduser().resolve()
        if (
            sha256_file(trajectory_path) != trajectory.get("sha256")
            or trajectory_path.stat().st_size != int(trajectory.get("size", -1))
        ):
            raise ValueError("Prediction shard trajectory artifact differs")
        predictions = load_prediction_file(trajectory_path, str(reference["protocol"]))
        if set(predictions) != set(assigned) or token_digest(predictions) != trajectory.get(
            "tokens_sha256"
        ):
            raise ValueError("Prediction shard output coverage differs")
        duplicate = set(merged) & set(predictions)
        if duplicate:
            raise ValueError(f"Prediction shard outputs duplicate tokens: {sorted(duplicate)[:3]}")
        merged.update(predictions)
        shard_artifacts.append(
            {"path": str(path), "sha256": sha256_file(path), "size": path.stat().st_size}
        )
    if (
        len(all_assigned) != int(reference["full_token_count"])
        or token_digest(all_assigned) != reference["full_tokens_sha256"]
        or set(merged) != all_assigned
    ):
        raise ValueError("Prediction shard union is not exact full-token coverage")
    protocol = str(reference["protocol"])
    canonical = {"predictions": [merged]} if protocol == "v1" else merged
    if output_path.exists():
        existing = load_prediction_file(output_path, protocol)
        if set(existing) != set(merged):
            raise RuntimeError("Existing merged trajectory has different coverage")
    else:
        _atomic_pickle(canonical, output_path)
    persisted = load_prediction_file(output_path, protocol)
    if set(persisted) != set(merged) or token_digest(persisted) != reference["full_tokens_sha256"]:
        raise RuntimeError("Merged trajectory failed exact full-token verification")
    merge_payload = {
        "schema": MERGE_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol,
        "split": reference["split"],
        "algorithm": SHARD_ALGORITHM,
        "source_commit": reference["source_commit"],
        "checkpoint": reference["checkpoint"],
        "num_shards": num_shards,
        "full_token_count": len(merged),
        "full_tokens_sha256": token_digest(merged),
        "shard_manifests": shard_artifacts,
        "trajectory": {
            "path": str(output_path.resolve()),
            "sha256": sha256_file(output_path),
            "size": output_path.stat().st_size,
            "token_count": len(persisted),
            "tokens_sha256": token_digest(persisted),
        },
        "coverage": {
            "all_shard_ids_present": True,
            "duplicate_tokens": 0,
            "missing_tokens": 0,
            "exact_full_token_union": True,
        },
    }
    existing = _match_existing_timestamped_manifest(merge_manifest_path, merge_payload)
    if existing is not None:
        return {"status": "matched", "manifest": existing}
    status = _exclusive_json(merge_manifest_path, merge_payload)
    return {"status": status, "manifest": merge_payload}
