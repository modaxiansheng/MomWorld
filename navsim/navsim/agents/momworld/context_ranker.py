"""Small set-wise trajectory ranker for NAVTRAIN-only offline learning.

The module deliberately consumes only quantities already produced by the
MomWorld rule candidate path.  It has no evaluator, map-cache, NAVTEST, or
NAVHARD dependency, which makes the same input construction reusable in the
offline cache trainer and in online trajectory selection.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


CONTEXT_RANKER_STATE_SCHEMA = "momworld-context-ranker-state-v1"
CONTEXT_RANKER_INJECTION_SCHEMA = "momworld-context-ranker-injection-v1"
CONTEXT_RANKER_ATTRIBUTE = "_context_ranker"
MAX_ADDED_RUNTIME_BYTES = 10 * 1024**3


@dataclass(frozen=True)
class ContextRankerConfig:
    """Architecture shared by offline training and checkpoint injection."""

    rule_feature_dim: int = 10
    hidden_dim: int = 64
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.05
    zscore_epsilon: float = 1e-4

    @property
    def input_dim(self) -> int:
        # raw features, within-scene z-score, normalized base score, descending
        # base rank, and a finite-candidate flag.
        return 2 * int(self.rule_feature_dim) + 3

    def validate(self) -> None:
        if self.rule_feature_dim < 1 or self.hidden_dim < 8:
            raise ValueError("rule_feature_dim and hidden_dim are too small")
        if self.num_layers < 1 or self.num_heads < 1:
            raise ValueError("num_layers and num_heads must be positive")
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not math.isfinite(float(self.zscore_epsilon)) or self.zscore_epsilon <= 0:
            raise ValueError("zscore_epsilon must be finite and positive")

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "ContextRankerConfig":
        expected = {
            "rule_feature_dim",
            "hidden_dim",
            "num_layers",
            "num_heads",
            "dropout",
            "zscore_epsilon",
        }
        if set(payload) != expected:
            raise ValueError(
                "Context-ranker config keys differ; missing={}, extra={}".format(
                    sorted(expected - set(payload)), sorted(set(payload) - expected)
                )
            )
        config = cls(
            rule_feature_dim=int(payload["rule_feature_dim"]),
            hidden_dim=int(payload["hidden_dim"]),
            num_layers=int(payload["num_layers"]),
            num_heads=int(payload["num_heads"]),
            dropout=float(payload["dropout"]),
            zscore_epsilon=float(payload["zscore_epsilon"]),
        )
        config.validate()
        return config


def _validate_candidate_tensors(
    rule_features: torch.Tensor,
    normalized_base: torch.Tensor,
    candidate_is_finite: torch.Tensor,
) -> None:
    if rule_features.ndim != 3:
        raise ValueError("rule_features must have shape [batch, candidates, features]")
    expected = rule_features.shape[:2]
    if normalized_base.shape != expected or candidate_is_finite.shape != expected:
        raise ValueError(
            "normalized_base and candidate_is_finite must match [batch, candidates]"
        )
    if rule_features.shape[1] < 2:
        raise ValueError("at least two candidates are required for set-wise ranking")


def effective_candidate_mask(candidate_is_finite: torch.Tensor) -> torch.Tensor:
    """Use finite candidates, falling back to all when an entire row is invalid."""

    if candidate_is_finite.ndim != 2:
        raise ValueError("candidate_is_finite must have shape [batch, candidates]")
    finite = candidate_is_finite.bool()
    return torch.where(finite.any(dim=1, keepdim=True), finite, torch.ones_like(finite))


def masked_scene_zscore(
    values: torch.Tensor, mask: torch.Tensor, epsilon: float
) -> torch.Tensor:
    """Standardize the last candidate axis without leaking across scenes."""

    if values.ndim != 3 or mask.shape != values.shape[:2]:
        raise ValueError("values/mask shapes must be [B,K,D] and [B,K]")
    if not math.isfinite(float(epsilon)) or float(epsilon) <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    values = torch.nan_to_num(values.float())
    effective = effective_candidate_mask(mask)
    weight = effective.unsqueeze(-1).to(values.dtype)
    count = weight.sum(dim=1, keepdim=True).clamp(min=1.0)
    mean = (values * weight).sum(dim=1, keepdim=True) / count
    variance = ((values - mean).square() * weight).sum(dim=1, keepdim=True) / count
    standardized = (values - mean) / variance.sqrt().clamp(min=float(epsilon))
    return standardized.masked_fill(~effective.unsqueeze(-1), 0.0)


def normalized_descending_rank(
    scores: torch.Tensor, candidate_is_finite: torch.Tensor
) -> torch.Tensor:
    """Return stable descending ranks in [0,1], with best candidate at zero."""

    if scores.ndim != 2 or candidate_is_finite.shape != scores.shape:
        raise ValueError("scores and candidate mask must have shape [batch, candidates]")
    effective = effective_candidate_mask(candidate_is_finite)
    clean = torch.nan_to_num(scores.float(), nan=-1e9, posinf=1e9, neginf=-1e9)
    clean = clean.masked_fill(~effective, torch.finfo(clean.dtype).min)
    order = torch.argsort(clean, dim=1, descending=True, stable=True)
    integer_rank = torch.empty_like(order)
    rank_values = torch.arange(order.shape[1], device=order.device).expand_as(order)
    integer_rank.scatter_(1, order, rank_values)
    denominator = max(order.shape[1] - 1, 1)
    result = integer_rank.to(clean.dtype) / float(denominator)
    return result.masked_fill(~effective, 1.0)


def build_context_ranker_inputs(
    rule_features: torch.Tensor,
    normalized_base: torch.Tensor,
    candidate_is_finite: torch.Tensor,
    epsilon: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the exact offline/online set-wise feature representation."""

    _validate_candidate_tensors(rule_features, normalized_base, candidate_is_finite)
    effective = effective_candidate_mask(candidate_is_finite)
    raw = torch.nan_to_num(rule_features.float()).masked_fill(
        ~effective.unsqueeze(-1), 0.0
    )
    scene_zscore = masked_scene_zscore(raw, effective, epsilon)
    clean_base = torch.nan_to_num(
        normalized_base.float(), nan=-1e9, posinf=1e9, neginf=-1e9
    )
    base_rank = normalized_descending_rank(clean_base, effective)
    clean_base = clean_base.masked_fill(~effective, 0.0)
    inputs = torch.cat(
        [
            raw,
            scene_zscore,
            clean_base.unsqueeze(-1),
            base_rank.unsqueeze(-1),
            effective.to(raw.dtype).unsqueeze(-1),
        ],
        dim=-1,
    )
    return inputs, effective


class _SetRankerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.feedforward_norm = nn.LayerNorm(hidden_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        normalized = self.attention_norm(hidden)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=~mask,
            need_weights=False,
        )
        hidden = hidden + self.dropout(attended)
        return hidden + self.dropout(self.feedforward(self.feedforward_norm(hidden)))


class ContextSetTrajectoryRanker(nn.Module):
    """Protocol-specific set ranker with candidate self-attention and pooling."""

    def __init__(self, config: ContextRankerConfig):
        super().__init__()
        config.validate()
        self.config = config
        # Offline training keeps this false. The sealed final artifact flips it
        # immediately before serialization. Online integration must require the
        # true buffer, preventing an enabled but unloaded random ranker.
        self.register_buffer("artifact_ready", torch.tensor(False), persistent=True)
        self.input_projection = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
        )
        self.blocks = nn.ModuleList(
            _SetRankerBlock(config.hidden_dim, config.num_heads, config.dropout)
            for _ in range(config.num_layers)
        )
        self.context_projection = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
        )
        self.protocol_embeddings = nn.ParameterDict(
            {
                protocol: nn.Parameter(torch.zeros(config.hidden_dim))
                for protocol in ("v1", "v2")
            }
        )
        self.heads = nn.ModuleDict(
            {
                protocol: nn.Sequential(
                    nn.LayerNorm(config.hidden_dim),
                    nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                    nn.GELU(),
                    nn.Linear(config.hidden_dim // 2, 1),
                )
                for protocol in ("v1", "v2")
            }
        )
        # A separate shared head learns NAVTRAIN future-agent collision risk.
        # Keeping it separate from protocol utility prevents rare unsafe
        # candidates from being hidden by high progress in the ranking head.
        self.collision_head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 1),
        )
        for head in self.heads.values():
            nn.init.zeros_(head[-1].bias)
        nn.init.zeros_(self.collision_head[-1].bias)

    def mark_artifact_ready(self) -> None:
        self.artifact_ready.fill_(True)

    def require_artifact_ready(self) -> None:
        if not bool(self.artifact_ready.item()):
            raise RuntimeError(
                "Context ranker is enabled but no sealed trained artifact was loaded"
            )

    def _encode(
        self,
        rule_features: torch.Tensor,
        normalized_base: torch.Tensor,
        candidate_is_finite: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if rule_features.shape[-1] != self.config.rule_feature_dim:
            raise ValueError(
                f"Expected {self.config.rule_feature_dim} rule features, "
                f"got {rule_features.shape[-1]}"
            )
        inputs, effective = build_context_ranker_inputs(
            rule_features,
            normalized_base,
            candidate_is_finite,
            self.config.zscore_epsilon,
        )
        hidden = self.input_projection(inputs)
        for block in self.blocks:
            hidden = block(hidden, effective)
        weight = effective.unsqueeze(-1).to(hidden.dtype)
        mean_pool = (hidden * weight).sum(dim=1) / weight.sum(dim=1).clamp(min=1.0)
        max_pool = hidden.masked_fill(~effective.unsqueeze(-1), -torch.inf).amax(dim=1)
        max_pool = torch.nan_to_num(max_pool)
        context = self.context_projection(torch.cat([mean_pool, max_pool], dim=-1))
        return hidden + context.unsqueeze(1), effective

    def forward_with_collision(
        self,
        rule_features: torch.Tensor,
        normalized_base: torch.Tensor,
        candidate_is_finite: torch.Tensor,
        protocol: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if protocol not in self.heads:
            raise ValueError(f"Unsupported ranking protocol: {protocol!r}")
        hidden, _ = self._encode(
            rule_features, normalized_base, candidate_is_finite
        )
        logits = self.heads[protocol](
            hidden + self.protocol_embeddings[protocol]
        ).squeeze(-1)
        collision_logits = self.collision_head(hidden).squeeze(-1)
        return (
            torch.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9),
            torch.nan_to_num(collision_logits, nan=0.0, posinf=30.0, neginf=-30.0),
        )

    def forward(
        self,
        rule_features: torch.Tensor,
        normalized_base: torch.Tensor,
        candidate_is_finite: torch.Tensor,
        protocol: str,
        return_collision: bool = False,
    ) -> Any:
        logits, collision_logits = self.forward_with_collision(
            rule_features, normalized_base, candidate_is_finite, protocol
        )
        if return_collision:
            return logits, collision_logits
        return logits

    def parameter_bytes(self) -> int:
        return sum(parameter.numel() * parameter.element_size() for parameter in self.parameters())

    def estimated_added_runtime_bytes(self, batch_size: int, candidates: int) -> int:
        """Conservative online estimate; enforced well below the 10 GiB budget."""

        if batch_size < 1 or candidates < 2:
            raise ValueError("batch_size and candidates must be positive")
        # Parameters plus attention matrices, hidden activations and temporary
        # projections.  The factor intentionally overestimates inference use.
        activations = batch_size * candidates * self.config.hidden_dim * 4
        attention = batch_size * self.config.num_heads * candidates * candidates * 4
        estimate = self.parameter_bytes() + self.config.num_layers * (
            24 * activations + 6 * attention
        )
        if estimate >= MAX_ADDED_RUNTIME_BYTES:
            raise MemoryError(
                f"Context ranker estimate {estimate} exceeds the 10 GiB added-memory cap"
            )
        return int(estimate)


def context_ranker_config_from_runtime(config: Any) -> ContextRankerConfig:
    """Build and validate the architecture named explicitly by Hydra config."""

    result = ContextRankerConfig(
        rule_feature_dim=10,
        hidden_dim=int(getattr(config, "context_ranker_hidden_dim")),
        num_layers=int(getattr(config, "context_ranker_num_layers")),
        num_heads=int(getattr(config, "context_ranker_num_heads")),
        dropout=float(getattr(config, "context_ranker_dropout")),
        zscore_epsilon=float(getattr(config, "context_ranker_zscore_epsilon")),
    )
    result.validate()
    if not bool(getattr(config, "rule_enabled", False)):
        raise ValueError("context_ranker_enabled requires rule_enabled")
    if bool(getattr(config, "rule_finetune_only", False)):
        raise ValueError("sealed context ranker cannot run in rule_finetune_only mode")
    if str(getattr(config, "rule_protocol", "")) not in ("v1", "v2"):
        raise ValueError("context ranker requires rule_protocol v1 or v2")
    candidate_count = int(getattr(config, "rule_candidate_topk"))
    if candidate_count < 2:
        raise ValueError("context ranker requires at least two retained candidates")
    if (
        str(getattr(config, "rule_rank_fusion_mode", "hybrid")) != "hybrid"
        or float(getattr(config, "rule_protocol_proxy_weight", 0.0)) != 0.0
        or float(getattr(config, "rule_learned_zscore_weight", 0.0)) != 0.0
    ):
        raise ValueError(
            "context ranker is an isolated selector and cannot be stacked with rank fusion"
        )
    base_weight = float(getattr(config, "context_ranker_base_weight"))
    ranker_weight = float(getattr(config, "context_ranker_weight"))
    collision_threshold = float(
        getattr(config, "context_ranker_collision_threshold")
    )
    collision_weight = float(getattr(config, "context_ranker_collision_weight"))
    learned_collision_threshold = float(
        getattr(config, "context_ranker_learned_collision_threshold")
    )
    fallback_risk_slack = float(
        getattr(config, "context_ranker_fallback_risk_slack")
    )
    if not all(
        math.isfinite(value)
        for value in (
            base_weight,
            ranker_weight,
            collision_threshold,
            collision_weight,
            learned_collision_threshold,
            fallback_risk_slack,
        )
    ):
        raise ValueError("context-ranker selection parameters must be finite")
    if (
        base_weight < 0.0
        or ranker_weight <= 0.0
        or collision_threshold <= 0.0
        or collision_weight < 0.0
        or not 0.0 < learned_collision_threshold <= 1.000001
        or not 0.0 <= fallback_risk_slack <= 1.0
    ):
        raise ValueError("context-ranker selection parameters are outside their domain")
    return result


def context_ranker_selection_from_runtime(config: Any) -> Dict[str, float]:
    """Return the exact frozen per-protocol selection values from Hydra."""

    context_ranker_config_from_runtime(config)
    return {
        "base_weight": float(getattr(config, "context_ranker_base_weight")),
        "ranker_weight": float(getattr(config, "context_ranker_weight")),
        "collision_threshold": float(
            getattr(config, "context_ranker_collision_threshold")
        ),
        "collision_weight": float(getattr(config, "context_ranker_collision_weight")),
        "learned_collision_threshold": float(
            getattr(config, "context_ranker_learned_collision_threshold")
        ),
        "fallback_risk_slack": float(
            getattr(config, "context_ranker_fallback_risk_slack")
        ),
    }


def validate_context_ranker_checkpoint_metadata(
    config: Any,
    checkpoint: Mapping[str, Any],
    raw_state_dict: Mapping[str, torch.Tensor],
    expected_model_state_keys: Sequence[str],
) -> Dict[str, Any]:
    """Reject silently ignored, partial, random, or misconfigured ranker weights."""

    def normalize(name: str) -> str:
        return name[len("agent.") :] if name.startswith("agent.") else name

    expected_context_keys = {
        name
        for name in expected_model_state_keys
        if name.startswith("model._context_ranker.")
    }
    raw_context = {
        name: value
        for name, value in raw_state_dict.items()
        if normalize(name).startswith("model._context_ranker.")
    }
    metadata = checkpoint.get("momworld_context_ranker_injection")
    enabled = bool(getattr(config, "context_ranker_enabled", False))
    if not enabled:
        if metadata is not None or raw_context or expected_context_keys:
            raise RuntimeError(
                "Context-ranker checkpoint/config mismatch: enable it explicitly"
            )
        return {
            "enabled": False,
            "checkpoint_context_keys": 0,
            "metadata_present": False,
        }

    architecture = context_ranker_config_from_runtime(config)
    selection = context_ranker_selection_from_runtime(config)
    if not isinstance(metadata, Mapping):
        raise RuntimeError("Enabled context ranker has no injection metadata")
    expected_metadata_keys = {
        "schema",
        "created_at",
        "base_checkpoint_sha256",
        "ranker_state_sha256",
        "ranker_prefix",
        "model_config",
        "selection_parameters",
        "candidate_count",
        "injected_keys",
        "parameter_bytes",
        "estimated_added_runtime_bytes_batch4",
        "max_added_runtime_bytes",
    }
    if set(metadata) != expected_metadata_keys:
        raise RuntimeError("Context-ranker injection metadata fields differ")
    if metadata.get("schema") != CONTEXT_RANKER_INJECTION_SCHEMA:
        raise RuntimeError("Unexpected context-ranker injection schema")
    for field in ("base_checkpoint_sha256", "ranker_state_sha256"):
        value = metadata.get(field)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise RuntimeError(f"Invalid context-ranker metadata digest: {field}")
    created_at = metadata.get("created_at")
    try:
        created = datetime.fromisoformat(created_at)
    except (TypeError, ValueError) as error:
        raise RuntimeError("Invalid context-ranker creation timestamp") from error
    if created.tzinfo is None:
        raise RuntimeError("Context-ranker creation timestamp has no timezone")
    try:
        recorded_architecture = ContextRankerConfig.from_mapping(
            metadata["model_config"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("Invalid context-ranker architecture metadata") from error
    if recorded_architecture != architecture:
        raise RuntimeError("Context-ranker Hydra architecture differs from checkpoint")
    raw_candidate_count = metadata.get("candidate_count")
    if type(raw_candidate_count) is not int:
        raise RuntimeError("Context-ranker candidate count is not an integer")
    candidate_count = raw_candidate_count
    if candidate_count != int(getattr(config, "rule_candidate_topk")):
        raise RuntimeError("Context-ranker candidate count differs from checkpoint")
    selections = metadata.get("selection_parameters")
    if not isinstance(selections, Mapping) or set(selections) != {"v1", "v2"}:
        raise RuntimeError("Context-ranker selection metadata differs")
    protocol = str(getattr(config, "rule_protocol"))
    parsed_selections: Dict[str, Dict[str, float]] = {}
    for recorded_protocol in ("v1", "v2"):
        recorded = selections.get(recorded_protocol)
        if not isinstance(recorded, Mapping) or set(recorded) != set(selection):
            raise RuntimeError("Context-ranker protocol selection metadata differs")
        parsed: Dict[str, float] = {}
        for name in selection:
            raw_value = recorded[name]
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise RuntimeError(
                    f"Context-ranker selection parameter is invalid: {name}"
                )
            parsed[name] = float(raw_value)
        if (
            not all(math.isfinite(value) for value in parsed.values())
            or parsed["base_weight"] < 0.0
            or parsed["ranker_weight"] <= 0.0
            or parsed["collision_threshold"] <= 0.0
            or parsed["collision_weight"] < 0.0
            or not 0.0 < parsed["learned_collision_threshold"] <= 1.000001
            or not 0.0 <= parsed["fallback_risk_slack"] <= 1.0
        ):
            raise RuntimeError(
                f"Context-ranker {recorded_protocol} selection is outside its domain"
            )
        parsed_selections[recorded_protocol] = parsed
    recorded_selection = parsed_selections[protocol]
    for name, expected_value in selection.items():
        actual_value = recorded_selection[name]
        if not math.isclose(
            actual_value, expected_value, rel_tol=0.0, abs_tol=1e-12
        ):
            raise RuntimeError(
                f"Context-ranker Hydra selection differs from checkpoint: {name}"
            )

    raw_keys = set(raw_context)
    injected_keys = metadata.get("injected_keys")
    if (
        not isinstance(injected_keys, list)
        or any(not isinstance(name, str) for name in injected_keys)
        or injected_keys != sorted(raw_keys)
    ):
        raise RuntimeError("Context-ranker injected-key audit differs")
    normalized_keys = {normalize(name) for name in raw_keys}
    if normalized_keys != expected_context_keys or len(normalized_keys) != len(raw_keys):
        raise RuntimeError("Context-ranker checkpoint tensors are missing or unexpected")
    prefix = str(metadata.get("ranker_prefix", ""))
    valid_prefixes = {
        "agent.model._context_ranker.",
        "model._context_ranker.",
    }
    if prefix not in valid_prefixes or any(
        not name.startswith(prefix) for name in raw_keys
    ):
        raise RuntimeError("Context-ranker checkpoint prefix differs")
    ready = next(
        (
            value
            for name, value in raw_context.items()
            if normalize(name) == "model._context_ranker.artifact_ready"
        ),
        None,
    )
    if not isinstance(ready, torch.Tensor) or ready.numel() != 1 or not bool(
        ready.item()
    ):
        raise RuntimeError("Context-ranker checkpoint is not a sealed artifact")
    for name, value in raw_context.items():
        if not isinstance(value, torch.Tensor) or (
            value.is_floating_point() and not torch.isfinite(value).all().item()
        ):
            raise RuntimeError(f"Invalid context-ranker tensor: {name}")

    # Reconstruct only to verify the sealed memory/parameter claim, while
    # preserving the process RNG state used by the surrounding agent.
    with torch.random.fork_rng(devices=[]):
        reference = ContextSetTrajectoryRanker(architecture)
    reference_state = reference.state_dict()
    context_prefix = "model._context_ranker."
    for name, value in raw_context.items():
        local_name = normalize(name)[len(context_prefix) :]
        expected_tensor = reference_state[local_name]
        if (
            value.shape != expected_tensor.shape
            or value.dtype != expected_tensor.dtype
        ):
            raise RuntimeError(f"Context-ranker tensor type/shape differs: {name}")
    parameter_bytes = reference.parameter_bytes()
    runtime_bytes = reference.estimated_added_runtime_bytes(4, candidate_count)
    recorded_memory = (
        metadata.get("parameter_bytes"),
        metadata.get("estimated_added_runtime_bytes_batch4"),
        metadata.get("max_added_runtime_bytes"),
    )
    if any(type(value) is not int for value in recorded_memory) or recorded_memory != (
        parameter_bytes,
        runtime_bytes,
        MAX_ADDED_RUNTIME_BYTES,
    ):
        raise RuntimeError("Context-ranker memory/parameter audit differs")
    return {
        "enabled": True,
        "protocol": protocol,
        "candidate_count": candidate_count,
        "checkpoint_context_keys": len(raw_keys),
        "model_config": architecture.as_dict(),
        "selection_parameters": dict(selection),
        "parameter_bytes": parameter_bytes,
        "estimated_added_runtime_bytes_batch4": runtime_bytes,
        "metadata_present": True,
    }


def candidate_score_zscore(
    scores: torch.Tensor,
    epsilon: float = 1e-4,
    candidate_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if scores.ndim != 2:
        raise ValueError("scores must have shape [batch, candidates]")
    scores = torch.nan_to_num(scores.float())
    if candidate_mask is None:
        candidate_mask = torch.ones_like(scores, dtype=torch.bool)
    if candidate_mask.shape != scores.shape:
        raise ValueError("candidate_mask must match scores")
    effective = effective_candidate_mask(candidate_mask)
    weight = effective.to(scores.dtype)
    count = weight.sum(dim=1, keepdim=True).clamp(min=1.0)
    mean = (scores * weight).sum(dim=1, keepdim=True) / count
    variance = ((scores - mean).square() * weight).sum(dim=1, keepdim=True) / count
    zscore = (scores - mean) / variance.sqrt().clamp(min=float(epsilon))
    return zscore.masked_fill(~effective, 0.0)


def eligible_candidate_mask(
    predicted_collision: torch.Tensor,
    candidate_is_finite: torch.Tensor,
    collision_threshold: float,
) -> torch.Tensor:
    """Mirror MomWorld safe/finite/all fallback for contextual selection."""

    if predicted_collision.ndim != 2 or candidate_is_finite.shape != predicted_collision.shape:
        raise ValueError("collision and finite tensors must share [batch, candidates]")
    if not math.isfinite(float(collision_threshold)):
        raise ValueError("collision_threshold must be finite")
    finite = candidate_is_finite.bool()
    safe = (predicted_collision < float(collision_threshold)) & finite
    return torch.where(
        safe.any(dim=1, keepdim=True),
        safe,
        torch.where(finite.any(dim=1, keepdim=True), finite, torch.ones_like(finite)),
    )


def select_context_ranker_candidates(
    logits: torch.Tensor,
    normalized_base: torch.Tensor,
    predicted_collision: torch.Tensor,
    candidate_is_finite: torch.Tensor,
    base_weight: float,
    ranker_weight: float,
    collision_threshold: float,
    learned_collision: Optional[torch.Tensor] = None,
    collision_weight: float = 0.0,
    learned_collision_threshold: float = 1.000001,
    fallback_risk_slack: float = 1.0,
    epsilon: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fuse base and contextual ranks, then apply the audited safety fallback."""

    combined, eligible = context_ranker_selection_scores(
        logits,
        normalized_base,
        predicted_collision,
        candidate_is_finite,
        base_weight,
        ranker_weight,
        collision_threshold,
        learned_collision,
        collision_weight,
        learned_collision_threshold,
        fallback_risk_slack,
        epsilon,
    )
    return combined.argmax(dim=1), eligible


def context_ranker_selection_scores(
    logits: torch.Tensor,
    normalized_base: torch.Tensor,
    predicted_collision: torch.Tensor,
    candidate_is_finite: torch.Tensor,
    base_weight: float,
    ranker_weight: float,
    collision_threshold: float,
    learned_collision: Optional[torch.Tensor] = None,
    collision_weight: float = 0.0,
    learned_collision_threshold: float = 1.000001,
    fallback_risk_slack: float = 1.0,
    epsilon: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return risk-aware fused scores and a stable lowest-risk fallback mask."""

    expected = logits.shape
    if logits.ndim != 2 or any(
        value.shape != expected
        for value in (normalized_base, predicted_collision, candidate_is_finite)
    ):
        raise ValueError("all selection tensors must share [batch, candidates]")
    if learned_collision is None:
        learned_collision = torch.zeros_like(logits)
    if learned_collision.shape != expected:
        raise ValueError("learned collision must share [batch, candidates]")
    if not all(
        math.isfinite(float(value))
        for value in (
            base_weight,
            ranker_weight,
            collision_weight,
            learned_collision_threshold,
            fallback_risk_slack,
        )
    ):
        raise ValueError("selection weights must be finite")
    if (
        float(collision_weight) < 0.0
        or not 0.0 < float(learned_collision_threshold) <= 1.000001
        or not 0.0 <= float(fallback_risk_slack) <= 1.0
    ):
        raise ValueError("risk-aware selection parameters are outside their domain")
    finite = candidate_is_finite.bool()
    learned_collision = torch.nan_to_num(
        learned_collision.float(), nan=1.0, posinf=1.0, neginf=0.0
    ).clamp(0.0, 1.0)
    combined = float(base_weight) * torch.nan_to_num(normalized_base.float())
    combined = combined + float(ranker_weight) * candidate_score_zscore(
        logits, epsilon, candidate_is_finite
    )
    if float(collision_weight) != 0.0:
        combined = combined - float(collision_weight) * candidate_score_zscore(
            learned_collision, epsilon, candidate_is_finite
        )

    geometry_safe = (
        torch.nan_to_num(predicted_collision.float(), nan=1.0)
        < float(collision_threshold)
    )
    learned_safe = learned_collision < float(learned_collision_threshold)
    safe = geometry_safe & learned_safe & finite
    has_safe = safe.any(dim=1, keepdim=True)

    # If every candidate is unsafe, do not fall back to arbitrary utility.
    # Restrict the fallback to the learned minimum-risk tier (plus a small
    # fixed slack), then let utility/progress break ties inside that tier.
    risk_for_min = learned_collision.masked_fill(~finite, torch.inf)
    minimum_risk = risk_for_min.amin(dim=1, keepdim=True)
    low_risk = finite & (
        learned_collision <= minimum_risk + float(fallback_risk_slack)
    )
    has_finite = finite.any(dim=1, keepdim=True)
    fallback = torch.where(
        has_finite,
        low_risk,
        torch.ones_like(finite),
    )
    eligible = torch.where(has_safe, safe, fallback)
    combined = torch.nan_to_num(combined, nan=-1e9, posinf=1e9, neginf=-1e9)
    combined = combined.masked_fill(~eligible, torch.finfo(combined.dtype).min)
    return combined, eligible


def listwise_soft_target_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    candidate_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """ListNet-style cross entropy over every candidate in each scene."""

    if logits.shape != targets.shape or candidate_mask.shape != logits.shape:
        raise ValueError("listwise tensors must share [batch, candidates]")
    if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise ValueError("temperature must be finite and positive")
    mask = effective_candidate_mask(candidate_mask)
    floor = torch.finfo(logits.dtype).min
    predicted_log_probability = F.log_softmax(logits.masked_fill(~mask, floor), dim=1)
    target_probability = F.softmax(
        (targets / float(temperature)).masked_fill(~mask, floor), dim=1
    )
    return -(target_probability * predicted_log_probability).sum(dim=1).mean()


def pairwise_ranknet_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    candidate_mask: torch.Tensor,
    minimum_target_gap: float = 1e-3,
) -> torch.Tensor:
    """All-pairs RankNet loss, weighted by normalized target-score gap."""

    if logits.shape != targets.shape or candidate_mask.shape != logits.shape:
        raise ValueError("pairwise tensors must share [batch, candidates]")
    mask = effective_candidate_mask(candidate_mask)
    target_difference = targets[:, :, None] - targets[:, None, :]
    logit_difference = logits[:, :, None] - logits[:, None, :]
    pair_mask = mask[:, :, None] & mask[:, None, :]
    pair_mask = pair_mask & (target_difference.abs() > float(minimum_target_gap))
    # Each unordered pair appears twice. Averaging both directions is exactly
    # symmetric and avoids sampling-dependent offline/online drift.
    target_sign = target_difference.sign()
    weights = target_difference.abs().detach()
    weights = weights / weights.amax(dim=(1, 2), keepdim=True).clamp(min=1e-6)
    losses = F.softplus(-target_sign * logit_difference) * weights
    return (losses * pair_mask.to(losses.dtype)).sum() / pair_mask.sum().clamp(min=1)
