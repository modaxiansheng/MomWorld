"""Momentum-aware latent world model built on the UniLAW GTRS-Dense agent."""

import hashlib
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from navsim.agents.gtrs_dense.hydra_model import HydraModel
from navsim.agents.momworld.context_ranker import (
    ContextSetTrajectoryRanker,
    context_ranker_config_from_runtime,
    context_ranker_selection_from_runtime,
    context_ranker_selection_scores,
)
from navsim.agents.momworld.momworld_config import MomWorldConfig
from navsim.agents.momworld.momworld_pairwise_gate import V2NavhardPairwiseGate


def _logit(probability: float) -> float:
    probability = min(max(float(probability), 1e-4), 1.0 - 1e-4)
    return math.log(probability / (1.0 - probability))


def _trajectory_to_flow_state(
    trajectory: torch.Tensor, position_scale: float
) -> torch.Tensor:
    """Encode ``(x, y, heading)`` without an angular discontinuity."""
    heading = trajectory[..., 2:3]
    return torch.cat(
        [trajectory[..., :2] / position_scale, torch.sin(heading), torch.cos(heading)],
        dim=-1,
    )


def _flow_state_to_trajectory(
    flow_state: torch.Tensor, position_scale: float
) -> torch.Tensor:
    """Decode normalized ``(x, y, sin(heading), cos(heading))`` state."""
    heading = torch.atan2(flow_state[..., 2:3], flow_state[..., 3:4])
    return torch.cat([flow_state[..., :2] * position_scale, heading], dim=-1)


RULE_METRIC_NAMES: Tuple[str, ...] = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "time_to_collision_within_bound",
    "ego_progress",
    "driving_direction_compliance",
    "lane_keeping",
    "traffic_light_compliance",
)

RELU_GUARD_WEIGHT_FIELDS: Tuple[str, ...] = (
    "rule_relu_collision_risk_weight",
    "rule_relu_no_collision_weight",
    "rule_relu_drivable_area_weight",
    "rule_relu_ttc_weight",
    "rule_relu_progress_weight",
    "rule_relu_direction_weight",
    "rule_relu_lane_weight",
    "rule_relu_traffic_light_weight",
    "rule_relu_kinematic_weight",
    "rule_relu_momentum_weight",
)

RULE_SAFETY_MIN_FIELDS: Tuple[str, ...] = (
    "rule_safety_no_collision_min",
    "rule_safety_drivable_area_min",
    "rule_safety_ttc_min",
    "rule_safety_progress_min",
    "rule_safety_direction_min",
    "rule_safety_lane_min",
    "rule_safety_traffic_light_min",
)

RULE_RELATIVE_SAFETY_FRACTION_FIELDS: Tuple[str, ...] = (
    "rule_relative_no_collision_fraction",
    "rule_relative_drivable_area_fraction",
    "rule_relative_ttc_fraction",
    "rule_relative_progress_fraction",
    "rule_relative_direction_fraction",
    "rule_relative_lane_fraction",
    "rule_relative_traffic_light_fraction",
    "rule_relative_collision_risk_fraction",
    "rule_relative_kinematic_fraction",
    "rule_relative_momentum_fraction",
)

RULE_MONOTONIC_WEIGHT_FIELDS: Tuple[str, ...] = (
    "rule_monotonic_no_collision_weight",
    "rule_monotonic_drivable_area_weight",
    "rule_monotonic_ttc_weight",
    "rule_monotonic_progress_weight",
    "rule_monotonic_direction_weight",
    "rule_monotonic_lane_weight",
    "rule_monotonic_traffic_light_weight",
    "rule_monotonic_collision_risk_weight",
    "rule_monotonic_kinematic_weight",
    "rule_monotonic_momentum_weight",
)


def monotonic_residual_weights_from_config(config: Any) -> Tuple[float, ...]:
    """Return a validated, opt-in monotonic residual configuration."""

    enabled = bool(getattr(config, "rule_monotonic_residual_enabled", False))
    weights = tuple(float(getattr(config, name, 0.0)) for name in RULE_MONOTONIC_WEIGHT_FIELDS)
    if any(not math.isfinite(value) or value < 0.0 for value in weights):
        raise ValueError("Monotonic residual weights must be finite and non-negative")
    if not enabled:
        if any(value != 0.0 for value in weights):
            raise ValueError("Non-zero monotonic residual weights require enabling it")
        return weights
    if not any(value > 0.0 for value in weights):
        raise ValueError("Enabled monotonic residual requires a positive weight")
    epsilon = float(getattr(config, "rule_monotonic_zscore_epsilon", 1e-4))
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("Monotonic residual epsilon must be finite and positive")
    return weights


def candidate_monotonic_residual(
    rule_features: torch.Tensor,
    weights: Tuple[float, ...],
    epsilon: float = 1e-4,
) -> torch.Tensor:
    """Score safer candidates monotonically using within-scene features."""

    if rule_features.ndim != 3 or rule_features.shape[-1] != 10:
        raise ValueError("rule_features must have shape [batch, candidates, 10]")
    if len(weights) != 10 or any(
        not math.isfinite(float(value)) or float(value) < 0.0 for value in weights
    ):
        raise ValueError("monotonic residual requires ten non-negative weights")
    if not math.isfinite(float(epsilon)) or float(epsilon) <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    features = rule_features.float()
    mean = features.mean(dim=1, keepdim=True)
    std = features.std(dim=1, keepdim=True, unbiased=False).clamp(min=float(epsilon))
    standardized = (features - mean) / std
    direction = standardized.new_tensor((1.0,) * 7 + (-1.0,) * 3)
    coefficient = standardized.new_tensor(weights)
    return (standardized * direction * coefficient).sum(dim=-1)


def candidate_rule_safety_filter(
    rule_features: torch.Tensor,
    candidate_is_finite: torch.Tensor,
    config: Any,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Delete unsafe candidates with a minimum-risk all-unsafe fallback.

    Returns ``eligible_mask``, ``strict_safe_mask``, and aggregate violation.
    The fallback never drops a scene: if no candidate satisfies every frozen
    threshold, only finite candidates with minimum violation remain eligible.
    The normal scorer then breaks ties using progress and the other metrics.
    """

    if rule_features.ndim != 3 or rule_features.shape[-1] < 10:
        raise ValueError("rule_features must have shape [batch, candidates, >=10]")
    if candidate_is_finite.shape != rule_features.shape[:2]:
        raise ValueError("candidate_is_finite shape must match candidate features")
    minimums = tuple(float(getattr(config, name)) for name in RULE_SAFETY_MIN_FIELDS)
    collision_max = float(getattr(config, "rule_safety_collision_risk_max"))
    kinematic_max = float(getattr(config, "rule_safety_kinematic_max"))
    momentum_max = float(getattr(config, "rule_safety_momentum_max"))
    fallback_slack = float(getattr(config, "rule_safety_fallback_risk_slack"))
    relative_enabled = bool(
        getattr(config, "rule_relative_safety_filter_enabled", False)
    )
    relative_fractions = tuple(
        float(getattr(config, name, 1.0))
        for name in RULE_RELATIVE_SAFETY_FRACTION_FIELDS
    )
    if (
        any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in minimums)
        or not math.isfinite(collision_max)
        or not 0.0 <= collision_max <= 1.000001
        or any(
            not math.isfinite(value) or value < 0.0
            for value in (kinematic_max, momentum_max, fallback_slack)
        )
        or any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in relative_fractions
        )
    ):
        raise ValueError("Invalid candidate safety-filter thresholds")
    if not relative_enabled and any(value != 1.0 for value in relative_fractions):
        raise ValueError("Relative safety fractions require the relative Rule")
    if relative_enabled and all(value == 1.0 for value in relative_fractions):
        raise ValueError("Enabled relative Rule requires a non-neutral fraction")

    features = rule_features.float()
    finite = candidate_is_finite.bool()
    strict = finite.clone()
    violation = features.new_zeros(features.shape[:2])
    for index, minimum in enumerate(minimums):
        strict &= features[..., index] >= minimum
        if minimum > 0.0:
            violation = violation + torch.relu(minimum - features[..., index]) / max(
                minimum, 0.1
            )
    for index, maximum in (
        (7, collision_max),
        (8, kinematic_max),
        (9, momentum_max),
    ):
        strict &= features[..., index] < maximum
        violation = violation + torch.relu(features[..., index] - maximum) / max(
            maximum, 0.1
        )

    if relative_enabled:
        has_finite = finite.any(dim=1, keepdim=True)
        minimum = features.masked_fill(~finite[..., None], torch.inf).amin(
            dim=1
        )
        maximum = features.masked_fill(~finite[..., None], -torch.inf).amax(
            dim=1
        )
        minimum = torch.where(has_finite, minimum, torch.zeros_like(minimum))
        maximum = torch.where(has_finite, maximum, torch.zeros_like(maximum))
        span = (maximum - minimum).clamp(min=0.0)
        denominator = span.clamp(min=1e-6)
        for index, fraction in enumerate(relative_fractions):
            if fraction == 1.0:
                continue
            if index < len(RULE_METRIC_NAMES):
                boundary = maximum[..., index : index + 1] - fraction * span[
                    ..., index : index + 1
                ]
                component_violation = torch.relu(
                    boundary - features[..., index]
                ) / denominator[..., index : index + 1]
                strict &= features[..., index] >= boundary - 1e-7
            else:
                boundary = minimum[..., index : index + 1] + fraction * span[
                    ..., index : index + 1
                ]
                component_violation = torch.relu(
                    features[..., index] - boundary
                ) / denominator[..., index : index + 1]
                strict &= features[..., index] <= boundary + 1e-7
            violation = violation + component_violation

    has_strict = strict.any(dim=1, keepdim=True)
    has_finite = finite.any(dim=1, keepdim=True)
    finite_violation = violation.masked_fill(~finite, torch.inf)
    minimum_violation = finite_violation.amin(dim=1, keepdim=True)
    fallback = finite & (violation <= minimum_violation + fallback_slack)
    eligible = torch.where(
        has_strict,
        strict,
        torch.where(has_finite, fallback, torch.ones_like(finite)),
    )
    return eligible, strict, violation


def candidate_score_zscore(
    scores: torch.Tensor, epsilon: float = 1e-4
) -> torch.Tensor:
    """Standardize candidate scores independently within each scene.

    Absolute scorer calibration varies substantially across scenes.  Rule
    fusion only needs the within-scene ordering, so this transformation keeps
    the candidate-relative signal while giving its configured weight a stable
    meaning.  A constant candidate row maps to zeros.
    """

    if scores.ndim != 2:
        raise ValueError("candidate scores must have shape [batch, candidates]")
    if not math.isfinite(float(epsilon)) or float(epsilon) <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    # The cache tuner always standardizes float32 values.  Explicitly mirror
    # that here because evaluation runs under mixed precision and float16
    # reductions can otherwise change the winner for nearly tied candidates.
    scores = scores.float()
    mean = scores.mean(dim=1, keepdim=True)
    std = scores.std(dim=1, keepdim=True, unbiased=False).clamp(
        min=float(epsilon)
    )
    return (scores - mean) / std


def candidate_calibrated_collision_filter(
    calibrated_collision_risk: torch.Tensor,
    candidate_is_finite: torch.Tensor,
    threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the frozen calibrator mask with the diagnostic's stable fallback."""

    if calibrated_collision_risk.ndim != 2:
        raise ValueError("calibrated collision risk must be [batch, candidates]")
    if candidate_is_finite.shape != calibrated_collision_risk.shape:
        raise ValueError("finite mask shape must match calibrated collision risk")
    if not math.isfinite(float(threshold)) or not 0.0 <= float(threshold) <= 1.000001:
        raise ValueError("calibrated collision threshold is invalid")
    finite = candidate_is_finite.bool()
    strict_safe = (calibrated_collision_risk < float(threshold)) & finite
    has_safe = strict_safe.any(dim=1, keepdim=True)
    has_finite = finite.any(dim=1, keepdim=True)
    eligible = torch.where(
        has_safe,
        strict_safe,
        torch.where(has_finite, finite, torch.ones_like(finite)),
    )
    return eligible, strict_safe


def candidate_protocol_proxy(
    rule_features: torch.Tensor, protocol: str
) -> torch.Tensor:
    """Compose predicted NAVSIM metric components into a rank proxy.

    ``rule_features`` starts with the seven entries in ``RULE_METRIC_NAMES``.
    Comfort signals are not available in the online candidate features, so
    their neutral value of one mirrors the cache builder's missing-component
    convention.  This proxy never reads evaluator output or benchmark labels.
    """

    if rule_features.ndim != 3 or rule_features.shape[-1] < len(RULE_METRIC_NAMES):
        raise ValueError(
            "rule_features must have shape [batch, candidates, >=7]"
        )
    # Cache records are persisted as float32 after the mixed-precision model
    # forward.  Perform the composition in float32 online as well so the
    # offline search and deployed selector use the same arithmetic.
    rule_features = rule_features.float()
    components = [
        rule_features[..., index].clamp(0.0, 1.0)
        for index in range(len(RULE_METRIC_NAMES))
    ]
    no_collision, drivable, ttc, progress, direction, lane, traffic_light = (
        components
    )
    if protocol == "v1":
        return no_collision * drivable * (
            5.0 * progress + 5.0 * ttc + 2.0
        ) / 12.0
    if protocol == "v2":
        return no_collision * drivable * direction * traffic_light * (
            5.0 * progress + 5.0 * ttc + 2.0 * lane + 4.0
        ) / 16.0
    raise ValueError(f"Unsupported rule protocol: {protocol!r}")


def v1_protocol_gate_candidates(
    normalized_base: torch.Tensor,
    protocol_proxy_zscore: torch.Tensor,
    candidate_is_finite: torch.Tensor,
    incumbent_weight: float,
    alternative_weights: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the frozen V1 incumbent and fixed-weight proposals."""

    base = normalized_base.float()
    proxy = protocol_proxy_zscore.float()
    finite = candidate_is_finite.bool()
    weights = alternative_weights.float()
    if base.ndim != 2 or proxy.shape != base.shape or finite.shape != base.shape:
        raise ValueError("V1 gate candidate inputs must be [batch,candidates]")
    if weights.ndim != 1 or len(weights) < 1:
        raise ValueError("V1 gate alternative weights must be one-dimensional")
    floor = torch.finfo(base.dtype).min
    incumbent = torch.where(
        finite, base + float(incumbent_weight) * proxy, floor
    ).argmax(dim=1)
    proposal_scores = base[:, None, :] + weights[None, :, None] * proxy[:, None, :]
    proposals = torch.where(finite[:, None, :], proposal_scores, floor).argmax(dim=2)
    return incumbent, proposals


def v1_protocol_gate_features(
    rule_features: torch.Tensor,
    normalized_base: torch.Tensor,
    protocol_proxy_zscore: torch.Tensor,
    candidate_is_finite: torch.Tensor,
    incumbent: torch.Tensor,
    proposals: torch.Tensor,
    alternative_weights: torch.Tensor,
) -> torch.Tensor:
    """Mirror the 63-D NAVTRAIN gate features with online tensors only."""

    values = torch.nan_to_num(
        rule_features.float(), nan=0.0, posinf=0.0, neginf=0.0
    )
    base = normalized_base.float()
    proxy = protocol_proxy_zscore.float()
    finite = candidate_is_finite.bool()
    if values.ndim != 3 or values.shape[-1] != 10:
        raise ValueError("V1 gate rule features must be [batch,candidates,10]")
    if base.shape != values.shape[:2] or proxy.shape != base.shape or finite.shape != base.shape:
        raise ValueError("V1 gate candidate feature shapes differ")
    if incumbent.shape != (len(values),) or proposals.ndim != 2:
        raise ValueError("V1 gate selection index shapes differ")
    if proposals.shape != (len(values), len(alternative_weights)):
        raise ValueError("V1 gate proposal/weight shapes differ")

    effective = torch.where(
        finite.any(dim=1, keepdim=True), finite, torch.ones_like(finite)
    )
    mask = effective.unsqueeze(-1)
    count = mask.sum(dim=1).clamp_min(1).to(values.dtype)
    scene_mean = (values * mask).sum(dim=1) / count
    centered = values - scene_mean[:, None, :]
    scene_std = torch.sqrt((centered.square() * mask).sum(dim=1) / count)
    positive_inf = torch.full_like(values, torch.inf)
    negative_inf = torch.full_like(values, -torch.inf)
    scene_min = torch.where(mask, values, positive_inf).amin(dim=1)
    scene_max = torch.where(mask, values, negative_inf).amax(dim=1)
    scene_span = scene_max - scene_min

    batch = torch.arange(len(values), device=values.device)
    incumbent_features = values[batch, incumbent]
    incumbent_base = base[batch, incumbent]
    incumbent_proxy = proxy[batch, incumbent]
    expanded = proposals.unsqueeze(-1).expand(-1, -1, values.shape[-1])
    proposal_features = values.gather(1, expanded)
    difference = proposal_features - incumbent_features[:, None, :]
    proposal_base = base.gather(1, proposals)
    proposal_proxy = proxy.gather(1, proposals)
    weights = alternative_weights.to(values).view(1, -1, 1).expand(len(values), -1, -1)
    scene = lambda tensor: tensor[:, None, :].expand(-1, len(alternative_weights), -1)
    result = torch.cat(
        (
            difference,
            (proposal_base - incumbent_base[:, None]).unsqueeze(-1),
            (proposal_proxy - incumbent_proxy[:, None]).unsqueeze(-1),
            weights,
            difference * weights,
            scene(incumbent_features),
            scene(scene_mean),
            scene(scene_std),
            scene(scene_span),
        ),
        dim=-1,
    )
    if result.shape != (len(values), len(alternative_weights), 63):
        raise RuntimeError(f"unexpected V1 protocol gate feature shape: {result.shape}")
    if not torch.isfinite(result).all():
        raise FloatingPointError("non-finite V1 protocol gate features")
    return result


class V1ProtocolSelectorGate(nn.Module):
    """Sealed pair of linear NAVTRAIN models for conservative V1 switching."""

    STATE_SCHEMA = "momworld-v1-protocol-selector-gate-state-v1"

    def __init__(self, config: Any) -> None:
        super().__init__()
        path = Path(str(getattr(config, "rule_v1_protocol_gate_state_path", ""))).expanduser()
        expected_sha256 = str(
            getattr(config, "rule_v1_protocol_gate_state_sha256", "")
        ).lower()
        if not path.is_file() or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
            raise ValueError("enabled V1 protocol gate requires a state path and SHA256")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected_sha256:
            raise RuntimeError("V1 protocol gate state SHA256 mismatch")
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict) or payload.get("schema") != self.STATE_SCHEMA:
            raise RuntimeError("invalid V1 protocol gate state")
        if int(payload.get("feature_dim", -1)) != 63:
            raise RuntimeError("invalid V1 protocol gate feature dimension")
        self.incumbent_weight = float(payload["incumbent_weight"])
        self.min_predicted_gain = float(payload["min_predicted_gain"])
        self.max_predicted_collision_delta = float(
            payload["max_predicted_collision_delta"]
        )
        if not (
            math.isclose(self.incumbent_weight, 1.6, rel_tol=0.0, abs_tol=1e-12)
            and math.isclose(self.min_predicted_gain, 0.002, rel_tol=0.0, abs_tol=1e-12)
            and math.isclose(
                self.max_predicted_collision_delta, 0.0, rel_tol=0.0, abs_tol=1e-12
            )
        ):
            raise RuntimeError("unexpected V1 protocol gate selection constants")
        self.register_buffer(
            "alternative_weights",
            torch.as_tensor(payload["alternative_weights"], dtype=torch.float32),
        )
        for prefix in ("gain", "collision"):
            state = payload[f"{prefix}_model"]
            for name in ("mean", "scale", "coefficient"):
                value = torch.as_tensor(state[name], dtype=torch.float64)
                if value.shape != (63,) or not torch.isfinite(value).all():
                    raise RuntimeError(f"invalid V1 protocol gate {prefix} {name}")
                self.register_buffer(f"{prefix}_{name}", value)
            intercept = torch.as_tensor(float(state["intercept"]), dtype=torch.float64)
            if not torch.isfinite(intercept):
                raise RuntimeError(f"invalid V1 protocol gate {prefix} intercept")
            self.register_buffer(f"{prefix}_intercept", intercept)

    def _predict(self, features: torch.Tensor, prefix: str) -> torch.Tensor:
        values = features.double()
        mean = getattr(self, f"{prefix}_mean")
        scale = getattr(self, f"{prefix}_scale")
        coefficient = getattr(self, f"{prefix}_coefficient")
        intercept = getattr(self, f"{prefix}_intercept")
        return (((values - mean) / scale) * coefficient).sum(dim=-1) + intercept

    def select(
        self,
        rule_features: torch.Tensor,
        normalized_base: torch.Tensor,
        protocol_proxy_zscore: torch.Tensor,
        candidate_is_finite: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        incumbent, proposals = v1_protocol_gate_candidates(
            normalized_base,
            protocol_proxy_zscore,
            candidate_is_finite,
            self.incumbent_weight,
            self.alternative_weights,
        )
        features = v1_protocol_gate_features(
            rule_features,
            normalized_base,
            protocol_proxy_zscore,
            candidate_is_finite,
            incumbent,
            proposals,
            self.alternative_weights,
        )
        gain = self._predict(features, "gain")
        collision = self._predict(features, "collision")
        eligible = collision <= self.max_predicted_collision_delta
        score = torch.where(eligible, gain, torch.full_like(gain, -torch.inf))
        best = score.argmax(dim=1)
        batch = torch.arange(len(score), device=score.device)
        switch = eligible[batch, best] & (score[batch, best] >= self.min_predicted_gain)
        selected = torch.where(switch, proposals[batch, best], incumbent)
        return selected, gain.float(), collision.float()


def v2_navtest_gate_features(
    rule_features: torch.Tensor,
    normalized_base: torch.Tensor,
    protocol_proxy_zscore: torch.Tensor,
    learned_score: torch.Tensor,
    candidate_is_finite: torch.Tensor,
    incumbent: torch.Tensor,
    proposals: torch.Tensor,
    configurations: torch.Tensor,
) -> torch.Tensor:
    """Mirror the 68-D frozen V2 NavTest selector features online."""

    values = torch.nan_to_num(
        rule_features.float(), nan=0.0, posinf=0.0, neginf=0.0
    )
    base = normalized_base.float()
    proxy = protocol_proxy_zscore.float()
    learned = learned_score.float()
    finite = candidate_is_finite.bool()
    configs = configurations.float()
    if values.ndim != 3 or values.shape[-1] != 10:
        raise ValueError("V2 gate rule features must be [batch,candidates,10]")
    if any(tensor.shape != values.shape[:2] for tensor in (base, proxy, learned, finite)):
        raise ValueError("V2 gate candidate input shapes differ")
    if proposals.shape != (len(values), len(configs)) or configs.shape[1:] != (5,):
        raise ValueError("V2 gate proposal/configuration shapes differ")
    has_finite = finite.any(dim=1, keepdim=True)
    mask = finite.unsqueeze(-1)
    count = mask.sum(dim=1).clamp_min(1).to(values.dtype)
    scene_mean = (values * mask).sum(dim=1) / count
    centered = values - scene_mean[:, None, :]
    scene_std = torch.sqrt((centered.square() * mask).sum(dim=1) / count)
    scene_min = torch.where(mask, values, torch.full_like(values, torch.inf)).amin(dim=1)
    scene_max = torch.where(mask, values, torch.full_like(values, -torch.inf)).amax(dim=1)
    scene_span = torch.where(
        has_finite, scene_max - scene_min, torch.zeros_like(scene_min)
    )
    batch = torch.arange(len(values), device=values.device)
    incumbent_features = values[batch, incumbent]
    expanded = proposals.unsqueeze(-1).expand(-1, -1, 10)
    proposal_features = values.gather(1, expanded)
    difference = proposal_features - incumbent_features[:, None, :]
    proposal_base = base.gather(1, proposals)
    proposal_proxy = proxy.gather(1, proposals)
    proposal_learned = learned.gather(1, proposals)
    incumbent_base = base[batch, incumbent]
    incumbent_proxy = proxy[batch, incumbent]
    incumbent_learned = learned[batch, incumbent]
    encoded = configs.to(values).unsqueeze(0).expand(len(values), -1, -1)
    proxy_weight = encoded[..., 3:4]
    scene = lambda tensor: tensor[:, None, :].expand(-1, len(configs), -1)
    result = torch.cat(
        (
            difference,
            (proposal_base - incumbent_base[:, None]).unsqueeze(-1),
            (proposal_proxy - incumbent_proxy[:, None]).unsqueeze(-1),
            (proposal_learned - incumbent_learned[:, None]).unsqueeze(-1),
            encoded,
            difference * proxy_weight,
            scene(incumbent_features),
            scene(scene_mean),
            scene(scene_std),
            scene(scene_span),
        ),
        dim=-1,
    )
    if result.shape != (len(values), len(configs), 68) or not torch.isfinite(result).all():
        raise RuntimeError(f"invalid V2 selector features: {result.shape}")
    return result


class V2NavtestSelectorGate(nn.Module):
    """Sealed NAVTRAIN ridge gate for the selected V2 NavTest incumbent."""

    STATE_SCHEMA = "momworld-v2-navtest-selector-gate-state-v1"

    def __init__(self, config: Any) -> None:
        super().__init__()
        path = Path(
            str(getattr(config, "rule_v2_navtest_selector_gate_state_path", ""))
        ).expanduser()
        expected_sha256 = str(
            getattr(config, "rule_v2_navtest_selector_gate_state_sha256", "")
        ).lower()
        if not path.is_file() or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
            raise ValueError("enabled V2 NavTest gate requires a state path and SHA256")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected_sha256:
            raise RuntimeError("V2 NavTest selector state SHA256 mismatch")
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict) or payload.get("schema") != self.STATE_SCHEMA:
            raise RuntimeError("invalid V2 NavTest selector state")
        if int(payload.get("feature_dim", -1)) != 68:
            raise RuntimeError("invalid V2 NavTest selector feature dimension")
        self.min_predicted_gain = float(payload["min_predicted_gain"])
        self.max_predicted_collision_delta = float(payload["max_predicted_collision_delta"])
        if not (
            math.isclose(self.min_predicted_gain, 0.002, rel_tol=0.0, abs_tol=1e-12)
            and math.isclose(
                self.max_predicted_collision_delta, 0.002, rel_tol=0.0, abs_tol=1e-12
            )
        ):
            raise RuntimeError("unexpected V2 NavTest selector thresholds")
        configurations = torch.as_tensor(payload["configurations"], dtype=torch.float32)
        if configurations.shape != (9, 5) or not torch.isfinite(configurations).all():
            raise RuntimeError("invalid V2 NavTest selector configurations")
        self.register_buffer("configurations", configurations)
        for prefix in ("gain", "collision"):
            state = payload[f"{prefix}_model"]
            for name in ("mean", "scale", "coefficient"):
                value = torch.as_tensor(state[name], dtype=torch.float64)
                if value.shape != (68,) or not torch.isfinite(value).all():
                    raise RuntimeError(f"invalid V2 selector {prefix} {name}")
                self.register_buffer(f"{prefix}_{name}", value)
            intercept = torch.as_tensor(float(state["intercept"]), dtype=torch.float64)
            if not torch.isfinite(intercept):
                raise RuntimeError(f"invalid V2 selector {prefix} intercept")
            self.register_buffer(f"{prefix}_intercept", intercept)

    def _predict(self, features: torch.Tensor, prefix: str) -> torch.Tensor:
        values = features.double()
        mean = getattr(self, f"{prefix}_mean")
        scale = getattr(self, f"{prefix}_scale")
        coefficient = getattr(self, f"{prefix}_coefficient")
        intercept = getattr(self, f"{prefix}_intercept")
        return (((values - mean) / scale) * coefficient).sum(dim=-1) + intercept

    def select(
        self,
        rule_features: torch.Tensor,
        normalized_base: torch.Tensor,
        protocol_proxy_zscore: torch.Tensor,
        learned_score: torch.Tensor,
        candidate_is_finite: torch.Tensor,
        eligible_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base = normalized_base.float()
        proxy = protocol_proxy_zscore.float()
        learned = learned_score.float()
        features = rule_features.float()
        collision = features[..., 7]
        momentum = features[..., 9]
        floor = torch.finfo(base.dtype).min
        incumbent_score = base + learned - 2.0 * collision - 0.1 * momentum
        incumbent = torch.where(eligible_mask, incumbent_score, floor).argmax(dim=1)
        configs = self.configurations.to(base)
        scores = (
            base[:, None, :]
            + configs[None, :, 0:1] * learned[:, None, :]
            - configs[None, :, 1:2] * collision[:, None, :]
            - configs[None, :, 2:3] * momentum[:, None, :]
            + configs[None, :, 3:4] * proxy[:, None, :]
        )
        proposals = torch.where(eligible_mask[:, None, :], scores, floor).argmax(dim=2)
        gate_features = v2_navtest_gate_features(
            rule_features,
            base,
            proxy,
            learned,
            candidate_is_finite,
            incumbent,
            proposals,
            configs,
        )
        gain_prediction = self._predict(gate_features, "gain")
        collision_prediction = self._predict(gate_features, "collision")
        eligible = collision_prediction <= self.max_predicted_collision_delta
        score = torch.where(
            eligible, gain_prediction, torch.full_like(gain_prediction, -torch.inf)
        )
        best = score.argmax(dim=1)
        batch = torch.arange(len(score), device=score.device)
        switch = eligible[batch, best] & (score[batch, best] >= self.min_predicted_gain)
        selected = torch.where(switch, proposals[batch, best], incumbent)
        return selected, gain_prediction.float(), collision_prediction.float()


def v2_navhard_candidate_selections(
    rule_features: torch.Tensor,
    normalized_base: torch.Tensor,
    protocol_proxy_zscore: torch.Tensor,
    candidate_is_finite: torch.Tensor,
    configurations: torch.Tensor,
    incumbent_proxy_weight: float = 3.2,
    incumbent_progress_min: float = 0.435,
    drivable_area_min: float = 0.55,
    lane_min: float = 0.55,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the frozen NavHard incumbent and proposal indices online."""

    features = rule_features.float()
    base = normalized_base.float()
    proxy = protocol_proxy_zscore.float()
    finite = candidate_is_finite.bool()
    configs = configurations.to(base)
    if features.ndim != 3 or features.shape[-1] != 10:
        raise ValueError("NavHard gate rule features must be [batch,candidates,10]")
    if any(tensor.shape != features.shape[:2] for tensor in (base, proxy, finite)):
        raise ValueError("NavHard gate candidate input shapes differ")
    if configs.ndim != 2 or configs.shape[1:] != (5,):
        raise ValueError("NavHard gate configurations must be [proposals,5]")

    progress_thresholds = torch.cat(
        (
            configs.new_tensor((incumbent_progress_min,)),
            configs[:, 4],
        )
    )
    values = features[:, None, :, :]
    finite_expanded = finite[:, None, :]
    strict = (
        finite_expanded
        & (values[..., 1] >= float(drivable_area_min))
        & (values[..., 5] >= float(lane_min))
        & (values[..., 3] >= progress_thresholds[None, :, None])
    )
    violation = (
        torch.relu(float(drivable_area_min) - values[..., 1])
        / max(float(drivable_area_min), 0.1)
        + torch.relu(float(lane_min) - values[..., 5])
        / max(float(lane_min), 0.1)
        + torch.relu(progress_thresholds[None, :, None] - values[..., 3])
        / progress_thresholds.clamp_min(0.1)[None, :, None]
    )
    has_strict = strict.any(dim=2, keepdim=True)
    has_finite = finite_expanded.any(dim=2, keepdim=True)
    minimum = violation.masked_fill(~finite_expanded, torch.inf).amin(
        dim=2, keepdim=True
    )
    fallback = finite_expanded & (violation <= minimum)
    eligible = torch.where(
        has_strict,
        strict,
        torch.where(has_finite, fallback, torch.ones_like(strict)),
    )
    floor = torch.finfo(base.dtype).min
    incumbent = torch.where(
        eligible[:, 0], base + float(incumbent_proxy_weight) * proxy, floor
    ).argmax(dim=1)
    proposal_scores = base[:, None, :] + configs[None, :, 3:4] * proxy[:, None, :]
    proposals = torch.where(eligible[:, 1:], proposal_scores, floor).argmax(dim=2)
    return incumbent, proposals


class V2NavhardNonlinearGate(nn.Module):
    """Sealed shallow histogram ensemble exported from NAVTRAIN sklearn."""

    STATE_SCHEMA = "momworld-v2-navhard-nonlinear-gate-state-v1"

    def __init__(self, config: Any) -> None:
        super().__init__()
        path = Path(
            str(getattr(config, "rule_v2_navhard_nonlinear_gate_state_path", ""))
        ).expanduser()
        expected_sha256 = str(
            getattr(config, "rule_v2_navhard_nonlinear_gate_state_sha256", "")
        ).lower()
        if not path.is_file() or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
            raise ValueError("enabled V2 NavHard gate requires a state path and SHA256")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected_sha256:
            raise RuntimeError("V2 NavHard gate state SHA256 mismatch")
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict) or payload.get("schema") != self.STATE_SCHEMA:
            raise RuntimeError("invalid V2 NavHard nonlinear gate state")
        if int(payload.get("feature_dim", -1)) != 68:
            raise RuntimeError("invalid V2 NavHard gate feature dimension")
        self.min_predicted_gain = float(payload["min_predicted_gain"])
        self.max_predicted_collision_delta = float(
            payload["max_predicted_collision_delta"]
        )
        self.incumbent_proxy_weight = float(payload["incumbent_proxy_weight"])
        self.incumbent_progress_min = float(payload["incumbent_progress_min"])
        self.drivable_area_min = float(payload["drivable_area_min"])
        self.lane_min = float(payload["lane_min"])
        expected = (0.0005, 0.0, 3.2, 0.435, 0.55, 0.55)
        actual = (
            self.min_predicted_gain,
            self.max_predicted_collision_delta,
            self.incumbent_proxy_weight,
            self.incumbent_progress_min,
            self.drivable_area_min,
            self.lane_min,
        )
        if any(
            not math.isclose(value, reference, rel_tol=0.0, abs_tol=1e-12)
            for value, reference in zip(actual, expected)
        ):
            raise RuntimeError("unexpected V2 NavHard gate thresholds")
        configurations = torch.as_tensor(payload["configurations"], dtype=torch.float32)
        if configurations.shape != (13, 5) or not torch.isfinite(configurations).all():
            raise RuntimeError("invalid V2 NavHard gate configurations")
        self.register_buffer("configurations", configurations)
        for prefix in ("gain", "collision"):
            state = payload[f"{prefix}_model"]
            tree_count = int(state["tree_count"])
            max_nodes = int(state["max_nodes"])
            max_depth = int(state["max_depth"])
            if tree_count != 80 or max_nodes < 1 or max_depth != 3:
                raise RuntimeError(f"invalid V2 NavHard {prefix} ensemble shape")
            self.register_buffer(
                f"{prefix}_baseline", torch.tensor(float(state["baseline"]), dtype=torch.float64)
            )
            for name, dtype in (
                ("feature_idx", torch.long),
                ("threshold", torch.float64),
                ("missing_left", torch.bool),
                ("left", torch.long),
                ("right", torch.long),
                ("is_leaf", torch.bool),
                ("value", torch.float64),
            ):
                value = torch.as_tensor(state[name], dtype=dtype)
                if value.shape != (tree_count, max_nodes):
                    raise RuntimeError(f"invalid V2 NavHard {prefix} {name}")
                self.register_buffer(f"{prefix}_{name}", value)
            setattr(self, f"{prefix}_tree_count", tree_count)
            setattr(self, f"{prefix}_max_depth", max_depth)

    def _predict(self, features: torch.Tensor, prefix: str) -> torch.Tensor:
        shape = features.shape[:-1]
        values = features.reshape(-1, features.shape[-1]).double()
        tree_count = int(getattr(self, f"{prefix}_tree_count"))
        tree = torch.arange(tree_count, device=values.device)[None, :].expand(
            len(values), -1
        )
        nodes = torch.zeros_like(tree)
        for _ in range(int(getattr(self, f"{prefix}_max_depth")) + 1):
            leaf = getattr(self, f"{prefix}_is_leaf")[tree, nodes]
            feature = getattr(self, f"{prefix}_feature_idx")[tree, nodes]
            observed = values.gather(1, feature)
            missing_left = getattr(self, f"{prefix}_missing_left")[tree, nodes]
            threshold = getattr(self, f"{prefix}_threshold")[tree, nodes]
            go_left = torch.where(torch.isnan(observed), missing_left, observed <= threshold)
            left = getattr(self, f"{prefix}_left")[tree, nodes]
            right = getattr(self, f"{prefix}_right")[tree, nodes]
            next_nodes = torch.where(go_left, left, right)
            nodes = torch.where(leaf, nodes, next_nodes)
        prediction = getattr(self, f"{prefix}_baseline") + getattr(
            self, f"{prefix}_value"
        )[tree, nodes].sum(dim=1)
        return prediction.reshape(shape)

    def select(
        self,
        rule_features: torch.Tensor,
        normalized_base: torch.Tensor,
        protocol_proxy_zscore: torch.Tensor,
        learned_score: torch.Tensor,
        candidate_is_finite: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        incumbent, proposals = v2_navhard_candidate_selections(
            rule_features,
            normalized_base,
            protocol_proxy_zscore,
            candidate_is_finite,
            self.configurations,
            self.incumbent_proxy_weight,
            self.incumbent_progress_min,
            self.drivable_area_min,
            self.lane_min,
        )
        gate_features = v2_navtest_gate_features(
            rule_features,
            normalized_base,
            protocol_proxy_zscore,
            learned_score,
            candidate_is_finite,
            incumbent,
            proposals,
            self.configurations,
        )
        gain = self._predict(gate_features, "gain")
        collision = self._predict(gate_features, "collision")
        permitted = collision <= self.max_predicted_collision_delta
        score = torch.where(permitted, gain, torch.full_like(gain, -torch.inf))
        best = score.argmax(dim=1)
        batch = torch.arange(len(score), device=score.device)
        switch = permitted[batch, best] & (score[batch, best] >= self.min_predicted_gain)
        selected = torch.where(switch, proposals[batch, best], incumbent)
        return selected, gain.float(), collision.float()


def relu_guard_weights_from_config(config: Any) -> Tuple[float, ...]:
    """Read and validate the opt-in ReLU guard weights."""

    weights = tuple(float(getattr(config, name, 0.0)) for name in RELU_GUARD_WEIGHT_FIELDS)
    if any(not math.isfinite(value) or value < 0.0 for value in weights):
        raise ValueError("ReLU guard weights must be finite and non-negative")
    return weights


def candidate_relu_guard_penalty(
    rule_features: torch.Tensor,
    weights: Tuple[float, ...],
    epsilon: float = 1e-4,
) -> torch.Tensor:
    """Compose standardized, thresholded candidate safety violations.

    Thresholds are semantic constants frozen before NAVTRAIN calibration.  A
    zero weight omits that term entirely, preserving the old arithmetic path.
    """

    if rule_features.ndim != 3 or rule_features.shape[-1] < 10:
        raise ValueError("rule_features must have shape [batch, candidates, >=10]")
    if len(weights) != len(RELU_GUARD_WEIGHT_FIELDS):
        raise ValueError("Unexpected ReLU guard weight count")
    if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in weights):
        raise ValueError("ReLU guard weights must be finite and non-negative")
    features = rule_features.float()
    raw_terms = (
        torch.relu(features[..., 7] - 0.50),
        torch.relu(0.50 - features[..., 0]),
        torch.relu(0.50 - features[..., 1]),
        torch.relu(0.50 - features[..., 2]),
        torch.relu(0.20 - features[..., 3]),
        torch.relu(0.50 - features[..., 4]),
        torch.relu(0.50 - features[..., 5]),
        torch.relu(0.50 - features[..., 6]),
        torch.relu(features[..., 8]),
        torch.relu(features[..., 9] - 0.10),
    )
    penalty = features.new_zeros(features.shape[:2])
    for weight, raw in zip(weights, raw_terms):
        if float(weight) != 0.0:
            penalty = penalty + float(weight) * candidate_score_zscore(raw, epsilon)
    return penalty


def validate_rank_fusion_configuration(
    config: Any,
) -> Tuple[str, float, float]:
    """Validate opt-in fusion semantics before they can affect selection."""

    protocol_proxy_weight = float(
        getattr(config, "rule_protocol_proxy_weight", 0.0)
    )
    learned_zscore_weight = float(
        getattr(config, "rule_learned_zscore_weight", 0.0)
    )
    rank_fusion_mode = str(getattr(config, "rule_rank_fusion_mode", "hybrid"))
    if rank_fusion_mode not in ("hybrid", "proxy_only", "proxy_relu"):
        raise ValueError(f"Unsupported rank-fusion mode: {rank_fusion_mode!r}")
    if not (
        math.isfinite(protocol_proxy_weight)
        and protocol_proxy_weight >= 0.0
        and math.isfinite(learned_zscore_weight)
        and learned_zscore_weight >= 0.0
    ):
        raise ValueError("rank-fusion weights must be finite and non-negative")
    relu_weights = relu_guard_weights_from_config(config)
    if rank_fusion_mode in ("proxy_only", "proxy_relu"):
        proxy_only_values = {
            "rule_learned_score_weight": float(config.rule_learned_score_weight),
            "rule_collision_weight": float(config.rule_collision_weight),
            "rule_kinematic_weight": float(config.rule_kinematic_weight),
            "rule_momentum_weight": float(config.rule_momentum_weight),
            "rule_learned_zscore_weight": learned_zscore_weight,
        }
        if any(
            not math.isfinite(value) or value != 0.0
            for value in proxy_only_values.values()
        ) or not math.isclose(
            float(config.rule_collision_filter_threshold),
            1.000001,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"{rank_fusion_mode} requires zero learned/collision/kinematic/"
                "momentum weights, zero learned z-score weight, and collision "
                "threshold 1.000001"
            )
        if rank_fusion_mode == "proxy_only" and any(value != 0.0 for value in relu_weights):
            raise ValueError("proxy_only requires all ReLU guard weights to be zero")
        if rank_fusion_mode == "proxy_relu" and (
            protocol_proxy_weight <= 0.0 or not any(value > 0.0 for value in relu_weights)
        ):
            raise ValueError("proxy_relu requires a positive proxy and at least one ReLU guard")
    return rank_fusion_mode, protocol_proxy_weight, learned_zscore_weight


def validate_collision_calibrator_configuration(config: Any) -> Tuple[bool, float, float]:
    enabled = bool(getattr(config, "rule_collision_calibrator_enabled", False))
    weight = float(getattr(config, "rule_calibrated_collision_weight", 0.0))
    threshold = float(
        getattr(config, "rule_calibrated_collision_threshold", 1.000001)
    )
    if (
        not math.isfinite(weight)
        or weight < 0.0
        or not math.isfinite(threshold)
        or not 0.0 <= threshold <= 1.000001
    ):
        raise ValueError("Invalid calibrated-collision selection parameters")
    if not enabled and (
        weight != 0.0
        or not math.isclose(threshold, 1.000001, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise ValueError("Calibrated-collision parameters require an enabled artifact")
    if enabled and bool(getattr(config, "context_ranker_enabled", False)):
        raise ValueError("Collision calibrator and context ranker are mutually exclusive")
    if enabled and bool(getattr(config, "rule_safety_filter_enabled", False)):
        raise ValueError(
            "Collision calibrator and threshold safety filter require separate frozen evaluations"
        )
    return enabled, weight, threshold


def _world_sample_indices(
    horizon: int, num_samples: int, device: torch.device
) -> torch.Tensor:
    """Match the 0.5 s supervision points used by MomWorld target building."""
    if horizon < 1 or num_samples < 1:
        raise ValueError("horizon and num_samples must be positive")
    stride = max(horizon // num_samples, 1)
    return ((torch.arange(num_samples, device=device) + 1) * stride - 1).clamp(
        max=horizon - 1
    )


def candidate_collision_risk(
    candidates: torch.Tensor,
    agent_positions: torch.Tensor,
    agent_presence: torch.Tensor,
    longitudinal_radius: float,
    lateral_radius: float,
    temperature: float,
) -> torch.Tensor:
    """Estimate collision probability for each candidate from predicted agents.

    candidates has shape [B, K, H, 3], agent_positions [B, S, A, 2],
    and agent_presence [B, S, A].
    """
    if candidates.ndim != 4 or candidates.shape[-1] != 3:
        raise ValueError("candidates must have shape [B, K, H, 3]")
    if agent_positions.ndim != 4 or agent_positions.shape[-1] != 2:
        raise ValueError("agent_positions must have shape [B, S, A, 2]")
    if agent_presence.shape != agent_positions.shape[:-1]:
        raise ValueError("agent_presence shape must match agent positions")
    if candidates.shape[0] != agent_positions.shape[0]:
        raise ValueError("candidate and agent batch dimensions must match")

    sample_indices = _world_sample_indices(
        candidates.shape[2], agent_positions.shape[1], candidates.device
    )
    ego = candidates.index_select(2, sample_indices)
    relative = agent_positions[:, None] - ego[:, :, :, None, :2]
    heading = ego[:, :, :, None, 2]
    cosine = torch.cos(heading)
    sine = torch.sin(heading)
    longitudinal = cosine * relative[..., 0] + sine * relative[..., 1]
    lateral = -sine * relative[..., 0] + cosine * relative[..., 1]
    normalized_distance = (
        longitudinal / max(float(longitudinal_radius), 1e-3)
    ).square() + (
        lateral / max(float(lateral_radius), 1e-3)
    ).square()
    overlap_probability = torch.sigmoid(
        (1.0 - normalized_distance) / max(float(temperature), 1e-3)
    )
    overlap_probability = overlap_probability * agent_presence[:, None].clamp(0.0, 1.0)
    return overlap_probability.amax(dim=(-1, -2))


def candidate_kinematic_penalty(
    candidates: torch.Tensor, dt: float
) -> torch.Tensor:
    """Softly penalize physically implausible speed, acceleration, jerk and yaw."""
    positions = candidates[..., :2]
    delta = positions[:, :, 1:] - positions[:, :, :-1]
    speed = delta.norm(dim=-1) / max(float(dt), 1e-3)
    penalty = torch.relu(speed - 18.0).mean(dim=-1) / 18.0

    if speed.shape[-1] > 1:
        acceleration = (speed[:, :, 1:] - speed[:, :, :-1]).abs() / max(
            float(dt), 1e-3
        )
        penalty = penalty + torch.relu(acceleration - 6.0).mean(dim=-1) / 6.0
    if speed.shape[-1] > 2:
        jerk = (acceleration[:, :, 1:] - acceleration[:, :, :-1]).abs() / max(
            float(dt), 1e-3
        )
        penalty = penalty + torch.relu(jerk - 10.0).mean(dim=-1) / 10.0

    heading_delta = torch.atan2(
        torch.sin(candidates[:, :, 1:, 2] - candidates[:, :, :-1, 2]),
        torch.cos(candidates[:, :, 1:, 2] - candidates[:, :, :-1, 2]),
    ).abs()
    yaw_rate = heading_delta / max(float(dt), 1e-3)
    penalty = penalty + torch.relu(yaw_rate - 1.2).mean(dim=-1) / 1.2
    return penalty


def candidate_momentum_error(
    candidates: torch.Tensor,
    world_momentum_xy: torch.Tensor,
    dt: float,
    velocity_scale: float,
) -> torch.Tensor:
    """Measure disagreement between candidate motion and predicted world momentum."""
    delta = candidates[:, :, 1:, :2] - candidates[:, :, :-1, :2]
    velocity = delta / max(float(dt), 1e-3)
    velocity = torch.cat([velocity[:, :, :1], velocity], dim=2)
    target_velocity = world_momentum_xy[:, None] * float(velocity_scale)
    return (
        (velocity - target_velocity).norm(dim=-1).mean(dim=-1)
        / max(float(velocity_scale), 1e-3)
    )


class RuleAwareTrajectoryScorer(nn.Module):
    """Learn protocol-specific utility from decomposed scores and rule signals."""

    def __init__(self, config: MomWorldConfig):
        super().__init__()
        raw_input_dim = len(RULE_METRIC_NAMES) + 3
        self.input_mode = str(getattr(config, "rule_scorer_input_mode", "raw"))
        if self.input_mode == "raw":
            input_dim = raw_input_dim
        elif self.input_mode == "raw_plus_scene_zscore":
            input_dim = 2 * raw_input_dim
        else:
            raise ValueError(
                "rule_scorer_input_mode must be raw or raw_plus_scene_zscore"
            )
        hidden_dim = int(config.rule_scorer_hidden_dim)
        self.heads = nn.ModuleDict(
            {
                protocol: nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, 1),
                )
                for protocol in ("v1", "v2")
            }
        )
        for head in self.heads.values():
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def forward(self, features: torch.Tensor, protocol: str) -> torch.Tensor:
        if protocol not in self.heads:
            raise ValueError(f"Unsupported rule protocol: {protocol!r}")
        if features.ndim != 3:
            raise ValueError("Rule scorer features must be [batch, candidates, features]")
        if self.input_mode == "raw_plus_scene_zscore":
            mean = features.mean(dim=1, keepdim=True)
            std = features.std(dim=1, keepdim=True, unbiased=False).clamp(min=1e-4)
            features = torch.cat((features, (features - mean) / std), dim=-1)
        return self.heads[protocol](features).squeeze(-1)


class CandidateCollisionCalibrator(nn.Module):
    """Small sealed NAVTRAIN-only head for candidate collision probability."""

    INPUT_DIM = 2 * (len(RULE_METRIC_NAMES) + 4)

    def __init__(self, config: MomWorldConfig):
        super().__init__()
        hidden_dim = int(config.rule_collision_calibrator_hidden_dim)
        if hidden_dim < 1:
            raise ValueError("rule_collision_calibrator_hidden_dim must be positive")
        self.network = nn.Sequential(
            nn.Linear(self.INPUT_DIM, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.register_buffer("input_mean", torch.zeros(self.INPUT_DIM))
        self.register_buffer("input_std", torch.ones(self.INPUT_DIM))
        self.register_buffer("artifact_ready", torch.tensor(False, dtype=torch.bool))

    def require_artifact_ready(self) -> None:
        if self.artifact_ready.numel() != 1 or not bool(self.artifact_ready.item()):
            raise RuntimeError("Collision calibrator is enabled without a sealed artifact")
        if (
            self.input_mean.shape != (self.INPUT_DIM,)
            or self.input_std.shape != (self.INPUT_DIM,)
            or not torch.isfinite(self.input_mean).all()
            or not torch.isfinite(self.input_std).all()
            or not torch.all(self.input_std > 0.0)
        ):
            raise RuntimeError("Collision calibrator normalization is invalid")

    def forward(
        self, rule_features: torch.Tensor, normalized_base: torch.Tensor
    ) -> torch.Tensor:
        self.require_artifact_ready()
        if rule_features.ndim != 3 or rule_features.shape[-1] != 10:
            raise ValueError("rule_features must have shape [batch, candidates, 10]")
        if normalized_base.shape != rule_features.shape[:2]:
            raise ValueError("normalized_base shape must match candidate dimensions")
        raw = torch.cat(
            (rule_features.float(), normalized_base.float().unsqueeze(-1)), dim=-1
        )
        scene_mean = raw.mean(dim=1, keepdim=True)
        scene_std = raw.std(dim=1, keepdim=True, unbiased=False).clamp(min=1e-4)
        inputs = torch.cat((raw, (raw - scene_mean) / scene_std), dim=-1)
        normalized = (inputs - self.input_mean) / self.input_std
        logits = self.network(normalized).squeeze(-1)
        return torch.nan_to_num(logits, nan=0.0, posinf=30.0, neginf=-30.0)

class MomentumLatentWorldModel(nn.Module):
    """Roll out paired configuration and momentum states in latent space.

    The transition follows a damped, scene-adaptive second-order update:

    ``p[k+1] = retain[k] * p[k] + (1 - retain[k]) * impulse[k]``
    ``z[k+1] = z[k] + dt * p[k+1]``

    A high scene-change gate lowers ``retain`` so that stale momentum is reset
    after abrupt changes, while smooth scenes preserve the previous trend.
    """

    def __init__(self, config: MomWorldConfig):
        super().__init__()
        dim = config.tf_d_model
        hidden = config.tf_d_ffn
        self.future_steps = config.world_future_steps
        self.num_agents = config.num_bounding_boxes
        self.dt = config.world_dt
        self.residual_max = config.world_residual_max

        self.kinematic_encoder = nn.Sequential(
            nn.Linear(4, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
        )
        self.configuration_init = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
        )
        self.momentum_init = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.Tanh(),
            nn.LayerNorm(dim),
        )
        self.time_embedding = nn.Parameter(torch.zeros(self.future_steps, dim))
        nn.init.normal_(self.time_embedding, std=0.02)

        transition_dim = dim * 4
        self.persistence_head = nn.Linear(transition_dim, 1)
        nn.init.zeros_(self.persistence_head.weight)
        nn.init.constant_(
            self.persistence_head.bias,
            _logit(config.momentum_persistence_init),
        )
        self.scene_change_head = nn.Sequential(
            nn.Linear(transition_dim, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )
        self.impulse_head = nn.Sequential(
            nn.Linear(transition_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
            nn.Tanh(),
        )
        self.configuration_norm = nn.LayerNorm(dim)
        self.momentum_norm = nn.LayerNorm(dim)
        self.memory_projection = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
        )

        self.ego_state_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 6),
        )
        self.momentum_xy_head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 2),
        )
        self.agent_queries = nn.Parameter(torch.zeros(self.num_agents, dim))
        nn.init.normal_(self.agent_queries, std=0.02)
        self.agent_state_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 6),
        )
        self.agent_presence_head = nn.Linear(dim, 1)

        self.flow_state_encoder = nn.Sequential(
            nn.Linear(4, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
        )
        self.flow_time_encoder = nn.Sequential(
            nn.Linear(1, dim),
            nn.GELU(),
        )
        self.flow_velocity_head = nn.Sequential(
            nn.Linear(dim * 4, hidden),
            nn.GELU(),
            nn.Linear(hidden, 4),
        )
        nn.init.zeros_(self.flow_velocity_head[-1].weight)
        nn.init.zeros_(self.flow_velocity_head[-1].bias)

    def predict_flow_velocity(
        self,
        flow_state: torch.Tensor,
        flow_time: torch.Tensor,
        world_memory: torch.Tensor,
        world_momenta: torch.Tensor,
    ) -> torch.Tensor:
        """Predict a momentum-conditioned conditional flow field."""
        state_encoding = self.flow_state_encoder(flow_state)
        time_encoding = self.flow_time_encoder(flow_time)
        flow_input = torch.cat(
            [state_encoding, time_encoding, world_memory, world_momenta], dim=-1
        )
        return torch.tanh(self.flow_velocity_head(flow_input)) * self.residual_max

    def forward(
        self,
        previous_tokens: torch.Tensor,
        current_tokens: torch.Tensor,
        status_encoding: torch.Tensor,
        ego_kinematics: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        previous_scene = previous_tokens.mean(dim=1)
        current_scene = current_tokens.mean(dim=1)
        visual_delta = current_scene - previous_scene
        kinematic_context = self.kinematic_encoder(ego_kinematics)

        configuration = self.configuration_init(
            torch.cat([current_scene, status_encoding], dim=-1)
        )
        momentum = self.momentum_init(
            torch.cat([visual_delta, kinematic_context, status_encoding], dim=-1)
        )
        transition_context = current_scene + status_encoding + kinematic_context

        configurations: List[torch.Tensor] = []
        momenta: List[torch.Tensor] = []
        change_gates: List[torch.Tensor] = []
        persistence_gates: List[torch.Tensor] = []

        for step in range(self.future_steps):
            time_context = self.time_embedding[step].unsqueeze(0).expand_as(configuration)
            transition_input = torch.cat(
                [configuration, momentum, transition_context, time_context], dim=-1
            )
            persistence = torch.sigmoid(self.persistence_head(transition_input))
            change_gate = torch.sigmoid(self.scene_change_head(transition_input))
            retention = persistence * (1.0 - change_gate)
            impulse = self.impulse_head(transition_input)

            momentum = self.momentum_norm(
                retention * momentum + (1.0 - retention) * impulse
            )
            configuration = self.configuration_norm(
                configuration + self.dt * momentum
            )

            configurations.append(configuration)
            momenta.append(momentum)
            change_gates.append(change_gate)
            persistence_gates.append(persistence)

        configuration_rollout = torch.stack(configurations, dim=1)
        momentum_rollout = torch.stack(momenta, dim=1)
        change_gate = torch.stack(change_gates, dim=1)
        persistence_gate = torch.stack(persistence_gates, dim=1)

        world_memory = self.memory_projection(
            torch.cat([configuration_rollout, momentum_rollout], dim=-1)
        )
        ego_states = self.ego_state_head(configuration_rollout)
        momentum_xy = self.momentum_xy_head(momentum_rollout)

        agent_latents = (
            configuration_rollout.unsqueeze(2)
            + self.agent_queries.view(1, 1, self.num_agents, -1)
        )
        agent_states = self.agent_state_head(agent_latents)
        agent_presence_logits = self.agent_presence_head(agent_latents).squeeze(-1)

        return {
            "world_memory": world_memory,
            "world_configurations": configuration_rollout,
            "world_momenta": momentum_rollout,
            "world_change_gate": change_gate,
            "world_persistence_gate": persistence_gate,
            "world_ego_states": ego_states,
            "world_momentum_xy": momentum_xy,
            "world_agent_states": agent_states,
            "world_agent_presence_logits": agent_presence_logits,
        }


class MomWorldModel(HydraModel):
    """GTRS-Dense trajectory scorer augmented with a latent world rollout."""

    def __init__(self, config: MomWorldConfig):
        super().__init__(config)
        if config.world_flow_steps < 1:
            raise ValueError("world_flow_steps must be at least 1")
        self._config = config
        self._world_model = MomentumLatentWorldModel(config)
        self._world_residual_gate = nn.Parameter(
            torch.tensor(float(config.world_residual_gate_init))
        )
        self._rule_scorer = RuleAwareTrajectoryScorer(config)
        self._v1_protocol_gate = None
        self._v2_navtest_selector_gate = None
        self._v2_navhard_nonlinear_gate = None
        self._v2_navhard_pairwise_gate = None
        enabled_selector_count = sum(
            bool(getattr(config, name, False))
            for name in (
                "rule_v1_protocol_gate_enabled",
                "rule_v2_navtest_selector_gate_enabled",
                "rule_v2_navhard_nonlinear_gate_enabled",
                "rule_v2_navhard_pairwise_gate_enabled",
            )
        )
        if enabled_selector_count > 1:
            raise ValueError("V1, V2 NavTest, and V2 NavHard gates are mutually exclusive")
        if bool(getattr(config, "rule_v1_protocol_gate_enabled", False)):
            if not (
                str(config.rule_protocol) == "v1"
                and str(config.rule_rank_fusion_mode) == "proxy_only"
                and math.isclose(
                    float(config.rule_protocol_proxy_weight),
                    1.6,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and not bool(getattr(config, "rule_safety_filter_enabled", False))
                and not bool(
                    getattr(config, "rule_relative_safety_filter_enabled", False)
                )
                and not bool(
                    getattr(config, "rule_monotonic_residual_enabled", False)
                )
            ):
                raise ValueError(
                    "V1 protocol gate requires the unmodified 1.6 proxy_only incumbent"
                )
            self._v1_protocol_gate = V1ProtocolSelectorGate(config)
            self._v1_protocol_gate.eval()
        if bool(getattr(config, "rule_v2_navtest_selector_gate_enabled", False)):
            if not (
                str(config.rule_protocol) == "v2"
                and str(config.rule_rank_fusion_mode) == "hybrid"
                and math.isclose(float(config.rule_learned_score_weight), 1.0)
                and math.isclose(float(config.rule_collision_weight), 2.0)
                and math.isclose(float(config.rule_kinematic_weight), 0.0)
                and math.isclose(float(config.rule_momentum_weight), 0.1)
                and math.isclose(float(config.rule_protocol_proxy_weight), 0.0)
                and math.isclose(float(config.rule_learned_zscore_weight), 0.0)
                and bool(getattr(config, "rule_safety_filter_enabled", False))
                and math.isclose(float(config.rule_safety_ttc_min), 0.55)
                and math.isclose(float(config.rule_safety_lane_min), 0.25)
                and not bool(
                    getattr(config, "rule_relative_safety_filter_enabled", False)
                )
                and not bool(getattr(config, "rule_monotonic_residual_enabled", False))
                and not bool(getattr(config, "rule_collision_calibrator_enabled", False))
                and not bool(getattr(config, "context_ranker_enabled", False))
            ):
                raise ValueError(
                    "V2 NavTest selector gate requires the frozen GTRS+Rule incumbent"
                )
            self._v2_navtest_selector_gate = V2NavtestSelectorGate(config)
            self._v2_navtest_selector_gate.eval()
        if bool(getattr(config, "rule_v2_navhard_nonlinear_gate_enabled", False)):
            if not (
                str(config.rule_protocol) == "v2"
                and str(config.rule_rank_fusion_mode) == "proxy_only"
                and math.isclose(float(config.rule_learned_score_weight), 0.0)
                and math.isclose(float(config.rule_collision_weight), 0.0)
                and math.isclose(float(config.rule_kinematic_weight), 0.0)
                and math.isclose(float(config.rule_momentum_weight), 0.0)
                and math.isclose(float(config.rule_protocol_proxy_weight), 3.2)
                and math.isclose(float(config.rule_learned_zscore_weight), 0.0)
                and math.isclose(float(config.rule_collision_filter_threshold), 1.000001)
                and bool(getattr(config, "rule_safety_filter_enabled", False))
                and math.isclose(float(config.rule_safety_drivable_area_min), 0.55)
                and math.isclose(float(config.rule_safety_progress_min), 0.435)
                and math.isclose(float(config.rule_safety_lane_min), 0.55)
                and not bool(getattr(config, "rule_relative_safety_filter_enabled", False))
                and not bool(getattr(config, "rule_monotonic_residual_enabled", False))
                and not bool(getattr(config, "rule_collision_calibrator_enabled", False))
                and not bool(getattr(config, "context_ranker_enabled", False))
            ):
                raise ValueError(
                    "V2 NavHard nonlinear gate requires the frozen 3.2 proxy incumbent"
                )
            self._v2_navhard_nonlinear_gate = V2NavhardNonlinearGate(config)
            self._v2_navhard_nonlinear_gate.eval()
        if bool(getattr(config, "rule_v2_navhard_pairwise_gate_enabled", False)):
            if not (
                str(config.rule_protocol) == "v2"
                and str(config.rule_rank_fusion_mode) == "proxy_only"
                and math.isclose(float(config.rule_learned_score_weight), 0.0)
                and math.isclose(float(config.rule_collision_weight), 0.0)
                and math.isclose(float(config.rule_kinematic_weight), 0.0)
                and math.isclose(float(config.rule_momentum_weight), 0.0)
                and math.isclose(float(config.rule_protocol_proxy_weight), 3.2)
                and math.isclose(float(config.rule_learned_zscore_weight), 0.0)
                and math.isclose(float(config.rule_collision_filter_threshold), 1.000001)
                and bool(getattr(config, "rule_safety_filter_enabled", False))
                and math.isclose(float(config.rule_safety_drivable_area_min), 0.55)
                and math.isclose(float(config.rule_safety_progress_min), 0.435)
                and math.isclose(float(config.rule_safety_lane_min), 0.55)
                and not bool(getattr(config, "rule_relative_safety_filter_enabled", False))
                and not bool(getattr(config, "rule_monotonic_residual_enabled", False))
                and not bool(getattr(config, "rule_collision_calibrator_enabled", False))
                and not bool(getattr(config, "context_ranker_enabled", False))
            ):
                raise ValueError(
                    "V2 NavHard pairwise gate requires the frozen 3.2 proxy incumbent"
                )
            self._v2_navhard_pairwise_gate = V2NavhardPairwiseGate(config)
            self._v2_navhard_pairwise_gate.eval()
        self._collision_calibrator = None
        if bool(getattr(config, "rule_collision_calibrator_enabled", False)):
            self._collision_calibrator = CandidateCollisionCalibrator(config)
            for parameter in self._collision_calibrator.parameters():
                parameter.requires_grad = False
            self._collision_calibrator.eval()
        self._context_ranker = None
        if bool(getattr(config, "context_ranker_enabled", False)):
            context_config = context_ranker_config_from_runtime(config)
            self._context_ranker = ContextSetTrajectoryRanker(context_config)
            self._context_ranker.estimated_added_runtime_bytes(
                batch_size=4,
                candidates=int(config.rule_candidate_topk),
            )
            for parameter in self._context_ranker.parameters():
                parameter.requires_grad = False
            self._context_ranker.eval()
        if config.rule_finetune_only:
            for parameter in self.parameters():
                parameter.requires_grad = False
            for parameter in self._rule_scorer.parameters():
                parameter.requires_grad = True

    def train(self, mode: bool = True):
        """Keep the frozen planner deterministic during scorer-only fine-tuning."""
        super().train(mode)
        if self._config.rule_finetune_only:
            for child in self.children():
                child.eval()
            # Keeping the trajectory head in training mode preserves the
            # memory-saving vocabulary subsampling augmentation. Its weights
            # remain frozen and it has no mutable normalization statistics.
            self._trajectory_head.train(mode)
            self._rule_scorer.train(mode)
        context_ranker = getattr(self, "_context_ranker", None)
        if context_ranker is not None:
            context_ranker.eval()
        collision_calibrator = getattr(self, "_collision_calibrator", None)
        if collision_calibrator is not None:
            collision_calibrator.eval()
        v1_protocol_gate = getattr(self, "_v1_protocol_gate", None)
        if v1_protocol_gate is not None:
            v1_protocol_gate.eval()
        v2_navtest_selector_gate = getattr(self, "_v2_navtest_selector_gate", None)
        if v2_navtest_selector_gate is not None:
            v2_navtest_selector_gate.eval()
        v2_navhard_nonlinear_gate = getattr(self, "_v2_navhard_nonlinear_gate", None)
        if v2_navhard_nonlinear_gate is not None:
            v2_navhard_nonlinear_gate.eval()
        v2_navhard_pairwise_gate = getattr(self, "_v2_navhard_pairwise_gate", None)
        if v2_navhard_pairwise_gate is not None:
            v2_navhard_pairwise_gate.eval()
        return self


    @staticmethod
    def _camera_sequence(camera_feature: torch.Tensor) -> List[torch.Tensor]:
        if isinstance(camera_feature, (list, tuple)):
            sequence = [frame for frame in camera_feature if frame is not None]
            if not sequence:
                raise ValueError("camera_feature contains no valid temporal frames")
            return sequence
        return [camera_feature]

    def _encode_temporal_scene(
        self, features: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        status_feature: torch.Tensor = features["status_feature"][0]
        if self._config.num_ego_status == 1 and status_feature.shape[1] == 32:
            status_input = status_feature[:, :8]
        else:
            status_input = status_feature
        status_encoding = self._status_encoding(status_input)

        camera_sequence = self._camera_sequence(features["camera_feature"])
        current_tokens = self.img_feat_blc(camera_sequence[-1])
        if len(camera_sequence) > 1:
            previous_tokens = self.img_feat_blc(camera_sequence[-2])
        else:
            previous_tokens = current_tokens.detach()

        positional_tokens = self._keyval_embedding.weight[None, ...]
        current_tokens = current_tokens + positional_tokens
        previous_tokens = previous_tokens + positional_tokens

        if "ego_velocity" in features and "ego_acceleration" in features:
            ego_velocity = torch.as_tensor(
                features["ego_velocity"], device=current_tokens.device, dtype=current_tokens.dtype
            )
            ego_acceleration = torch.as_tensor(
                features["ego_acceleration"],
                device=current_tokens.device,
                dtype=current_tokens.dtype,
            )
            ego_kinematics = torch.cat([ego_velocity, ego_acceleration], dim=-1)
        else:
            ego_kinematics = status_input[:, -4:]

        world_output = self._world_model(
            previous_tokens=previous_tokens,
            current_tokens=current_tokens,
            status_encoding=status_encoding,
            ego_kinematics=ego_kinematics,
        )
        memory = torch.cat([current_tokens, world_output["world_memory"]], dim=1)
        return memory, status_encoding, current_tokens, world_output

    @staticmethod
    def _gather_candidates(
        candidate_vocab: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        if candidate_vocab.ndim == 3:
            return candidate_vocab[indices]
        if candidate_vocab.ndim == 4:
            batch_indices = torch.arange(
                indices.shape[0], device=indices.device
            )[:, None]
            return candidate_vocab[batch_indices, indices]
        raise ValueError("candidate vocabulary must have shape [N,H,3] or [B,N,H,3]")

    def _refine_candidate_trajectories(
        self,
        base_candidates: torch.Tensor,
        world_output: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply momentum flow editing to every retained candidate."""
        batch_size, num_candidates, horizon, _ = base_candidates.shape
        if world_output["world_memory"].shape[1] != horizon:
            raise ValueError(
                "world_future_steps must match the trajectory vocabulary horizon"
            )

        flow_state = _trajectory_to_flow_state(
            base_candidates, self._config.world_position_scale
        ).reshape(batch_size * num_candidates, horizon, 4)
        world_memory = (
            world_output["world_memory"][:, None]
            .expand(-1, num_candidates, -1, -1)
            .reshape(batch_size * num_candidates, horizon, -1)
        )
        world_momenta = (
            world_output["world_momenta"][:, None]
            .expand(-1, num_candidates, -1, -1)
            .reshape(batch_size * num_candidates, horizon, -1)
        )

        for step in range(self._config.world_flow_steps):
            flow_time = flow_state.new_full(
                (batch_size * num_candidates, horizon, 1),
                (step + 0.5) / self._config.world_flow_steps,
            )
            velocity = self._world_model.predict_flow_velocity(
                flow_state=flow_state,
                flow_time=flow_time,
                world_memory=world_memory,
                world_momenta=world_momenta,
            )
            flow_state = flow_state + velocity / self._config.world_flow_steps
            heading_vector = F.normalize(flow_state[..., 2:4], dim=-1, eps=1e-6)
            flow_state = torch.cat([flow_state[..., :2], heading_vector], dim=-1)

        raw_refined = _flow_state_to_trajectory(
            flow_state, self._config.world_position_scale
        ).reshape_as(base_candidates)
        heading_residual = torch.atan2(
            torch.sin(raw_refined[..., 2:3] - base_candidates[..., 2:3]),
            torch.cos(raw_refined[..., 2:3] - base_candidates[..., 2:3]),
        )
        residual = torch.cat(
            [
                raw_refined[..., :2] - base_candidates[..., :2],
                heading_residual,
            ],
            dim=-1,
        ).clamp(-self._config.world_residual_max, self._config.world_residual_max)
        time_gate = torch.linspace(
            0.05,
            1.0,
            horizon,
            device=residual.device,
            dtype=residual.dtype,
        ).view(1, 1, horizon, 1)
        residual_gate = torch.sigmoid(self._world_residual_gate)
        refined = base_candidates + residual_gate * time_gate * residual
        return refined, residual

    def _apply_rule_aware_selection(
        self,
        trajectory_output: Dict[str, torch.Tensor],
        world_output: Dict[str, torch.Tensor],
    ) -> None:
        """Edit top candidates, remove predicted collisions, then rerank."""
        if "selection_scores" not in trajectory_output:
            self._apply_world_residual(trajectory_output, world_output)
            return

        if "trajectory_vocab_dropout" in trajectory_output:
            candidate_vocab = trajectory_output["trajectory_vocab_dropout"]
        else:
            candidate_vocab = trajectory_output["trajectory_vocab"]
        selection_scores = torch.nan_to_num(
            trajectory_output["selection_scores"],
            nan=-1e9,
            posinf=1e9,
            neginf=-1e9,
        )
        topk = min(int(self._config.rule_candidate_topk), selection_scores.shape[1])
        top_scores, top_indices = torch.topk(selection_scores, k=topk, dim=1)
        base_candidates = self._gather_candidates(candidate_vocab, top_indices)
        refined_candidates, residuals = self._refine_candidate_trajectories(
            base_candidates, world_output
        )
        base_is_finite = torch.isfinite(base_candidates).flatten(
            start_dim=2
        ).all(dim=-1)
        refined_is_finite = torch.isfinite(refined_candidates).flatten(
            start_dim=2
        ).all(dim=-1)
        refined_candidates = torch.where(
            refined_is_finite[..., None, None], refined_candidates, base_candidates
        )
        residuals = torch.where(
            refined_is_finite[..., None, None], residuals, torch.zeros_like(residuals)
        )
        candidate_is_finite = base_is_finite & torch.isfinite(
            refined_candidates
        ).flatten(start_dim=2).all(dim=-1)
        refined_candidates = torch.nan_to_num(refined_candidates)

        sample_indices = _world_sample_indices(
            refined_candidates.shape[2],
            int(self._config.world_agent_steps),
            refined_candidates.device,
        )
        agent_positions = (
            world_output["world_agent_states"]
            .index_select(1, sample_indices)[..., :2]
            * float(self._config.world_position_scale)
        )
        agent_presence = torch.sigmoid(
            world_output["world_agent_presence_logits"].index_select(
                1, sample_indices
            )
        )
        collision_risk = candidate_collision_risk(
            refined_candidates,
            torch.nan_to_num(agent_positions),
            torch.nan_to_num(agent_presence),
            self._config.rule_collision_longitudinal_radius,
            self._config.rule_collision_lateral_radius,
            self._config.rule_collision_temperature,
        )
        kinematic_penalty = candidate_kinematic_penalty(
            refined_candidates, self._config.world_dt
        )
        momentum_error = candidate_momentum_error(
            refined_candidates,
            torch.nan_to_num(world_output["world_momentum_xy"]),
            self._config.world_dt,
            self._config.world_velocity_scale,
        )

        metric_features = [
            torch.nan_to_num(
                trajectory_output[name].gather(1, top_indices)
            ).sigmoid()
            for name in RULE_METRIC_NAMES
        ]
        rule_features = torch.stack(metric_features, dim=-1)
        rule_features = torch.cat(
            [
                rule_features,
                collision_risk.unsqueeze(-1),
                kinematic_penalty.unsqueeze(-1),
                momentum_error.unsqueeze(-1),
            ],
            dim=-1,
        )
        context_ranker = getattr(self, "_context_ranker", None)
        if context_ranker is None:
            v1_logits = self._rule_scorer(rule_features, "v1")
            v2_logits = self._rule_scorer(rule_features, "v2")
        else:
            # The sealed set-wise selector is independent of the first-round
            # scorer.  Do not even execute that module when context ranking is
            # enabled; zero placeholders preserve the diagnostic output keys.
            v1_logits = rule_features.new_zeros(rule_features.shape[:2])
            v2_logits = rule_features.new_zeros(rule_features.shape[:2])
        protocol = str(self._config.rule_protocol)
        learned_logits = v1_logits if protocol == "v1" else v2_logits
        protocol_proxy = candidate_protocol_proxy(rule_features, protocol)
        protocol_proxy_zscore = candidate_score_zscore(
            protocol_proxy,
            float(getattr(self._config, "rule_score_zscore_epsilon", 1e-4)),
        )
        if context_ranker is None:
            (
                _rank_fusion_mode,
                protocol_proxy_weight,
                learned_zscore_weight,
            ) = validate_rank_fusion_configuration(self._config)
            relu_guard_weights = relu_guard_weights_from_config(self._config)
            # The offline tuner loads the frozen scorer on CPU in float32. AMP
            # logits are not bitwise-equivalent, so recompute only the selected
            # head in float32 when that term is active.
            learned_logits_for_zscore = learned_logits
            if learned_zscore_weight != 0.0:
                with torch.autocast(
                    device_type=rule_features.device.type, enabled=False
                ):
                    learned_logits_for_zscore = self._rule_scorer(
                        rule_features.float(), protocol
                    )
            learned_logit_zscore = candidate_score_zscore(
                learned_logits_for_zscore,
                float(getattr(self._config, "rule_score_zscore_epsilon", 1e-4)),
            )
        else:
            protocol_proxy_weight = 0.0
            learned_zscore_weight = 0.0
            learned_logit_zscore = torch.zeros_like(protocol_proxy_zscore)
            relu_guard_weights = (0.0,) * len(RELU_GUARD_WEIGHT_FIELDS)

        base_mean = top_scores.mean(dim=1, keepdim=True)
        base_std = top_scores.std(dim=1, keepdim=True, unbiased=False).clamp(min=1e-4)
        normalized_base = (top_scores - base_mean) / base_std
        (
            collision_calibrator_enabled,
            calibrated_collision_weight,
            calibrated_collision_threshold,
        ) = validate_collision_calibrator_configuration(self._config)
        calibrated_collision_logits = None
        calibrated_collision_risk = None
        if collision_calibrator_enabled:
            collision_calibrator = self._collision_calibrator
            if collision_calibrator is None:
                raise RuntimeError("Enabled collision calibrator module is missing")
            with torch.autocast(device_type=rule_features.device.type, enabled=False):
                calibrated_collision_logits = collision_calibrator(
                    rule_features.float(), normalized_base.float()
                )
                calibrated_collision_risk = torch.sigmoid(
                    calibrated_collision_logits
                )
        context_ranker_logits = None
        context_ranker_collision_risk = None
        if context_ranker is not None:
            context_ranker.require_artifact_ready()
            context_ranker.estimated_added_runtime_bytes(
                batch_size=int(rule_features.shape[0]),
                candidates=int(rule_features.shape[1]),
            )
            context_selection = context_ranker_selection_from_runtime(self._config)
            with torch.autocast(
                device_type=rule_features.device.type, enabled=False
            ):
                (
                    context_ranker_logits,
                    context_ranker_collision_logits,
                ) = context_ranker.forward_with_collision(
                    rule_features.float(),
                    normalized_base.float(),
                    candidate_is_finite,
                    protocol,
                )
                context_ranker_collision_risk = torch.sigmoid(
                    context_ranker_collision_logits
                )
            combined_scores, eligible_mask = context_ranker_selection_scores(
                context_ranker_logits,
                normalized_base,
                collision_risk,
                candidate_is_finite,
                context_selection["base_weight"],
                context_selection["ranker_weight"],
                context_selection["collision_threshold"],
                learned_collision=context_ranker_collision_risk,
                collision_weight=context_selection["collision_weight"],
                learned_collision_threshold=context_selection[
                    "learned_collision_threshold"
                ],
                fallback_risk_slack=context_selection["fallback_risk_slack"],
                epsilon=float(self._config.context_ranker_zscore_epsilon),
            )
            safe_mask = (
                (collision_risk < context_selection["collision_threshold"])
                & (
                    context_ranker_collision_risk
                    < context_selection["learned_collision_threshold"]
                )
                & candidate_is_finite
            )
        else:
            raw_learned_weight = float(self._config.rule_learned_score_weight)
            collision_weight = float(self._config.rule_collision_weight)
            kinematic_weight = float(self._config.rule_kinematic_weight)
            momentum_weight = float(self._config.rule_momentum_weight)
            combined_scores = normalized_base
            # Zero-weight signals are omitted, rather than multiplied by zero,
            # so proxy_only cannot be affected by an unused signal's dtype or
            # non-finite value.
            if raw_learned_weight != 0.0:
                combined_scores = combined_scores + raw_learned_weight * torch.sigmoid(
                    learned_logits
                )
            if collision_weight != 0.0:
                combined_scores = combined_scores - collision_weight * collision_risk
            if kinematic_weight != 0.0:
                combined_scores = combined_scores - kinematic_weight * kinematic_penalty
            if momentum_weight != 0.0:
                combined_scores = combined_scores - momentum_weight * momentum_error
            if protocol_proxy_weight != 0.0:
                combined_scores = (
                    combined_scores + protocol_proxy_weight * protocol_proxy_zscore
                )
            if learned_zscore_weight != 0.0:
                combined_scores = (
                    combined_scores + learned_zscore_weight * learned_logit_zscore
                )
            if any(value != 0.0 for value in relu_guard_weights):
                combined_scores = combined_scores - candidate_relu_guard_penalty(
                    rule_features,
                    relu_guard_weights,
                    float(getattr(self._config, "rule_score_zscore_epsilon", 1e-4)),
                )
            if calibrated_collision_risk is not None and calibrated_collision_weight != 0.0:
                combined_scores = combined_scores - calibrated_collision_weight * (
                    candidate_score_zscore(
                        calibrated_collision_risk,
                        float(getattr(self._config, "rule_score_zscore_epsilon", 1e-4)),
                    )
                )
            monotonic_weights = monotonic_residual_weights_from_config(self._config)
            if bool(
                getattr(self._config, "rule_monotonic_residual_enabled", False)
            ):
                combined_scores = combined_scores + candidate_monotonic_residual(
                    rule_features,
                    monotonic_weights,
                    float(
                        getattr(
                            self._config, "rule_monotonic_zscore_epsilon", 1e-4
                        )
                    ),
                )
            combined_scores = torch.nan_to_num(
                combined_scores, nan=-1e9, posinf=1e9, neginf=-1e9
            )
            if bool(getattr(self._config, "rule_safety_filter_enabled", False)):
                eligible_mask, safe_mask, safety_violation = (
                    candidate_rule_safety_filter(
                        rule_features, candidate_is_finite, self._config
                    )
                )
                trajectory_output["rule_safety_violation"] = safety_violation
                trajectory_output["rule_safety_eligible_mask"] = eligible_mask
            else:
                safe_mask = (
                    collision_risk
                    < float(self._config.rule_collision_filter_threshold)
                ) & candidate_is_finite
                has_safe_candidate = safe_mask.any(dim=1, keepdim=True)
                has_finite_candidate = candidate_is_finite.any(dim=1, keepdim=True)
                eligible_mask = torch.where(
                    has_safe_candidate,
                    safe_mask,
                    torch.where(
                        has_finite_candidate,
                        candidate_is_finite,
                        torch.ones_like(candidate_is_finite),
                    ),
                )
            if calibrated_collision_risk is not None:
                eligible_mask, calibrated_safe = (
                    candidate_calibrated_collision_filter(
                        calibrated_collision_risk,
                        candidate_is_finite,
                        calibrated_collision_threshold,
                    )
                )
                safe_mask = calibrated_safe
            combined_scores = combined_scores.masked_fill(
                ~eligible_mask, torch.finfo(combined_scores.dtype).min
            )
        v1_protocol_gate = getattr(self, "_v1_protocol_gate", None)
        v2_navtest_selector_gate = getattr(
            self, "_v2_navtest_selector_gate", None
        )
        v2_navhard_nonlinear_gate = getattr(
            self, "_v2_navhard_nonlinear_gate", None
        )
        v2_navhard_pairwise_gate = getattr(
            self, "_v2_navhard_pairwise_gate", None
        )
        v1_protocol_gate_gain = None
        v1_protocol_gate_collision = None
        v2_navtest_gate_gain = None
        v2_navtest_gate_collision = None
        v2_navhard_gate_gain = None
        v2_navhard_gate_collision = None
        v2_navhard_pairwise_gain = None
        v2_navhard_pairwise_incumbent_topk = None
        v2_navhard_pairwise_alternative_topk = None
        if (
            v1_protocol_gate is None
            and v2_navtest_selector_gate is None
            and v2_navhard_nonlinear_gate is None
            and v2_navhard_pairwise_gate is None
        ):
            selected_topk = combined_scores.argmax(dim=1)
        elif v1_protocol_gate is not None:
            if context_ranker is not None:
                raise RuntimeError("V1 protocol gate cannot be combined with context ranking")
            (
                selected_topk,
                v1_protocol_gate_gain,
                v1_protocol_gate_collision,
            ) = v1_protocol_gate.select(
                rule_features,
                normalized_base,
                protocol_proxy_zscore,
                candidate_is_finite,
            )
        elif v2_navtest_selector_gate is not None:
            if context_ranker is not None:
                raise RuntimeError(
                    "V2 NavTest selector gate cannot be combined with context ranking"
                )
            (
                selected_topk,
                v2_navtest_gate_gain,
                v2_navtest_gate_collision,
            ) = v2_navtest_selector_gate.select(
                rule_features,
                normalized_base,
                protocol_proxy_zscore,
                torch.sigmoid(v2_logits),
                candidate_is_finite,
                eligible_mask,
            )
        elif v2_navhard_pairwise_gate is not None:
            if context_ranker is not None:
                raise RuntimeError(
                    "V2 NavHard pairwise gate cannot be combined with context ranking"
                )
            (
                selected_topk,
                v2_navhard_pairwise_gain,
                v2_navhard_pairwise_incumbent_topk,
                v2_navhard_pairwise_alternative_topk,
            ) = (
                v2_navhard_pairwise_gate.select(
                    rule_features,
                    normalized_base,
                    protocol_proxy_zscore,
                    candidate_is_finite,
                )
            )
        else:
            if context_ranker is not None:
                raise RuntimeError(
                    "V2 NavHard nonlinear gate cannot be combined with context ranking"
                )
            (
                selected_topk,
                v2_navhard_gate_gain,
                v2_navhard_gate_collision,
            ) = v2_navhard_nonlinear_gate.select(
                rule_features,
                normalized_base,
                protocol_proxy_zscore,
                torch.sigmoid(v2_logits),
                candidate_is_finite,
            )
        batch_indices = torch.arange(
            refined_candidates.shape[0], device=refined_candidates.device
        )
        selected_local_indices = top_indices[batch_indices, selected_topk]
        base_trajectory = base_candidates[batch_indices, selected_topk]

        dropout_indices = trajectory_output.get("dropout_indices")
        if dropout_indices is not None and dropout_indices.ndim == 1:
            rule_vocab_indices = dropout_indices[top_indices]
        else:
            rule_vocab_indices = top_indices

        trajectory_output["selected_indices"] = selected_local_indices
        trajectory_output["base_trajectory"] = base_trajectory
        trajectory_output["trajectory_residual"] = residuals[
            batch_indices, selected_topk
        ]
        trajectory_output["trajectory"] = refined_candidates[
            batch_indices, selected_topk
        ]
        trajectory_output["world_residual_gate"] = torch.sigmoid(
            self._world_residual_gate
        )
        trajectory_output["rule_candidate_trajectories"] = refined_candidates
        trajectory_output["rule_vocab_indices"] = rule_vocab_indices
        trajectory_output["rule_collision_risk"] = collision_risk
        trajectory_output["rule_kinematic_penalty"] = kinematic_penalty
        trajectory_output["rule_momentum_error"] = momentum_error
        trajectory_output["rule_features"] = rule_features
        # Expose the two inputs that cannot be reconstructed from rule_features.
        # The offline rule tuner caches these values so that validation-time
        # selection exactly mirrors the inference-time fallback/masking logic.
        trajectory_output["rule_normalized_base"] = normalized_base
        trajectory_output["rule_candidate_is_finite"] = candidate_is_finite
        trajectory_output["rule_v1_logits"] = v1_logits
        trajectory_output["rule_v2_logits"] = v2_logits
        if calibrated_collision_logits is not None:
            trajectory_output["rule_calibrated_collision_logits"] = (
                calibrated_collision_logits
            )
            trajectory_output["rule_calibrated_collision_risk"] = (
                calibrated_collision_risk
            )
        trajectory_output["rule_protocol_proxy"] = protocol_proxy
        trajectory_output["rule_protocol_proxy_zscore"] = protocol_proxy_zscore
        trajectory_output["rule_learned_logit_zscore"] = learned_logit_zscore
        if bool(getattr(self._config, "rule_monotonic_residual_enabled", False)):
            trajectory_output["rule_monotonic_residual"] = candidate_monotonic_residual(
                rule_features,
                monotonic_residual_weights_from_config(self._config),
                float(getattr(self._config, "rule_monotonic_zscore_epsilon", 1e-4)),
            )
        trajectory_output["rule_combined_scores"] = combined_scores
        trajectory_output["rule_selected_topk"] = selected_topk
        if v1_protocol_gate_gain is not None:
            trajectory_output["rule_v1_protocol_gate_gain"] = v1_protocol_gate_gain
            trajectory_output["rule_v1_protocol_gate_collision_delta"] = (
                v1_protocol_gate_collision
            )
            trajectory_output["rule_v1_protocol_gate_selected_topk"] = selected_topk
        if v2_navtest_gate_gain is not None:
            trajectory_output["rule_v2_navtest_gate_gain"] = v2_navtest_gate_gain
            trajectory_output["rule_v2_navtest_gate_collision_delta"] = (
                v2_navtest_gate_collision
            )
            trajectory_output["rule_v2_navtest_gate_selected_topk"] = selected_topk
        if v2_navhard_gate_gain is not None:
            trajectory_output["rule_v2_navhard_gate_gain"] = v2_navhard_gate_gain
            trajectory_output["rule_v2_navhard_gate_collision_delta"] = (
                v2_navhard_gate_collision
            )
            trajectory_output["rule_v2_navhard_gate_selected_topk"] = selected_topk
        if v2_navhard_pairwise_gain is not None:
            trajectory_output["rule_v2_navhard_pairwise_gain"] = (
                v2_navhard_pairwise_gain
            )
            trajectory_output["rule_v2_navhard_pairwise_selected_topk"] = selected_topk
            trajectory_output["rule_v2_navhard_pairwise_incumbent_topk"] = (
                v2_navhard_pairwise_incumbent_topk
            )
            trajectory_output["rule_v2_navhard_pairwise_alternative_topk"] = (
                v2_navhard_pairwise_alternative_topk
            )
        trajectory_output["rule_num_filtered"] = (~safe_mask).sum(dim=1)
        if context_ranker_logits is not None:
            trajectory_output["context_ranker_logits"] = context_ranker_logits
            trajectory_output["context_ranker_collision_risk"] = (
                context_ranker_collision_risk
            )
            trajectory_output["context_ranker_eligible_mask"] = eligible_mask
            trajectory_output["context_ranker_selected_topk"] = selected_topk

    def _apply_world_residual(
        self,
        trajectory_output: Dict[str, torch.Tensor],
        world_output: Dict[str, torch.Tensor],
    ) -> None:
        if "selected_indices" in trajectory_output:
            selected_indices = trajectory_output["selected_indices"]
            batch_indices = torch.arange(
                selected_indices.shape[0], device=selected_indices.device
            )
            candidate_vocab = trajectory_output.get(
                "trajectory_vocab_dropout", trajectory_output["trajectory_vocab"]
            )
            if candidate_vocab.ndim == 3:
                base_trajectory = candidate_vocab[selected_indices]
            else:
                base_trajectory = candidate_vocab[batch_indices, selected_indices]
        else:
            base_trajectory = trajectory_output["trajectory"]

        if world_output["world_memory"].shape[1] != base_trajectory.shape[1]:
            raise ValueError(
                "world_future_steps must match the trajectory vocabulary horizon: "
                f"{world_output['world_memory'].shape[1]} != {base_trajectory.shape[1]}"
            )

        flow_state = _trajectory_to_flow_state(
            base_trajectory, self._config.world_position_scale
        )
        batch_size, horizon, _ = flow_state.shape
        for step in range(self._config.world_flow_steps):
            flow_time = flow_state.new_full(
                (batch_size, horizon, 1),
                (step + 0.5) / self._config.world_flow_steps,
            )
            velocity = self._world_model.predict_flow_velocity(
                flow_state=flow_state,
                flow_time=flow_time,
                world_memory=world_output["world_memory"],
                world_momenta=world_output["world_momenta"],
            )
            flow_state = flow_state + velocity / self._config.world_flow_steps
            heading_vector = F.normalize(flow_state[..., 2:4], dim=-1, eps=1e-6)
            flow_state = torch.cat([flow_state[..., :2], heading_vector], dim=-1)

        refined_trajectory = _flow_state_to_trajectory(
            flow_state, self._config.world_position_scale
        )
        heading_residual = torch.atan2(
            torch.sin(refined_trajectory[..., 2:3] - base_trajectory[..., 2:3]),
            torch.cos(refined_trajectory[..., 2:3] - base_trajectory[..., 2:3]),
        )
        residual = torch.cat(
            [
                refined_trajectory[..., :2] - base_trajectory[..., :2],
                heading_residual,
            ],
            dim=-1,
        ).clamp(-self._config.world_residual_max, self._config.world_residual_max)
        time_gate = torch.linspace(
            0.05,
            1.0,
            residual.shape[1],
            device=residual.device,
            dtype=residual.dtype,
        ).view(1, -1, 1)
        residual_gate = torch.sigmoid(self._world_residual_gate)
        trajectory_output["base_trajectory"] = base_trajectory
        trajectory_output["trajectory_residual"] = residual
        trajectory_output["trajectory"] = (
            base_trajectory + residual_gate * time_gate * residual
        )
        trajectory_output["world_residual_gate"] = residual_gate

    def flow_matching_outputs(
        self,
        base_trajectory: torch.Tensor,
        target_trajectory: torch.Tensor,
        world_memory: torch.Tensor,
        world_momenta: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Build a stochastic conditional-flow matching training pair."""
        base_state = _trajectory_to_flow_state(
            base_trajectory, self._config.world_position_scale
        )
        target_state = _trajectory_to_flow_state(
            target_trajectory, self._config.world_position_scale
        )
        flow_time = torch.rand(
            (base_state.shape[0], 1, 1),
            device=base_state.device,
            dtype=base_state.dtype,
        ).expand(-1, base_state.shape[1], -1)
        interpolated_state = (1.0 - flow_time) * base_state + flow_time * target_state
        flow_target = target_state - base_state
        flow_prediction = self._world_model.predict_flow_velocity(
            flow_state=interpolated_state,
            flow_time=flow_time,
            world_memory=world_memory,
            world_momenta=world_momenta,
        )
        return {
            "world_flow_velocity": flow_prediction,
            "world_flow_target": flow_target,
            "world_flow_time": flow_time,
        }

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        interpolated_traj=None,
    ) -> Dict[str, torch.Tensor]:
        if self._config.rule_finetune_only and self.training:
            with torch.no_grad():
                memory, status_encoding, _, world_output = self._encode_temporal_scene(
                    features
                )
                output = self._trajectory_head(
                    memory, status_encoding, interpolated_traj
                )
        else:
            memory, status_encoding, _, world_output = self._encode_temporal_scene(
                features
            )
            output = self._trajectory_head(memory, status_encoding, interpolated_traj)
        if self._config.rule_enabled:
            self._apply_rule_aware_selection(output, world_output)
        else:
            self._apply_world_residual(output, world_output)
        output.update(world_output)
        return output

    def evaluate_dp_proposals(
        self,
        features: Dict[str, torch.Tensor],
        dp_proposals: torch.Tensor,
        topk: int = 10,
        dp_only_inference: bool = False,
    ) -> Dict[str, torch.Tensor]:
        memory, status_encoding, _, world_output = self._encode_temporal_scene(features)
        output = self._trajectory_head.eval_dp_proposals(
            memory,
            status_encoding,
            dp_proposals,
            topk=topk,
            dp_only_inference=dp_only_inference,
        )
        if self._config.rule_enabled:
            self._apply_rule_aware_selection(output, world_output)
        else:
            self._apply_world_residual(output, world_output)
        output.update(world_output)
        return output
