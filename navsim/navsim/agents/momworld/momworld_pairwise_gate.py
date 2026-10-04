"""Frozen conservative pairwise selector for the V2 NavHard incumbent.

The number of eligible alternatives is sealed in the checksummed NAVTRAIN
artifact.  No evaluator information is consumed during inference.
"""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Any, Tuple

import torch
import torch.nn as nn


def _gather(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    rows = torch.arange(len(values), device=values.device)[:, None]
    return values[rows, indices]


def _pair_features(
    rule_features: torch.Tensor,
    normalized_base: torch.Tensor,
    proxy_zscore: torch.Tensor,
    combined_score: torch.Tensor,
    incumbent: torch.Tensor,
    alternatives: torch.Tensor,
) -> torch.Tensor:
    """Mirror the 59-D NAVTRAIN pair representation exactly."""

    values = torch.nan_to_num(
        rule_features.float(), nan=0.0, posinf=0.0, neginf=0.0
    )
    rows = torch.arange(len(values), device=values.device)[:, None]
    incumbent_column = incumbent[:, None]
    incumbent_features = values[rows, incumbent_column].expand(
        -1, alternatives.shape[1], -1
    )
    alternative_features = values[rows, alternatives]
    scene_mean = values.mean(dim=1, keepdim=True).expand_as(alternative_features)
    scene_std = values.std(dim=1, keepdim=True, unbiased=False).expand_as(
        alternative_features
    )

    def scalar_triplet(scores: torch.Tensor) -> torch.Tensor:
        incumbent_value = _gather(scores.float(), incumbent_column).expand(
            -1, alternatives.shape[1]
        )
        alternative_value = _gather(scores.float(), alternatives)
        return torch.stack(
            (
                incumbent_value,
                alternative_value,
                alternative_value - incumbent_value,
            ),
            dim=-1,
        )

    return torch.cat(
        (
            incumbent_features,
            alternative_features,
            alternative_features - incumbent_features,
            scene_mean,
            scene_std,
            scalar_triplet(normalized_base),
            scalar_triplet(proxy_zscore),
            scalar_triplet(combined_score),
        ),
        dim=-1,
    )


class V2NavhardPairwiseGate(nn.Module):
    """Sealed NAVTRAIN histogram regressor with conservative abstention."""

    STATE_SCHEMA = "momworld-v2-navhard-pairwise-gate-state-v1"

    def __init__(self, config: Any) -> None:
        super().__init__()
        path = Path(
            str(getattr(config, "rule_v2_navhard_pairwise_gate_state_path", ""))
        ).expanduser()
        expected_sha256 = str(
            getattr(config, "rule_v2_navhard_pairwise_gate_state_sha256", "")
        ).lower()
        if not path.is_file() or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
            raise ValueError("enabled pairwise gate requires a state path and SHA256")
        digest_builder = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest_builder.update(chunk)
        digest = digest_builder.hexdigest()
        if digest != expected_sha256:
            raise RuntimeError("V2 NavHard pairwise gate state SHA256 mismatch")
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict) or payload.get("schema") != self.STATE_SCHEMA:
            raise RuntimeError("invalid V2 NavHard pairwise gate state")
        if int(payload.get("feature_dim", -1)) != 59:
            raise RuntimeError("invalid V2 NavHard pairwise feature dimension")
        self.alternative_count = int(payload.get("alternatives", -1))
        if not 1 <= self.alternative_count <= 31:
            raise RuntimeError("invalid V2 NavHard pairwise alternative count")
        self.min_predicted_gain = float(payload["min_predicted_gain"])
        if not math.isfinite(self.min_predicted_gain):
            raise RuntimeError("invalid V2 NavHard pairwise threshold")
        model = payload["gain_model"]
        self.tree_count = int(model["tree_count"])
        self.max_depth = int(model["max_depth"])
        max_nodes = int(model["max_nodes"])
        if self.tree_count < 1 or max_nodes < 1 or self.max_depth < 1:
            raise RuntimeError("invalid V2 NavHard pairwise ensemble shape")
        self.register_buffer(
            "baseline", torch.tensor(float(model["baseline"]), dtype=torch.float64)
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
            value = torch.as_tensor(model[name], dtype=dtype)
            if value.shape != (self.tree_count, max_nodes):
                raise RuntimeError(f"invalid pairwise ensemble field: {name}")
            self.register_buffer(name, value)

    def _predict(self, features: torch.Tensor) -> torch.Tensor:
        shape = features.shape[:-1]
        values = features.reshape(-1, features.shape[-1]).double()
        tree = torch.arange(self.tree_count, device=values.device)[None, :].expand(
            len(values), -1
        )
        nodes = torch.zeros_like(tree)
        for _ in range(self.max_depth + 1):
            leaf = self.is_leaf[tree, nodes]
            feature = self.feature_idx[tree, nodes]
            observed = values.gather(1, feature)
            go_left = torch.where(
                torch.isnan(observed),
                self.missing_left[tree, nodes],
                observed <= self.threshold[tree, nodes],
            )
            next_nodes = torch.where(
                go_left, self.left[tree, nodes], self.right[tree, nodes]
            )
            nodes = torch.where(leaf, nodes, next_nodes)
        prediction = self.baseline + self.value[tree, nodes].sum(dim=1)
        return prediction.reshape(shape)

    def select(
        self,
        rule_features: torch.Tensor,
        normalized_base: torch.Tensor,
        protocol_proxy_zscore: torch.Tensor,
        candidate_is_finite: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return selection diagnostics for target-distribution calibration."""

        values = rule_features.float()
        finite = candidate_is_finite.bool()
        strict = (
            finite
            & (values[..., 1] >= 0.55)
            & (values[..., 3] >= 0.435)
            & (values[..., 5] >= 0.55)
        )
        violation = (
            torch.relu(0.55 - values[..., 1]) / 0.55
            + torch.relu(0.435 - values[..., 3]) / 0.435
            + torch.relu(0.55 - values[..., 5]) / 0.55
        )
        has_strict = strict.any(dim=1, keepdim=True)
        has_finite = finite.any(dim=1, keepdim=True)
        minimum = violation.masked_fill(~finite, torch.inf).amin(dim=1, keepdim=True)
        fallback = finite & (violation <= minimum)
        eligible = torch.where(
            has_strict,
            strict,
            torch.where(has_finite, fallback, torch.ones_like(finite)),
        )
        combined = normalized_base.float() + 3.2 * protocol_proxy_zscore.float()
        order = combined.masked_fill(~eligible, -torch.inf).argsort(
            dim=1, descending=True
        )
        incumbent = order[:, 0]
        alternatives = order[:, 1 : 1 + self.alternative_count]
        alternative_eligible = _gather(eligible, alternatives)
        features = _pair_features(
            values,
            normalized_base,
            protocol_proxy_zscore,
            combined,
            incumbent,
            alternatives,
        )
        prediction = self._predict(features).masked_fill(
            ~alternative_eligible, -torch.inf
        )
        best = prediction.argmax(dim=1)
        batch = torch.arange(len(prediction), device=prediction.device)
        switch = prediction[batch, best] >= self.min_predicted_gain
        selected = torch.where(switch, alternatives[batch, best], incumbent)
        return selected, prediction.float(), incumbent, alternatives
