"""Trainable MomWorld agent and auxiliary losses."""

import os
from typing import Any, Dict, List, Mapping, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint

from navsim.agents.gtrs_dense.gtrs_agent import GTRSAgent
from navsim.agents.momworld.context_ranker import (
    validate_context_ranker_checkpoint_metadata,
)
from navsim.agents.momworld.momworld_config import MomWorldConfig
from navsim.agents.momworld.momworld_features import MomWorldTargetBuilder
from navsim.agents.momworld.momworld_model import (
    MomWorldModel,
    candidate_collision_risk,
)
from navsim.planning.training.abstract_feature_target_builder import AbstractTargetBuilder


def _ego_world_targets(
    trajectory: torch.Tensor,
    config: MomWorldConfig,
) -> torch.Tensor:
    """Convert a 10 Hz ``(x, y, heading)`` trajectory to normalized state."""
    positions = trajectory[..., :2]
    previous_position = torch.cat(
        [torch.zeros_like(positions[:, :1]), positions[:, :-1]], dim=1
    )
    velocity = (positions - previous_position) / config.world_dt
    heading = trajectory[..., 2:3]
    return torch.cat(
        [
            positions / config.world_position_scale,
            velocity / config.world_velocity_scale,
            torch.sin(heading),
            torch.cos(heading),
        ],
        dim=-1,
    )


def momworld_auxiliary_loss(
    targets: Dict[str, torch.Tensor],
    predictions: Dict[str, torch.Tensor],
    config: MomWorldConfig,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Supervise configuration, momentum, abrupt changes, and residual planning."""
    interpolated = targets["interpolated_traj"].to(
        device=predictions["world_ego_states"].device,
        dtype=predictions["world_ego_states"].dtype,
    )
    ego_target = _ego_world_targets(interpolated, config)

    ego_state_loss = F.smooth_l1_loss(
        predictions["world_ego_states"], ego_target
    )
    momentum_loss = F.smooth_l1_loss(
        predictions["world_momentum_xy"], ego_target[..., 2:4]
    )

    velocity = ego_target[..., 2:4] * config.world_velocity_scale
    previous_velocity = torch.cat(
        [velocity[:, :1], velocity[:, :-1]], dim=1
    )
    acceleration = (velocity - previous_velocity) / config.world_dt
    change_target = (
        acceleration.norm(dim=-1, keepdim=True)
        / config.scene_change_accel_threshold
    ).clamp(0.0, 1.0)
    change_gate_loss = F.smooth_l1_loss(
        predictions["world_change_gate"], change_target
    )

    momentum_delta = (
        predictions["world_momenta"][:, 1:]
        - predictions["world_momenta"][:, :-1]
    ).abs()
    smooth_weight = 1.0 - predictions["world_change_gate"][:, 1:]
    momentum_smoothness_loss = (momentum_delta * smooth_weight).mean()

    trajectory_position_loss = F.smooth_l1_loss(
        predictions["trajectory"][..., :2], interpolated[..., :2]
    )
    trajectory_heading_loss = (
        1.0
        - torch.cos(
            predictions["trajectory"][..., 2] - interpolated[..., 2]
        )
    ).mean()
    trajectory_residual_loss = trajectory_position_loss + 0.2 * trajectory_heading_loss

    flow_matching_loss = F.mse_loss(
        predictions["world_flow_velocity"], predictions["world_flow_target"]
    )

    zero = ego_state_loss.new_zeros(())
    agent_state_loss = zero
    agent_presence_loss = zero
    if "world_agent_states" in targets and "world_agent_masks" in targets:
        sample_indices = torch.arange(
            4,
            config.world_future_steps,
            5,
            device=predictions["world_agent_states"].device,
        )[: config.world_agent_steps]
        predicted_agent_states = predictions["world_agent_states"][:, sample_indices]
        predicted_agent_states = predicted_agent_states.permute(0, 2, 1, 3)
        predicted_presence = predictions["world_agent_presence_logits"][:, sample_indices]
        predicted_presence = predicted_presence.permute(0, 2, 1)

        agent_target = targets["world_agent_states"].to(
            device=predicted_agent_states.device,
            dtype=predicted_agent_states.dtype,
        )
        agent_mask = targets["world_agent_masks"].to(
            device=predicted_agent_states.device,
            dtype=torch.bool,
        )
        normalized_agent_target = agent_target.clone()
        normalized_agent_target[..., :2] /= config.world_position_scale
        normalized_agent_target[..., 2:4] /= config.world_velocity_scale
        agent_state_error = F.smooth_l1_loss(
            predicted_agent_states,
            normalized_agent_target,
            reduction="none",
        ).mean(dim=-1)
        agent_state_loss = (
            agent_state_error * agent_mask.to(agent_state_error.dtype)
        ).sum() / agent_mask.sum().clamp(min=1)
        agent_presence_loss = F.binary_cross_entropy_with_logits(
            predicted_presence, agent_mask.to(predicted_presence.dtype)
        )

    weighted_losses = {
        "world_ego_state_loss": config.world_ego_state_weight * ego_state_loss,
        "world_agent_state_loss": config.world_agent_state_weight * agent_state_loss,
        "world_agent_presence_loss": config.world_agent_presence_weight * agent_presence_loss,
        "world_momentum_loss": config.world_momentum_weight * momentum_loss,
        "world_change_gate_loss": config.world_change_gate_weight * change_gate_loss,
        "world_momentum_smoothness_loss": (
            config.world_momentum_smoothness_weight * momentum_smoothness_loss
        ),
        "world_trajectory_residual_loss": (
            config.world_trajectory_residual_weight * trajectory_residual_loss
        ),
        "world_flow_matching_loss": (
            config.world_flow_matching_weight * flow_matching_loss
        ),
    }
    total = sum(weighted_losses.values(), zero)
    weighted_losses["world_residual_gate"] = predictions["world_residual_gate"].detach()
    weighted_losses["world_change_gate_mean"] = predictions[
        "world_change_gate"
    ].detach().mean()
    weighted_losses["world_persistence_mean"] = predictions[
        "world_persistence_gate"
    ].detach().mean()
    return total, weighted_losses


def _rule_protocol_score(
    components: Dict[str, torch.Tensor], protocol: str
) -> torch.Tensor:
    ones = torch.ones_like(components["ego_progress"])
    no_collision = components["no_at_fault_collisions"].clamp(0.0, 1.0)
    drivable = components["drivable_area_compliance"].clamp(0.0, 1.0)
    ttc = components["time_to_collision_within_bound"].clamp(0.0, 1.0)
    progress = components["ego_progress"].clamp(0.0, 1.0)
    comfort = components.get("history_comfort", ones).clamp(0.0, 1.0)

    if protocol == "v1":
        return no_collision * drivable * (
            5.0 * progress + 5.0 * ttc + 2.0 * comfort
        ) / 12.0
    if protocol != "v2":
        raise ValueError(f"Unsupported rule protocol: {protocol!r}")

    direction = components["driving_direction_compliance"].clamp(0.0, 1.0)
    traffic_light = components["traffic_light_compliance"].clamp(0.0, 1.0)
    lane_keeping = components["lane_keeping"].clamp(0.0, 1.0)
    extended_comfort = components.get("extended_comfort", comfort).clamp(0.0, 1.0)
    return no_collision * drivable * direction * traffic_light * (
        5.0 * progress
        + 5.0 * ttc
        + 2.0 * lane_keeping
        + 2.0 * comfort
        + 2.0 * extended_comfort
    ) / 16.0


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


def momworld_rule_scorer_loss(
    targets: Dict[str, torch.Tensor],
    predictions: Dict[str, torch.Tensor],
    config: MomWorldConfig,
    rule_targets: Dict[str, torch.Tensor],
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Train both NAVSIM protocol heads and collision-aware candidate ranking."""
    device = predictions["rule_v1_logits"].device
    dtype = predictions["rule_v1_logits"].dtype
    components = {
        name: value.to(device=device, dtype=dtype)
        for name, value in rule_targets.items()
    }
    target_v1 = _rule_protocol_score(components, "v1")
    target_v2 = _rule_protocol_score(components, "v2")

    candidate_trajectories = predictions["rule_candidate_trajectories"]
    agent_positions = (
        targets["world_agent_states"][..., :2]
        .to(device=device, dtype=dtype)
        .permute(0, 2, 1, 3)
    )
    agent_presence = (
        targets["world_agent_masks"]
        .to(device=device, dtype=dtype)
        .permute(0, 2, 1)
    )
    target_collision_risk = candidate_collision_risk(
        candidate_trajectories,
        agent_positions,
        agent_presence,
        config.rule_collision_longitudinal_radius,
        config.rule_collision_lateral_radius,
        config.rule_collision_temperature,
    ).detach()
    target_v1 = target_v1 * (1.0 - target_collision_risk)
    target_v2 = target_v2 * (1.0 - target_collision_risk)

    v1_loss = F.binary_cross_entropy_with_logits(
        predictions["rule_v1_logits"], target_v1
    )
    v2_loss = F.binary_cross_entropy_with_logits(
        predictions["rule_v2_logits"], target_v2
    )
    collision_loss = F.smooth_l1_loss(
        predictions["rule_collision_risk"], target_collision_risk
    )
    rank_loss = 0.5 * (
        _pairwise_best_rank_loss(
            predictions["rule_v1_logits"],
            target_v1,
            config.rule_rank_margin,
        )
        + _pairwise_best_rank_loss(
            predictions["rule_v2_logits"],
            target_v2,
            config.rule_rank_margin,
        )
    )

    scorer_loss = 0.5 * (v1_loss + v2_loss)
    weighted = {
        "rule_scorer_loss": config.rule_scorer_loss_weight * scorer_loss,
        "rule_collision_loss": config.rule_collision_loss_weight * collision_loss,
        "rule_rank_loss": config.rule_rank_loss_weight * rank_loss,
    }
    predictions["rule_target_v1"] = target_v1.detach()
    predictions["rule_target_v2"] = target_v2.detach()
    predictions["rule_target_collision_risk"] = target_collision_risk
    return sum(weighted.values()), weighted


class MomWorldAgent(GTRSAgent):
    """GTRS-Dense agent with momentum-aware latent world modeling."""

    def __init__(
        self,
        config: MomWorldConfig,
        lr: float,
        checkpoint_path: str = None,
        pdm_gt_path=None,
    ):
        super().__init__(
            config=config,
            lr=lr,
            checkpoint_path=checkpoint_path,
            pdm_gt_path=pdm_gt_path,
            model_cls=MomWorldModel,
        )

    def initialize(self) -> None:
        """Load optional context weights with an exact architecture/key audit."""

        try:
            checkpoint = torch.load(
                self._checkpoint_path,
                map_location=torch.device("cpu"),
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(
                self._checkpoint_path, map_location=torch.device("cpu")
            )
        if not isinstance(checkpoint, Mapping):
            raise RuntimeError("MomWorld checkpoint is not a mapping")
        raw_state = checkpoint.get("state_dict")
        if not isinstance(raw_state, Mapping):
            raise RuntimeError("MomWorld checkpoint has no state_dict mapping")
        state_dict: Dict[str, Any] = {
            str(name): value
            for name, value in raw_state.items()
            if "model._trajectory_head.vocab" not in str(name)
        }
        self._context_ranker_load_audit = validate_context_ranker_checkpoint_metadata(
            self._config,
            checkpoint,
            state_dict,
            tuple(self.state_dict()),
        )
        normalized_state = {
            name.replace("agent.", ""): value for name, value in state_dict.items()
        }
        message = self.load_state_dict(normalized_state, strict=False)
        collision_prefix = "model._collision_calibrator."
        collision_enabled = bool(
            getattr(self._config, "rule_collision_calibrator_enabled", False)
        )
        checkpoint_has_collision = any(
            name.startswith(collision_prefix) for name in normalized_state
        )
        if collision_enabled:
            collision_missing = [
                name for name in message.missing_keys if name.startswith(collision_prefix)
            ]
            collision_unexpected = [
                name for name in message.unexpected_keys if name.startswith(collision_prefix)
            ]
            if collision_missing or collision_unexpected or not checkpoint_has_collision:
                raise RuntimeError(
                    "Collision-calibrator load was incomplete: "
                    f"missing={collision_missing}, unexpected={collision_unexpected}"
                )
            if self.model._collision_calibrator is None:
                raise RuntimeError("Collision-calibrator config did not construct its module")
            self.model._collision_calibrator.require_artifact_ready()
        elif checkpoint_has_collision:
            raise RuntimeError(
                "Checkpoint contains collision-calibrator state but runtime config disables it"
            )
        if bool(getattr(self._config, "context_ranker_enabled", False)):
            context_prefix = "model._context_ranker."
            missing = [
                name for name in message.missing_keys if name.startswith(context_prefix)
            ]
            unexpected = [
                name
                for name in message.unexpected_keys
                if name.startswith(context_prefix)
            ]
            if missing or unexpected:
                raise RuntimeError(
                    "Context-ranker load was incomplete: "
                    f"missing={missing}, unexpected={unexpected}"
                )
            if self.model._context_ranker is None:
                raise RuntimeError("Context-ranker config did not construct its module")
            self.model._context_ranker.require_artifact_ready()
        print("Loading full MomWorld model", message)

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        return [MomWorldTargetBuilder(config=self._config)]

    def _gather_rule_targets(
        self,
        tokens: List[str],
        predictions: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Gather train-only vocabulary labels for the retained candidates."""
        indices = predictions["rule_vocab_indices"].detach().cpu().numpy()
        metric_names = list(
            dict.fromkeys(
                list(self.metrics) + ["history_comfort", "extended_comfort"]
            )
        )
        gathered: Dict[str, torch.Tensor] = {}
        for metric_name in metric_names:
            rows = []
            for token, token_indices in zip(tokens, indices):
                token_scores = self.vocab_pdm_score_full[token]
                if metric_name in token_scores:
                    values = np.asarray(token_scores[metric_name])
                    rows.append(values[token_indices])
                else:
                    rows.append(
                        np.ones(token_indices.shape, dtype=np.float32)
                    )
            gathered[metric_name] = torch.from_numpy(
                np.stack(rows).astype(np.float32)
            ).to(predictions["rule_v1_logits"].device)
        return gathered


    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
        tokens=None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if self._config.rule_finetune_only:
            rule_targets = self._gather_rule_targets(tokens, predictions)
            return momworld_rule_scorer_loss(
                targets=targets,
                predictions=predictions,
                config=self._config,
                rule_targets=rule_targets,
            )

        base_loss, loss_dict = super().compute_loss(
            features=features,
            targets=targets,
            predictions=predictions,
            tokens=tokens,
        )
        target_trajectory = targets["interpolated_traj"].to(
            device=predictions["base_trajectory"].device,
            dtype=predictions["base_trajectory"].dtype,
        )
        predictions.update(
            self.model.flow_matching_outputs(
                base_trajectory=predictions["base_trajectory"],
                target_trajectory=target_trajectory,
                world_memory=predictions["world_memory"],
                world_momenta=predictions["world_momenta"],
            )
        )
        world_loss, world_loss_dict = momworld_auxiliary_loss(
            targets=targets,
            predictions=predictions,
            config=self._config,
        )
        loss_dict.update(world_loss_dict)
        rule_loss = base_loss.new_zeros(())
        if self._config.rule_enabled:
            rule_targets = self._gather_rule_targets(tokens, predictions)
            rule_loss, rule_loss_dict = momworld_rule_scorer_loss(
                targets=targets,
                predictions=predictions,
                config=self._config,
                rule_targets=rule_targets,
            )
            loss_dict.update(rule_loss_dict)

        if self._config.rule_finetune_only:
            total_loss = rule_loss
        else:
            total_loss = base_loss + world_loss + rule_loss
        return total_loss, loss_dict

    def get_optimizers(self):
        if not self._config.rule_finetune_only:
            return super().get_optimizers()
        parameters = [
            parameter
            for parameter in self.model._rule_scorer.parameters()
            if parameter.requires_grad
        ]
        if not parameters:
            raise RuntimeError("rule_finetune_only enabled without trainable scorer parameters")
        return torch.optim.AdamW(
            parameters,
            lr=self._lr,
            weight_decay=self._config.weight_decay,
        )


    def get_training_callbacks(self) -> List[pl.Callback]:
        """Keep a bounded number of large checkpoints on disk."""
        return [
            ModelCheckpoint(
                save_top_k=self._config.world_checkpoint_top_k,
                save_last=True,
                monitor="val/loss_epoch",
                mode="min",
                dirpath=(
                    f"{os.environ.get('NAVSIM_EXP_ROOT')}/"
                    f"{self._config.ckpt_path}/"
                ),
                filename="{epoch:02d}-{step:04d}",
            )
        ]
