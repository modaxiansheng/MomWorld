"""Configuration for the NAVSIM MomWorld agent."""

from dataclasses import dataclass

from navsim.agents.gtrs_dense.hydra_config import HydraConfig


@dataclass
class MomWorldConfig(HydraConfig):
    """Extends GTRS-Dense with an explicit configuration--momentum world state."""

    world_future_steps: int = 40
    world_agent_steps: int = 8
    world_dt: float = 0.1

    world_position_scale: float = 50.0
    world_velocity_scale: float = 20.0
    world_residual_max: float = 2.0
    world_residual_gate_init: float = -4.0
    world_flow_steps: int = 4
    momentum_persistence_init: float = 0.9
    scene_change_accel_threshold: float = 3.0

    world_ego_state_weight: float = 1.0
    world_agent_state_weight: float = 0.25
    world_agent_presence_weight: float = 0.1
    world_momentum_weight: float = 0.5
    world_change_gate_weight: float = 0.1
    world_momentum_smoothness_weight: float = 0.05
    world_trajectory_residual_weight: float = 1.0
    world_flow_matching_weight: float = 1.0
    world_checkpoint_top_k: int = 5

    # Rule-aware candidate refinement. The official NAVSIM evaluator is never
    # queried here: all signals come from the learned scene/world state and the
    # train-only vocabulary labels.
    rule_enabled: bool = False
    rule_protocol: str = "v2"
    rule_candidate_topk: int = 32
    rule_collision_longitudinal_radius: float = 4.5
    rule_collision_lateral_radius: float = 2.2
    rule_collision_temperature: float = 0.25
    rule_collision_filter_threshold: float = 0.65
    rule_collision_weight: float = 4.0
    rule_kinematic_weight: float = 0.25
    rule_momentum_weight: float = 0.10
    rule_learned_score_weight: float = 1.0
    # Optional NAVTRAIN-only rank fusion.  Both defaults are zero so existing
    # checkpoints/evaluation configs retain their original selection exactly.
    # The protocol proxy composes the seven predicted NAVSIM sub-metrics using
    # the corresponding v1/v2 aggregation rule, then standardizes the result
    # across each scene's candidate set.  The learned term standardizes the
    # protocol-specific MLP logits in the same way.
    rule_rank_fusion_mode: str = "hybrid"
    rule_protocol_proxy_weight: float = 0.0
    rule_learned_zscore_weight: float = 0.0
    rule_score_zscore_epsilon: float = 1e-4
    # Optional frozen NAVTRAIN-only V1 gate. It chooses among a fixed set of
    # ProtocolProxy weights and abstains to the 1.6 incumbent unless both a
    # predicted-gain and predicted-collision guard pass.
    rule_v1_protocol_gate_enabled: bool = False
    rule_v1_protocol_gate_state_path: str = ""
    rule_v1_protocol_gate_state_sha256: str = ""
    # Optional frozen NAVTRAIN-only selector for the selected V2 NavTest
    # GTRS+Rule incumbent. Defaults are neutral for every existing run.
    rule_v2_navtest_selector_gate_enabled: bool = False
    rule_v2_navtest_selector_gate_state_path: str = ""
    rule_v2_navtest_selector_gate_state_sha256: str = ""
    # Optional frozen NAVTRAIN-only nonlinear selector for V2 NavHard.
    rule_v2_navhard_nonlinear_gate_enabled: bool = False
    rule_v2_navhard_nonlinear_gate_state_path: str = ""
    rule_v2_navhard_nonlinear_gate_state_sha256: str = ""
    # Optional conservative NAVTRAIN-only pairwise selector for V2 NavHard.
    # It preserves the frozen 3.2-progress incumbent unless a sealed model is
    # highly confident that one of the next three eligible candidates is better.
    rule_v2_navhard_pairwise_gate_enabled: bool = False
    rule_v2_navhard_pairwise_gate_state_path: str = ""
    rule_v2_navhard_pairwise_gate_state_sha256: str = ""
    # Optional NAVTRAIN-calibrated, candidate-relative ReLU safety penalties.
    # All defaults are zero, preserving every existing checkpoint/config.
    rule_relu_collision_risk_weight: float = 0.0
    rule_relu_no_collision_weight: float = 0.0
    rule_relu_drivable_area_weight: float = 0.0
    rule_relu_ttc_weight: float = 0.0
    rule_relu_progress_weight: float = 0.0
    rule_relu_direction_weight: float = 0.0
    rule_relu_lane_weight: float = 0.0
    rule_relu_traffic_light_weight: float = 0.0
    rule_relu_kinematic_weight: float = 0.0
    rule_relu_momentum_weight: float = 0.0
    # Optional candidate-deletion rule.  It only changes which of the existing
    # top-k trajectories may win; it never removes an evaluation scene.  When
    # every candidate violates at least one threshold, selection falls back to
    # the minimum aggregate violation and then the normal combined score.
    rule_safety_filter_enabled: bool = False
    rule_safety_no_collision_min: float = 0.0
    rule_safety_drivable_area_min: float = 0.0
    rule_safety_ttc_min: float = 0.0
    rule_safety_progress_min: float = 0.0
    rule_safety_direction_min: float = 0.0
    rule_safety_lane_min: float = 0.0
    rule_safety_traffic_light_min: float = 0.0
    rule_safety_collision_risk_max: float = 1.000001
    rule_safety_kinematic_max: float = 1e6
    rule_safety_momentum_max: float = 1e6
    rule_safety_fallback_risk_slack: float = 0.0
    # Optional NAVTRAIN-only monotonic residual scorer.  The first seven
    # coefficients reward higher predicted protocol components; the final
    # three coefficients penalize collision, kinematic, and momentum risk.
    # All defaults are neutral, preserving every existing evaluation.
    rule_monotonic_residual_enabled: bool = False
    rule_monotonic_zscore_epsilon: float = 1e-4
    rule_monotonic_no_collision_weight: float = 0.0
    rule_monotonic_drivable_area_weight: float = 0.0
    rule_monotonic_ttc_weight: float = 0.0
    rule_monotonic_progress_weight: float = 0.0
    rule_monotonic_direction_weight: float = 0.0
    rule_monotonic_lane_weight: float = 0.0
    rule_monotonic_traffic_light_weight: float = 0.0
    rule_monotonic_collision_risk_weight: float = 0.0
    rule_monotonic_kinematic_weight: float = 0.0
    rule_monotonic_momentum_weight: float = 0.0
    # Optional scene-relative hard Rule.  Fractions describe how much of each
    # within-scene feature range remains eligible: for the first seven
    # higher-is-safer features, 0.2 keeps the top 20% of the range; for the
    # final three lower-is-safer features, it keeps the bottom 20%.  Defaults
    # are neutral and therefore preserve every existing evaluation exactly.
    rule_relative_safety_filter_enabled: bool = False
    rule_relative_no_collision_fraction: float = 1.0
    rule_relative_drivable_area_fraction: float = 1.0
    rule_relative_ttc_fraction: float = 1.0
    rule_relative_progress_fraction: float = 1.0
    rule_relative_direction_fraction: float = 1.0
    rule_relative_lane_fraction: float = 1.0
    rule_relative_traffic_light_fraction: float = 1.0
    rule_relative_collision_risk_fraction: float = 1.0
    rule_relative_kinematic_fraction: float = 1.0
    rule_relative_momentum_fraction: float = 1.0
    # Optional NAVTRAIN-only collision calibrator.  The head consumes the
    # existing rule features plus normalized base score; it never changes or
    # retrains the planner.  Defaults preserve every existing checkpoint.
    rule_collision_calibrator_enabled: bool = False
    rule_collision_calibrator_hidden_dim: int = 64
    rule_calibrated_collision_weight: float = 0.0
    rule_calibrated_collision_threshold: float = 1.000001
    rule_scorer_hidden_dim: int = 64
    # ``raw_plus_scene_zscore`` augments each candidate with the same ten
    # features standardized within its scene.  The default remains ``raw`` so
    # existing checkpoints and launch configurations are bit-compatible.
    rule_scorer_input_mode: str = "raw"
    rule_scorer_loss_weight: float = 1.0
    rule_collision_loss_weight: float = 0.5
    rule_rank_loss_weight: float = 0.25
    rule_rank_margin: float = 0.05
    rule_finetune_only: bool = False

    # Optional NAVTRAIN-only set-wise ranker.  This stays disabled for every
    # existing checkpoint.  Enabling it requires an injected, sealed ranker
    # artifact whose architecture and per-protocol selection values exactly
    # match these explicit Hydra fields.
    context_ranker_enabled: bool = False
    context_ranker_hidden_dim: int = 64
    context_ranker_num_layers: int = 2
    context_ranker_num_heads: int = 4
    context_ranker_dropout: float = 0.05
    context_ranker_zscore_epsilon: float = 1e-4
    context_ranker_base_weight: float = 0.0
    context_ranker_weight: float = 1.0
    context_ranker_collision_threshold: float = 0.8
    context_ranker_collision_weight: float = 0.0
    context_ranker_learned_collision_threshold: float = 1.000001
    context_ranker_fallback_risk_slack: float = 1.0
