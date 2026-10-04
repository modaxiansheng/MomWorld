"""Agent-aware latent world model for 6-second open-loop planning.

The module deliberately has no recurrent cache of its own.  MomAD's
``InstanceQueue`` already supplies scene-aware temporal features; keeping a
second cache inside the planning head would mix samples when batches are
shuffled or distributed over several GPUs.
"""

import torch
import torch.nn as nn


class LatentWorldModelMomAD6s(nn.Module):
    """Roll out future scene latents and decode multi-modal ego trajectories.

    The rollout is conditioned on three sources:

    * current ego, detected-agent and HD-map latent features;
    * MomAD's multi-modal future motion prediction for surrounding agents;
    * command/mode-specific planning queries.

    Args:
        embed_dims: Feature dimension used by MomAD.
        future_steps: Number of 0.5-second rollout steps (12 for nuScenes 6s).
        num_plan_modes: Three commands times the per-command mode count.
        max_dynamic_agents: Maximum number of confident agents used by the
            future-scene encoder.
        dropout: Dropout probability in context fusion blocks.
        trajectory_residual_scale: Initial scale of the auxiliary trajectory
            residual around MomAD's original planning prediction.
        fusion_scale_init: Initial residual scale applied to the planning
            query.  Zero makes a baseline checkpoint's first forward pass
            unchanged while auxiliary losses can still train the rollout.
    """

    def __init__(
        self,
        embed_dims=256,
        future_steps=12,
        num_plan_modes=18,
        max_dynamic_agents=32,
        dropout=0.1,
        trajectory_residual_scale=0.1,
        fusion_scale_init=0.0,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.future_steps = future_steps
        self.num_plan_modes = num_plan_modes
        self.max_dynamic_agents = max_dynamic_agents
        self.trajectory_residual_scale = nn.Parameter(
            torch.tensor(float(trajectory_residual_scale))
        )

        def projection():
            return nn.Sequential(
                nn.Linear(embed_dims, embed_dims),
                nn.ReLU(inplace=True),
                nn.LayerNorm(embed_dims),
            )

        self.ego_projection = projection()
        self.agent_projection = projection()
        self.map_projection = projection()
        self.context_fusion = nn.Sequential(
            nn.Linear(embed_dims * 3, embed_dims),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.LayerNorm(embed_dims),
        )

        # Each predicted agent position becomes a future perception token.
        self.agent_position_encoder = nn.Sequential(
            nn.Linear(2, embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims // 2, embed_dims),
            nn.LayerNorm(embed_dims),
        )
        self.future_scene_fusion = nn.Sequential(
            nn.Linear(embed_dims * 2, embed_dims),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.LayerNorm(embed_dims),
        )

        self.time_embedding = nn.Parameter(
            torch.zeros(future_steps, embed_dims)
        )
        nn.init.normal_(self.time_embedding, std=0.02)
        self.action_encoder = nn.Sequential(
            nn.Linear(4, embed_dims),
            nn.ReLU(inplace=True),
            nn.LayerNorm(embed_dims),
        )
        self.initial_state = nn.Sequential(
            nn.Linear(embed_dims * 2, embed_dims),
            nn.Tanh(),
            nn.LayerNorm(embed_dims),
        )
        self.rollout_cell = nn.GRUCell(embed_dims, embed_dims)
        self.rollout_norm = nn.LayerNorm(embed_dims)

        # Auxiliary heads make the latent dynamics explicitly trainable.
        self.trajectory_decoder = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, 2),
        )
        self.mode_decoder = nn.Sequential(
            nn.Linear(embed_dims * 2, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, 1),
        )
        self.scene_flow_decoder = nn.Sequential(
            nn.Linear(embed_dims, embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims // 2, 2),
        )

        # Future-state teacher: it is trained through reconstruction, while
        # the rollout matches a stop-gradient copy of its latent target.
        self.target_agent_state_encoder = nn.Sequential(
            nn.Linear(7, embed_dims),
            nn.ReLU(inplace=True),
            nn.LayerNorm(embed_dims),
        )
        self.target_ego_state_encoder = nn.Sequential(
            nn.Linear(7, embed_dims),
            nn.ReLU(inplace=True),
            nn.LayerNorm(embed_dims),
        )
        self.target_state_fusion = nn.Sequential(
            nn.Linear(embed_dims * 2, embed_dims),
            nn.ReLU(inplace=True),
            nn.LayerNorm(embed_dims),
        )
        self.future_state_decoder = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, 14),
        )

        self.query_fusion = nn.Sequential(
            nn.Linear(embed_dims * 2, embed_dims),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.LayerNorm(embed_dims),
            nn.Linear(embed_dims, embed_dims),
        )
        self.fusion_scale = nn.Parameter(torch.tensor(float(fusion_scale_init)))

    @staticmethod
    def _weighted_pool(features, confidence):
        """Confidence-weighted pooling that also handles empty features."""
        if features is None or features.shape[1] == 0:
            return None
        if confidence is None:
            return features.mean(dim=1)
        weight = confidence.clamp(min=0).unsqueeze(-1)
        weight = weight / weight.sum(dim=1, keepdim=True).clamp(min=1e-6)
        return (features * weight).sum(dim=1)

    @staticmethod
    def _gather_agents(tensor, indices):
        """Gather the agent dimension while retaining all trailing axes."""
        view_shape = list(indices.shape) + [1] * (tensor.dim() - 2)
        expand_shape = list(indices.shape) + list(tensor.shape[2:])
        gather_index = indices.view(*view_shape).expand(*expand_shape)
        return torch.gather(tensor, 1, gather_index)

    def _encode_future_agents(
        self,
        agent_features,
        agent_confidence,
        motion_logits,
        motion_deltas,
    ):
        """Convert multi-modal agent forecasts into one token per future step."""
        batch_size = agent_features.shape[0]
        if motion_deltas is None or motion_logits is None:
            empty_token = agent_features.new_zeros(
                batch_size, 1, self.future_steps, self.embed_dims
            )
            empty_weight = agent_features.new_ones(batch_size, 1)
            return empty_token, empty_weight

        num_agents = motion_deltas.shape[1]
        topk_count = min(self.max_dynamic_agents, num_agents)
        confidence, indices = torch.topk(
            agent_confidence, topk_count, dim=1, sorted=False
        )
        selected_features = self._gather_agents(agent_features, indices)
        selected_logits = self._gather_agents(motion_logits, indices)
        selected_deltas = self._gather_agents(motion_deltas, indices)

        # Encode every motion mode before mixture pooling. This avoids creating
        # a physically implausible mean trajectory when left/right futures are
        # both likely.
        mode_weight = torch.softmax(selected_logits, dim=-1)
        absolute_trajs = selected_deltas.cumsum(dim=-2)
        absolute_trajs = absolute_trajs[..., : self.future_steps, :]
        future_position = self.agent_position_encoder(absolute_trajs)
        agent_token = self.agent_projection(selected_features)[:, :, None, None]
        future_token = self.future_scene_fusion(
            torch.cat(
                [agent_token.expand_as(future_position), future_position],
                dim=-1,
            )
        )

        confidence = confidence.clamp(min=0)
        confidence = confidence / confidence.sum(dim=1, keepdim=True).clamp(min=1e-6)
        token_weight = confidence[:, :, None] * mode_weight
        future_token = future_token.flatten(1, 2)
        token_weight = token_weight.flatten(1, 2)
        token_weight = token_weight / token_weight.sum(dim=1, keepdim=True).clamp(
            min=1e-6
        )
        return future_token, token_weight

    @staticmethod
    def _normalize_world_state(states):
        """Normalize x/y/v/speed and represent heading without wraparound."""
        position = states[..., 0:2] / 50.0
        velocity = states[..., 2:4] / 20.0
        speed = states[..., 4:5] / 20.0
        heading = states[..., 5:6]
        return torch.cat(
            [position, velocity, speed, torch.sin(heading), torch.cos(heading)],
            dim=-1,
        )

    def encode_future_world_targets(
        self,
        agent_states,
        agent_masks,
        ego_states,
        ego_masks,
        device,
    ):
        """Encode real future world states into reconstruction-backed latents."""
        if not torch.is_tensor(ego_states):
            ego_states = torch.stack([state.to(device) for state in ego_states])
        else:
            ego_states = ego_states.to(device)
        if not torch.is_tensor(ego_masks):
            ego_masks = torch.stack([mask.to(device) for mask in ego_masks])
        else:
            ego_masks = ego_masks.to(device)

        batch_size = ego_states.shape[0]
        steps = min(self.future_steps, ego_states.shape[1])
        ego_normalized = self._normalize_world_state(ego_states[:, :steps])
        agent_summary = ego_normalized.new_zeros(batch_size, steps, 7)
        agent_valid = ego_masks.new_zeros(batch_size, steps)

        for batch_idx in range(batch_size):
            states = agent_states[batch_idx].to(device)[:, :steps]
            masks = agent_masks[batch_idx].to(device)[:, :steps]
            if states.numel() == 0:
                continue
            normalized = self._normalize_world_state(states)
            valid_count = masks.sum(dim=0)
            agent_summary[batch_idx] = (
                normalized * masks.unsqueeze(-1)
            ).sum(dim=0) / valid_count.clamp(min=1).unsqueeze(-1)
            agent_valid[batch_idx] = valid_count.gt(0).to(agent_valid.dtype)

        agent_latent = self.target_agent_state_encoder(agent_summary)
        ego_latent = self.target_ego_state_encoder(ego_normalized)
        target_latent = self.target_state_fusion(
            torch.cat([agent_latent, ego_latent], dim=-1)
        )
        valid_mask = ego_masks[:, :steps].to(target_latent.dtype)
        return {
            "latent": target_latent,
            "ego_state": ego_normalized,
            "agent_state": agent_summary,
            "valid_mask": valid_mask,
            "agent_valid_mask": agent_valid.to(target_latent.dtype),
        }

    def decode_future_world_states(self, latent):
        decoded = self.future_state_decoder(latent)
        return decoded[..., :7], decoded[..., 7:]

    def forward(
        self,
        plan_query,
        ego_feature,
        agent_features,
        agent_confidence,
        map_features,
        map_confidence,
        motion_logits,
        motion_deltas,
        base_plan_logits,
        base_plan_deltas,
    ):
        """Run a 12-step latent rollout.

        Returns tensors in the same command-major mode layout used by MomAD's
        ``PlanningTarget`` and ``HierarchicalPlanningDecoder``.
        """
        batch_size, _, num_modes, channels = plan_query.shape
        if num_modes != self.num_plan_modes:
            raise ValueError(
                "Expected %d planning modes, got %d"
                % (self.num_plan_modes, num_modes)
            )

        ego_context = self.ego_projection(ego_feature.squeeze(1))
        agent_context = self._weighted_pool(agent_features, agent_confidence)
        map_context = self._weighted_pool(map_features, map_confidence)
        if agent_context is None:
            agent_context = ego_context.new_zeros(ego_context.shape)
        else:
            agent_context = self.agent_projection(agent_context)
        if map_context is None:
            map_context = ego_context.new_zeros(ego_context.shape)
        else:
            map_context = self.map_projection(map_context)
        scene_context = self.context_fusion(
            torch.cat([ego_context, agent_context, map_context], dim=-1)
        )

        future_agent_tokens, future_agent_prior = self._encode_future_agents(
            agent_features,
            agent_confidence,
            motion_logits,
            motion_deltas,
        )

        plan_deltas = base_plan_deltas.detach().squeeze(1)
        plan_absolute = plan_deltas.cumsum(dim=-2) / 50.0
        action_token = self.action_encoder(
            torch.cat([plan_deltas, plan_absolute], dim=-1)
        )

        mode_query = plan_query.squeeze(1)
        # Query every agent-motion-mode token separately. The probability is a
        # prior, not an early trajectory average, so distinct futures remain
        # available to different ego planning modes.
        attention_score = torch.einsum(
            "bqc,bstc->bqts", mode_query, future_agent_tokens
        ) * (channels ** -0.5)
        attention_score = attention_score + torch.log(
            future_agent_prior.clamp(min=1e-6)
        )[:, None, None, :]
        attention_weight = torch.softmax(attention_score, dim=-1)
        future_scene = torch.einsum(
            "bqts,bstc->bqtc", attention_weight, future_agent_tokens
        )
        future_scene = future_scene + scene_context[:, None, None]

        context_per_mode = scene_context.unsqueeze(1).expand(-1, num_modes, -1)
        hidden = self.initial_state(
            torch.cat([mode_query, context_per_mode], dim=-1)
        ).reshape(batch_size * num_modes, channels)

        future_latents = []
        for step in range(self.future_steps):
            step_input = (
                future_scene[:, :, step]
                + action_token[:, :, step]
                + self.time_embedding[step]
            )
            hidden = self.rollout_cell(
                step_input.reshape(batch_size * num_modes, channels), hidden
            )
            latent = self.rollout_norm(hidden).reshape(
                batch_size, num_modes, channels
            )
            future_latents.append(latent)
        future_latents = torch.stack(future_latents, dim=2)

        trajectory_residual = self.trajectory_decoder(future_latents)
        residual_scale = torch.tanh(self.trajectory_residual_scale)
        world_plan_deltas = (
            base_plan_deltas.detach()
            + residual_scale * trajectory_residual.unsqueeze(1)
        )
        pooled_future = future_latents.mean(dim=2)
        mode_residual = self.mode_decoder(
            torch.cat([mode_query, pooled_future], dim=-1)
        ).squeeze(-1)
        world_plan_logits = (
            base_plan_logits.detach()
            + residual_scale * mode_residual.unsqueeze(1)
        )

        # A shared scene-flow prediction directly supervises future perception.
        scene_latent = future_latents.mean(dim=1)
        scene_flow = self.scene_flow_decoder(scene_latent)

        query_residual = self.query_fusion(
            torch.cat([mode_query, pooled_future], dim=-1)
        )
        enhanced_query = mode_query + torch.tanh(self.fusion_scale) * query_residual

        return {
            "enhanced_plan_query": enhanced_query.unsqueeze(1),
            "future_latents": future_latents,
            "world_plan_logits": world_plan_logits,
            "world_plan_deltas": world_plan_deltas,
            "scene_flow": scene_flow,
            "residual_scale": residual_scale,
        }
