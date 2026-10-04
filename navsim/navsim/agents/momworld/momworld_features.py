"""NAVSIM target construction for MomWorld."""

from typing import Dict, List, Tuple

import numpy as np
import torch

from navsim.agents.gtrs_dense.hydra_features import HydraTargetBuilder
from navsim.agents.momworld.momworld_config import MomWorldConfig
from navsim.common.dataclasses import Annotations, Scene


def _rotation(angle: float) -> np.ndarray:
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float32)


def _wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


class MomWorldTargetBuilder(HydraTargetBuilder):
    """Add track-aligned multi-agent futures to the standard GTRS targets."""

    def __init__(self, config: MomWorldConfig):
        super().__init__(config)
        self._config = config

    def get_unique_name(self) -> str:
        return "momworld_target"

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        targets = super().compute_targets(scene)
        agent_states, agent_masks = self._compute_future_agent_targets(scene)
        targets["world_agent_states"] = torch.from_numpy(agent_states)
        targets["world_agent_masks"] = torch.from_numpy(agent_masks)
        return targets

    def _current_vehicle_tracks(
        self, annotations: Annotations
    ) -> List[str]:
        candidates: List[Tuple[float, str]] = []
        for box, name, track_token in zip(
            annotations.boxes, annotations.names, annotations.track_tokens
        ):
            if name != "vehicle" or not track_token:
                continue
            x_position, y_position = float(box[0]), float(box[1])
            if not np.isfinite([x_position, y_position]).all():
                continue
            if not (
                self._config.lidar_min_x <= x_position <= self._config.lidar_max_x
                and self._config.lidar_min_y <= y_position <= self._config.lidar_max_y
            ):
                continue
            candidates.append((float(np.hypot(x_position, y_position)), track_token))
        candidates.sort(key=lambda item: item[0])
        return [token for _, token in candidates[: self._config.num_bounding_boxes]]

    def _compute_future_agent_targets(
        self, scene: Scene
    ) -> Tuple[np.ndarray, np.ndarray]:
        num_agents = self._config.num_bounding_boxes
        future_steps = self._config.world_agent_steps
        states = np.zeros((num_agents, future_steps, 6), dtype=np.float32)
        masks = np.zeros((num_agents, future_steps), dtype=np.bool_)

        current_index = scene.scene_metadata.num_history_frames - 1
        current_frame = scene.frames[current_index]
        current_pose = np.asarray(current_frame.ego_status.ego_pose, dtype=np.float32)
        current_rotation_inv = _rotation(-float(current_pose[2]))
        selected_tracks = self._current_vehicle_tracks(current_frame.annotations)
        selected_lookup = {token: index for index, token in enumerate(selected_tracks)}

        max_available_steps = min(
            future_steps,
            len(scene.frames) - current_index - 1,
        )
        for step in range(max_available_steps):
            frame = scene.frames[current_index + step + 1]
            annotations = frame.annotations
            if annotations is None:
                continue
            frame_pose = np.asarray(frame.ego_status.ego_pose, dtype=np.float32)
            frame_rotation = _rotation(float(frame_pose[2]))
            velocity_rotation = current_rotation_inv @ frame_rotation

            for box, name, velocity, track_token in zip(
                annotations.boxes,
                annotations.names,
                annotations.velocity_3d,
                annotations.track_tokens,
            ):
                if name != "vehicle" or track_token not in selected_lookup:
                    continue
                agent_index = selected_lookup[track_token]
                local_position = np.asarray(box[:2], dtype=np.float32)
                local_velocity = np.asarray(velocity[:2], dtype=np.float32)
                if not np.isfinite(local_position).all():
                    continue
                local_velocity = np.nan_to_num(local_velocity)
                global_position = frame_rotation @ local_position + frame_pose[:2]
                current_local_position = current_rotation_inv @ (
                    global_position - current_pose[:2]
                )
                current_local_velocity = velocity_rotation @ local_velocity
                relative_heading = _wrap_angle(
                    float(frame_pose[2]) + float(box[-1]) - float(current_pose[2])
                )

                states[agent_index, step] = np.asarray(
                    [
                        current_local_position[0],
                        current_local_position[1],
                        current_local_velocity[0],
                        current_local_velocity[1],
                        np.sin(relative_heading),
                        np.cos(relative_heading),
                    ],
                    dtype=np.float32,
                )
                masks[agent_index, step] = True

        return states, masks
