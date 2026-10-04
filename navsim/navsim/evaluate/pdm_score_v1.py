"""Exact classic NAVSIM v1.1 PDM Score on the current metric-cache format."""

import copy
from typing import Dict, List

import numpy as np
import numpy.typing as npt
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.simulation.observation.idm.utils import is_agent_ahead, is_agent_behind
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from shapely import creation

from navsim.common.dataclasses import Trajectory
from navsim.evaluate.pdm_comfort_v1 import ego_is_comfortable_v1
from navsim.evaluate.pdm_score import (
    get_trajectory_as_array,
    transform_trajectory,
)
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.planning.simulation.planner.pdm_planner.observation.pdm_observation import (
    PDMObservation,
)
from navsim.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import (
    PDMDrivableMap,
)
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
    PDMSimulator,
)
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    BBCoordsIndex,
    EgoAreaIndex,
    MultiMetricIndex,
    StateIndex,
    WeightedMetricIndex,
)
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath


def _normalize_v1_progress(
    progress_raw: np.ndarray,
    no_collision: np.ndarray,
    drivable_area: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Apply the normalization used by the official NAVSIM v1.1 scorer."""
    multiplicative = no_collision * drivable_area
    masked_progress = progress_raw * multiplicative
    max_progress = float(np.max(masked_progress))
    if max_progress > threshold:
        return masked_progress / max_progress

    normalized = np.ones_like(masked_progress, dtype=np.float64)
    normalized[multiplicative == 0.0] = 0.0
    return normalized


class PDMScorerV1(PDMScorer):
    """Port of the NAVSIM v1.1 scorer using current cache dataclasses.

    Collision, drivable-area, and progress implementations are unchanged in
    UniLAW, so they are inherited. TTC and Comfort deliberately restore the
    official v1.1 behavior from commit 3e8291b.
    """

    def _calculate_ttc_v1(self) -> None:
        ttc_scores = np.ones(self._num_proposals, dtype=np.float64)
        collided_track_ids = {
            proposal_idx: copy.deepcopy(self._observation.collided_track_ids)
            for proposal_idx in range(self._num_proposals)
        }

        future_time_indices = np.arange(0, 10, 3)
        coordinates = self._ego_coords.copy()
        coordinates[:, :, BBCoordsIndex.CENTER, :] = coordinates[
            :, :, BBCoordsIndex.FRONT_LEFT, :
        ]
        projected_coordinates = np.repeat(
            coordinates[:, :, None], len(future_time_indices), axis=2
        )

        speeds = np.hypot(
            self._states[..., StateIndex.VELOCITY_X],
            self._states[..., StateIndex.VELOCITY_Y],
        )
        displacement_per_second = np.stack(
            [
                np.cos(self._states[..., StateIndex.HEADING]) * speeds,
                np.sin(self._states[..., StateIndex.HEADING]) * speeds,
            ],
            axis=-1,
        )
        for index, future_time_index in enumerate(future_time_indices):
            delta_t = float(future_time_index) * self.proposal_sampling.interval_length
            projected_coordinates[:, :, index] += (
                displacement_per_second[:, :, None] * delta_t
            )
        polygons = creation.polygons(projected_coordinates)

        # NAVSIM v1 evaluates every proposal time and uses the observation's
        # extra one-second TTC horizon (41 proposal states, 51 observations).
        for time_index in range(self.proposal_sampling.num_poses + 1):
            for step_index, future_time_index in enumerate(future_time_indices):
                observation_index = time_index + future_time_index
                intersecting = self._observation[observation_index].query(
                    polygons[:, time_index, step_index], predicate="intersects"
                )
                if len(intersecting) == 0:
                    continue

                for proposal_index, geometry_index in zip(
                    intersecting[0], intersecting[1]
                ):
                    token = self._observation[observation_index].tokens[geometry_index]
                    if (
                        self._observation.red_light_token in token
                        or token in collided_track_ids[proposal_index]
                        or speeds[proposal_index, time_index]
                        < self._config.stopped_speed_threshold
                    ):
                        continue

                    invalid_area = (
                        self._ego_areas[
                            proposal_index,
                            time_index,
                            EgoAreaIndex.MULTIPLE_LANES,
                        ]
                        or self._ego_areas[
                            proposal_index,
                            time_index,
                            EgoAreaIndex.NON_DRIVABLE_AREA,
                        ]
                    )
                    ego_state = StateSE2(
                        *self._states[
                            proposal_index, time_index, StateIndex.STATE_SE2
                        ]
                    )
                    centroid = self._observation[observation_index][token].centroid
                    track_heading = self._observation.unique_objects[
                        token
                    ].box.center.heading
                    track_state = StateSE2(centroid.x, centroid.y, track_heading)
                    if is_agent_ahead(ego_state, track_state) or (
                        (
                            invalid_area
                            or self._drivable_area_map.is_in_layer(
                                ego_state.point,
                                layer=SemanticMapLayer.INTERSECTION,
                            )
                        )
                        and not is_agent_behind(ego_state, track_state)
                    ):
                        ttc_scores[proposal_index] = 0.0
                        self._ttc_time_idcs[proposal_index] = min(
                            time_index, self._ttc_time_idcs[proposal_index]
                        )
                    else:
                        collided_track_ids[proposal_index].append(token)

        self._weighted_metrics[WeightedMetricIndex.TTC] = ttc_scores

    def score_proposals_v1(
        self,
        states: npt.NDArray[np.float64],
        observation: PDMObservation,
        centerline: PDMPath,
        route_lane_ids: List[str],
        drivable_area_map: PDMDrivableMap,
    ) -> List[Dict[str, float]]:
        """Return classic v1 subscores for all simulated proposals."""
        self._reset(
            states,
            observation,
            centerline,
            route_lane_ids,
            drivable_area_map,
            None,
        )
        self._calculate_ego_area()
        self._calculate_no_at_fault_collision()
        self._calculate_drivable_area_compliance()
        self._calculate_progress()
        self._calculate_ttc_v1()

        no_collision = self._multi_metrics[MultiMetricIndex.NO_COLLISION]
        drivable_area = self._multi_metrics[MultiMetricIndex.DRIVABLE_AREA]
        progress = _normalize_v1_progress(
            self._progress_raw,
            no_collision,
            drivable_area,
            self._config.progress_distance_threshold,
        )
        ttc = self._weighted_metrics[WeightedMetricIndex.TTC]
        time_points = (
            np.arange(self.proposal_sampling.num_poses + 1, dtype=np.float64)
            * self.proposal_sampling.interval_length
        )
        comfort = ego_is_comfortable_v1(states, time_points).all(axis=-1).astype(float)

        results: List[Dict[str, float]] = []
        for index in range(self._num_proposals):
            score = no_collision[index] * drivable_area[index] * (
                5.0 * progress[index] + 5.0 * ttc[index] + 2.0 * comfort[index]
            ) / 12.0
            results.append(
                {
                    "no_at_fault_collisions": float(no_collision[index]),
                    "drivable_area_compliance": float(drivable_area[index]),
                    "ego_progress": float(progress[index]),
                    "time_to_collision_within_bound": float(ttc[index]),
                    "comfort": float(comfort[index]),
                    "score": float(score),
                }
            )
        return results


def pdm_score_v1(
    metric_cache: MetricCache,
    model_trajectory: Trajectory,
    future_sampling: TrajectorySampling,
    simulator: PDMSimulator,
    scorer: PDMScorerV1,
) -> Dict[str, float]:
    """Run the exact v1 non-reactive simulation for one predicted trajectory."""
    initial_ego_state = metric_cache.ego_state
    predicted_trajectory = transform_trajectory(model_trajectory, initial_ego_state)
    pdm_states = get_trajectory_as_array(
        metric_cache.trajectory,
        future_sampling,
        initial_ego_state.time_point,
    )
    predicted_states = get_trajectory_as_array(
        predicted_trajectory,
        future_sampling,
        initial_ego_state.time_point,
    )
    proposal_states = np.stack([pdm_states, predicted_states], axis=0)
    simulated_states = simulator.simulate_proposals(
        proposal_states, initial_ego_state
    )
    return scorer.score_proposals_v1(
        simulated_states,
        metric_cache.observation,
        metric_cache.centerline,
        metric_cache.route_lane_ids,
        metric_cache.drivable_area_map,
    )[1]
