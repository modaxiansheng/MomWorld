import numpy as np

from navsim.evaluate.pdm_comfort_v1 import ego_is_comfortable_v1
from navsim.evaluate.pdm_score_v1 import _normalize_v1_progress
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex


def test_v1_progress_uses_only_collision_and_drivable_area() -> None:
    progress = np.array([4.0, 8.0])
    collision = np.array([1.0, 1.0])
    drivable = np.array([1.0, 1.0])
    normalized = _normalize_v1_progress(progress, collision, drivable, threshold=5.0)
    np.testing.assert_allclose(normalized, np.array([0.5, 1.0]))


def test_v1_progress_zeroes_invalid_short_proposal() -> None:
    progress = np.array([2.0, 3.0])
    collision = np.array([1.0, 0.0])
    drivable = np.array([1.0, 1.0])
    normalized = _normalize_v1_progress(progress, collision, drivable, threshold=5.0)
    np.testing.assert_allclose(normalized, np.array([1.0, 0.0]))


def test_v1_comfort_accepts_stationary_trajectory() -> None:
    states = np.zeros((1, 41, StateIndex.size()), dtype=np.float64)
    time_points = np.arange(41, dtype=np.float64) * 0.1
    assert ego_is_comfortable_v1(states, time_points).all()
