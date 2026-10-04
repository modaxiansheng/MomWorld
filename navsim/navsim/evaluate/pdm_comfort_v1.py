"""Original NAVSIM v1.1 comfort metric.

This is a compact compatibility port of autonomousvision/navsim commit
3e8291bfa89ff247231e0227778840cd0a036896 (Apache-2.0). It intentionally
keeps the v1 rear-axle reference point and smoothing windows.
"""

from typing import Optional

import numpy as np
import numpy.typing as npt
from scipy.signal import savgol_filter

from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex

MAX_ABS_MAG_JERK = 8.37
MAX_ABS_LAT_ACCEL = 4.89
MAX_LON_ACCEL = 2.40
MIN_LON_ACCEL = -4.05
MAX_ABS_YAW_ACCEL = 1.93
MAX_ABS_LON_JERK = 4.13
MAX_ABS_YAW_RATE = 0.95


def _extract_acceleration(
    states: npt.NDArray[np.float64],
    coordinate: str,
    decimals: int = 8,
    poly_order: int = 2,
    window_length: int = 8,
) -> npt.NDArray[np.float64]:
    _, n_time, _ = states.shape
    if coordinate == "x":
        acceleration = states[..., StateIndex.ACCELERATION_X]
    elif coordinate == "y":
        acceleration = states[..., StateIndex.ACCELERATION_Y]
    elif coordinate == "magnitude":
        acceleration = np.hypot(
            states[..., StateIndex.ACCELERATION_X],
            states[..., StateIndex.ACCELERATION_Y],
        )
    else:
        raise ValueError(f"Unsupported acceleration coordinate: {coordinate}")
    acceleration = savgol_filter(
        acceleration,
        polyorder=poly_order,
        window_length=min(window_length, n_time),
        axis=-1,
    )
    return np.round(acceleration, decimals=decimals)


def _phase_unwrap(headings: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    two_pi = 2.0 * np.pi
    adjustments = np.zeros_like(headings)
    adjustments[..., 1:] = np.cumsum(
        np.round(np.diff(headings, axis=-1) / two_pi), axis=-1
    )
    return headings - two_pi * adjustments


def _approximate_derivatives(
    y: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
    window_length: int = 5,
    poly_order: int = 2,
    deriv_order: int = 1,
) -> npt.NDArray[np.float64]:
    window_length = min(window_length, len(x))
    if not poly_order < window_length:
        raise ValueError(f"{poly_order} < {window_length} does not hold")
    dx = np.diff(x, axis=-1)
    if not (dx > 0).all():
        raise RuntimeError("Time points are not strictly increasing")
    return savgol_filter(
        y,
        polyorder=poly_order,
        window_length=window_length,
        deriv=deriv_order,
        delta=dx.mean(),
        axis=-1,
    )


def _extract_jerk(
    states: npt.NDArray[np.float64],
    coordinate: str,
    time_steps_s: npt.NDArray[np.float64],
    decimals: int = 8,
    deriv_order: int = 1,
    poly_order: int = 2,
    window_length: int = 15,
) -> npt.NDArray[np.float64]:
    _, n_time, _ = states.shape
    acceleration = _extract_acceleration(states, coordinate)
    jerk = _approximate_derivatives(
        acceleration,
        time_steps_s,
        deriv_order=deriv_order,
        poly_order=poly_order,
        window_length=min(window_length, n_time),
    )
    return np.round(jerk, decimals=decimals)


def _extract_yaw_rate(
    states: npt.NDArray[np.float64],
    time_steps_s: npt.NDArray[np.float64],
    deriv_order: int = 1,
    poly_order: int = 2,
    decimals: int = 8,
) -> npt.NDArray[np.float64]:
    yaw_rate = _approximate_derivatives(
        _phase_unwrap(states[..., StateIndex.HEADING]),
        time_steps_s,
        deriv_order=deriv_order,
        poly_order=poly_order,
    )
    return np.round(yaw_rate, decimals=decimals)


def _within_bound(
    metric: npt.NDArray[np.float64],
    min_bound: Optional[float] = None,
    max_bound: Optional[float] = None,
) -> npt.NDArray[np.bool_]:
    lower = min_bound if min_bound else float(-np.inf)
    upper = max_bound if max_bound else float(np.inf)
    return np.all((metric > lower) & (metric < upper), axis=-1)


def ego_is_comfortable_v1(
    states: npt.NDArray[np.float64],
    time_points_s: npt.NDArray[np.float64],
) -> npt.NDArray[np.bool_]:
    """Return the six original v1 comfort checks for each proposal."""
    _, n_time, n_states = states.shape
    if n_time != len(time_points_s) or n_states != StateIndex.size():
        raise ValueError("Invalid state or time-point shape for NAVSIM v1 comfort")

    longitude_acceleration = _extract_acceleration(
        states, "x", window_length=n_time
    )
    lateral_acceleration = _extract_acceleration(
        states, "y", window_length=n_time
    )
    magnitude_jerk = _extract_jerk(
        states, "magnitude", time_points_s, window_length=n_time
    )
    longitude_jerk = _extract_jerk(
        states, "x", time_points_s, window_length=n_time
    )
    yaw_acceleration = _extract_yaw_rate(
        states, time_points_s, deriv_order=2, poly_order=3
    )
    yaw_rate = _extract_yaw_rate(states, time_points_s)

    return np.stack(
        [
            _within_bound(longitude_acceleration, MIN_LON_ACCEL, MAX_LON_ACCEL),
            _within_bound(lateral_acceleration, -MAX_ABS_LAT_ACCEL, MAX_ABS_LAT_ACCEL),
            _within_bound(magnitude_jerk, -MAX_ABS_MAG_JERK, MAX_ABS_MAG_JERK),
            _within_bound(longitude_jerk, -MAX_ABS_LON_JERK, MAX_ABS_LON_JERK),
            _within_bound(yaw_acceleration, -MAX_ABS_YAW_ACCEL, MAX_ABS_YAW_ACCEL),
            _within_bound(yaw_rate, -MAX_ABS_YAW_RATE, MAX_ABS_YAW_RATE),
        ],
        axis=-1,
    )
