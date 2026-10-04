from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from nuplan.planning.simulation.trajectory.trajectory_sampling import (
    TrajectorySampling,
)

from navsim.common.dataclasses import Trajectory
from navsim.planning.script.run_pdm_score_gpu_v1 import (
    SCORE_CACHE_VERSION,
    TrajectoryPredictionModule,
    _load_cached_trajectories,
    _load_cached_score,
    _prediction_cache_file,
    _save_cached_score,
    _save_cached_trajectory,
)


def _trajectory(offset: float) -> Trajectory:
    sampling = TrajectorySampling(time_horizon=4.0, interval_length=0.1)
    poses = np.full((sampling.num_poses, 3), offset, dtype=np.float32)
    return Trajectory(poses=poses, trajectory_sampling=sampling)


def test_prediction_cache_round_trip_and_corrupt_file_recovery(
    tmp_path: Path,
) -> None:
    _save_cached_trajectory(tmp_path, "token-a", _trajectory(1.0))
    _save_cached_trajectory(tmp_path, "token-b", _trajectory(2.0))
    (tmp_path / "token-corrupt.pkl").write_bytes(b"not-a-pickle")

    loaded = _load_cached_trajectories(
        tmp_path, ["token-a", "token-b", "token-corrupt", "token-missing"]
    )

    assert set(loaded) == {"token-a", "token-b"}
    np.testing.assert_allclose(loaded["token-a"].poses, 1.0)
    assert not list(tmp_path.glob("*.tmp"))


def test_prediction_module_persists_each_token_for_resume(tmp_path: Path) -> None:
    sampling = TrajectorySampling(time_horizon=4.0, interval_length=0.1)
    agent = SimpleNamespace(
        _trajectory_sampling=sampling,
        forward=lambda features: {
            "trajectory": torch.zeros(2, sampling.num_poses, 3)
        },
    )
    module = TrajectoryPredictionModule(agent, tmp_path)

    returned = module.predict_step(({}, {}, ["token-a", "token-b"]), 0)
    loaded = _load_cached_trajectories(tmp_path, ["token-a", "token-b"])

    assert returned == {}
    assert set(loaded) == {"token-a", "token-b"}


def test_score_cache_reuses_only_valid_versioned_rows(tmp_path: Path) -> None:
    row = {"token": "token-a", "valid": True, "score": 0.75}
    _save_cached_score(tmp_path, "token-a", row)

    assert _load_cached_score(tmp_path, "token-a") == row

    cache_file = tmp_path / "token-a.pkl"
    cache_file.write_bytes(b"corrupt")
    assert _load_cached_score(tmp_path, "token-a") is None

    _save_cached_score(
        tmp_path, "token-a", {"token": "token-a", "valid": False}
    )
    assert _load_cached_score(tmp_path, "token-a") is None
    assert SCORE_CACHE_VERSION


@pytest.mark.parametrize("token", ["", ".", "..", "../escape", "a/b"])
def test_prediction_cache_rejects_unsafe_tokens(tmp_path: Path, token: str) -> None:
    with pytest.raises(ValueError):
        _prediction_cache_file(tmp_path, token)
