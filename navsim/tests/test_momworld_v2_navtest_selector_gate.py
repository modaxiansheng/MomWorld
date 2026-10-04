import numpy as np
import torch

from navsim.agents.momworld.momworld_model import (
    v2_navhard_candidate_selections,
    v2_navtest_gate_features,
)
from scripts.training.train_momworld_v2_navtest_selector_gate_all_navtrain import (
    select_proposal,
    selector_features,
)
from scripts.training.tune_momworld_candidate_safety_rule_all_navtrain import (
    apply_safety_rule,
    neutral_rule,
)


def test_v2_selector_features_are_finite() -> None:
    rng = np.random.default_rng(3)
    features = rng.random((2, 4, 10), dtype=np.float32)
    base = rng.normal(size=(2, 4)).astype(np.float32)
    proxy = rng.normal(size=(2, 4)).astype(np.float32)
    learned = rng.random((2, 4), dtype=np.float32)
    incumbent = np.asarray([0, 1])
    proposals = np.asarray([[1, 2], [3, 0]])
    configs = np.asarray([[1, 2, .1, 0, 0], [0, 0, 0, 2.4, 0]], np.float32)
    result = selector_features(features, base, proxy, learned, np.ones((2, 4), bool), incumbent, proposals, configs)
    assert result.shape == (2, 2, 68)
    assert np.isfinite(result).all()


def test_v2_selector_abstains_on_gain_or_collision() -> None:
    gain = np.asarray([[.003, .005], [.001, .004]], np.float32)
    collision = np.asarray([[.001, .003], [-.1, .003]], np.float32)
    proposals = np.asarray([[2, 3], [4, 5]])
    incumbent = np.asarray([0, 1])
    np.testing.assert_array_equal(select_proposal(gain, collision, proposals, incumbent), [2, 1])


def test_v2_selector_online_features_match_offline() -> None:
    rng = np.random.default_rng(17)
    features = rng.random((7, 11, 10), dtype=np.float32)
    base = rng.normal(size=(7, 11)).astype(np.float32)
    proxy = rng.normal(size=(7, 11)).astype(np.float32)
    learned = rng.random((7, 11), dtype=np.float32)
    finite = rng.random((7, 11)) > 0.15
    finite[:, 0] = True
    finite[0] = False
    incumbent = rng.integers(0, 11, size=7, dtype=np.int64)
    proposals = rng.integers(0, 11, size=(7, 9), dtype=np.int64)
    configs = rng.normal(size=(9, 5)).astype(np.float32)
    offline = selector_features(
        features, base, proxy, learned, finite, incumbent, proposals, configs
    )
    online = v2_navtest_gate_features(
        torch.from_numpy(features),
        torch.from_numpy(base),
        torch.from_numpy(proxy),
        torch.from_numpy(learned),
        torch.from_numpy(finite),
        torch.from_numpy(incumbent),
        torch.from_numpy(proposals),
        torch.from_numpy(configs),
    ).numpy()
    np.testing.assert_allclose(online, offline, rtol=1e-6, atol=2e-6)


def test_v2_navhard_online_candidates_match_offline_safety_rule() -> None:
    rng = np.random.default_rng(29)
    features = rng.random((31, 32, 10), dtype=np.float32)
    base = rng.normal(size=(31, 32)).astype(np.float32)
    proxy = rng.normal(size=(31, 32)).astype(np.float32)
    finite = rng.random((31, 32)) > 0.12
    finite[0] = False
    configurations = np.asarray(
        [
            (0.0, 0.0, 0.0, 2.4, 0.435),
            (0.0, 0.0, 0.0, 2.8, 0.435),
            (0.0, 0.0, 0.0, 3.6, 0.435),
            (0.0, 0.0, 0.0, 4.0, 0.435),
            (0.0, 0.0, 0.0, 3.2, 0.0),
            (0.0, 0.0, 0.0, 3.2, 0.4),
            (0.0, 0.0, 0.0, 3.2, 0.425),
            (0.0, 0.0, 0.0, 3.2, 0.445),
            (0.0, 0.0, 0.0, 3.2, 0.46),
            (0.0, 0.0, 0.0, 2.8, 0.425),
            (0.0, 0.0, 0.0, 3.6, 0.425),
            (0.0, 0.0, 0.0, 2.8, 0.445),
            (0.0, 0.0, 0.0, 3.6, 0.445),
        ],
        dtype=np.float32,
    )

    def eligibility(progress: float) -> np.ndarray:
        parameters = neutral_rule()
        parameters.update(
            drivable_area_min=0.55, lane_min=0.55, progress_min=progress
        )
        return apply_safety_rule(features, finite, parameters)[0]

    offline_incumbent = np.where(
        eligibility(0.435), base + 3.2 * proxy, -1e9
    ).argmax(axis=1)
    offline_proposals = np.stack(
        [
            np.where(
                eligibility(float(configuration[4])),
                base + float(configuration[3]) * proxy,
                -1e9,
            ).argmax(axis=1)
            for configuration in configurations
        ],
        axis=1,
    )
    online_incumbent, online_proposals = v2_navhard_candidate_selections(
        torch.from_numpy(features),
        torch.from_numpy(base),
        torch.from_numpy(proxy),
        torch.from_numpy(finite),
        torch.from_numpy(configurations),
    )
    np.testing.assert_array_equal(online_incumbent.numpy(), offline_incumbent)
    np.testing.assert_array_equal(online_proposals.numpy(), offline_proposals)
