import hashlib
from types import SimpleNamespace

import numpy as np
import torch

from navsim.agents.momworld.momworld_pairwise_gate import (
    V2NavhardPairwiseGate,
    _pair_features,
)
from scripts.training.refit_momworld_v2_navhard_pairwise_gate import pair_features


def test_pairwise_online_features_match_offline() -> None:
    rng = np.random.default_rng(20260812)
    features = rng.random((17, 32, 10), dtype=np.float32)
    base = rng.normal(size=(17, 32)).astype(np.float32)
    proxy = rng.normal(size=(17, 32)).astype(np.float32)
    combined = base + 3.2 * proxy
    incumbent = rng.integers(0, 32, size=17, dtype=np.int64)
    alternatives = rng.integers(0, 32, size=(17, 3), dtype=np.int64)
    offline = pair_features(
        torch.from_numpy(features),
        torch.from_numpy(base),
        torch.from_numpy(proxy),
        torch.from_numpy(combined),
        torch.from_numpy(incumbent),
        torch.from_numpy(alternatives),
    ).reshape(17, 3, 59)
    online = _pair_features(
        torch.from_numpy(features),
        torch.from_numpy(base),
        torch.from_numpy(proxy),
        torch.from_numpy(combined),
        torch.from_numpy(incumbent),
        torch.from_numpy(alternatives),
    ).numpy()
    np.testing.assert_array_equal(online, offline)


def test_pairwise_gate_loads_sealed_histogram_state(tmp_path) -> None:
    model = {
        "baseline": 0.2,
        "tree_count": 1,
        "max_nodes": 1,
        "max_depth": 1,
        "feature_idx": torch.zeros((1, 1), dtype=torch.long),
        "threshold": torch.zeros((1, 1), dtype=torch.float64),
        "missing_left": torch.zeros((1, 1), dtype=torch.bool),
        "left": torch.zeros((1, 1), dtype=torch.long),
        "right": torch.zeros((1, 1), dtype=torch.long),
        "is_leaf": torch.ones((1, 1), dtype=torch.bool),
        "value": torch.full((1, 1), 0.1, dtype=torch.float64),
    }
    state_path = tmp_path / "pairwise.pt"
    torch.save(
        {
            "schema": V2NavhardPairwiseGate.STATE_SCHEMA,
            "feature_dim": 59,
            "alternatives": 3,
            "min_predicted_gain": 0.25,
            "gain_model": model,
        },
        state_path,
    )
    digest = hashlib.sha256(state_path.read_bytes()).hexdigest()
    gate = V2NavhardPairwiseGate(
        SimpleNamespace(
            rule_v2_navhard_pairwise_gate_state_path=str(state_path),
            rule_v2_navhard_pairwise_gate_state_sha256=digest,
        )
    )
    prediction = gate._predict(torch.zeros((4, 3, 59)))
    torch.testing.assert_close(prediction, torch.full((4, 3), 0.3, dtype=torch.float64))

    features = torch.ones((4, 32, 10), dtype=torch.float32)
    base = torch.arange(32, dtype=torch.float32)[None].expand(4, -1)
    proxy = torch.zeros_like(base)
    finite = torch.ones_like(base, dtype=torch.bool)
    selected, gain, incumbent, alternatives = gate.select(
        features, base, proxy, finite
    )
    torch.testing.assert_close(gain, torch.full((4, 3), 0.3))
    torch.testing.assert_close(incumbent, torch.full((4,), 31, dtype=torch.long))
    torch.testing.assert_close(
        alternatives,
        torch.tensor([[30, 29, 28]], dtype=torch.long).expand(4, -1),
    )
    torch.testing.assert_close(selected, torch.full((4,), 30, dtype=torch.long))


def test_pairwise_gate_supports_and_masks_full_candidate_pool(tmp_path) -> None:
    model = {
        "baseline": 0.2,
        "tree_count": 1,
        "max_nodes": 1,
        "max_depth": 1,
        "feature_idx": torch.zeros((1, 1), dtype=torch.long),
        "threshold": torch.zeros((1, 1), dtype=torch.float64),
        "missing_left": torch.zeros((1, 1), dtype=torch.bool),
        "left": torch.zeros((1, 1), dtype=torch.long),
        "right": torch.zeros((1, 1), dtype=torch.long),
        "is_leaf": torch.ones((1, 1), dtype=torch.bool),
        "value": torch.full((1, 1), 0.1, dtype=torch.float64),
    }
    state_path = tmp_path / "wide_pairwise.pt"
    torch.save(
        {
            "schema": V2NavhardPairwiseGate.STATE_SCHEMA,
            "feature_dim": 59,
            "alternatives": 31,
            "min_predicted_gain": 0.25,
            "gain_model": model,
        },
        state_path,
    )
    gate = V2NavhardPairwiseGate(
        SimpleNamespace(
            rule_v2_navhard_pairwise_gate_state_path=str(state_path),
            rule_v2_navhard_pairwise_gate_state_sha256=hashlib.sha256(
                state_path.read_bytes()
            ).hexdigest(),
        )
    )

    features = torch.ones((2, 32, 10), dtype=torch.float32)
    features[:, 30, 1] = 0.0
    base = torch.arange(32, dtype=torch.float32)[None].expand(2, -1)
    proxy = torch.zeros_like(base)
    finite = torch.ones_like(base, dtype=torch.bool)
    selected, gain, incumbent, alternatives = gate.select(
        features, base, proxy, finite
    )

    assert gain.shape == (2, 31)
    assert alternatives.shape == (2, 31)
    assert torch.isneginf(gain).sum().item() == 2
    torch.testing.assert_close(incumbent, torch.full((2,), 31, dtype=torch.long))
    torch.testing.assert_close(selected, torch.full((2,), 29, dtype=torch.long))
