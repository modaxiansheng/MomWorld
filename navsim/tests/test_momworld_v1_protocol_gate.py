import numpy as np
import torch

from navsim.agents.momworld.momworld_model import (
    v1_protocol_gate_candidates,
    v1_protocol_gate_features,
)

from scripts.training.train_momworld_v1_protocol_gate_all_navtrain import (
    protocol_gate_features,
    select_protocol_proposal,
)


def test_protocol_gate_features_are_online_and_finite() -> None:
    values = np.arange(2 * 3 * 10, dtype=np.float32).reshape(2, 3, 10) / 100.0
    base = np.asarray([[0.1, 0.4, 0.2], [0.3, 0.2, 0.5]], dtype=np.float32)
    proxy = np.asarray([[0.5, 0.2, 0.1], [0.2, 0.6, 0.3]], dtype=np.float32)
    incumbent = np.asarray([1, 2])
    proposals = np.asarray([[0, 2], [1, 0]])
    weights = np.asarray([0.0, 2.0], dtype=np.float32)
    result = protocol_gate_features(
        values, base, proxy, incumbent, proposals, weights, np.ones((2, 3), bool)
    )
    assert result.shape == (2, 2, 63)
    assert np.isfinite(result).all()
    np.testing.assert_allclose(result[:, 0, 13:23], 0.0)


def test_protocol_gate_abstains_unless_gain_and_safety_pass() -> None:
    gain = np.asarray([[0.003, 0.004], [0.001, 0.005]], dtype=np.float32)
    collision = np.asarray([[-0.1, 0.1], [-0.1, 0.2]], dtype=np.float32)
    proposals = np.asarray([[3, 4], [5, 6]])
    incumbent = np.asarray([1, 2])
    selected = select_protocol_proposal(gain, collision, proposals, incumbent)
    np.testing.assert_array_equal(selected, np.asarray([3, 2]))


def test_online_gate_features_match_navtrain_builder() -> None:
    generator = np.random.default_rng(20260811)
    values = generator.random((3, 5, 10), dtype=np.float32)
    base = generator.normal(size=(3, 5)).astype(np.float32)
    proxy = generator.normal(size=(3, 5)).astype(np.float32)
    finite = np.ones((3, 5), dtype=bool)
    weights = np.asarray([0.0, 0.8, 2.0, 3.2], dtype=np.float32)
    incumbent, proposals = v1_protocol_gate_candidates(
        torch.from_numpy(base),
        torch.from_numpy(proxy),
        torch.from_numpy(finite),
        1.6,
        torch.from_numpy(weights),
    )
    expected = protocol_gate_features(
        values,
        base,
        proxy,
        incumbent.numpy(),
        proposals.numpy(),
        weights,
        finite,
    )
    actual = v1_protocol_gate_features(
        torch.from_numpy(values),
        torch.from_numpy(base),
        torch.from_numpy(proxy),
        torch.from_numpy(finite),
        incumbent,
        proposals,
        torch.from_numpy(weights),
    )
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-6, atol=1e-6)
