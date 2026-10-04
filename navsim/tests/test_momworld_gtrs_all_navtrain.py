from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from navsim.agents.momworld.momworld_model import RuleAwareTrajectoryScorer
from scripts.training.refit_momworld_gtrs_scorer_all_navtrain import (
    combine_all_unique_navtrain,
    gtrs_loss,
)
from scripts.training.train_momworld_rule_scorer_from_cache import CacheRecord


def _record(index: int, dataset: str) -> CacheRecord:
    rng = np.random.default_rng(index)
    features = rng.random((32, 10), dtype=np.float32)
    return CacheRecord(
        path=Path(f"{dataset}-{index}.npz"), token=f"token-{index}", dataset=dataset,
        features=features, target_v1=rng.random(32, dtype=np.float32),
        target_v2=rng.random(32, dtype=np.float32),
        target_collision=rng.random(32, dtype=np.float32),
        predicted_collision=features[:, 7], kinematic_penalty=features[:, 8],
        momentum_error=features[:, 9], normalized_base=np.zeros(32, dtype=np.float32),
        candidate_is_finite=np.ones(32, dtype=np.bool_),
    )


def test_all_navtrain_union_uses_each_validation_token_once() -> None:
    train = [_record(index, "train") for index in range(10)]
    val = [_record(index, "val") for index in range(5, 15)]
    combined, audit = combine_all_unique_navtrain({"train": train, "val": val})
    assert len(combined) == 15
    assert len({record.token for record in combined}) == 15
    assert audit["raw_train"] == 10
    assert audit["effective_train"] == 5
    assert audit["val"] == 10
    assert audit["overlap_excluded"] == 5


def test_gtrs_loss_is_finite_and_updates_both_protocol_heads() -> None:
    torch.manual_seed(4)
    model = RuleAwareTrajectoryScorer(SimpleNamespace(rule_scorer_hidden_dim=16))
    features = torch.rand(3, 32, 10)
    targets_v1 = torch.rand(3, 32)
    targets_v2 = torch.rand(3, 32)
    collision = torch.rand(3, 32)
    finite = torch.ones(3, 32, dtype=torch.bool)
    loss, metrics = gtrs_loss(model, features, targets_v1, targets_v2, collision, finite)
    assert torch.isfinite(loss)
    assert metrics["v1_listnet"] > 0
    assert metrics["v2_ranknet"] > 0
    assert metrics["v1_top1_cross_entropy"] > 0
    assert 0 <= metrics["v2_top1_accuracy"] <= 1
    assert metrics["v2_top1_regret"] >= 0
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_scene_normalized_scorer_appends_exact_candidate_zscores() -> None:
    model = RuleAwareTrajectoryScorer(
        SimpleNamespace(
            rule_scorer_hidden_dim=16,
            rule_scorer_input_mode="raw_plus_scene_zscore",
        )
    )
    features = torch.rand(2, 32, 10)
    captured = []
    handle = model.heads["v2"][0].register_forward_pre_hook(
        lambda _module, values: captured.append(values[0].detach().clone())
    )
    try:
        output = model(features, "v2")
    finally:
        handle.remove()
    assert output.shape == (2, 32)
    assert captured[0].shape == (2, 32, 20)
    expected = (features - features.mean(dim=1, keepdim=True)) / features.std(
        dim=1, keepdim=True, unbiased=False
    ).clamp(min=1e-4)
    torch.testing.assert_close(captured[0][..., :10], features)
    torch.testing.assert_close(captured[0][..., 10:], expected)


def test_invalid_scorer_input_mode_is_rejected() -> None:
    try:
        RuleAwareTrajectoryScorer(
            SimpleNamespace(
                rule_scorer_hidden_dim=16,
                rule_scorer_input_mode="unknown",
            )
        )
    except ValueError as error:
        assert "rule_scorer_input_mode" in str(error)
    else:
        raise AssertionError("invalid scorer input mode was accepted")
