import numpy as np

from scripts.training.evaluate_momworld_gtrs_scorer_all_navtrain import (
    _metrics,
    _select,
    _zscore,
)


def test_selection_falls_back_to_finite_candidates() -> None:
    score = np.asarray([[1.0, 4.0, 3.0]], dtype=np.float32)
    finite = np.asarray([[True, True, False]])
    risk = np.asarray([[0.9, 0.8, 0.0]], dtype=np.float32)
    selected = _select(score, finite, risk, threshold=0.5)
    assert selected.tolist() == [1]


def test_gate_metrics_and_zscore_are_finite() -> None:
    values = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)
    normalized = _zscore(values)
    np.testing.assert_allclose(normalized.mean(axis=1), 0.0, atol=1e-6)
    selected = np.asarray([2])
    report = _metrics(
        selected,
        values,
        np.asarray([[0.0, 0.2, 0.7]], dtype=np.float32),
        np.asarray([0]),
    )
    assert report["mean_target_score"] == 3.0
    assert report["target_collision_rate_ge_0_5"] == 1.0
    assert report["changed_vs_reference_rate"] == 1.0
