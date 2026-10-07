"""Explicit summaries of the existing twelve half-second planning metrics.

This module has no framework dependencies. Inputs to ``cumulative_metric_summary``
are the evaluator's already-computed prefix means, not endpoint errors. Collision
values remain fractions here; conversion to percent belongs only in display code.
"""

import math
from numbers import Real


def validate_valid_samples(valid_samples):
    """Reject an empty metric population instead of reporting an artificial zero."""
    if (isinstance(valid_samples, bool) or not isinstance(valid_samples, Real)
            or not math.isfinite(float(valid_samples))
            or valid_samples <= 0 or int(valid_samples) != valid_samples):
        raise ValueError("Planning metrics require a positive integer valid-sample count")
    return int(valid_samples)


def validate_metric_values(half_second_values, valid_samples):
    """Require exactly twelve finite scalar observations at 0.5s through 6s."""
    validate_valid_samples(valid_samples)
    values = tuple(half_second_values)
    if len(values) != 12:
        raise ValueError("Planning metrics require exactly 12 half-second values")
    if any(isinstance(value, bool) or not isinstance(value, Real)
           or not math.isfinite(float(value)) for value in values):
        raise ValueError("Planning metric values must be finite real scalars")
    return tuple(float(value) for value in values)


def cumulative_metric_summary(metric_name, cumulative_half_second_values,
                              valid_samples):
    """Return flat scalar keys without changing the evaluator's cumulative means.

    The six horizon values select the original 1, 2, ..., 6 second prefix means.
    The 1s-to-6s and 4s-to-6s summaries are arithmetic means of those selected
    values, not means of the twelve instantaneous errors or endpoint metrics.
    """
    if not isinstance(metric_name, str) or not metric_name:
        raise ValueError("A nonempty metric name is required")
    values = validate_metric_values(cumulative_half_second_values, valid_samples)
    horizons = values[1::2]
    result = {
        metric_name + "_cumulative_%ds" % horizon: value
        for horizon, value in enumerate(horizons, start=1)
    }
    result[metric_name + "_cumulative_mean_1s_to_6s"] = sum(horizons) / 6
    result[metric_name + "_cumulative_mean_4s_to_6s"] = sum(horizons[3:]) / 3
    if any(not math.isfinite(value) for value in result.values()):
        raise ValueError("Planning metric summary must remain finite")
    return result
