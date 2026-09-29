"""Contract for `crc_threshold`'s infeasible-alpha fallback (R4.1, C40/C41).

Given calibration scores and alpha such that alpha*(n+1) < 1 (too few points
for the requested confidence to be feasible as a finite-sample order statistic),
`crc_threshold` must return exactly 0.0 (the most conservative threshold:
accept everything), not the empirical minimum score or any other order
statistic. For alpha*(n+1) >= 1, the existing order-statistic behavior must be
byte-for-byte unchanged: index = floor(alpha*(n+1) - 1) clamped to [0, n-1],
returning sorted(scores)[index].

Source: plans/round-04/plan-a.md, claims C40 and C41; src/calibration.py:57-67.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.calibration import crc_threshold


# ---------------------------------------------------------------------------
# Infeasible regime: alpha * (n + 1) < 1 must return exactly 0.0.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n", "alpha"),
    [
        (5, 0.1),   # alpha*(n+1) = 0.6
        (9, 0.05),  # alpha*(n+1) = 0.5
        (2, 0.2),   # alpha*(n+1) = 0.6
        (1, 0.4),   # alpha*(n+1) = 0.8
    ],
)
def test_infeasible_alpha_returns_exact_zero_regardless_of_score_distribution(n: int, alpha: float) -> None:
    """Given n, alpha with alpha*(n+1) < 1, when crc_threshold is called with two
    different synthetic score arrays of size n, then both return exactly 0.0.

    This is the primary mutation target: a mutant that instead returns
    sorted(scores)[0] (the current buggy behavior, clamping the negative index
    to 0) would return the arrays' minimum score here, not 0.0. Using two very
    different distributions (all-high vs all-low) rules out either array's
    minimum happening to equal 0.0 by coincidence.
    """
    assert alpha * (n + 1) < 1.0, "test setup invariant: must be in the infeasible regime"

    high_scores = np.linspace(0.9, 0.99, n)
    low_scores = np.linspace(0.01, 0.1, n)

    assert crc_threshold(high_scores, alpha) == 0.0
    assert crc_threshold(low_scores, alpha) == 0.0


def test_infeasible_alpha_returns_zero_even_when_zero_is_not_in_the_score_array() -> None:
    """Given scores that contain no value anywhere near 0.0, when alpha*(n+1) < 1,
    then crc_threshold still returns exactly 0.0 -- proving the result is a
    fallback constant, not an interpolation or clamp derived from the data.
    """
    n, alpha = 5, 0.1
    assert alpha * (n + 1) < 1.0
    scores = np.array([0.55, 0.60, 0.65, 0.70, 0.75])

    result = crc_threshold(scores, alpha)

    assert result == 0.0
    assert result not in scores


def test_infeasible_alpha_single_calibration_point() -> None:
    """Given the smallest legal calibration set (n=1) and an alpha too small to
    make even one point feasible (alpha*(2) < 1 i.e. alpha < 0.5), when
    crc_threshold is called, then it returns exactly 0.0 rather than the sole
    score in the array.
    """
    n, alpha = 1, 0.3
    assert alpha * (n + 1) < 1.0
    scores_a = np.array([0.12])
    scores_b = np.array([0.98])

    assert crc_threshold(scores_a, alpha) == 0.0
    assert crc_threshold(scores_b, alpha) == 0.0


# ---------------------------------------------------------------------------
# Feasible regime: alpha * (n + 1) >= 1 must be byte-for-byte unchanged.
# ---------------------------------------------------------------------------


def test_feasible_boundary_alpha_times_n_plus_one_equals_one_selects_index_zero() -> None:
    """Given n=4, alpha=0.2 (alpha*(n+1) = 1.0 exactly, the feasibility boundary),
    when crc_threshold is called, then it returns sorted(scores)[0] -- the
    hand-computed index floor(1.0 - 1.0) = 0, clamped to [0, n-1] = [0, 3].

    This pins the boundary itself into the feasible (order-statistic) branch,
    not the infeasible (0.0) branch, since C40 requires strict "< 1" for the
    fallback.
    """
    n, alpha = 4, 0.2
    assert math.isclose(alpha * (n + 1), 1.0)
    scores = np.array([0.7, 0.3, 0.9, 0.5])

    expected_index = min(max(math.floor(alpha * (n + 1) - 1.0), 0), n - 1)
    assert expected_index == 0
    expected_value = sorted(scores)[expected_index]

    result = crc_threshold(scores, alpha)

    assert result == expected_value
    assert result == 0.3


def test_feasible_just_above_boundary_selects_hand_computed_index() -> None:
    """Given n=5, alpha=0.2 (alpha*(n+1) = 1.2, just above the feasibility
    boundary), when crc_threshold is called, then it returns the order
    statistic at the hand-computed index floor(1.2 - 1) = 0, clamped to
    [0, n-1] = [0, 4] -- i.e. sorted(scores)[0], the calibration minimum.
    """
    n, alpha = 5, 0.2
    assert alpha * (n + 1) == pytest.approx(1.2)
    scores = np.array([0.44, 0.10, 0.77, 0.63, 0.29])

    expected_index = min(max(math.floor(alpha * (n + 1) - 1.0), 0), n - 1)
    assert expected_index == 0
    expected_value = sorted(scores)[expected_index]

    result = crc_threshold(scores, alpha)

    assert result == expected_value
    assert result == 0.10


def test_feasible_mid_range_selects_hand_computed_nonzero_index() -> None:
    """Given n=9, alpha=0.5 (alpha*(n+1) = 5.0, well into the feasible regime
    with a non-trivial index), when crc_threshold is called, then it returns
    the order statistic at the hand-computed index floor(5.0 - 1) = 4, clamped
    to [0, 8].

    This is the second mutation target: a mutant that shifts the feasible-case
    formula by one (e.g. drops the "-1" or changes the floor/clamp) would
    select index 3 or 5 instead of 4, returning a different sorted value from
    this 9-element array.
    """
    n, alpha = 9, 0.5
    assert alpha * (n + 1) == pytest.approx(5.0)
    scores = np.array([0.81, 0.12, 0.55, 0.63, 0.29, 0.90, 0.47, 0.38, 0.71])

    expected_index = min(max(math.floor(alpha * (n + 1) - 1.0), 0), n - 1)
    assert expected_index == 4
    expected_value = sorted(scores)[expected_index]

    result = crc_threshold(scores, alpha)

    assert result == expected_value
    assert result == 0.55


def test_feasible_upper_clamp_selects_last_order_statistic() -> None:
    """Given n=4, alpha=0.9 (alpha*(n+1) = 4.5, whose unclamped index 3 already
    sits at n-1), when crc_threshold is called, then it returns the maximum
    calibration score, hand-computed as sorted(scores)[min(3, n-1)] = sorted(scores)[3].

    This guards the upper clamp boundary of the feasible formula separately
    from the lower (index-0) boundary already covered above.
    """
    n, alpha = 4, 0.9
    assert alpha * (n + 1) == pytest.approx(4.5)
    scores = np.array([0.20, 0.95, 0.40, 0.60])

    expected_index = min(max(math.floor(alpha * (n + 1) - 1.0), 0), n - 1)
    assert expected_index == n - 1 == 3
    expected_value = sorted(scores)[expected_index]

    result = crc_threshold(scores, alpha)

    assert result == expected_value
    assert result == 0.95
