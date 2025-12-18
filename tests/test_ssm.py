import math

import pytest

from ssmproxy.ssm import compute_ssm


def test_symmetry_and_diagonal():
    # Two bars with distinct non-zero features should produce a symmetric matrix
    pch = [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    onh = [
        [1.0] + [0.0] * 15,
        [0.0, 1.0] + [0.0] * 14,
    ]

    ssm = compute_ssm(pch, onh)

    assert ssm[0][0] == pytest.approx(1.0)
    assert ssm[1][1] == pytest.approx(1.0)
    assert ssm[0][1] == pytest.approx(ssm[1][0])
    assert 0.0 <= ssm[0][1] <= 1.0


def test_silent_bar_similarity():
    # A silent bar (zero vectors) should yield zero similarity to others and zero diagonal
    pch = [
        [1.0] + [0.0] * 11,
        [0.0] * 12,
    ]
    onh = [
        [1.0] + [0.0] * 15,
        [0.0] * 16,
    ]

    ssm = compute_ssm(pch, onh)

    assert ssm[1][1] == pytest.approx(0.0)
    assert ssm[0][1] == pytest.approx(0.0)
    assert ssm[1][0] == pytest.approx(0.0)


def test_mapping_to_unit_interval():
    # Using the unit-interval mapping should transform similarity scores accordingly
    pch = [
        [1.0] + [0.0] * 11,
        [0.0, 1.0] + [0.0] * 10,
    ]
    onh = [
        [1.0] + [0.0] * 15,
        [0.0, 1.0] + [0.0] * 14,
    ]

    ssm_raw = compute_ssm(pch, onh, map_to_unit_interval=False)
    ssm_unit = compute_ssm(pch, onh, map_to_unit_interval=True)

    off_diag_raw = ssm_raw[0][1]
    off_diag_unit = ssm_unit[0][1]

    assert off_diag_unit == pytest.approx((off_diag_raw + 1.0) / 2.0)
    assert 0.0 <= off_diag_unit <= 1.0
    assert math.isclose(ssm_unit[0][0], 1.0)
    assert math.isclose(ssm_unit[1][1], 1.0)
