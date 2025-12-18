import numpy as np
import pytest

from ssmproxy.novelty import compute_novelty


def test_block_diagonal_peak_detection():
    # Construct a block-diagonal SSM with a clear boundary
    block = np.ones((3, 3))
    zero = np.zeros((3, 3))
    ssm = np.block([[block, zero], [zero, block]])

    L = 1
    result = compute_novelty(ssm, L)

    assert result.novelty.shape == (6,)
    assert np.all(result.novelty >= 0.0)
    assert np.all(result.novelty <= 1.0)

    # Expect a strong novelty response near the block boundary
    assert any(peak in (2, 3) for peak in result.peaks)
    assert len(result.peaks) >= 1
    assert result.stats["peak_rate"] == pytest.approx(len(result.peaks) / ssm.shape[0])
    assert result.stats["prom_mean"] >= 0.10
    assert result.stats["prom_median"] >= 0.10


def test_interval_stats_single_peak():
    block = np.ones((2, 2))
    ssm = np.block([[block, np.zeros((2, 2))], [np.zeros((2, 2)), block]])
    result = compute_novelty(ssm, L=1)

    assert result.stats["interval_mean"] == 0.0
    assert result.stats["interval_cv"] == 0.0
