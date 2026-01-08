import pytest

from ssmproxy.novelty import compute_novelty


def _block_diagonal_ssm(block_size: int) -> list[list[float]]:
    block = [[1.0 for _ in range(block_size)] for _ in range(block_size)]
    zero = [[0.0 for _ in range(block_size)] for _ in range(block_size)]
    top = [block_row + zero_row for block_row, zero_row in zip(block, zero)]
    bottom = [zero_row + block_row for zero_row, block_row in zip(zero, block)]
    return top + bottom


def test_block_diagonal_peak_detection():
    ssm = _block_diagonal_ssm(3)

    L = 1
    result = compute_novelty(ssm, L)

    assert len(result.novelty) == 6
    assert all(0.0 <= value <= 1.0 for value in result.novelty)

    # Expect a strong novelty response near the block boundary
    assert any(peak in (2, 3) for peak in result.peaks)
    assert len(result.peaks) >= 1
    valid_len = max(0, len(ssm) - 2 * L)
    assert result.stats["peak_rate"] == pytest.approx(len(result.peaks) / valid_len)
    assert result.stats["peak_rate_raw"] == pytest.approx(len(result.peaks) / len(ssm))
    assert result.stats["prom_mean"] >= 0.10
    assert result.stats["prom_median"] >= 0.10


def test_interval_stats_single_peak():
    ssm = _block_diagonal_ssm(2)
    result = compute_novelty(ssm, L=1)

    assert result.stats["interval_mean"] == 0.0
    assert result.stats["interval_cv"] == 0.0
