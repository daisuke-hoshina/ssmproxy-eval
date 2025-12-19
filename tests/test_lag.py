import math

from ssmproxy.lag import compute_lag_energy


def _build_periodic_ssm(num_bars: int = 32, period: int = 8) -> list[list[float]]:
    base = 0.05
    strong = 0.95
    secondary = 0.6

    ssm = [[base for _ in range(num_bars)] for _ in range(num_bars)]

    # Identity diagonal
    for i in range(num_bars):
        ssm[i][i] = 1.0

    # Strong periodicity at ``period`` and a weaker one at twice the period.
    for lag, value in ((period, strong), (period * 2, secondary)):
        for i in range(num_bars - lag):
            ssm[i][i + lag] = value
            ssm[i + lag][i] = value

    return ssm


def test_detects_strongest_periodic_lag():
    ssm = _build_periodic_ssm()
    energy, best_lag, lag_energies = compute_lag_energy(ssm, top_k=2, return_full=True)

    assert best_lag == 8
    assert math.isclose(energy, 0.95 + 0.6)
    assert all(lag_energies[i] is None for i in range(4))
    assert math.isclose(lag_energies[8], 0.95)
    assert math.isclose(lag_energies[16], 0.6)


def test_respects_minimum_lag_threshold():
    ssm = _build_periodic_ssm()
    energy, best_lag, lag_energies = compute_lag_energy(ssm, min_lag=9, return_full=True)

    assert best_lag == 16
    assert math.isclose(energy, 0.6)
    assert lag_energies[8] is None
