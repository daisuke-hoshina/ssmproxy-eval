import pytest
from ssmproxy.lag import estimate_base_period

def test_estimate_base_period_simple_peak():
    # lags: 0, 1, 2, 3, 4, 5...
    # Energy: 0, 0, 0.2, 0.8, 0.4, ...
    # Peak at lag 3
    lag_energies = [0.0, 0.0, 0.2, 0.8, 0.4, 0.1]
    # min_lag=1
    l0 = estimate_base_period(lag_energies, min_lag=1)
    assert l0 == 3

def test_estimate_base_period_multiple_peaks():
    # Peak at 3 (0.8), Peak at 10 (0.9)
    # Since 0.9 > 0.8, both are candidates.
    # Top 5 will include both.
    # We pick smallest lag -> 3.
    lag_energies = [0.0] * 20
    lag_energies[3] = 0.8
    lag_energies[2] = 0.5
    lag_energies[4] = 0.5
    
    lag_energies[10] = 0.9
    lag_energies[9] = 0.5
    lag_energies[11] = 0.5
    
    l0 = estimate_base_period(lag_energies, min_lag=1)
    assert l0 == 3

def test_estimate_base_period_fallback_max():
    # Monotonic increasing, no local peak
    lag_energies = [0.1, 0.2, 0.3, 0.4, 0.5]
    l0 = estimate_base_period(lag_energies, min_lag=1)
    assert l0 == 4 # Max valid index

def test_estimate_base_period_min_lag_peak():
    # Peak at min_lag (lag 2)
    # E[2]=0.8, E[3]=0.5
    lag_energies = [0.0, 0.0, 0.8, 0.5, 0.2]
    l0 = estimate_base_period(lag_energies, min_lag=2)
    assert l0 == 2

def test_estimate_base_period_none_values():
    # Some None values
    lag_energies = [None, None, 0.5, 0.8, 0.4, None]
    l0 = estimate_base_period(lag_energies, min_lag=2)
    assert l0 == 3

def test_estimate_base_period_plateau():
    # 0.5, 0.8, 0.8, 0.5
    # indices 1, 2, 3, 4
    # local peak logic:
    # 2: left(1)=0.5 < 0.8. right(3)=0.8 <= 0.8 (left peak condition met, right strict check?)
    # code: right check: if energy <= right_energy: is_peak=False
    # So 2 is NOT a peak because 0.8 <= 0.8.
    # 3: left(2)=0.8 !< 0.8 -> is_peak=False (left check: energy < left implied NOT peak? check logic)
    # Code: if energy < left: is_peak = False. 0.8 is not < 0.8. So continues.
    # right(4)=0.5. 3 > 4 condition? 
    # Code: if energy <= right: is_peak = False. 0.8 is not <= 0.5. So remains True.
    # So 3 is a peak.
    lag_energies = [0.5, 0.5, 0.8, 0.8, 0.5]
    l0 = estimate_base_period(lag_energies, min_lag=1)
    assert l0 == 3

