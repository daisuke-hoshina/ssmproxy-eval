
import math
import sys
from pathlib import Path
import unittest

# Ensure we can import from ssmproxy
sys.path.append(str(Path.cwd()))

from ssmproxy.lag import (
    compute_lag_prominence,
    estimate_base_period_comb,
    compute_hierarchy_index_auto_slope,
    compute_hierarchy_index_auto_adjdiff
)

class TestLagMetrics(unittest.TestCase):
    
    def test_compute_prominence(self):
        # Setup: A peak at index 10 surrounded by lower values
        lag_energies = [0.1] * 20
        lag_energies[10] = 0.5 # Peak
        
        # Window adaptive: For lag 10, window = max(2, min(16, 5)) = 5
        # Neighborhood [5, 15], median is 0.1
        # Prominence should be 0.5 - 0.1 = 0.4
        
        prom = compute_lag_prominence(lag_energies)
        self.assertAlmostEqual(prom[10], 0.4)
        self.assertAlmostEqual(prom[9], 0.0)
        
    def test_comb_filter_hierarchical(self):
        # Scenario: Peaks at 8, 16, 32 (Hierarchical structure, L0=8)
        # Prominence map
        prom = [0.0] * 50
        prom[8] = 0.5
        prom[16] = 0.5
        prom[32] = 0.5
        
        # Energy (dummy, not used for candidates if prominence peaks exist)
        energies = [0.1] * 50
        
        l0, fallback, _ = estimate_base_period_comb(energies, prom, min_lag=4)
        
        # Should pick 8 because harmonics 1, 2, 4 align
        self.assertEqual(l0, 8)
        self.assertFalse(fallback)
        
    def test_comb_filter_repeat_vs_random(self):
        # Scenario: Peak ONLY at 13 (Random/Prime repeat)
        prom = [0.0] * 50
        prom[13] = 0.8
        
        energies = [0.1] * 50
        energies[13] = 0.9 # Best energy
        
        l0, fallback, _ = estimate_base_period_comb(energies, prom, min_lag=4, min_hits=1)
        
        # Harmonics [1,2,4,8] -> 13, 26, 52.. 
        # 13 has prom 0.8. 26 is 0. 
        # Score = 1.0*0.8 + 0 + ... = 0.8
        
        # Is this enough to beat fallback? Fallback means NO candidates found or NO score.
        # But we DO have a candidate at 13 (since it's a prominence peak).
        # And we set best_lag_idx = 13.
        # So it should return 13. Is it fallback? 
        # Code says: if candidates exist and we scored them, returns best_l0.
        # It's not a fallback if logic succeeded, even if score is low-ish.
        # BUT: fallback flag is defined as "did we fail to find ANY alignment?".
        # In current impl, it returns False (success) as long as candidates existed.
        
        self.assertEqual(l0, 13)
        self.assertFalse(fallback)
        
    def test_comb_fallback_if_no_peaks(self):
        # Flat energies
        prom = [0.0] * 50
        energies = [0.1] * 50
        energies[5] = 0.15 # Slight max
        
        # compute_lag_prominence will likely return 0s if perfectly flat or near flat.
        
        l0, fallback, _ = estimate_base_period_comb(energies, prom, min_lag=4)
        
        self.assertIsNone(l0) # New behavior: return None on failure
        self.assertTrue(fallback)

    def test_hierarchy_slope_hierarchical(self):
        # L0 = 8. Multipliers 1, 2, 4
        # Prominence grows: P[8]=0.2, P[16]=0.4, P[32]=0.6
        # log2(m): 0, 1, 2
        # y: 0.2, 0.4, 0.6
        # Slope should be 0.2
        
        prom = [None] * 100
        prom[8] = 0.2
        prom[16] = 0.4
        prom[32] = 0.6
        # index 64 remains None, so it should be skipped
        
        slope = compute_hierarchy_index_auto_slope(prom, l0=8, max_levels=4) # 1,2,4,8
        self.assertAlmostEqual(slope, 0.2)
        
    def test_hierarchy_slope_fine_grained(self):
        # L0 = 4. Multipliers 1, 2, 4 (scale up)
        # Prominence decays: P[4]=0.8, P[8]=0.3, P[16]=0.1
        # log2(m): 0, 1, 2
        # y: 0.8, 0.3, 0.1
        # Slope should be negative (~ -0.35)
        
        prom = [None] * 100
        prom[4] = 0.8
        prom[8] = 0.3
        prom[16] = 0.1
        
        slope = compute_hierarchy_index_auto_slope(prom, l0=4, max_levels=4)
        self.assertLess(slope, 0.0)

    def test_length_dependence(self):
        # Verify that adding trailing Nones (simulating running out of data)
        # doesn't drastically flip the sign, provided we have enough points.
        
        prom = [None] * 100
        l0 = 8
        for i in [1, 2, 4]:
            prom[l0*i] = 0.5 # Flat/Repeated structure
            
        slope_full = compute_hierarchy_index_auto_slope(prom, l0, max_levels=5)
        self.assertAlmostEqual(slope_full, 0.0) # Flat
        
        # Now cut off 4*L0 (idx 32)
        prom_cut = list(prom)
        prom_cut[32] = None # Simulating end of validity
        
        # With 1, 2 still valid (points (0, 0.5), (1, 0.5))
        slope_cut = compute_hierarchy_index_auto_slope(prom_cut, l0, max_levels=5)
        self.assertAlmostEqual(slope_cut, 0.0) # Still flat

if __name__ == '__main__':
    unittest.main()
