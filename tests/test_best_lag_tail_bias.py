
import unittest
import numpy as np
from ssmproxy.lag import compute_lag_energy

class TestBestLagTailBias(unittest.TestCase):
    def test_tail_bias_lcb(self):
        """Verify that LCB mode prevents selecting tail lags when signal is flat noise."""
        # Create an SSM that results in flat lag energies with noise
        # 100 bars. Constant SSM = 1.0 (flat energy 1.0)
        # Add random noise to make some lags accidentally higher
        B = 100
        # Perfect SSM would be all 1s.
        # Let's mock a diagonal-like structure where energy is roughly constant
        ssm = [[1.0] * B for _ in range(B)]
        
        # In a real noisy scenario, small support leads to high variance.
        # Constant signal (1.0) has 0 variance.
        # We need variance. Let's make the SSM noisy.
        rng = np.random.RandomState(42)
        ssm_noisy = rng.normal(1.0, 0.5, (B, B)).tolist()
        
        # Case 1: Mean mode (default). Small support (large lag) might get lucky high mean.
        # With min_support=None/Small, tail bias exists.
        # However, purely random normal(1, 0.5) implies mean is 1. Standard Error = 0.5/sqrt(N).
        # For N=1 (lag=99), SE=0.5. Range [0, 2].
        # For N=100 (lag=0), SE=0.05. Range [0.9, 1.1].
        # So "luck" is much more likely at tail.
        
        # We want to see if mode="mean" picks a large lag, and "lcb" picks a smaller/safer one.
        
        # Actually, let's just test that LCB penalizes tail.
        # We can construct a specific case.
        # Lag 10: mean=1.0, N=90. SE ~ 0.5/sqrt(90) ~ 0.05. LCB ~ 1.0 - 1.0*0.05 = 0.95
        # Lag 90: mean=1.2, N=10. SE ~ 0.5/sqrt(10) ~ 0.15. LCB ~ 1.2 - 1.0*0.15 = 1.05 (still higher?)
        # Let's make Lag 90 have high mean but high variance.
        
        # Construct specific lag diagonals
        # We can mock this by manipulating SSM manually or just trusting the math.
        # Let's trust the math implementation and verify it works on a noisy matrix.
        
        # Run with mode="mean"
        _, best_lag_mean, _, _ = compute_lag_energy(
             ssm_noisy, min_lag=4, max_lag=95, best_lag_mode="mean"
        )
        
        # Run with mode="lcb", z=1.0
        _, best_lag_lcb, _, _ = compute_lag_energy(
             ssm_noisy, min_lag=4, max_lag=95, best_lag_mode="lcb", best_lag_lcb_z=2.0
        )
        
        # We expect LCB to potentially pick a smaller lag (higher support) or at least different.
        # In random noise, ArgMax(Mean) tends to happen at N=small.
        # Let's verify 'best_lag_lcb' has higher support than 'best_lag_mean' typically.
        # Calculating support: B - lag.
        # Smaller lag = Higher support.
        
        # We expect best_lag_lcb < best_lag_mean (prefer smaller lag / higher support)
        # But random is random. Let's check statistics or just run once?
        # A single run might fail if seed is unlucky.
        
        # Let's use a deterministic construction.
        # Lag 10 (High Support): Mean 0.8, Std 0.1. N=90. SE=0.01. LCB=0.79.
        # Lag 90 (Low Support): Mean 0.9, Std 0.1. N=10. SE=0.03. LCB=0.87. 
        # Here Lag 90 wins both.
        
        # Construct:
        # Lag 10: Mean 0.8. N=90. SE=0.01. LCB=0.79.
        # Lag 90: Mean 0.85. N=10. SE=0.20. LCB=0.65.
        # Here Lag 90 wins Mean (0.85 > 0.8), but Lag 10 wins LCB (0.79 > 0.65).
        
        B = 100
        ssm_synth = [[0.0] * B for _ in range(B)]
        
        # Fill Lag 10 diagonal
        lag10 = 10
        count10 = B - lag10
        # Target mean 0.8, low variance
        vals10 = [0.8] * count10
        # Add tiny variance
        vals10[0] += 0.01
        vals10[1] -= 0.01
        for i, v in enumerate(vals10):
            ssm_synth[i][i+lag10] = v
            ssm_synth[i+lag10][i] = v
            
        # Fill Lag 90 diagonal
        lag90 = 90
        count90 = B - lag90 # 10
        # Target mean 0.85 (higher), high variance (SE large)
        # To get high SE with count 10, we need high std dev.
        # SE = std / sqrt(10). We want SE > (0.85-0.79)=0.06.
        # std > 0.06 * 3.16 = 0.19.
        # Let's make values oscillate: 0.85 +/- 0.3
        vals90 = [0.85 + (0.3 if i % 2 == 0 else -0.3) for i in range(count90)]
        for i, v in enumerate(vals90):
            ssm_synth[i][i+lag90] = v
            ssm_synth[i+lag90][i] = v
             
        # Run Mean
        _, best_lag_mean, _, _ = compute_lag_energy(
             ssm_synth, min_lag=4, best_lag_mode="mean"
        )
        self.assertEqual(best_lag_mean, 90, "Mean mode should pick higher energy despite low support")
        
        # Run LCB
        _, best_lag_lcb, _, _ = compute_lag_energy(
             ssm_synth, min_lag=4, best_lag_mode="lcb", best_lag_lcb_z=1.0
        )
        self.assertEqual(best_lag_lcb, 10, "LCB mode should pick high support candidate due to penalty on low support")

    def test_tie_breaking_small_lag(self):
        """Verify that tie breaking prefers smaller lag."""
        B = 50
        ssm = [[0.0] * B for _ in range(B)]
        
        # Lag 10: Energy 0.8 (perfectly constant)
        for i in range(B - 10):
            ssm[i][i+10] = 0.8
            
        # Lag 20: Energy 0.8 (perfectly constant)
        for i in range(B - 20):
            ssm[i][i+20] = 0.8
            
        # Tie epsilon 1e-6
        _, best_lag, _, _ = compute_lag_energy(
            ssm, min_lag=4, best_lag_mode="mean", best_lag_tie_eps=1e-5
        )
        
        self.assertEqual(best_lag, 10, "Should pick smaller lag (10) over larger (20) in a tie")

if __name__ == "__main__":
    unittest.main()
