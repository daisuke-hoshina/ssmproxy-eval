
import pytest
from ssmproxy.config import _resolve_ssm_weights, get_run_defaults

def test_resolve_ssm_weights_prevents_double_counting():
    """
    Ensure that ssm_weights dictionary never contains both 'onh' and 'onh_bin'.
    It should prioritize 'onh_bin' but optionally respect 'weight_onh' from legacy config.
    """
    
    # Case 1: Default (no input)
    w = _resolve_ssm_weights({})
    # Defaults in config.py: onh_bin=0.1, onh=missing (by default implementation)
    assert "onh_bin" in w
    assert "onh" not in w
    
    # Case 2: Explicit weight_onh (legacy)
    # The requirement is: "Prefer onh_bin for backward compatibility. Only include one."
    # If user provides weight_onh=0.8, we want that to be effective (mapped to onh_bin usually)
    # OR if mapped to "onh", then "onh_bin" should be absent.
    
    cfg_legacy = {"weight_onh": 0.8}
    w_legacy = _resolve_ssm_weights(cfg_legacy)
    
    has_onh = "onh" in w_legacy
    has_onh_bin = "onh_bin" in w_legacy
    
    assert not (has_onh and has_onh_bin), "Should not have both onh and onh_bin"
    
    # Ideallly, if I set legacy, I expect the weight to be 0.8.
    assert w_legacy["onh_bin"] == 0.8, "Legacy weight_onh should map to onh_bin"
    assert "onh" not in w_legacy, "onh key should not exist"
    
    # Case 3: Explicit weight_onh_bin
    cfg_new = {"weight_onh_bin": 0.7}
    w_new = _resolve_ssm_weights(cfg_new)
    assert w_new["onh_bin"] == 0.7
    assert "onh" not in w_new
    
    # Case 4: Both present (New should win)
    cfg_both = {"weight_onh": 0.3, "weight_onh_bin": 0.9}
    w_both = _resolve_ssm_weights(cfg_both)
    assert w_both["onh_bin"] == 0.9, "weight_onh_bin should override weight_onh"

def test_get_run_defaults_structure():
    defaults = get_run_defaults({})
    weights = defaults["ssm_weights"]
    
    assert "onh" not in weights
    assert "onh_bin" in weights
    # Also density should be there
    assert "density" in weights

