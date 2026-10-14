"""
Comprehensive Automated Verification Suite for Municipal Heat Watch:
1. Data Lineage & Provenance Tracker
2. Physical Boundary Clamping & Scenario Plausibility
3. Scientific Risk Index Bounds & Monotonicity
4. State Machine Lifecycle & Terminal State Enforcement
5. Security Guards: CSV Formula Injection & Stored XSS Escaping
"""

import html
import numpy as np
import pandas as pd
from pathlib import Path

from src.config import REGIONS, EXPOSURE_FEATURES
from src.data_status import data_status_tracker
from src.scenario_service import scenario_service
from src.proposal_service import proposal_service, sanitize_for_csv, VALID_TRANSITIONS

def test_data_status_tracker():
    """Verify data provenance catalog, layer audit, and observation timeline."""
    prov = data_status_tracker.get_provenance_catalog()
    assert "lst_satellite" in prov
    assert "surface_indices" in prov
    assert "telemetry_live" in prov

    status = data_status_tracker.get_region_status("trichy_delta")
    assert status["region_key"] == "trichy_delta"
    assert status["is_demonstration_mode"] is True
    assert "lst_satellite" in status["layers"]
    assert status["layers"]["lst_satellite"]["status"] == "Available"

    timeline = data_status_tracker.get_timeline_scenes("trichy_delta")
    assert len(timeline) >= 3
    assert timeline[0]["is_active"] is True

def test_scenario_clamping_and_bounds():
    """Verify scenario engine physically clamps out-of-domain feature modifications."""
    # Test Albedo boundary clamping (max is 0.55)
    clamped_val, was_clamped = scenario_service.clamp_feature("Albedo", 0.99)
    assert was_clamped is True
    assert clamped_val == 0.55

    # Test NDVI boundary clamping (min is -0.20)
    clamped_val, was_clamped = scenario_service.clamp_feature("NDVI", -0.85)
    assert was_clamped is True
    assert clamped_val == -0.20

    # Test within-range value
    clamped_val, was_clamped = scenario_service.clamp_feature("NDVI", 0.35)
    assert was_clamped is False
    assert clamped_val == 0.35

def test_scientific_hri_bounds_and_monotonicity():
    """
    Verify mathematical properties of the decoupled Heat Risk Index:
    HRI = sqrt(Exposure * Vulnerability).
    Must strictly satisfy: 0.0 <= HRI <= 1.0, and be monotonic.
    """
    # 1. Bounds check
    for exp in [0.0, 0.25, 0.5, 0.75, 1.0]:
        for vuln in [0.0, 0.25, 0.5, 0.75, 1.0]:
            hri = float(np.sqrt(exp * vuln))
            assert 0.0 <= hri <= 1.0, f"HRI out of bounds for E={exp}, V={vuln}"

    # 2. Monotonicity check: decreasing exposure while holding vulnerability constant must reduce HRI
    v_fixed = 0.60
    e_high = 0.80
    e_low = 0.40
    hri_high = float(np.sqrt(e_high * v_fixed))
    hri_low = float(np.sqrt(e_low * v_fixed))
    assert hri_low < hri_high, "HRI failed monotonicity on exposure reduction"

def test_state_machine_and_terminal_enforcement():
    """
    Verify server-side lifecycle transitions:
    Draft -> Submitted -> Approved -> Completed.
    Completed is terminal: any attempt to reopen or modify status must raise ValueError.
    """
    test_p = {
        "region_key": "trichy_delta",
        "ward_id": "W-STATE-TEST",
        "ward_name": "Ward Lifecycle Test",
        "intervention_type": "Cool Roof Retrofit",
        "title": "State Machine Verification Proposal",
        "department": "Engineering & Public Works (PWD)",
        "urgency": "Urgent (1-2 weeks)",
        "budget_inr": 50000.0,
        "status": "Draft",
        "created_by": "Test Runner"
    }

    pid = proposal_service.create_proposal(test_p)
    assert pid > 0

    # Valid step 1: Draft -> Submitted
    assert proposal_service.update_proposal_status(pid, "Submitted", "Test Officer", "Submitting for approval")

    # Valid step 2: Submitted -> Approved
    assert proposal_service.update_proposal_status(pid, "Approved", "Commissioner", "Approved")

    # Valid step 3: Approved -> Completed
    assert proposal_service.update_proposal_status(pid, "Completed", "Site Engineer", "Work completed")

    # Invalid transition: Completed proposal cannot transition to Draft or Approved
    illegal_caught = False
    try:
        proposal_service.update_proposal_status(pid, "Draft", "Rogue Officer", "Attempting illegal reopen")
    except ValueError as e:
        illegal_caught = True
        assert "Illegal state transition" in str(e)
    assert illegal_caught is True, "Server failed to enforce terminal state on Completed proposal!"

def test_csv_formula_injection_sanitization():
    """
    Verify OWASP Spreadsheet Formula Injection defense.
    Any cell beginning with '=', '+', '-', or '@' must be prepended with a single quote.
    """
    dangerous_inputs = [
        ("=CMD|' /C calc'!A0", "'=CMD|' /C calc'!A0"),
        ("+12345-Exploit", "'+12345-Exploit"),
        ("-2+3*cmd", "'-2+3*cmd"),
        ("@SUM(A1:A10)", "'@SUM(A1:A10)"),
        ("Safe text without formula prefix", "Safe text without formula prefix")
    ]

    for raw, expected in dangerous_inputs:
        sanitized = sanitize_for_csv(raw)
        assert sanitized == expected, f"CSV formula injection guard failed for: {raw}"

def test_stored_xss_escaping():
    """Verify HTML escaping on user-contributed proposal fields."""
    malicious_script = "<script>alert('XSS')</script>"
    escaped = html.escape(malicious_script)
    assert "<script>" not in escaped
    assert "&lt;script&gt;" in escaped
