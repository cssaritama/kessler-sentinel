"""Unit tests for FastAPI conjunction triage microservice endpoints."""

import pytest
from fastapi.testclient import TestClient

from predict import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert data["service"] == "kessler-sentinel-triage"
    assert data["version"] == "1.0.0"
    assert data["model_loaded"] is True
    assert "decision_threshold" in data


def test_root_console_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "KESSLER-SENTINEL" in response.text
    assert "Conjunction Triage Console" in response.text


def test_predict_critical_conjunction():
    payload = {
        "conjunction_id": "CDM-TEST-CRITICAL-001",
        "orbital_regime_band": "Sun_Synchronous_LEO",
        "chaser_object_type": "Fragmentation_Debris",
        "primary_propulsion_class": "Monopropellant_Hydrazine",
        "solar_weather_storm_tier": "Severe_Geomagnetic_Storm_G3_Plus",
        "tracking_network_geometry": "Single_Pass_Radar_Sparse",
        "time_to_tca_hours": 36.0,
        "altitude_perigee_km": 485.0,
        "inclination_deg": 97.4,
        "relative_velocity_km_s": 14.8,
        "miss_distance_total_m": 42.0,
        "radial_separation_abs_m": 8.5,
        "in_track_separation_abs_m": 35.0,
        "cross_track_separation_abs_m": 21.0,
        "combined_sigma_radial_m": 12.0,
        "combined_sigma_in_track_m": 180.0,
        "combined_sigma_cross_track_m": 45.0,
        "chaser_rcs_m2": 2.5,
        "solar_flux_f10_7_sfu": 215.0,
        "geomagnetic_ap_index": 78.0,
        "cdm_update_count": 5,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["conjunction_event_id"] == "CDM-TEST-CRITICAL-001"
    assert data["maneuver_escalation_required"] is True
    assert "burn" in data["recommended_flight_action"].lower() or "radar" in data["recommended_flight_action"].lower()
    assert "mahalanobis_rtn_distance" in data["engineered_diagnostics"]


def test_predict_nominal_conjunction():
    # Use standard CASE 3 NOMINAL from test_service.py
    payload = {
        "conjunction_id": "CDM-2026-LEO-NOMINAL-003",
        "orbital_regime_band": "Equatorial_Low_Inclination",
        "chaser_object_type": "Uncharacterized_Tracklet",
        "primary_propulsion_class": "Hall_Effect_Ion_Electric",
        "solar_weather_storm_tier": "Quiet_Geomagnetic",
        "tracking_network_geometry": "Multi_Station_Phased_Array",
        "time_to_tca_hours": 78.0,
        "altitude_perigee_km": 780.0,
        "inclination_deg": 12.4,
        "relative_velocity_km_s": 5.4,
        "miss_distance_total_m": 8400.0,
        "radial_separation_abs_m": 1250.0,
        "in_track_separation_abs_m": 7200.0,
        "cross_track_separation_abs_m": 3800.0,
        "combined_sigma_radial_m": 22.0,
        "combined_sigma_in_track_m": 310.0,
        "combined_sigma_cross_track_m": 65.0,
        "chaser_rcs_m2": 0.08,
        "solar_flux_f10_7_sfu": 72.0,
        "geomagnetic_ap_index": 5.0,
        "cdm_update_count": 22,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["conjunction_event_id"] == "CDM-2026-LEO-NOMINAL-003"
    assert data["maneuver_escalation_required"] is False
    assert data["operational_risk_tier"] in ["GREEN_NOMINAL_PASS", "YELLOW_MONITORING_TRACK"]


def test_validation_error_on_invalid_payload():
    invalid_payload = {
        "conjunction_id": "INVALID",
        "time_to_tca_hours": -5.0,  # Negative lead time is invalid in schema
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422
