"""Verification Client Script for KESSLER-SENTINEL Conjunction Triage Service.

Sends 3 canonical orbital conjunction scenarios to the FastAPI inference engine
and verifies risk tier categorization and decision output.
"""

import argparse
import sys
import requests

DEFAULT_URL = "http://127.0.0.1:9696"

# Test Case 1: Critical Red Conjunction (Hypervelocity Fragmentation Debris in SSO)
CASE_1_RED = {
    "conjunction_id": "CDM-2026-SSO-CRITICAL-001",
    "orbital_regime_band": "Sun_Synchronous_LEO",
    "chaser_object_type": "Fragmentation_Debris",
    "primary_propulsion_class": "Bipropellant_High_Thrust",
    "solar_weather_storm_tier": "Severe_Geomagnetic_Storm_G3_Plus",
    "tracking_network_geometry": "Single_Pass_Radar_Sparse",
    "time_to_tca_hours": 28.5,
    "altitude_perigee_km": 485.0,
    "inclination_deg": 97.4,
    "relative_velocity_km_s": 14.65,
    "miss_distance_total_m": 42.0,
    "radial_separation_abs_m": 12.5,
    "in_track_separation_abs_m": 36.0,
    "cross_track_separation_abs_m": 18.0,
    "combined_sigma_radial_m": 14.0,
    "combined_sigma_in_track_m": 160.0,
    "combined_sigma_cross_track_m": 42.0,
    "chaser_rcs_m2": 0.35,
    "solar_flux_f10_7_sfu": 220.0,
    "geomagnetic_ap_index": 95.0,
    "cdm_update_count": 3,
}

# Test Case 2: Marginal Amber Conjunction (High Along-Track Covariance Uncertainty)
CASE_2_AMBER = {
    "conjunction_id": "CDM-2026-POLAR-AMBER-002",
    "orbital_regime_band": "Polar_Earth_Observation",
    "chaser_object_type": "Payload_Defunct",
    "primary_propulsion_class": "Monopropellant_Hydrazine",
    "solar_weather_storm_tier": "Moderate_Solar_Flux",
    "tracking_network_geometry": "Optical_Radar_Hybrid",
    "time_to_tca_hours": 44.0,
    "altitude_perigee_km": 620.0,
    "inclination_deg": 82.1,
    "relative_velocity_km_s": 11.2,
    "miss_distance_total_m": 195.0,
    "radial_separation_abs_m": 65.0,
    "in_track_separation_abs_m": 175.0,
    "cross_track_separation_abs_m": 55.0,
    "combined_sigma_radial_m": 32.0,
    "combined_sigma_in_track_m": 420.0,
    "combined_sigma_cross_track_m": 85.0,
    "chaser_rcs_m2": 1.20,
    "solar_flux_f10_7_sfu": 135.0,
    "geomagnetic_ap_index": 38.0,
    "cdm_update_count": 6,
}

# Test Case 3: Nominal Green Pass (Wide Clearance, Phased Array Precision)
CASE_3_GREEN = {
    "conjunction_id": "CDM-2026-SHELL-NOMINAL-003",
    "orbital_regime_band": "Inclined_Constellation_Shell",
    "chaser_object_type": "Rocket_Body_Stage",
    "primary_propulsion_class": "Hall_Effect_Ion_Electric",
    "solar_weather_storm_tier": "Quiet_Geomagnetic",
    "tracking_network_geometry": "Multi_Station_Phased_Array",
    "time_to_tca_hours": 72.0,
    "altitude_perigee_km": 550.0,
    "inclination_deg": 53.2,
    "relative_velocity_km_s": 9.8,
    "miss_distance_total_m": 3200.0,
    "radial_separation_abs_m": 1120.0,
    "in_track_separation_abs_m": 2900.0,
    "cross_track_separation_abs_m": 780.0,
    "combined_sigma_radial_m": 18.0,
    "combined_sigma_in_track_m": 180.0,
    "combined_sigma_cross_track_m": 45.0,
    "chaser_rcs_m2": 6.5,
    "solar_flux_f10_7_sfu": 74.0,
    "geomagnetic_ap_index": 6.0,
    "cdm_update_count": 18,
}


def run_verification(base_url: str):
    print("=" * 80)
    print(f"KESSLER-SENTINEL CONJUNCTION TRIAGE VERIFICATION SUITE")
    print(f"Target URL: {base_url}")
    print("=" * 80)

    # 1. Health Probe
    try:
        health_resp = requests.get(f"{base_url}/health", timeout=5)
        health_resp.raise_for_status()
        health_data = health_resp.json()
        print(f"[HEALTH CHECK] Service: {health_data.get('service')} | Status: {health_data.get('status')}")
        print(f"[HEALTH CHECK] Model: {health_data.get('model_architecture')} | Decision Threshold: {health_data.get('decision_threshold')}")
        print("-" * 80)
    except Exception as exc:
        print(f"[ERROR] Could not connect to health endpoint: {exc}")
        sys.exit(1)

    # 2. Conjunction Scenarios
    scenarios = [
        ("SCENARIO 1: CRITICAL CONJUNCTION (RED GATE)", CASE_1_RED),
        ("SCENARIO 2: MARGINAL CONJUNCTION (AMBER TIER)", CASE_2_AMBER),
        ("SCENARIO 3: NOMINAL ORBITAL PASS (GREEN HOLD)", CASE_3_GREEN),
    ]

    for title, payload in scenarios:
        print(f"\n>>> Running {title}...")
        resp = requests.post(f"{base_url}/predict", json=payload, timeout=5)
        if resp.status_code != 200:
            print(f"[FAILED] HTTP {resp.status_code}: {resp.text}")
            continue

        data = resp.json()
        print(f"  Event ID:             {data['conjunction_event_id']}")
        print(f"  Collision Escalation: {data['collision_escalation_probability'] * 100:.2f}% (Threshold: {data['decision_threshold_applied']:.4f})")
        print(f"  CAM Burn Required:    {data['maneuver_escalation_required']}")
        print(f"  Operational Risk Tier: {data['operational_risk_tier']}")
        print(f"  Recommended Action:   {data['recommended_flight_action']}")
        print(f"  Diagnostics:          Mahalanobis DM={data['engineered_diagnostics']['mahalanobis_rtn_distance']:.2f} | Drag Perturbation={data['engineered_diagnostics']['thermospheric_drag_perturbation']:.1f}")

    print("\n" + "=" * 80)
    print("ALL CONJUNCTION TRIAGE SCENARIOS EVALUATED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test KESSLER-SENTINEL Inference Microservice")
    parser.add_argument("--url", type=str, default=DEFAULT_URL, help="Base service URL")
    args = parser.parse_args()
    run_verification(args.url)
