"""Unit tests for serialized model artifact and decision engine logic."""

import os
import pickle
import pytest

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model_kessler_sentinel_v1.bin")


@pytest.fixture(scope="session")
def loaded_artifact():
    assert os.path.exists(MODEL_PATH), f"Model artifact not found at {MODEL_PATH}"
    with open(MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)
    return artifact


def test_artifact_structure(loaded_artifact):
    expected_keys = [
        "dv",
        "model",
        "threshold",
        "medians",
        "feature_names",
        "metrics",
        "timestamp",
    ]
    for key in expected_keys:
        assert key in loaded_artifact, f"Artifact missing key: {key}"


def test_threshold_value(loaded_artifact):
    tau = loaded_artifact["threshold"]
    assert 0.01 <= tau <= 0.20, f"Optimal threshold {tau} out of expected operational range"


def test_metrics_integrity(loaded_artifact):
    metrics = loaded_artifact["metrics"]
    assert "test_roc_auc" in metrics
    assert "test_recall" in metrics
    assert metrics["test_roc_auc"] > 0.90, "Expected Test ROC-AUC > 0.90"
    assert metrics["test_recall"] > 0.80, "Expected Test Recall > 0.80"


def test_model_inference_bounds(loaded_artifact):
    dv = loaded_artifact["dv"]
    model = loaded_artifact["model"]
    tau = loaded_artifact["threshold"]

    sample_dict = {
        "orbital_regime_band": "Sun_Synchronous_LEO",
        "chaser_object_type": "Fragmentation_Debris",
        "primary_propulsion_class": "Monopropellant_Hydrazine",
        "solar_weather_storm_tier": "Quiet_Geomagnetic",
        "tracking_network_geometry": "Multi_Station_Phased_Array",
        "time_to_tca_hours": 48.0,
        "altitude_perigee_km": 500.0,
        "inclination_deg": 97.4,
        "relative_velocity_km_s": 14.5,
        "miss_distance_total_m": 120.0,
        "radial_separation_abs_m": 35.0,
        "in_track_separation_abs_m": 85.0,
        "cross_track_separation_abs_m": 60.0,
        "combined_sigma_radial_m": 25.0,
        "combined_sigma_in_track_m": 350.0,
        "combined_sigma_cross_track_m": 80.0,
        "chaser_rcs_m2": 0.45,
        "solar_flux_f10_7_sfu": 135.0,
        "geomagnetic_ap_index": 12.0,
        "cdm_update_count": 14,
        "mahalanobis_rtn_distance": 1.45,
        "kinetic_lethality_log_index": 1.98,
        "thermospheric_drag_perturbation": 167.4,
        "radial_to_sigma_margin": 1.35,
    }

    X = dv.transform([sample_dict])
    prob = float(model.predict_proba(X)[0, 1])
    assert 0.0 <= prob <= 1.0

    action = "EXECUTE_CAM" if prob >= tau else "MAINTAIN_TRAJECTORY"
    assert action in ["EXECUTE_CAM", "MAINTAIN_TRAJECTORY"]
