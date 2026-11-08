"""Unit tests for astrodynamics feature engineering module."""

import math
import numpy as np
import pandas as pd
import pytest

from src.features import (
    CATEGORICAL_FEATURES,
    NUMERICAL_RAW_FEATURES,
    ENGINEERED_FEATURES,
    compute_astrodynamics_features,
    extract_telemetry_dict,
)


@pytest.fixture
def sample_cdm_record():
    return {
        "cdm_id": "CDM-TEST-001",
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
    }


def test_feature_columns_defined():
    assert len(CATEGORICAL_FEATURES) == 5
    assert len(NUMERICAL_RAW_FEATURES) == 15
    assert len(ENGINEERED_FEATURES) == 4


def test_mahalanobis_distance_calculation(sample_cdm_record):
    df = pd.DataFrame([sample_cdm_record])
    processed = compute_astrodynamics_features(df)

    dr, sr = sample_cdm_record["radial_separation_abs_m"], sample_cdm_record["combined_sigma_radial_m"]
    dt, st = sample_cdm_record["in_track_separation_abs_m"], sample_cdm_record["combined_sigma_in_track_m"]
    dn, sn = sample_cdm_record["cross_track_separation_abs_m"], sample_cdm_record["combined_sigma_cross_track_m"]

    expected_dm = math.sqrt((dr / sr) ** 2 + (dt / st) ** 2 + (dn / sn) ** 2)
    computed_dm = processed.loc[0, "mahalanobis_rtn_distance"]

    assert pytest.approx(computed_dm, rel=1e-5) == expected_dm
    assert computed_dm > 0.0


def test_kinetic_lethality_log_index(sample_cdm_record):
    df = pd.DataFrame([sample_cdm_record])
    processed = compute_astrodynamics_features(df)

    rcs = sample_cdm_record["chaser_rcs_m2"]
    v_rel = sample_cdm_record["relative_velocity_km_s"]
    expected_lethality = math.log10(1.0 + rcs * (v_rel**2))
    computed_lethality = processed.loc[0, "kinetic_lethality_log_index"]

    assert pytest.approx(computed_lethality, rel=1e-5) == expected_lethality


def test_thermospheric_drag_perturbation(sample_cdm_record):
    df = pd.DataFrame([sample_cdm_record])
    processed = compute_astrodynamics_features(df)

    f10_7 = sample_cdm_record["solar_flux_f10_7_sfu"]
    ap = sample_cdm_record["geomagnetic_ap_index"]
    alt = sample_cdm_record["altitude_perigee_km"]
    alt_factor = (alt / 500.0) ** 2
    expected_drag = (f10_7 * (1.0 + 0.02 * ap)) / alt_factor
    computed_drag = processed.loc[0, "thermospheric_drag_perturbation"]

    assert pytest.approx(computed_drag, rel=1e-5) == expected_drag


def test_imputation_with_medians(sample_cdm_record):
    record_with_nans = sample_cdm_record.copy()
    record_with_nans["chaser_rcs_m2"] = np.nan
    record_with_nans["solar_flux_f10_7_sfu"] = np.nan

    medians = {"chaser_rcs_m2": 0.35, "solar_flux_f10_7_sfu": 130.0}
    df = pd.DataFrame([record_with_nans])
    processed = compute_astrodynamics_features(df, medians=medians)

    assert processed.loc[0, "chaser_rcs_m2"] == 0.35
    assert processed.loc[0, "solar_flux_f10_7_sfu"] == 130.0
    assert not processed.isna().any().any()


def test_extract_telemetry_dict(sample_cdm_record):
    medians = {"chaser_rcs_m2": 0.35, "solar_flux_f10_7_sfu": 130.0}
    res = extract_telemetry_dict(sample_cdm_record, medians=medians)

    assert isinstance(res, dict)
    assert "mahalanobis_rtn_distance" in res
    assert "kinetic_lethality_log_index" in res
    assert "thermospheric_drag_perturbation" in res
    assert "radial_to_sigma_margin" in res
