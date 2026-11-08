"""Astrodynamics Feature Engineering Module for LEO Conjunction Triage.

Computes physical interaction features from Conjunction Data Message (CDM)
telemetry expressed in the RTN (Radial, Transverse/In-Track, Normal/Cross-Track)
coordinate frame.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

# Canonical feature definitions
CATEGORICAL_FEATURES: List[str] = [
    "orbital_regime_band",
    "chaser_object_type",
    "primary_propulsion_class",
    "solar_weather_storm_tier",
    "tracking_network_geometry",
]

NUMERICAL_RAW_FEATURES: List[str] = [
    "time_to_tca_hours",
    "altitude_perigee_km",
    "inclination_deg",
    "relative_velocity_km_s",
    "miss_distance_total_m",
    "radial_separation_abs_m",
    "in_track_separation_abs_m",
    "cross_track_separation_abs_m",
    "combined_sigma_radial_m",
    "combined_sigma_in_track_m",
    "combined_sigma_cross_track_m",
    "chaser_rcs_m2",
    "solar_flux_f10_7_sfu",
    "geomagnetic_ap_index",
    "cdm_update_count",
]

ENGINEERED_FEATURES: List[str] = [
    "mahalanobis_rtn_distance",
    "kinetic_lethality_log_index",
    "thermospheric_drag_perturbation",
    "radial_to_sigma_margin",
]

TARGET_COLUMN: str = "maneuver_escalation_required"


def compute_astrodynamics_features(
    df: pd.DataFrame,
    medians: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Compute domain-specific physical metrics for orbital conjunction triage.

    Derived Metrics:
    1. mahalanobis_rtn_distance: Dimensionless 3D covariance-normalized proximity
       in the RTN relative orbital frame.
    2. kinetic_lethality_log_index: Logarithmic kinetic energy proxy based on
       chaser radar cross section and relative hypervelocity.
    3. thermospheric_drag_perturbation: Atmospheric density variability proxy
       scaled by solar radio flux F10.7 and geomagnetic index Ap at altitude.
    4. radial_to_sigma_margin: Critical radial separation normalized by 1-sigma
       radial covariance uncertainty.

    Args:
        df: Input DataFrame containing raw CDM telemetry.
        medians: Optional dict of training set medians for deterministic imputation.

    Returns:
        DataFrame with engineered features appended and missing values imputed.
    """
    out = df.copy()

    # Deterministic missing-value imputation if medians are provided
    if medians is not None:
        for col, val in medians.items():
            if col in out.columns:
                out[col] = out[col].fillna(val)
    else:
        # Fallback to local column medians
        for col in ["chaser_rcs_m2", "solar_flux_f10_7_sfu"]:
            if col in out.columns and out[col].isna().any():
                local_med = float(out[col].median())
                out[col] = out[col].fillna(local_med)

    # 1. Mahalanobis RTN Distance
    dr = out["radial_separation_abs_m"]
    dt = out["in_track_separation_abs_m"]
    dn = out["cross_track_separation_abs_m"]

    sr = np.maximum(out["combined_sigma_radial_m"], 1e-3)
    st = np.maximum(out["combined_sigma_in_track_m"], 1e-3)
    sn = np.maximum(out["combined_sigma_cross_track_m"], 1e-3)

    out["mahalanobis_rtn_distance"] = np.sqrt(
        (dr / sr) ** 2 + (dt / st) ** 2 + (dn / sn) ** 2
    )

    # 2. Kinetic Lethality Log Index: log10(1 + rcs * v_rel^2)
    rcs = np.maximum(out["chaser_rcs_m2"], 1e-4)
    v_rel = out["relative_velocity_km_s"]
    out["kinetic_lethality_log_index"] = np.log10(1.0 + rcs * (v_rel**2))

    # 3. Thermospheric Drag Perturbation: F10.7 * (1 + 0.02 * Ap) / (h / 500)^2
    f10_7 = out["solar_flux_f10_7_sfu"]
    ap = out["geomagnetic_ap_index"]
    alt = np.maximum(out["altitude_perigee_km"], 100.0)
    alt_factor = (alt / 500.0) ** 2
    out["thermospheric_drag_perturbation"] = (f10_7 * (1.0 + 0.02 * ap)) / np.maximum(alt_factor, 1e-2)

    # 4. Radial to Sigma Margin: |dr| / (sigma_r + 1.0)
    out["radial_to_sigma_margin"] = dr / (sr + 1.0)

    return out


def extract_telemetry_dict(
    raw_dict: Dict[str, Any],
    medians: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Process a single CDM telemetry record into a model-ready dictionary.

    Args:
        raw_dict: Single JSON record conforming to CDM schema.
        medians: Imputation medians from training split.

    Returns:
        Dictionary containing raw categorical, numerical, and engineered values.
    """
    df = pd.DataFrame([raw_dict])
    processed_df = compute_astrodynamics_features(df, medians=medians)
    return processed_df.to_dict(orient="records")[0]
