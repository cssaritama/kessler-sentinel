"""FastAPI Orbital Conjunction Triage Microservice for LEO Collision Avoidance.

Provides real-time Go/No-Go screening against collision escalation
risk for incoming Conjunction Data Messages (CDMs).
"""

import os
import pickle
from typing import Any, Dict, Literal, Optional
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import uvicorn

from src.features import compute_astrodynamics_features

MODEL_PATH = os.getenv("MODEL_PATH", "model_kessler_sentinel_v1.bin")

# Telemetry Schema Definition with Pydantic
OrbitalRegime = Literal[
    "Sun_Synchronous_LEO",
    "Polar_Earth_Observation",
    "Inclined_Constellation_Shell",
    "Equatorial_Low_Inclination",
]

ChaserType = Literal[
    "Fragmentation_Debris",
    "Payload_Defunct",
    "Rocket_Body_Stage",
    "Uncharacterized_Tracklet",
]

PropulsionClass = Literal[
    "Hall_Effect_Ion_Electric",
    "Monopropellant_Hydrazine",
    "Bipropellant_High_Thrust",
    "Differential_Drag_Only",
]

StormTier = Literal[
    "Quiet_Geomagnetic",
    "Moderate_Solar_Flux",
    "Severe_Geomagnetic_Storm_G3_Plus",
]

TrackingGeometry = Literal[
    "Multi_Station_Phased_Array",
    "Single_Pass_Radar_Sparse",
    "Optical_Radar_Hybrid",
]


class ConjunctionTelemetryRequest(BaseModel):
    conjunction_id: str = Field(default="CDM-UNKNOWN", description="Unique CDM identifier")
    orbital_regime_band: OrbitalRegime
    chaser_object_type: ChaserType
    primary_propulsion_class: PropulsionClass
    solar_weather_storm_tier: StormTier
    tracking_network_geometry: TrackingGeometry
    time_to_tca_hours: float = Field(..., ge=0.0, le=240.0, description="Lead time to TCA (hours)")
    altitude_perigee_km: float = Field(..., ge=150.0, le=2500.0, description="Perigee altitude (km)")
    inclination_deg: float = Field(..., ge=0.0, le=180.0, description="Orbital inclination (degrees)")
    relative_velocity_km_s: float = Field(..., ge=0.1, le=25.0, description="Encounter relative speed (km/s)")
    miss_distance_total_m: float = Field(..., ge=0.0, description="Euclidean separation at TCA (m)")
    radial_separation_abs_m: float = Field(..., ge=0.0, description="|Delta R| radial clearance (m)")
    in_track_separation_abs_m: float = Field(..., ge=0.0, description="|Delta T| along-track clearance (m)")
    cross_track_separation_abs_m: float = Field(..., ge=0.0, description="|Delta N| normal clearance (m)")
    combined_sigma_radial_m: float = Field(..., gt=0.0, description="1-sigma radial covariance uncertainty (m)")
    combined_sigma_in_track_m: float = Field(..., gt=0.0, description="1-sigma in-track covariance uncertainty (m)")
    combined_sigma_cross_track_m: float = Field(..., gt=0.0, description="1-sigma normal covariance uncertainty (m)")
    chaser_rcs_m2: Optional[float] = Field(default=None, description="Radar cross section (m^2)")
    solar_flux_f10_7_sfu: Optional[float] = Field(default=None, description="10.7cm solar radio flux (sfu)")
    geomagnetic_ap_index: float = Field(..., ge=0.0, description="Planetary geomagnetic perturbation Ap")
    cdm_update_count: int = Field(..., ge=1, le=100, description="Number of sequential radar tracks")

    model_config = {
        "json_schema_extra": {
            "example": {
                "conjunction_id": "CDM-2026-SSO-08491",
                "orbital_regime_band": "Sun_Synchronous_LEO",
                "chaser_object_type": "Fragmentation_Debris",
                "primary_propulsion_class": "Bipropellant_High_Thrust",
                "solar_weather_storm_tier": "Severe_Geomagnetic_Storm_G3_Plus",
                "tracking_network_geometry": "Single_Pass_Radar_Sparse",
                "time_to_tca_hours": 32.5,
                "altitude_perigee_km": 492.0,
                "inclination_deg": 97.4,
                "relative_velocity_km_s": 14.65,
                "miss_distance_total_m": 84.2,
                "radial_separation_abs_m": 18.5,
                "in_track_separation_abs_m": 76.0,
                "cross_track_separation_abs_m": 31.4,
                "combined_sigma_radial_m": 12.8,
                "combined_sigma_in_track_m": 145.0,
                "combined_sigma_cross_track_m": 38.0,
                "chaser_rcs_m2": 0.45,
                "solar_flux_f10_7_sfu": 215.0,
                "geomagnetic_ap_index": 82.0,
                "cdm_update_count": 4,
            }
        }
    }


class ConjunctionTriageResponse(BaseModel):
    conjunction_event_id: str
    collision_escalation_probability: float
    decision_threshold_applied: float
    maneuver_escalation_required: bool
    operational_risk_tier: str
    recommended_flight_action: str
    engineered_diagnostics: Dict[str, float]


# Initialize FastAPI application
app = FastAPI(
    title="KESSLER-SENTINEL: Orbital Conjunction Triage Service",
    description="Autonomous Low Earth Orbit (LEO) Collision Avoidance & Maneuver Decision API",
    version="1.0.0",
)

# Global model state
bundle: Optional[Dict[str, Any]] = None


def load_model_bundle() -> Dict[str, Any]:
    global bundle
    if bundle is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(f"Trained model artifact not found at {MODEL_PATH}")
        with open(MODEL_PATH, "rb") as f:
            bundle = pickle.load(f)
    return bundle


@app.on_event("startup")
def startup_event():
    """Warm up and validate model artifact on application boot."""
    try:
        b = load_model_bundle()
        print(f"[SYSTEM] Loaded model artifact from {MODEL_PATH}")
        print(f"[SYSTEM] Calibrated decision threshold: {b['threshold']:.4f}")
    except Exception as e:
        print(f"[WARNING] Model artifact could not be loaded at startup: {e}")


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Health check endpoint for container orchestrators and probes."""
    b = bundle
    return {
        "status": "HEALTHY",
        "service": "kessler-sentinel-triage",
        "version": "1.0.0",
        "model_loaded": b is not None,
        "decision_threshold": float(b["threshold"]) if b else None,
        "model_architecture": type(b["model"]).__name__ if b else None,
    }


@app.post("/predict", response_model=ConjunctionTriageResponse, status_code=status.HTTP_200_OK)
def predict_conjunction_risk(telemetry: ConjunctionTelemetryRequest):
    """Evaluate collision escalation probability and operational action for a CDM."""
    try:
        b = load_model_bundle()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Inference engine unavailable: {str(exc)}",
        )

    dv = b["dv"]
    model = b["model"]
    tau = float(b["threshold"])
    medians = b["medians"]

    # Extract dict and compute astrodynamics features
    raw_dict = telemetry.model_dump()
    c_id = raw_dict.pop("conjunction_id", "CDM-UNKNOWN")

    import pandas as pd
    df_raw = pd.DataFrame([raw_dict])
    df_feat = compute_astrodynamics_features(df_raw, medians=medians)
    feat_dict = df_feat.to_dict(orient="records")[0]

    # Vectorize and score
    X = dv.transform([feat_dict])
    prob = float(model.predict_proba(X)[0, 1])

    escalate = prob >= tau

    # Classify operational risk tier and flight action
    if prob >= 0.70:
        tier = "RED_CRITICAL_MANEUVER_GATE"
        action = "EXECUTE_RADIAL_PROGRADE_CAM_BURN_AND_TASK_HPR_RADAR"
    elif prob >= tau:
        tier = "AMBER_ELEVATED_COVARIANCE_WARNING"
        action = "PREPARE_PROPULSION_SUB-SYSTEM_AND_REQUEST_URGENT_RADAR_TASKING"
    elif prob >= 0.10:
        tier = "YELLOW_MONITORING_TRACK"
        action = "CONTINUE_ROUTINE_EPHEMERIS_PROPAGATION"
    else:
        tier = "GREEN_NOMINAL_PASS"
        action = "NOMINAL_PASS_HOLD_PROPULSION_EXPENDITURE"

    diagnostics = {
        "mahalanobis_rtn_distance": round(float(feat_dict["mahalanobis_rtn_distance"]), 3),
        "kinetic_lethality_log_index": round(float(feat_dict["kinetic_lethality_log_index"]), 3),
        "thermospheric_drag_perturbation": round(float(feat_dict["thermospheric_drag_perturbation"]), 2),
        "radial_to_sigma_margin": round(float(feat_dict["radial_to_sigma_margin"]), 3),
    }

    return ConjunctionTriageResponse(
        conjunction_event_id=c_id,
        collision_escalation_probability=round(prob, 4),
        decision_threshold_applied=round(tau, 4),
        maneuver_escalation_required=escalate,
        operational_risk_tier=tier,
        recommended_flight_action=action,
        engineered_diagnostics=diagnostics,
    )


if __name__ == "__main__":
    uvicorn.run("predict:app", host="0.0.0.0", port=9696, reload=False)
