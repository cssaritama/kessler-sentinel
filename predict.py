"""FastAPI Orbital Conjunction Triage Microservice & Flight Operations Console.

Provides real-time Go/No-Go screening against collision escalation
risk for incoming Conjunction Data Messages (CDMs) in Low Earth Orbit (LEO).
Includes an embedded minimalist Flight Dynamics Operations Web Console at GET /.
"""

import os
import pickle
from typing import Any, Dict, Literal, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse
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


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Health check endpoint for container orchestrators and probes."""
    b = bundle if bundle is not None else (load_model_bundle() if os.path.exists(MODEL_PATH) else None)
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
    elif prob >= 0.02:
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


@app.get("/", response_class=HTMLResponse)
def flight_dynamics_console():
    """Minimalist dark-slate flight dynamics operations console."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>KESSLER-SENTINEL | Conjunction Triage Console</title>
    <style>
        :root {
            --bg-primary: #0F172A;
            --bg-secondary: #1E293B;
            --bg-card: #182234;
            --border: #334155;
            --text-primary: #F8FAFC;
            --text-secondary: #94A3B8;
            --accent-blue: #3B82F6;
            --accent-red: #EF4444;
            --accent-green: #10B981;
            --accent-amber: #F59E0B;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        body { background-color: var(--bg-primary); color: var(--text-primary); padding: 24px; line-height: 1.5; }
        .container { max-width: 1100px; margin: 0 auto; }
        header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); padding-bottom: 16px; margin-bottom: 24px; }
        h1 { font-size: 20px; font-weight: 700; letter-spacing: 0.5px; }
        .badge { background: #064E3B; color: #34D399; font-size: 11px; padding: 4px 10px; border-radius: 9999px; font-weight: 600; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
        @media (max-width: 800px) { .grid { grid-template-columns: 1fr; } }
        .card { background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; padding: 20px; }
        .card-title { font-size: 13px; text-transform: uppercase; letter-spacing: 1px; color: var(--text-secondary); margin-bottom: 16px; font-weight: 600; }
        .scenario-bar { display: flex; gap: 8px; margin-bottom: 16px; }
        .btn-scenario { background: var(--bg-secondary); border: 1px solid var(--border); color: var(--text-primary); padding: 6px 12px; font-size: 12px; border-radius: 4px; cursor: pointer; transition: 0.2s; }
        .btn-scenario:hover { border-color: var(--accent-blue); }
        .form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
        .form-group { display: flex; flex-direction: column; gap: 4px; }
        label { font-size: 11px; color: var(--text-secondary); }
        input, select { background: var(--bg-secondary); border: 1px solid var(--border); color: var(--text-primary); padding: 8px; border-radius: 4px; font-size: 12px; }
        input:focus, select:focus { outline: none; border-color: var(--accent-blue); }
        .btn-submit { background: var(--accent-blue); color: white; border: none; padding: 10px; border-radius: 4px; font-size: 13px; font-weight: 600; cursor: pointer; width: 100%; margin-top: 16px; transition: 0.2s; }
        .btn-submit:hover { opacity: 0.9; }
        .gauge-container { text-align: center; padding: 24px 0; border-bottom: 1px solid var(--border); margin-bottom: 20px; }
        .prob-number { font-size: 48px; font-weight: 800; font-variant-numeric: tabular-nums; }
        .prob-label { font-size: 12px; color: var(--text-secondary); }
        .tier-pill { display: inline-block; padding: 6px 16px; border-radius: 4px; font-size: 12px; font-weight: 700; margin-top: 12px; letter-spacing: 0.5px; }
        .tier-red { background: #7F1D1D; color: #FCA5A5; border: 1px solid var(--accent-red); }
        .tier-amber { background: #78350F; color: #FCD34D; border: 1px solid var(--accent-amber); }
        .tier-green { background: #064E3B; color: #6EE7B7; border: 1px solid var(--accent-green); }
        .action-box { background: var(--bg-secondary); border-left: 3px solid var(--accent-blue); padding: 12px; margin-bottom: 16px; font-size: 12px; font-family: monospace; }
        .diag-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
        .diag-item { background: var(--bg-secondary); padding: 10px; border-radius: 4px; }
        .diag-name { font-size: 10px; color: var(--text-secondary); text-transform: uppercase; }
        .diag-val { font-size: 16px; font-weight: 700; font-family: monospace; }
        footer { margin-top: 24px; text-align: center; font-size: 11px; color: var(--text-secondary); border-top: 1px solid var(--border); padding-top: 16px; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>KESSLER-SENTINEL | LEO Conjunction Triage Engine</h1>
                <div style="font-size: 12px; color: var(--text-secondary); margin-top: 2px;">Autonomous Flight Dynamics Screening Gate (T - 48h)</div>
            </div>
            <div class="badge">SYSTEM READY (tau* = 0.0550)</div>
        </header>

        <div class="grid">
            <!-- Left: Telemetry Input -->
            <div class="card">
                <div class="card-title">Conjunction Telemetry Ingestion</div>
                <div class="scenario-bar">
                    <button class="btn-scenario" onclick="loadScenario('critical')">Scenario 1: Critical (Red)</button>
                    <button class="btn-scenario" onclick="loadScenario('amber')">Scenario 2: Marginal (Amber)</button>
                    <button class="btn-scenario" onclick="loadScenario('nominal')">Scenario 3: Nominal (Green)</button>
                </div>
                <form id="cdmForm" onsubmit="event.preventDefault(); evaluateTriage();">
                    <div class="form-grid">
                        <div class="form-group">
                            <label>Event Identifier</label>
                            <input type="text" id="conjunction_id" value="CDM-2026-SSO-CRITICAL-001">
                        </div>
                        <div class="form-group">
                            <label>Orbital Regime</label>
                            <select id="orbital_regime_band">
                                <option value="Sun_Synchronous_LEO">Sun Synchronous LEO</option>
                                <option value="Polar_Earth_Observation">Polar Earth Observation</option>
                                <option value="Inclined_Constellation_Shell">Inclined Constellation Shell</option>
                                <option value="Equatorial_Low_Inclination">Equatorial Low Inclination</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Chaser Object Type</label>
                            <select id="chaser_object_type">
                                <option value="Fragmentation_Debris">Fragmentation Debris</option>
                                <option value="Payload_Defunct">Payload Defunct</option>
                                <option value="Rocket_Body_Stage">Rocket Body Stage</option>
                                <option value="Uncharacterized_Tracklet">Uncharacterized Tracklet</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Tracking Geometry</label>
                            <select id="tracking_network_geometry">
                                <option value="Single_Pass_Radar_Sparse">Single Pass Radar Sparse</option>
                                <option value="Multi_Station_Phased_Array">Multi Station Phased Array</option>
                                <option value="Optical_Radar_Hybrid">Optical Radar Hybrid</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Time to TCA (hours)</label>
                            <input type="number" step="0.1" id="time_to_tca_hours" value="28.5">
                        </div>
                        <div class="form-group">
                            <label>Perigee Altitude (km)</label>
                            <input type="number" step="0.1" id="altitude_perigee_km" value="485.0">
                        </div>
                        <div class="form-group">
                            <label>Relative Speed (km/s)</label>
                            <input type="number" step="0.01" id="relative_velocity_km_s" value="14.65">
                        </div>
                        <div class="form-group">
                            <label>Radial Separation |Delta R| (m)</label>
                            <input type="number" step="0.1" id="radial_separation_abs_m" value="12.5">
                        </div>
                        <div class="form-group">
                            <label>Along-Track |Delta T| (m)</label>
                            <input type="number" step="0.1" id="in_track_separation_abs_m" value="36.0">
                        </div>
                        <div class="form-group">
                            <label>Cross-Track |Delta N| (m)</label>
                            <input type="number" step="0.1" id="cross_track_separation_abs_m" value="18.0">
                        </div>
                        <div class="form-group">
                            <label>Sigma Radial (m)</label>
                            <input type="number" step="0.1" id="combined_sigma_radial_m" value="14.0">
                        </div>
                        <div class="form-group">
                            <label>Sigma Along-Track (m)</label>
                            <input type="number" step="0.1" id="combined_sigma_in_track_m" value="160.0">
                        </div>
                        <div class="form-group">
                            <label>Radar Cross Section (m^2)</label>
                            <input type="number" step="0.01" id="chaser_rcs_m2" value="0.35">
                        </div>
                        <div class="form-group">
                            <label>Solar Flux F10.7 (sfu)</label>
                            <input type="number" step="0.1" id="solar_flux_f10_7_sfu" value="220.0">
                        </div>
                        <div class="form-group">
                            <label>Geomagnetic Ap Index</label>
                            <input type="number" step="0.1" id="geomagnetic_ap_index" value="95.0">
                        </div>
                        <div class="form-group">
                            <label>Radar Update Count</label>
                            <input type="number" id="cdm_update_count" value="3">
                        </div>
                    </div>
                    <button type="submit" class="btn-submit">SCREEN CONJUNCTION RISK</button>
                </form>
            </div>

            <!-- Right: Decision & Diagnostics -->
            <div class="card">
                <div class="card-title">Flight Avoidance Decision</div>
                <div class="gauge-container">
                    <div class="prob-number" id="probDisplay">--%</div>
                    <div class="prob-label">Collision Escalation Probability P(Escalation)</div>
                    <div id="tierBadge" class="tier-pill tier-green">AWAITING TELEMETRY</div>
                </div>

                <div class="card-title" style="margin-top: 16px;">Flight Action Directive</div>
                <div class="action-box" id="actionDisplay">Select a scenario or enter telemetry to evaluate Go/No-Go screening.</div>

                <div class="card-title" style="margin-top: 16px;">Astrodynamics Diagnostics</div>
                <div class="diag-grid">
                    <div class="diag-item">
                        <div class="diag-name">Mahalanobis Proximity</div>
                        <div class="diag-val" id="diagDm">--</div>
                    </div>
                    <div class="diag-item">
                        <div class="diag-name">Kinetic Lethality</div>
                        <div class="diag-val" id="diagLethal">--</div>
                    </div>
                    <div class="diag-item">
                        <div class="diag-name">Drag Perturbation</div>
                        <div class="diag-val" id="diagDrag">--</div>
                    </div>
                    <div class="diag-item">
                        <div class="diag-name">Radial Margin</div>
                        <div class="diag-val" id="diagMargin">--</div>
                    </div>
                </div>
            </div>
        </div>

        <footer>
            KESSLER-SENTINEL Astrodynamics ML Decision Engine | REST API Documentation available at <a href="/docs" style="color: var(--accent-blue);">/docs</a>
        </footer>
    </div>

    <script>
        const scenarios = {
            critical: {
                conjunction_id: "CDM-2026-SSO-CRITICAL-001",
                orbital_regime_band: "Sun_Synchronous_LEO",
                chaser_object_type: "Fragmentation_Debris",
                tracking_network_geometry: "Single_Pass_Radar_Sparse",
                time_to_tca_hours: 28.5,
                altitude_perigee_km: 485.0,
                relative_velocity_km_s: 14.65,
                radial_separation_abs_m: 12.5,
                in_track_separation_abs_m: 36.0,
                cross_track_separation_abs_m: 18.0,
                combined_sigma_radial_m: 14.0,
                combined_sigma_in_track_m: 160.0,
                chaser_rcs_m2: 0.35,
                solar_flux_f10_7_sfu: 220.0,
                geomagnetic_ap_index: 95.0,
                cdm_update_count: 3
            },
            amber: {
                conjunction_id: "CDM-2026-POLAR-AMBER-002",
                orbital_regime_band: "Polar_Earth_Observation",
                chaser_object_type: "Payload_Defunct",
                tracking_network_geometry: "Optical_Radar_Hybrid",
                time_to_tca_hours: 44.0,
                altitude_perigee_km: 620.0,
                relative_velocity_km_s: 11.20,
                radial_separation_abs_m: 65.0,
                in_track_separation_abs_m: 175.0,
                cross_track_separation_abs_m: 55.0,
                combined_sigma_radial_m: 32.0,
                combined_sigma_in_track_m: 420.0,
                chaser_rcs_m2: 1.20,
                solar_flux_f10_7_sfu: 135.0,
                geomagnetic_ap_index: 38.0,
                cdm_update_count: 6
            },
            nominal: {
                conjunction_id: "CDM-2026-SHELL-NOMINAL-003",
                orbital_regime_band: "Inclined_Constellation_Shell",
                chaser_object_type: "Rocket_Body_Stage",
                tracking_network_geometry: "Multi_Station_Phased_Array",
                time_to_tca_hours: 72.0,
                altitude_perigee_km: 550.0,
                relative_velocity_km_s: 9.80,
                radial_separation_abs_m: 1120.0,
                in_track_separation_abs_m: 2900.0,
                cross_track_separation_abs_m: 780.0,
                combined_sigma_radial_m: 18.0,
                combined_sigma_in_track_m: 180.0,
                chaser_rcs_m2: 6.50,
                solar_flux_f10_7_sfu: 74.0,
                geomagnetic_ap_index: 6.0,
                cdm_update_count: 18
            }
        };

        function loadScenario(name) {
            const s = scenarios[name];
            for (const key in s) {
                const el = document.getElementById(key);
                if (el) el.value = s[key];
            }
            evaluateTriage();
        }

        async function evaluateTriage() {
            const dr = parseFloat(document.getElementById("radial_separation_abs_m").value);
            const dt = parseFloat(document.getElementById("in_track_separation_abs_m").value);
            const dn = parseFloat(document.getElementById("cross_track_separation_abs_m").value);
            const miss = Math.sqrt(dr*dr + dt*dt + dn*dn);

            const payload = {
                conjunction_id: document.getElementById("conjunction_id").value,
                orbital_regime_band: document.getElementById("orbital_regime_band").value,
                chaser_object_type: document.getElementById("chaser_object_type").value,
                primary_propulsion_class: "Bipropellant_High_Thrust",
                solar_weather_storm_tier: parseFloat(document.getElementById("geomagnetic_ap_index").value) > 60 ? "Severe_Geomagnetic_Storm_G3_Plus" : "Quiet_Geomagnetic",
                tracking_network_geometry: document.getElementById("tracking_network_geometry").value,
                time_to_tca_hours: parseFloat(document.getElementById("time_to_tca_hours").value),
                altitude_perigee_km: parseFloat(document.getElementById("altitude_perigee_km").value),
                inclination_deg: 97.4,
                relative_velocity_km_s: parseFloat(document.getElementById("relative_velocity_km_s").value),
                miss_distance_total_m: miss,
                radial_separation_abs_m: dr,
                in_track_separation_abs_m: dt,
                cross_track_separation_abs_m: dn,
                combined_sigma_radial_m: parseFloat(document.getElementById("combined_sigma_radial_m").value),
                combined_sigma_in_track_m: parseFloat(document.getElementById("combined_sigma_in_track_m").value),
                combined_sigma_cross_track_m: 40.0,
                chaser_rcs_m2: parseFloat(document.getElementById("chaser_rcs_m2").value),
                solar_flux_f10_7_sfu: parseFloat(document.getElementById("solar_flux_f10_7_sfu").value),
                geomagnetic_ap_index: parseFloat(document.getElementById("geomagnetic_ap_index").value),
                cdm_update_count: parseInt(document.getElementById("cdm_update_count").value)
            };

            try {
                const res = await fetch("/predict", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload)
                });
                const data = await res.json();
                
                const pct = (data.collision_escalation_probability * 100).toFixed(1);
                const probEl = document.getElementById("probDisplay");
                probEl.innerText = pct + "%";

                const badgeEl = document.getElementById("tierBadge");
                badgeEl.innerText = data.operational_risk_tier.replace(/_/g, " ");
                badgeEl.className = "tier-pill " + (data.maneuver_escalation_required ? "tier-red" : "tier-green");
                probEl.style.color = data.maneuver_escalation_required ? "var(--accent-red)" : "var(--accent-green)";

                document.getElementById("actionDisplay").innerText = data.recommended_flight_action;
                document.getElementById("diagDm").innerText = data.engineered_diagnostics.mahalanobis_rtn_distance;
                document.getElementById("diagLethal").innerText = data.engineered_diagnostics.kinetic_lethality_log_index;
                document.getElementById("diagDrag").innerText = data.engineered_diagnostics.thermospheric_drag_perturbation;
                document.getElementById("diagMargin").innerText = data.engineered_diagnostics.radial_to_sigma_margin;
            } catch(e) {
                alert("Error calling inference service: " + e);
            }
        }

        window.onload = () => loadScenario('critical');
    </script>
</body>
</html>"""
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    uvicorn.run("predict:app", host="0.0.0.0", port=9696, reload=False)
