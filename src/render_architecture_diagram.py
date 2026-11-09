"""Generates publication-grade architecture diagrams for KESSLER-SENTINEL.

Renders both:
- docs/figures/system_architecture.png (300 DPI minimalist aerospace diagram)
- docs/figures/system_architecture.svg (Ultra-crisp responsive vector graphic)
"""

import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def generate_architecture_png(output_path: str = "docs/figures/system_architecture.png"):
    fig = plt.figure(figsize=(16, 7.2), dpi=300, facecolor="#0B0F19")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#0B0F19")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    # Title Banner
    ax.text(
        0.8, 6.6,
        "KESSLER-SENTINEL  //  AUTONOMOUS ORBITAL CONJUNCTION DECISION PIPELINE",
        fontsize=12,
        fontweight="bold",
        color="#38BDF8",
        family="sans-serif"
    )
    ax.text(
        0.8, 6.25,
        "Real-Time Low Earth Orbit (LEO) Collision Escalation Triage at T-48h Operational Screening Gate",
        fontsize=9,
        color="#94A3B8",
        family="sans-serif"
    )

    # Helper function to draw rounded cards
    def draw_card(x, y, w, h, bg_color, border_color, tag, title, bullets, tag_color="#38BDF8", title_color="#F8FAFC"):
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.08,rounding_size=0.15",
            facecolor=bg_color,
            edgecolor=border_color,
            linewidth=1.4,
            zorder=2
        )
        ax.add_patch(box)

        # Tag pill
        tag_bg = FancyBboxPatch(
            (x + 0.15, y + h - 0.38), len(tag) * 0.085 + 0.25, 0.24,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            facecolor="#1E293B",
            edgecolor=tag_color,
            linewidth=0.8,
            zorder=3
        )
        ax.add_patch(tag_bg)
        ax.text(
            x + 0.22, y + h - 0.26,
            tag,
            fontsize=6.5,
            fontweight="bold",
            color=tag_color,
            va="center",
            family="sans-serif",
            zorder=4
        )

        # Title
        ax.text(
            x + 0.15, y + h - 0.65,
            title,
            fontsize=8.5,
            fontweight="bold",
            color=title_color,
            va="top",
            family="sans-serif",
            zorder=4
        )

        # Bullets / Subtext
        curr_y = y + h - 1.05
        for line in bullets:
            ax.text(
                x + 0.15, curr_y,
                f"• {line}",
                fontsize=7.2,
                color="#94A3B8",
                va="top",
                family="sans-serif",
                zorder=4
            )
            curr_y -= 0.26

    # Flow arrows
    def draw_arrow(x1, y1, x2, y2, color="#475569", label=""):
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.8,
            color=color,
            zorder=5
        )
        ax.add_patch(arrow)
        if label:
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2 + 0.15
            ax.text(
                mid_x, mid_y, label,
                fontsize=7,
                fontweight="bold",
                color=color,
                ha="center",
                va="bottom",
                family="sans-serif",
                zorder=6
            )

    # 1. Raw CDM Ingestion
    draw_card(
        x=0.8, y=2.5, w=2.4, h=3.1,
        bg_color="#0F172A", border_color="#334155",
        tag="STEP 1 : TELEMETRY",
        title="Raw Conjunction\nMessage (CDM)",
        bullets=[
            "SSN / Space-Track Radar",
            "RTN State Vectors (r, v)",
            "Joint Covariance 3x3",
            "Solar Flux & Geomagnetic Ap",
            "Time-to-TCA (12h - 96h)"
        ]
    )

    # Arrow 1 -> 2
    draw_arrow(3.2, 4.05, 3.7, 4.05, color="#38BDF8")

    # 2. Astrodynamics Feature Engine
    draw_card(
        x=3.7, y=2.5, w=2.5, h=3.1,
        bg_color="#0F172A", border_color="#0284C7",
        tag="STEP 2 : ASTRODYNAMICS",
        title="Physics Feature\nEngine (src/features.py)",
        bullets=[
            "Mahalanobis Proximity D_M",
            "Kinetic Lethality log(I_leth)",
            "Thermospheric Drag P_drag",
            "Radial Margin |DR| / (s_R+1)",
            "Strict Leak-Free Imputation"
        ],
        tag_color="#38BDF8"
    )

    # Arrow 2 -> 3
    draw_arrow(6.2, 4.05, 6.7, 4.05, color="#38BDF8")

    # 3. Vectorization & XGBoost
    draw_card(
        x=6.7, y=2.5, w=2.5, h=3.1,
        bg_color="#0F172A", border_color="#6366F1",
        tag="STEP 3 : ML PIPELINE",
        title="Covariance Vectorizer\n& Tuned XGBoost",
        bullets=[
            "DictVectorizer (37 Feats)",
            "180 Trees, Max Depth 5",
            "Learning Rate eta = 0.08",
            "5-Fold CV: 0.9290 AUC",
            "Calibrated Prob P(Escalation)"
        ],
        tag_color="#818CF8"
    )

    # Arrow 3 -> 4 (Decision Gate)
    draw_arrow(9.2, 4.05, 9.7, 4.05, color="#818CF8")

    # 4. Asymmetric Decision Gate
    draw_card(
        x=9.7, y=2.9, w=2.3, h=2.3,
        bg_color="#1E1B4B", border_color="#A855F7",
        tag="STEP 4 : DECISION GATE",
        title="Cost Threshold\nP >= tau* (0.0550)",
        bullets=[
            "Asymmetric Risk Policy",
            "Cost Ratio: C_FN / C_FP = 25",
            "Safety Recall: 89.6%",
            "Propellant Loss Minimized"
        ],
        tag_color="#C084FC"
    )

    # Branching Arrows: 4 -> Red/Amber & 4 -> Green
    arrow_red = FancyArrowPatch(
        (12.0, 4.4), (12.4, 5.0),
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.8,
        color="#EF4444",
        zorder=5
    )
    ax.add_patch(arrow_red)
    ax.text(12.05, 4.8, "P >= 0.0550\n(Risk High)", fontsize=6.8, fontweight="bold", color="#EF4444", va="bottom")

    arrow_green = FancyArrowPatch(
        (12.0, 3.6), (12.4, 2.9),
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.8,
        color="#10B981",
        zorder=5
    )
    ax.add_patch(arrow_green)
    ax.text(12.05, 3.1, "P < 0.0550\n(Nominal)", fontsize=6.8, fontweight="bold", color="#10B981", va="top")

    # Red/Amber Decision Card
    draw_card(
        x=12.4, y=4.1, w=2.8, h=1.85,
        bg_color="#450A0A", border_color="#DC2626",
        tag="DIRECTIVE : ESCALATION",
        title="RED / AMBER: Trigger CAM",
        bullets=[
            "Sequence thruster warm-up",
            "Attitude slew to Delta-V burn",
            "Task high-priority radar"
        ],
        tag_color="#F87171",
        title_color="#FCA5A5"
    )

    # Green Decision Card
    draw_card(
        x=12.4, y=1.9, w=2.8, h=1.85,
        bg_color="#064E3B", border_color="#059669",
        tag="DIRECTIVE : NOMINAL",
        title="GREEN: Preserve Propellant",
        bullets=[
            "Nominal orbital passage",
            "Hold Delta-V propellant",
            "Retain science payload point"
        ],
        tag_color="#34D399",
        title_color="#6EE7B7"
    )

    # Delivery Footer Bar
    footer_box = FancyBboxPatch(
        (0.8, 0.45), 14.4, 1.05,
        boxstyle="round,pad=0.06,rounding_size=0.12",
        facecolor="#0F172A",
        edgecolor="#1E293B",
        linewidth=1.2,
        zorder=2
    )
    ax.add_patch(footer_box)

    ax.text(
        1.1, 1.15,
        "PRODUCTION DEPLOYMENT & VERIFICATION INTERFACES",
        fontsize=8.5,
        fontweight="bold",
        color="#F8FAFC",
        family="sans-serif"
    )
    ax.text(
        1.1, 0.72,
        "• REST API: POST /predict (FastAPI :9696, Pydantic strict schema)    • Health: GET /health    • Web Console: GET /    • Docker: python:3.11-slim    • Cloud: Render & GCP Cloud Run",
        fontsize=7.3,
        color="#94A3B8",
        family="sans-serif"
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="#0B0F19")
    plt.close(fig)
    print(f"[SUCCESS] Architecture PNG generated: {output_path}")


def generate_architecture_svg(output_path: str = "docs/figures/system_architecture.svg"):
    svg_content = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1280 540" width="100%" height="100%" style="background:#0B0F19; font-family:-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">
  <defs>
    <linearGradient id="cardGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#141E33"/>
      <stop offset="100%" stop-color="#0E1726"/>
    </linearGradient>
    <linearGradient id="gateGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#241B44"/>
      <stop offset="100%" stop-color="#191230"/>
    </linearGradient>
    <linearGradient id="redGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#3B1219"/>
      <stop offset="100%" stop-color="#230B0F"/>
    </linearGradient>
    <linearGradient id="greenGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#0E3326"/>
      <stop offset="100%" stop-color="#091E17"/>
    </linearGradient>
    <filter id="subtleShadow" x="-5%" y="-5%" width="110%" height="110%">
      <feDropShadow dx="0" dy="4" stdDeviation="6" flood-color="#000000" flood-opacity="0.4"/>
    </filter>
    <marker id="arrowCyan" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#38BDF8"/>
    </marker>
    <marker id="arrowIndigo" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#818CF8"/>
    </marker>
    <marker id="arrowRed" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#EF4444"/>
    </marker>
    <marker id="arrowGreen" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#10B981"/>
    </marker>
  </defs>

  <!-- Header Title -->
  <g transform="translate(50, 42)">
    <text x="0" y="0" fill="#38BDF8" font-size="14" font-weight="700" letter-spacing="1.2">KESSLER-SENTINEL  //  AUTONOMOUS ORBITAL CONJUNCTION DECISION PIPELINE</text>
    <text x="0" y="22" fill="#94A3B8" font-size="12">Real-Time Low Earth Orbit (LEO) Collision Escalation Triage at T-48h Operational Screening Gate</text>
  </g>

  <!-- CARD 1: RAW CDM TELEMETRY -->
  <g transform="translate(50, 95)" filter="url(#subtleShadow)">
    <rect width="210" height="260" rx="10" fill="url(#cardGrad)" stroke="#334155" stroke-width="1.2"/>
    <rect x="14" y="14" width="112" height="20" rx="4" fill="#1E293B" stroke="#64748B" stroke-width="0.8"/>
    <text x="22" y="28" fill="#94A3B8" font-size="9" font-weight="700" letter-spacing="0.5">STEP 1 : TELEMETRY</text>
    <text x="14" y="58" fill="#F8FAFC" font-size="13" font-weight="700">Raw Conjunction</text>
    <text x="14" y="76" fill="#F8FAFC" font-size="13" font-weight="700">Message (CDM)</text>
    <g fill="#94A3B8" font-size="10.5">
      <text x="14" y="112">• SSN / Space-Track Radar</text>
      <text x="14" y="134">• RTN State Vectors (r, v)</text>
      <text x="14" y="156">• Joint Covariance 3x3</text>
      <text x="14" y="178">• Solar Flux &amp; Geomagnetic</text>
      <text x="14" y="200">• Lead Time to TCA (12-96h)</text>
      <text x="14" y="222">• Object Class &amp; Thrusters</text>
    </g>
  </g>

  <!-- CONNECTOR 1 -> 2 -->
  <line x1="262" y1="225" x2="298" y2="225" stroke="#38BDF8" stroke-width="2" marker-end="url(#arrowCyan)"/>

  <!-- CARD 2: ASTRODYNAMICS ENGINE -->
  <g transform="translate(302, 95)" filter="url(#subtleShadow)">
    <rect width="220" height="260" rx="10" fill="url(#cardGrad)" stroke="#0284C7" stroke-width="1.4"/>
    <rect x="14" y="14" width="138" height="20" rx="4" fill="#0C2D48" stroke="#0284C7" stroke-width="0.8"/>
    <text x="22" y="28" fill="#38BDF8" font-size="9" font-weight="700" letter-spacing="0.5">STEP 2 : ASTRODYNAMICS</text>
    <text x="14" y="58" fill="#F8FAFC" font-size="13" font-weight="700">Physics Feature Engine</text>
    <text x="14" y="76" fill="#38BDF8" font-size="11" font-weight="600">(src/features.py)</text>
    <g fill="#94A3B8" font-size="10.5">
      <text x="14" y="112">• Mahalanobis Proximity D_M</text>
      <text x="14" y="134">• Kinetic Lethality log(I_leth)</text>
      <text x="14" y="156">• Thermospheric Drag P_drag</text>
      <text x="14" y="178">• Radial Margin |ΔR|/(σ_R+1)</text>
      <text x="14" y="200">• Training-Fitted Imputation</text>
      <text x="14" y="222">• Strict Data-Leak Prevention</text>
    </g>
  </g>

  <!-- CONNECTOR 2 -> 3 -->
  <line x1="524" y1="225" x2="560" y2="225" stroke="#38BDF8" stroke-width="2" marker-end="url(#arrowCyan)"/>

  <!-- CARD 3: ML PIPELINE -->
  <g transform="translate(564, 95)" filter="url(#subtleShadow)">
    <rect width="215" height="260" rx="10" fill="url(#cardGrad)" stroke="#6366F1" stroke-width="1.4"/>
    <rect x="14" y="14" width="124" height="20" rx="4" fill="#201C47" stroke="#6366F1" stroke-width="0.8"/>
    <text x="22" y="28" fill="#818CF8" font-size="9" font-weight="700" letter-spacing="0.5">STEP 3 : ML PIPELINE</text>
    <text x="14" y="58" fill="#F8FAFC" font-size="13" font-weight="700">DictVectorizer &amp;</text>
    <text x="14" y="76" fill="#F8FAFC" font-size="13" font-weight="700">Tuned XGBoost</text>
    <g fill="#94A3B8" font-size="10.5">
      <text x="14" y="112">• 37 Orthogonal Features</text>
      <text x="14" y="134">• 180 Estimators, Depth 5</text>
      <text x="14" y="156">• Learning Rate η = 0.08</text>
      <text x="14" y="178">• 5-Fold CV: 0.9290 AUC</text>
      <text x="14" y="200">• Calibrated P(Escalation)</text>
      <text x="14" y="222">• Atomic Artifact Serialization</text>
    </g>
  </g>

  <!-- CONNECTOR 3 -> 4 -->
  <line x1="781" y1="225" x2="817" y2="225" stroke="#818CF8" stroke-width="2" marker-end="url(#arrowIndigo)"/>

  <!-- CARD 4: ASYMMETRIC DECISION GATE -->
  <g transform="translate(821, 115)" filter="url(#subtleShadow)">
    <rect width="180" height="220" rx="10" fill="url(#gateGrad)" stroke="#A855F7" stroke-width="1.5"/>
    <rect x="12" y="14" width="128" height="20" rx="4" fill="#3B1C5A" stroke="#C084FC" stroke-width="0.8"/>
    <text x="18" y="28" fill="#E9D5FF" font-size="8.5" font-weight="700" letter-spacing="0.5">STEP 4 : DECISION GATE</text>
    <text x="12" y="58" fill="#F8FAFC" font-size="12" font-weight="700">Operational Gate</text>
    <text x="12" y="76" fill="#C084FC" font-size="12" font-weight="700">P ≥ τ* (0.0550)</text>
    <g fill="#CBD5E1" font-size="10">
      <text x="12" y="108">• Asymmetric Cost Loss</text>
      <text x="12" y="128">• Ratio C_FN / C_FP = 25</text>
      <text x="12" y="148">• Safety Recall: 89.6%</text>
      <text x="12" y="168">• F2-Score: 0.7708</text>
      <text x="12" y="188">• Protects Delta-V Budget</text>
    </g>
  </g>

  <!-- BRANCHING ARROWS -->
  <!-- Upper: To RED/AMBER -->
  <path d="M 1003 190 L 1028 155 L 1048 155" fill="none" stroke="#EF4444" stroke-width="2" marker-end="url(#arrowRed)"/>
  <text x="1012" y="145" fill="#EF4444" font-size="9" font-weight="700">P ≥ 0.0550</text>

  <!-- Lower: To GREEN -->
  <path d="M 1003 260 L 1028 295 L 1048 295" fill="none" stroke="#10B981" stroke-width="2" marker-end="url(#arrowGreen)"/>
  <text x="1012" y="315" fill="#10B981" font-size="9" font-weight="700">P &lt; 0.0550</text>

  <!-- CARD 5A: RED / AMBER DIRECTIVE -->
  <g transform="translate(1052, 95)" filter="url(#subtleShadow)">
    <rect width="180" height="120" rx="8" fill="url(#redGrad)" stroke="#EF4444" stroke-width="1.4"/>
    <text x="12" y="24" fill="#FCA5A5" font-size="8.5" font-weight="700" letter-spacing="0.5">DIRECTIVE : ESCALATION</text>
    <text x="12" y="44" fill="#FEE2E2" font-size="11.5" font-weight="700">RED / AMBER : CAM</text>
    <g fill="#FCA5A5" font-size="9.5">
      <text x="12" y="66">• Sequence thruster warm-up</text>
      <text x="12" y="84">• Attitude slew for Δv burn</text>
      <text x="12" y="102">• Task high-priority radar</text>
    </g>
  </g>

  <!-- CARD 5B: GREEN DIRECTIVE -->
  <g transform="translate(1052, 235)" filter="url(#subtleShadow)">
    <rect width="180" height="120" rx="8" fill="url(#greenGrad)" stroke="#10B981" stroke-width="1.4"/>
    <text x="12" y="24" fill="#6EE7B7" font-size="8.5" font-weight="700" letter-spacing="0.5">DIRECTIVE : NOMINAL PASS</text>
    <text x="12" y="44" fill="#D1FAE5" font-size="11.5" font-weight="700">GREEN : Hold Maneuver</text>
    <g fill="#6EE7B7" font-size="9.5">
      <text x="12" y="66">• Nominal clearance verified</text>
      <text x="12" y="84">• Conserve onboard propellant</text>
      <text x="12" y="102">• Maintain scientific pointing</text>
    </g>
  </g>

  <!-- FOOTER: INTERFACES & DEPLOYMENT -->
  <g transform="translate(50, 395)" filter="url(#subtleShadow)">
    <rect width="1182" height="100" rx="10" fill="#0E1726" stroke="#1E293B" stroke-width="1.2"/>
    <text x="20" y="32" fill="#F8FAFC" font-size="12" font-weight="700" letter-spacing="0.8">PRODUCTION INTERFACES &amp; CLOUD DEPLOYMENT</text>
    <g fill="#94A3B8" font-size="11">
      <text x="20" y="62">
        <tspan fill="#38BDF8" font-weight="600">REST API :</tspan> POST /predict (FastAPI :9696, Pydantic Schema)    
        <tspan fill="#38BDF8" font-weight="600">Health :</tspan> GET /health    
        <tspan fill="#38BDF8" font-weight="600">Flight Console :</tspan> GET / (Interactive Web UI)
      </text>
      <text x="20" y="84">
        <tspan fill="#818CF8" font-weight="600">Container :</tspan> Docker (python:3.11-slim)    
        <tspan fill="#818CF8" font-weight="600">Cloud :</tspan> Render Blueprint (render.yaml) &amp; Google Cloud Run    
        <tspan fill="#10B981" font-weight="600">Verification :</tspan> test_service.py (Red/Amber/Green Scenarios)
      </text>
    </g>
  </g>
</svg>
"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg_content.strip())
    print(f"[SUCCESS] Architecture SVG generated: {output_path}")

if __name__ == "__main__":
    generate_architecture_png()
    generate_architecture_svg()
