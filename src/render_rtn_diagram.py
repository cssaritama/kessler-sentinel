"""Generates publication-grade RTN Coordinate Frame diagram for KESSLER-SENTINEL.

Renders docs/figures/rtn_frame_geometry.svg with generous spacing,
clean typography, and no overlapping elements.
"""

import os

def generate_rtn_svg(output_path: str = "docs/figures/rtn_frame_geometry.svg"):
    svg_content = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 940 400" width="100%" height="100%" style="background:#0B0F19; font-family:-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">
  <defs>
    <radialGradient id="covGrad" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#38BDF8" stop-opacity="0.30"/>
      <stop offset="65%" stop-color="#0284C7" stop-opacity="0.12"/>
      <stop offset="100%" stop-color="#0284C7" stop-opacity="0.0"/>
    </radialGradient>
    <radialGradient id="satGlow" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#38BDF8" stop-opacity="0.9"/>
      <stop offset="100%" stop-color="#38BDF8" stop-opacity="0"/>
    </radialGradient>
    <radialGradient id="chaserGlow" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#EF4444" stop-opacity="0.95"/>
      <stop offset="100%" stop-color="#EF4444" stop-opacity="0"/>
    </radialGradient>
    <marker id="arrowCyanRTN" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#38BDF8"/>
    </marker>
    <filter id="badgeShadow" x="-10%" y="-10%" width="120%" height="120%">
      <feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="#000000" flood-opacity="0.5"/>
    </filter>
  </defs>

  <!-- Title & Frame Header Bar -->
  <g transform="translate(36, 28)">
    <rect x="0" y="0" width="168" height="24" rx="5" fill="#1E293B" stroke="#0284C7" stroke-width="1"/>
    <text x="12" y="16" fill="#38BDF8" font-size="10" font-weight="700" letter-spacing="1.0">ASTRODYNAMICS FRAME</text>
    <text x="186" y="17" fill="#F8FAFC" font-size="13.5" font-weight="700">Relative Orbital Geometry at TCA (RTN Coordinate System)</text>
    <text x="0" y="44" fill="#94A3B8" font-size="10">Local-vertical / local-horizontal reference frame centered at primary spacecraft at Time of Closest Approach</text>
  </g>

  <!-- Main Reference Coordinate System Origin at Primary Spacecraft -->
  <!-- Shifted down and right to give generous clearance for all axis labels -->
  <g transform="translate(140, 275)">

    <!-- ================================================================ -->
    <!-- AXIS 1: RADIAL (+R) VERTICAL ZENITH                              -->
    <!-- ================================================================ -->
    <line x1="0" y1="0" x2="0" y2="-175" stroke="#38BDF8" stroke-width="2.2" marker-end="url(#arrowCyanRTN)"/>
    
    <!-- Clean Pill Badge for Radial Axis Label (Spaced safely above arrow) -->
    <g transform="translate(-18, -208)" filter="url(#badgeShadow)">
      <rect x="0" y="0" width="124" height="24" rx="5" fill="#0C2D48" stroke="#38BDF8" stroke-width="1.2"/>
      <text x="10" y="16" fill="#38BDF8" font-size="11" font-weight="700" letter-spacing="0.5">+R : Radial Axis</text>
    </g>
    <text x="114" y="-192" fill="#E2E8F0" font-size="9.5" font-weight="600">[ Altitude Separation ΔR ]</text>
    <text x="14" y="-168" fill="#94A3B8" font-size="8.5">Governs semi-major axis intersection</text>
    <text x="14" y="-154" fill="#64748B" font-size="8.5">Most critical flight collision risk vector</text>

    <!-- ================================================================ -->
    <!-- AXIS 2: TRANSVERSE / IN-TRACK (+T) HORIZONTAL VELOCITY           -->
    <!-- ================================================================ -->
    <line x1="0" y1="0" x2="520" y2="0" stroke="#38BDF8" stroke-width="2.2" marker-end="url(#arrowCyanRTN)"/>
    
    <g transform="translate(535, -12)" filter="url(#badgeShadow)">
      <rect x="0" y="0" width="144" height="24" rx="5" fill="#0C2D48" stroke="#38BDF8" stroke-width="1.2"/>
      <text x="10" y="16" fill="#38BDF8" font-size="11" font-weight="700" letter-spacing="0.5">+T : Transverse (In-Track)</text>
    </g>
    <text x="542" y="26" fill="#E2E8F0" font-size="9.5" font-weight="600">[ Along-Track Separation ΔT ]</text>
    <text x="542" y="40" fill="#94A3B8" font-size="8.5">Orbital velocity vector direction</text>
    <text x="542" y="52" fill="#64748B" font-size="8.5">Dominant atmospheric drag uncertainty (σ_T ≫ σ_R)</text>

    <!-- ================================================================ -->
    <!-- AXIS 3: NORMAL / CROSS-TRACK (+N) DEPTH NORMAL                   -->
    <!-- ================================================================ -->
    <line x1="0" y1="0" x2="-68" y2="58" stroke="#38BDF8" stroke-width="2.2" marker-end="url(#arrowCyanRTN)"/>
    
    <g transform="translate(-130, 72)" filter="url(#badgeShadow)">
      <rect x="0" y="0" width="112" height="24" rx="5" fill="#0C2D48" stroke="#38BDF8" stroke-width="1.2"/>
      <text x="10" y="16" fill="#38BDF8" font-size="11" font-weight="700" letter-spacing="0.5">+N : Normal Axis</text>
    </g>
    <text x="-130" y="110" fill="#E2E8F0" font-size="9.5" font-weight="600">[ Cross-Track ΔN ]</text>
    <text x="-130" y="124" fill="#64748B" font-size="8.5">Orbit plane normal vector</text>

    <!-- ================================================================ -->
    <!-- PRIMARY SPACECRAFT AT ORIGIN (0, 0, 0)                           -->
    <!-- ================================================================ -->
    <circle cx="0" cy="0" r="16" fill="url(#satGlow)"/>
    <circle cx="0" cy="0" r="5" fill="#38BDF8" stroke="#FFFFFF" stroke-width="1.5"/>
    <g transform="translate(-20, 20)">
      <rect x="0" y="0" width="155" height="22" rx="4" fill="#1E293B" stroke="#475569" stroke-width="0.8"/>
      <text x="10" y="15" fill="#F8FAFC" font-size="9.5" font-weight="700">Primary Satellite (0, 0, 0)</text>
    </g>

    <!-- ================================================================ -->
    <!-- CHASER OBJECT & COVARIANCE DISPERSION ELLIPSOID                  -->
    <!-- Encounter Position: dT = 310m, dR = -75m                         -->
    <!-- ================================================================ -->
    <g transform="translate(310, -75)">
      <!-- 3D Covariance Dispersion Ellipse (along-track elongation) -->
      <ellipse cx="0" cy="0" rx="155" ry="38" fill="url(#covGrad)" stroke="#38BDF8" stroke-width="1.4" stroke-dasharray="6,4"/>
      <ellipse cx="0" cy="8" rx="150" ry="34" fill="none" stroke="#0284C7" stroke-width="0.8" stroke-dasharray="3,3" opacity="0.6"/>

      <!-- Chaser Core Point with Red Glow Indicator -->
      <circle cx="0" cy="0" r="14" fill="url(#chaserGlow)"/>
      <circle cx="0" cy="0" r="5" fill="#EF4444" stroke="#FFFFFF" stroke-width="1.8"/>

      <!-- Chaser Object Badge (Pill above point) -->
      <g transform="translate(-75, -50)" filter="url(#badgeShadow)">
        <rect x="0" y="0" width="150" height="26" rx="5" fill="#1E1B4B" stroke="#A855F7" stroke-width="1.2"/>
        <circle cx="14" cy="13" r="4" fill="#EF4444"/>
        <text x="25" y="17" fill="#F8FAFC" font-size="10" font-weight="700">Chaser Object (*)</text>
      </g>

      <!-- Covariance Ellipsoid Spec Tag (Cleanly positioned below ellipse) -->
      <g transform="translate(-115, 48)">
        <rect x="0" y="0" width="230" height="34" rx="4" fill="#0E1726" stroke="#1E293B" stroke-width="1"/>
        <text x="12" y="15" fill="#38BDF8" font-size="9" font-weight="700">Covariance Ellipsoid: σ_T ≫ σ_R</text>
        <text x="12" y="28" fill="#94A3B8" font-size="8">Thermospheric drag inflates along-track uncertainty</text>
      </g>

      <!-- Miss Distance Vector from Origin to Chaser -->
      <line x1="-310" y1="75" x2="0" y2="0" stroke="#F59E0B" stroke-width="1.4" stroke-dasharray="4,4"/>
    </g>

    <!-- Miss Distance Label on Vector -->
    <g transform="translate(110, -22)" filter="url(#badgeShadow)">
      <rect x="0" y="0" width="132" height="20" rx="4" fill="#1C1917" stroke="#F59E0B" stroke-width="0.8"/>
      <text x="8" y="14" fill="#F59E0B" font-size="8.5" font-weight="700">Miss Distance |d| at TCA</text>
    </g>

    <!-- Coordinate Projection Helper Guides -->
    <!-- Along-Track (ΔT) Guide -->
    <line x1="0" y1="52" x2="310" y2="52" stroke="#475569" stroke-width="1" stroke-dasharray="3,3"/>
    <line x1="0" y1="47" x2="0" y2="57" stroke="#475569" stroke-width="1.2"/>
    <line x1="310" y1="47" x2="310" y2="57" stroke="#475569" stroke-width="1.2"/>
    <text x="155" y="66" fill="#94A3B8" font-size="9" font-weight="600" text-anchor="middle">Along-Track Separation (ΔT)</text>

    <!-- Radial Separation (ΔR) Guide on the Left -->
    <line x1="-42" y1="0" x2="-42" y2="-75" stroke="#475569" stroke-width="1" stroke-dasharray="3,3"/>
    <line x1="-47" y1="0" x2="-37" y2="0" stroke="#475569" stroke-width="1.2"/>
    <line x1="-47" y1="-75" x2="-37" y2="-75" stroke="#475569" stroke-width="1.2"/>
    <text x="-52" y="-35" fill="#94A3B8" font-size="9" font-weight="600" text-anchor="end">Radial (ΔR)</text>

  </g>
</svg>
"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg_content.strip())
    print(f"[SUCCESS] High-clarity RTN SVG generated: {output_path}")

if __name__ == "__main__":
    generate_rtn_svg()
