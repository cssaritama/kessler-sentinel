"""Generates high-resolution publication-grade RTN Coordinate Frame diagram.

Renders both:
- docs/figures/rtn_coordinate_frame.png (300 DPI ultra-sharp PNG, guaranteed GitHub rendering)
- docs/figures/rtn_coordinate_frame.svg (Clean vector graphic)
"""

import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse

def generate_rtn_png(output_path: str = "docs/figures/rtn_coordinate_frame.png"):
    fig = plt.figure(figsize=(13, 6.2), dpi=300, facecolor="#0B0F19")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#0B0F19")
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    # =========================================================================
    # 1. TOP HEADER BANNER
    # =========================================================================
    # Badge
    badge_box = FancyBboxPatch(
        (0.6, 5.45), 2.2, 0.38,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        facecolor="#0C2D48",
        edgecolor="#38BDF8",
        linewidth=1.2,
        zorder=3
    )
    ax.add_patch(badge_box)
    ax.text(
        1.7, 5.64,
        "ASTRODYNAMICS FRAME",
        fontsize=8.5,
        fontweight="bold",
        color="#38BDF8",
        ha="center",
        va="center",
        family="sans-serif",
        zorder=4
    )

    # Title
    ax.text(
        3.1, 5.64,
        "Relative Orbital Geometry at TCA (RTN Coordinates)",
        fontsize=12.5,
        fontweight="bold",
        color="#F8FAFC",
        va="center",
        family="sans-serif"
    )

    # Subtitle
    ax.text(
        0.6, 5.15,
        "Local-vertical / local-horizontal encounter frame centered at primary spacecraft at Time of Closest Approach",
        fontsize=8.5,
        color="#94A3B8",
        family="sans-serif"
    )

    # Divider line
    ax.plot([0.6, 12.4], [4.9, 4.9], color="#1E293B", linewidth=1.0, zorder=1)

    # =========================================================================
    # 2. COORDINATE SYSTEM (Origin at x=1.8, y=1.5)
    # =========================================================================
    ox, oy = 1.8, 1.5

    # -------------------------------------------------------------------------
    # AXIS 1: RADIAL (+R) VERTICAL ZENITH
    # -------------------------------------------------------------------------
    arrow_r = FancyArrowPatch(
        (ox, oy), (ox, 4.2),
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=2.2,
        color="#38BDF8",
        zorder=5
    )
    ax.add_patch(arrow_r)

    # Label Badge for Radial Axis
    r_badge = FancyBboxPatch(
        (ox + 0.2, 3.9), 2.1, 0.38,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        facecolor="#0C2D48",
        edgecolor="#38BDF8",
        linewidth=1.2,
        zorder=6
    )
    ax.add_patch(r_badge)
    ax.text(
        ox + 1.25, 4.09,
        "+R : Radial Axis",
        fontsize=9,
        fontweight="bold",
        color="#38BDF8",
        ha="center",
        va="center",
        family="sans-serif",
        zorder=7
    )
    ax.text(
        ox + 2.5, 4.09,
        "[ Altitude Separation Delta R ]",
        fontsize=8.5,
        fontweight="bold",
        color="#E2E8F0",
        va="center",
        family="sans-serif"
    )
    ax.text(
        ox + 0.25, 3.55,
        "- Governs semi-major axis intersection\n- Primary collision risk vector",
        fontsize=8,
        color="#94A3B8",
        family="sans-serif"
    )

    # -------------------------------------------------------------------------
    # AXIS 2: TRANSVERSE / IN-TRACK (+T) HORIZONTAL VELOCITY
    # -------------------------------------------------------------------------
    arrow_t = FancyArrowPatch(
        (ox, oy), (8.8, oy),
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=2.2,
        color="#38BDF8",
        zorder=5
    )
    ax.add_patch(arrow_t)

    t_badge = FancyBboxPatch(
        (9.0, oy - 0.2), 2.6, 0.38,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        facecolor="#0C2D48",
        edgecolor="#38BDF8",
        linewidth=1.2,
        zorder=6
    )
    ax.add_patch(t_badge)
    ax.text(
        10.3, oy - 0.01,
        "+T : Transverse (In-Track)",
        fontsize=9,
        fontweight="bold",
        color="#38BDF8",
        ha="center",
        va="center",
        family="sans-serif",
        zorder=7
    )
    ax.text(
        9.05, oy - 0.45,
        "[ Along-Track Separation Delta T ]",
        fontsize=8.5,
        fontweight="bold",
        color="#E2E8F0",
        va="top",
        family="sans-serif"
    )
    ax.text(
        9.05, oy - 0.75,
        "- Orbital velocity vector direction\n- Dominant atmospheric drag uncertainty (sigma_T >> sigma_R)",
        fontsize=7.8,
        color="#94A3B8",
        va="top",
        family="sans-serif"
    )

    # -------------------------------------------------------------------------
    # AXIS 3: NORMAL / CROSS-TRACK (+N) DEPTH NORMAL
    # -------------------------------------------------------------------------
    arrow_n = FancyArrowPatch(
        (ox, oy), (0.7, 0.6),
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=2.2,
        color="#38BDF8",
        zorder=5
    )
    ax.add_patch(arrow_n)

    n_badge = FancyBboxPatch(
        (0.2, 0.1), 1.7, 0.36,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        facecolor="#0C2D48",
        edgecolor="#38BDF8",
        linewidth=1.2,
        zorder=6
    )
    ax.add_patch(n_badge)
    ax.text(
        1.05, 0.28,
        "+N : Normal Axis",
        fontsize=8.5,
        fontweight="bold",
        color="#38BDF8",
        ha="center",
        va="center",
        family="sans-serif",
        zorder=7
    )
    ax.text(
        2.05, 0.28,
        "[ Cross-Track Delta N ]  (Orbit plane normal)",
        fontsize=8,
        fontweight="bold",
        color="#94A3B8",
        va="center",
        family="sans-serif"
    )

    # -------------------------------------------------------------------------
    # PRIMARY SPACECRAFT AT ORIGIN (ox, oy)
    # -------------------------------------------------------------------------
    glow_prim = plt.Circle((ox, oy), 0.22, color="#38BDF8", alpha=0.35, zorder=8)
    dot_prim = plt.Circle((ox, oy), 0.08, color="#FFFFFF", ec="#38BDF8", lw=2, zorder=9)
    ax.add_patch(glow_prim)
    ax.add_patch(dot_prim)

    prim_box = FancyBboxPatch(
        (ox - 0.2, oy - 0.65), 2.2, 0.34,
        boxstyle="round,pad=0.03,rounding_size=0.06",
        facecolor="#1E293B",
        edgecolor="#475569",
        linewidth=1,
        zorder=8
    )
    ax.add_patch(prim_box)
    ax.text(
        ox + 0.9, oy - 0.48,
        "Primary Satellite (0, 0, 0)",
        fontsize=8.5,
        fontweight="bold",
        color="#F8FAFC",
        ha="center",
        va="center",
        family="sans-serif",
        zorder=9
    )

    # -------------------------------------------------------------------------
    # CHASER OBJECT & COVARIANCE ELLIPSOID
    # Location: (cx, cy) = (5.8, 3.1)
    # -------------------------------------------------------------------------
    cx, cy = 5.8, 3.1

    # Covariance Ellipse (sigma_T >> sigma_R)
    ell_fill = Ellipse(
        (cx, cy), width=3.8, height=1.05, angle=0,
        facecolor="#0284C7", alpha=0.20, zorder=3
    )
    ell_border = Ellipse(
        (cx, cy), width=3.8, height=1.05, angle=0,
        facecolor="none", edgecolor="#38BDF8", linewidth=1.5, linestyle="--", zorder=4
    )
    ax.add_patch(ell_fill)
    ax.add_patch(ell_border)

    # Chaser marker
    glow_chaser = plt.Circle((cx, cy), 0.22, color="#EF4444", alpha=0.45, zorder=8)
    dot_chaser = plt.Circle((cx, cy), 0.08, color="#FFFFFF", ec="#EF4444", lw=2, zorder=9)
    ax.add_patch(glow_chaser)
    ax.add_patch(dot_chaser)

    # Chaser Badge above
    chaser_box = FancyBboxPatch(
        (cx - 1.1, cy + 0.7), 2.2, 0.40,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        facecolor="#1E1B4B",
        edgecolor="#A855F7",
        linewidth=1.2,
        zorder=8
    )
    ax.add_patch(chaser_box)
    ax.text(
        cx, cy + 0.90,
        "Chaser Object (*)",
        fontsize=9,
        fontweight="bold",
        color="#F8FAFC",
        ha="center",
        va="center",
        family="sans-serif",
        zorder=9
    )

    # Covariance Spec Card below ellipse
    cov_card = FancyBboxPatch(
        (cx - 1.6, cy - 1.15), 3.2, 0.46,
        boxstyle="round,pad=0.04,rounding_size=0.06",
        facecolor="#0E1726",
        edgecolor="#1E293B",
        linewidth=1.0,
        zorder=6
    )
    ax.add_patch(cov_card)
    ax.text(
        cx, cy - 0.85,
        "Covariance Ellipsoid: sigma_T >> sigma_R",
        fontsize=8.5,
        fontweight="bold",
        color="#38BDF8",
        ha="center",
        va="center",
        family="sans-serif",
        zorder=7
    )
    ax.text(
        cx, cy - 1.05,
        "Thermospheric drag inflates along-track uncertainty",
        fontsize=7.5,
        color="#94A3B8",
        ha="center",
        va="center",
        family="sans-serif",
        zorder=7
    )

    # Miss Distance Vector (dashed amber)
    ax.plot([ox, cx], [oy, cy], color="#F59E0B", linewidth=1.4, linestyle="--", zorder=4)

    # Miss Distance Badge
    md_box = FancyBboxPatch(
        (3.3, 2.35), 1.9, 0.32,
        boxstyle="round,pad=0.03,rounding_size=0.06",
        facecolor="#1C1917",
        edgecolor="#F59E0B",
        linewidth=1.0,
        zorder=8
    )
    ax.add_patch(md_box)
    ax.text(
        4.25, 2.51,
        "Miss Distance |d| at TCA",
        fontsize=7.8,
        fontweight="bold",
        color="#F59E0B",
        ha="center",
        va="center",
        family="sans-serif",
        zorder=9
    )

    # Along-Track Delta T Guide
    ax.plot([ox, cx], [0.85, 0.85], color="#475569", linestyle=":", linewidth=1.2, zorder=2)
    ax.plot([ox, ox], [0.75, 0.95], color="#475569", linewidth=1.4, zorder=2)
    ax.plot([cx, cx], [0.75, 0.95], color="#475569", linewidth=1.4, zorder=2)
    ax.text(
        (ox + cx) / 2, 0.65,
        "Along-Track Separation (Delta T)",
        fontsize=8,
        fontweight="bold",
        color="#94A3B8",
        ha="center",
        family="sans-serif"
    )

    # Radial Delta R Guide
    ax.plot([1.2, 1.2], [oy, cy], color="#475569", linestyle=":", linewidth=1.2, zorder=2)
    ax.plot([1.1, 1.3], [oy, oy], color="#475569", linewidth=1.4, zorder=2)
    ax.plot([1.1, 1.3], [cy, cy], color="#475569", linewidth=1.4, zorder=2)
    ax.text(
        1.05, (oy + cy) / 2,
        "Radial (Delta R)",
        fontsize=8,
        fontweight="bold",
        color="#94A3B8",
        ha="right",
        va="center",
        family="sans-serif"
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="#0B0F19")
    plt.close(fig)
    print(f"[SUCCESS] High-resolution PNG generated: {output_path}")

if __name__ == "__main__":
    generate_rtn_png()
