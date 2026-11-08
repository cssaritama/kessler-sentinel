"""Minimalist Scientific & Editorial Figures Generator for KESSLER-SENTINEL.

Renders 5 publication-grade figures conforming to international editorial standards
(Nature Physics / Financial Times visual philosophy: high whitespace ratio,
refined typography, despine framing, and disciplined chromatic palette).
"""

import os
import sys

# Ensure repository root is on python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mutual_info_score, roc_auc_score, fbeta_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from src.features import (
    compute_astrodynamics_features,
    CATEGORICAL_FEATURES,
    NUMERICAL_RAW_FEATURES,
    ENGINEERED_FEATURES,
    TARGET_COLUMN,
)

# Refined Editorial Color Palette
COLOR_SLATE_DARK = "#0F172A"
COLOR_SLATE_MED = "#475569"
COLOR_SLATE_LIGHT = "#94A3B8"
COLOR_GRID = "#F1F5F9"
COLOR_ACCENT_RED = "#E11D48"
COLOR_ACCENT_BLUE = "#2563EB"
COLOR_ACCENT_TEAL = "#0D9488"
COLOR_ACCENT_AMBER = "#D97706"


def configure_editorial_style():
    """Apply minimalist editorial styling with clean margins and no bounding boxes."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "600",
        "axes.titlecolor": COLOR_SLATE_DARK,
        "axes.labelsize": 10,
        "axes.labelweight": "500",
        "axes.labelcolor": COLOR_SLATE_MED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.color": COLOR_SLATE_MED,
        "ytick.color": COLOR_SLATE_MED,
        "axes.edgecolor": "#CBD5E1",
        "axes.linewidth": 0.7,
        "axes.facecolor": "#FFFFFF",
        "figure.facecolor": "#FFFFFF",
        "grid.color": COLOR_GRID,
        "grid.linestyle": "-",
        "grid.linewidth": 0.8,
        "grid.alpha": 1.0,
    })


def generate_all_figures(data_path: str = "data/leo_conjunction_cdm_events.csv", output_dir: str = "docs/figures"):
    os.makedirs(output_dir, exist_ok=True)
    configure_editorial_style()

    print(f"[FIGURES] Loading telemetry from {data_path}...")
    df_raw = pd.read_csv(data_path)
    df = compute_astrodynamics_features(df_raw)

    # -------------------------------------------------------------------------
    # FIGURE 1: Escalation Prevalence by Orbital Shell & Object Type
    # -------------------------------------------------------------------------
    print("[FIGURES] Generating Figure 1 (Minimalist Regime & Object Breakdown)...")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), dpi=300)

    # Left: Orbital Regime
    reg_df = (
        df.groupby("orbital_regime_band")[TARGET_COLUMN]
        .mean()
        .mul(100.0)
        .sort_values(ascending=True)
        .reset_index()
    )
    # Clean display labels
    reg_df["label"] = reg_df["orbital_regime_band"].str.replace("_", " ")

    bars1 = axes[0].barh(reg_df["label"], reg_df[TARGET_COLUMN], color="#334155", height=0.55, edgecolor="none")
    # Highlight highest regime
    bars1[-1].set_color(COLOR_ACCENT_RED)

    axes[0].axvline(df[TARGET_COLUMN].mean() * 100, color=COLOR_SLATE_LIGHT, linestyle="--", linewidth=1.0, label="Global Average (16.5%)")
    axes[0].set_title("Conjunction Escalation Rate by LEO Regime Band", pad=14, loc="left")
    axes[0].set_xlabel("Escalation Rate (%)", labelpad=8)
    axes[0].set_xlim(0, 30)
    axes[0].grid(axis="x", zorder=0)
    axes[0].grid(axis="y", visible=False)
    sns.despine(ax=axes[0], left=True, top=True, right=True)

    for p in axes[0].patches:
        w = p.get_width()
        if w > 0:
            axes[0].annotate(f"{w:.1f}%", (w + 0.6, p.get_y() + p.get_height() / 2.0), va="center", fontsize=9, color=COLOR_SLATE_DARK, weight="500")

    axes[0].legend(frameon=False, loc="lower right", fontsize=9)

    # Right: Chaser Object Type
    chaser_df = (
        df.groupby("chaser_object_type")[TARGET_COLUMN]
        .mean()
        .mul(100.0)
        .sort_values(ascending=True)
        .reset_index()
    )
    chaser_df["label"] = chaser_df["chaser_object_type"].str.replace("_", " ")

    bars2 = axes[1].barh(chaser_df["label"], chaser_df[TARGET_COLUMN], color="#334155", height=0.55, edgecolor="none")
    bars2[-1].set_color(COLOR_ACCENT_RED)

    axes[1].axvline(df[TARGET_COLUMN].mean() * 100, color=COLOR_SLATE_LIGHT, linestyle="--", linewidth=1.0)
    axes[1].set_title("Conjunction Escalation Rate by Approaching Object", pad=14, loc="left")
    axes[1].set_xlabel("Escalation Rate (%)", labelpad=8)
    axes[1].set_xlim(0, 30)
    axes[1].grid(axis="x", zorder=0)
    axes[1].grid(axis="y", visible=False)
    sns.despine(ax=axes[1], left=True, top=True, right=True)

    for p in axes[1].patches:
        w = p.get_width()
        if w > 0:
            axes[1].annotate(f"{w:.1f}%", (w + 0.6, p.get_y() + p.get_height() / 2.0), va="center", fontsize=9, color=COLOR_SLATE_DARK, weight="500")

    plt.tight_layout()
    fig1_path = os.path.join(output_dir, "fig1_target_and_regime_breakdown.png")
    fig.savefig(fig1_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {fig1_path}")

    # -------------------------------------------------------------------------
    # FIGURE 2: RTN Orbit Separation & Mahalanobis Density
    # -------------------------------------------------------------------------
    print("[FIGURES] Generating Figure 2 (Minimalist RTN Clearance & Density)...")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), dpi=300)

    sample = df.sample(min(3500, len(df)), random_state=42)
    nom = sample[sample[TARGET_COLUMN] == 0]
    esc = sample[sample[TARGET_COLUMN] == 1]

    # Scatter: In-track vs Radial
    axes[0].scatter(nom["in_track_separation_abs_m"], nom["radial_separation_abs_m"], c="#94A3B8", alpha=0.35, s=12, edgecolors="none", label="Nominal Hold (Class 0)")
    axes[0].scatter(esc["in_track_separation_abs_m"], esc["radial_separation_abs_m"], c=COLOR_ACCENT_RED, alpha=0.75, s=18, edgecolors="none", label="Maneuver Required (Class 1)")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_title("RTN Frame Clearance at TCA Screening Gate", pad=14, loc="left")
    axes[0].set_xlabel("Along-Track Separation |Delta T| (meters)", labelpad=8)
    axes[0].set_ylabel("Radial Separation |Delta R| (meters)", labelpad=8)
    axes[0].axhline(80.0, color=COLOR_SLATE_DARK, linestyle=":", linewidth=1.0, alpha=0.7)
    axes[0].annotate("Critical Radial Boundary (|Delta R| < 80 m)", xy=(25, 90), xytext=(25, 150), fontsize=8.5, color=COLOR_SLATE_MED)
    axes[0].legend(frameon=False, loc="upper left", fontsize=8.5)
    axes[0].grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    sns.despine(ax=axes[0], top=True, right=True)

    # KDE: Mahalanobis RTN Proximity
    sns.kdeplot(df[df[TARGET_COLUMN] == 0]["mahalanobis_rtn_distance"], ax=axes[1], color="#64748B", fill=True, alpha=0.25, linewidth=1.5, clip=(0, 16), label="Nominal Pass")
    sns.kdeplot(df[df[TARGET_COLUMN] == 1]["mahalanobis_rtn_distance"], ax=axes[1], color=COLOR_ACCENT_RED, fill=True, alpha=0.45, linewidth=1.8, clip=(0, 16), label="Maneuver Required")
    axes[1].axvline(3.0, color=COLOR_SLATE_DARK, linestyle="--", linewidth=1.1, alpha=0.8)
    axes[1].annotate("3-Sigma Ellipsoid Gate (D_M = 3.0)", xy=(3.1, 0.16), fontsize=8.5, color=COLOR_SLATE_MED)
    axes[1].set_title("3D Covariance-Normalized Proximity Distribution", pad=14, loc="left")
    axes[1].set_xlabel("Mahalanobis Distance D_M (sigma units)", labelpad=8)
    axes[1].set_ylabel("Probability Density", labelpad=8)
    axes[1].set_xlim(0, 15)
    axes[1].legend(frameon=False, loc="upper right", fontsize=8.5)
    axes[1].grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    sns.despine(ax=axes[1], top=True, right=True)

    plt.tight_layout()
    fig2_path = os.path.join(output_dir, "fig2_rtn_kinematics_distributions.png")
    fig.savefig(fig2_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {fig2_path}")

    # -------------------------------------------------------------------------
    # FIGURE 3: Mutual Information & Physical Correlation Heatmap
    # -------------------------------------------------------------------------
    print("[FIGURES] Generating Figure 3 (Minimalist Mutual Information & Correlations)...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # Left: Mutual Info
    mi = {}
    for c in CATEGORICAL_FEATURES:
        clean_name = c.replace("_", " ").title()
        mi[clean_name] = mutual_info_score(df[c], df[TARGET_COLUMN])
    mi_s = pd.Series(mi).sort_values(ascending=True)

    axes[0].barh(mi_s.index, mi_s.values, color="#334155", height=0.5, edgecolor="none")
    axes[0].set_title("Categorical Telemetry: Mutual Information with Target", pad=14, loc="left")
    axes[0].set_xlabel("Mutual Information (nats)", labelpad=8)
    axes[0].set_xlim(0, max(mi_s.values) * 1.3)
    axes[0].grid(axis="x", zorder=0)
    axes[0].grid(axis="y", visible=False)
    sns.despine(ax=axes[0], left=True, top=True, right=True)

    for p in axes[0].patches:
        w = p.get_width()
        axes[0].annotate(f"{w:.4f}", (w + 0.0006, p.get_y() + p.get_height() / 2.0), va="center", fontsize=9, color=COLOR_SLATE_MED)

    # Right: Correlation Matrix
    corr_vars = [
        "radial_separation_abs_m",
        "in_track_separation_abs_m",
        "combined_sigma_radial_m",
        "mahalanobis_rtn_distance",
        "kinetic_lethality_log_index",
        "thermospheric_drag_perturbation",
        TARGET_COLUMN,
    ]
    labels_clean = ["Delta R", "Delta T", "Sigma R", "D_M", "Lethality", "Drag", "Target"]
    cm = df[corr_vars].corr()
    cm.index = labels_clean
    cm.columns = labels_clean

    mask = np.triu(np.ones_like(cm, dtype=bool))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    sns.heatmap(
        cm,
        mask=mask,
        ax=axes[1],
        cmap=cmap,
        vmin=-0.6,
        vmax=0.6,
        center=0,
        square=True,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8.5},
        linewidths=1.0,
        linecolor="#FFFFFF",
        cbar_kws={"shrink": 0.75, "label": "Pearson r"},
    )
    axes[1].set_title("Physical Interaction Correlation Matrix", pad=14, loc="left")
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].tick_params(axis="y", rotation=0)

    plt.tight_layout()
    fig3_path = os.path.join(output_dir, "fig3_mutual_information_and_correlations.png")
    fig.savefig(fig3_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {fig3_path}")

    # -------------------------------------------------------------------------
    # FIGURE 4: Hyperparameter Tuning Response Curves
    # -------------------------------------------------------------------------
    print("[FIGURES] Generating Figure 4 (Minimalist Hyperparameter Curves)...")
    from sklearn.model_selection import train_test_split
    df_train, df_val = train_test_split(df, test_size=0.25, random_state=42, stratify=df[TARGET_COLUMN])
    cols = CATEGORICAL_FEATURES + NUMERICAL_RAW_FEATURES + ENGINEERED_FEATURES

    dv = DictVectorizer(sparse=False)
    X_tr = dv.fit_transform(df_train[cols].to_dict(orient="records"))
    X_va = dv.transform(df_val[cols].to_dict(orient="records"))
    y_tr, y_va = df_train[TARGET_COLUMN].values, df_val[TARGET_COLUMN].values

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), dpi=300)

    # 4A: Logistic Regression C sweep
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    c_list = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
    lr_scores = []
    for c in c_list:
        m = LogisticRegression(C=c, max_iter=500, random_state=42)
        m.fit(X_tr_s, y_tr)
        lr_scores.append(roc_auc_score(y_va, m.predict_proba(X_va_s)[:, 1]))

    axes[0].plot(c_list, lr_scores, color=COLOR_ACCENT_BLUE, marker="o", markersize=4.5, linewidth=1.6)
    axes[0].set_xscale("log")
    axes[0].axvline(0.1, color=COLOR_ACCENT_RED, linestyle="--", linewidth=1.1, label="Selected C = 0.1")
    axes[0].set_title("Logistic Regression: L2 Regularization", pad=12, loc="left")
    axes[0].set_xlabel("Inverse Regularization Strength C", labelpad=6)
    axes[0].set_ylabel("Validation ROC-AUC", labelpad=6)
    axes[0].legend(frameon=False, fontsize=8.5)
    axes[0].grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    sns.despine(ax=axes[0], top=True, right=True)

    # 4B: Random Forest Depth
    depths = [4, 6, 8, 10, 12]
    rf_scores = []
    for d in depths:
        m = RandomForestClassifier(n_estimators=150, max_depth=d, min_samples_leaf=5, random_state=42, n_jobs=-1)
        m.fit(X_tr, y_tr)
        rf_scores.append(roc_auc_score(y_va, m.predict_proba(X_va)[:, 1]))

    axes[1].plot(depths, rf_scores, color=COLOR_ACCENT_TEAL, marker="s", markersize=5, linewidth=1.6)
    axes[1].axvline(10, color=COLOR_ACCENT_RED, linestyle="--", linewidth=1.1, label="Selected Depth = 10")
    axes[1].set_title("Random Forest: Maximum Tree Depth", pad=12, loc="left")
    axes[1].set_xlabel("Tree Depth", labelpad=6)
    axes[1].set_ylabel("Validation ROC-AUC", labelpad=6)
    axes[1].legend(frameon=False, fontsize=8.5)
    axes[1].grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    sns.despine(ax=axes[1], top=True, right=True)

    # 4C: XGBoost Learning Rate Comparison
    etas = [0.03, 0.08, 0.20]
    n_trees = [40, 80, 120, 160, 200]
    colors_eta = [COLOR_SLATE_LIGHT, COLOR_ACCENT_RED, COLOR_ACCENT_AMBER]

    for eta, col in zip(etas, colors_eta):
        curve = []
        for n in n_trees:
            m = xgb.XGBClassifier(n_estimators=n, learning_rate=eta, max_depth=5, min_child_weight=3, random_state=42, n_jobs=-1, eval_metric="logloss")
            m.fit(X_tr, y_tr)
            curve.append(roc_auc_score(y_va, m.predict_proba(X_va)[:, 1]))
        axes[2].plot(n_trees, curve, marker="o", markersize=4, color=col, linewidth=1.6, label=f"eta = {eta}")

    axes[2].set_title("XGBoost: Estimators across Learning Rates", pad=12, loc="left")
    axes[2].set_xlabel("Boosting Rounds", labelpad=6)
    axes[2].set_ylabel("Validation ROC-AUC", labelpad=6)
    axes[2].legend(frameon=False, fontsize=8.5)
    axes[2].grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    sns.despine(ax=axes[2], top=True, right=True)

    plt.tight_layout()
    fig4_path = os.path.join(output_dir, "fig4_hyperparameter_tuning_curves.png")
    fig.savefig(fig4_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {fig4_path}")

    # -------------------------------------------------------------------------
    # FIGURE 5: Operational Cost & Threshold Calibration
    # -------------------------------------------------------------------------
    print("[FIGURES] Generating Figure 5 (Minimalist Cost Optimization)...")
    xgb_final = xgb.XGBClassifier(
        n_estimators=180,
        max_depth=5,
        learning_rate=0.08,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    xgb_final.fit(X_tr, y_tr)
    probs_val = xgb_final.predict_proba(X_va)[:, 1]

    taus = np.linspace(0.02, 0.90, 89)
    costs, recalls, precisions, f2s = [], [], [], []
    c_fn, c_fp = 25.0, 1.0

    for tau in taus:
        preds = (probs_val >= tau).astype(int)
        tp = np.sum((y_va == 1) & (preds == 1))
        fp = np.sum((y_va == 0) & (preds == 1))
        fn = np.sum((y_va == 1) & (preds == 0))
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f2 = 5 * p * r / max(4 * p + r, 1e-8)
        cost_per_k = ((fn * c_fn + fp * c_fp) / len(y_va)) * 1000.0

        precisions.append(p)
        recalls.append(r)
        f2s.append(f2)
        costs.append(cost_per_k)

    opt_idx = int(np.argmin(costs))
    opt_tau = taus[opt_idx]

    fig, ax1 = plt.subplots(figsize=(10, 4.5), dpi=300)

    ax1.plot(taus, recalls, color=COLOR_ACCENT_TEAL, linewidth=1.8, label="Safety Recall (Collision Avoidance)")
    ax1.plot(taus, f2s, color=COLOR_ACCENT_BLUE, linestyle="--", linewidth=1.5, label="F2-Score (Safety-Weighted)")
    ax1.plot(taus, precisions, color=COLOR_SLATE_LIGHT, linestyle=":", linewidth=1.4, label="Precision (Propellant Preservation)")
    ax1.set_xlabel("Operational Decision Threshold (tau)", labelpad=8)
    ax1.set_ylabel("Classification Metric Score", labelpad=8, color=COLOR_SLATE_DARK)
    ax1.set_ylim(-0.02, 1.05)
    ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    ax2 = ax1.twinx()
    ax2.plot(taus, costs, color=COLOR_ACCENT_RED, linewidth=2.0, label="Expected Operational Cost / 1k Passes")
    ax2.set_ylabel("Operational Loss (Cost Units per 1,000 CDMs)", labelpad=8, color=COLOR_ACCENT_RED)
    ax2.grid(False)

    ax1.axvline(opt_tau, color=COLOR_ACCENT_RED, linestyle="-.", linewidth=1.2, label=f"Cost-Optimal tau* = {opt_tau:.3f}")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="center left", fontsize=8.5)

    plt.title("Asymmetric Risk Calibration: Safety Recall vs Propellant Conservation", pad=14, loc="left")
    sns.despine(ax=ax1, top=True, right=False)
    sns.despine(ax=ax2, top=True, left=True)

    plt.tight_layout()
    fig5_path = os.path.join(output_dir, "fig5_propellant_cost_threshold_optimization.png")
    fig.savefig(fig5_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {fig5_path}")

    print("[SUCCESS] All 5 minimalist editorial figures rendered.")


if __name__ == "__main__":
    generate_all_figures()
