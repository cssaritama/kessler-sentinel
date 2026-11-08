"""Publication-Grade Astrodynamics Figures Generator for KESSLER-SENTINEL.

Generates 5 high-resolution figures (dpi=200) for docs/figures/ and README.md:
1. Target prevalence and orbital regime breakdown
2. RTN kinematic distributions (Radial vs In-Track log scales, Mahalanobis)
3. Mutual information and numerical correlation matrix
4. Hyperparameter tuning response curves
5. Operational propellant cost vs decision threshold curve
"""

import os
import sys

# Ensure repository root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mutual_info_score, roc_auc_score, precision_recall_curve, f1_score, fbeta_score
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


def apply_publication_theme():
    """Apply clean, minimalist scientific theme without decorative clutter."""
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.titlesize": 13,
        "figure.autolayout": True,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "grid.color": "#e0e0e0",
        "grid.linestyle": "--",
        "grid.alpha": 0.6,
    })


def generate_all_figures(data_path: str = "data/leo_conjunction_cdm_events.csv", output_dir: str = "docs/figures"):
    """Generate and save the 5 canonical technical figures."""
    os.makedirs(output_dir, exist_ok=True)
    apply_publication_theme()

    print(f"[FIGURES] Ingesting dataset from {data_path}...")
    df_raw = pd.read_csv(data_path)
    df = compute_astrodynamics_features(df_raw)

    palette = {"0": "#4A6FA5", "1": "#D9534F"}

    # -------------------------------------------------------------
    # FIGURE 1: Target Prevalence and Orbital Regime Breakdown
    # -------------------------------------------------------------
    print("[FIGURES] Generating Figure 1: Target & Regime Breakdown...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=200)

    # 1A: Escalation rate by Orbital Regime Band
    reg_summary = (
        df.groupby("orbital_regime_band")[TARGET_COLUMN]
        .agg(["mean", "count"])
        .reset_index()
        .sort_values(by="mean", ascending=False)
    )
    reg_summary["escalation_pct"] = reg_summary["mean"] * 100.0

    sns.barplot(
        data=reg_summary,
        x="orbital_regime_band",
        y="escalation_pct",
        color="#2B5B84",
        ax=axes[0],
    )
    axes[0].set_title("Operational Escalation Rate by Orbital Regime Band", pad=12)
    axes[0].set_xlabel("LEO Orbital Regime Band")
    axes[0].set_ylabel("Maneuver Escalation Rate (%)")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=15, ha="right")
    axes[0].axhline(df[TARGET_COLUMN].mean() * 100, color="#D9534F", linestyle="--", linewidth=1.2, label=f"Global Mean ({df[TARGET_COLUMN].mean()*100:.1f}%)")
    axes[0].legend(loc="upper right", frameon=True)

    for p in axes[0].patches:
        height = p.get_height()
        axes[0].annotate(f"{height:.1f}%", (p.get_x() + p.get_width() / 2.0, height + 0.6), ha="center", fontsize=9)

    # 1B: Escalation rate by Chaser Object Type
    chaser_summary = (
        df.groupby("chaser_object_type")[TARGET_COLUMN]
        .agg(["mean", "count"])
        .reset_index()
        .sort_values(by="mean", ascending=False)
    )
    chaser_summary["escalation_pct"] = chaser_summary["mean"] * 100.0

    sns.barplot(
        data=chaser_summary,
        x="chaser_object_type",
        y="escalation_pct",
        color="#3E8E7E",
        ax=axes[1],
    )
    axes[1].set_title("Operational Escalation Rate by Chaser Object Type", pad=12)
    axes[1].set_xlabel("Approaching Chaser Class")
    axes[1].set_ylabel("Maneuver Escalation Rate (%)")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=15, ha="right")
    axes[1].axhline(df[TARGET_COLUMN].mean() * 100, color="#D9534F", linestyle="--", linewidth=1.2, label=f"Global Mean ({df[TARGET_COLUMN].mean()*100:.1f}%)")
    axes[1].legend(loc="upper right", frameon=True)

    for p in axes[1].patches:
        height = p.get_height()
        axes[1].annotate(f"{height:.1f}%", (p.get_x() + p.get_width() / 2.0, height + 0.6), ha="center", fontsize=9)

    plt.tight_layout()
    fig1_path = os.path.join(output_dir, "fig1_target_and_regime_breakdown.png")
    fig.savefig(fig1_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {fig1_path}")

    # -------------------------------------------------------------
    # FIGURE 2: RTN Kinematics & Mahalanobis Distributions
    # -------------------------------------------------------------
    print("[FIGURES] Generating Figure 2: RTN Kinematics Distributions...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=200)

    # 2A: Radial vs In-Track Separation Scatter (Log-Scale)
    sample_df = df.sample(min(4000, len(df)), random_state=42)
    sample_df["target_str"] = sample_df[TARGET_COLUMN].astype(str)

    sns.scatterplot(
        data=sample_df[sample_df[TARGET_COLUMN] == 0],
        x="in_track_separation_abs_m",
        y="radial_separation_abs_m",
        color="#4A6FA5",
        alpha=0.35,
        s=15,
        label="Nominal Hold (Class 0)",
        ax=axes[0],
    )
    sns.scatterplot(
        data=sample_df[sample_df[TARGET_COLUMN] == 1],
        x="in_track_separation_abs_m",
        y="radial_separation_abs_m",
        color="#D9534F",
        alpha=0.65,
        s=25,
        label="Maneuver Required (Class 1)",
        ax=axes[0],
    )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_title("RTN Orbit Separation: In-Track vs Radial Clearance", pad=12)
    axes[0].set_xlabel("Absolute In-Track Separation |Delta T| (m, log scale)")
    axes[0].set_ylabel("Absolute Radial Separation |Delta R| (m, log scale)")
    axes[0].axhline(50.0, color="#333333", linestyle=":", linewidth=1.0, label="Radial Warning Margin (50m)")
    axes[0].legend(loc="upper left", frameon=True)

    # 2B: 3D Mahalanobis Distance Distribution
    sns.kdeplot(
        data=df[df[TARGET_COLUMN] == 0],
        x="mahalanobis_rtn_distance",
        color="#4A6FA5",
        fill=True,
        alpha=0.4,
        label="Nominal Hold (Class 0)",
        clip=(0, 20),
        ax=axes[1],
    )
    sns.kdeplot(
        data=df[df[TARGET_COLUMN] == 1],
        x="mahalanobis_rtn_distance",
        color="#D9534F",
        fill=True,
        alpha=0.5,
        label="Maneuver Required (Class 1)",
        clip=(0, 20),
        ax=axes[1],
    )
    axes[1].set_xlim(0, 15)
    axes[1].set_title("3D Mahalanobis RTN Covariance-Normalized Proximity", pad=12)
    axes[1].set_xlabel("Mahalanobis Distance D_M (dimensionless sigma)")
    axes[1].set_ylabel("Probability Density")
    axes[1].axvline(3.0, color="#333333", linestyle="--", linewidth=1.2, label="Classical Screening Threshold (D_M=3.0)")
    axes[1].legend(loc="upper right", frameon=True)

    plt.tight_layout()
    fig2_path = os.path.join(output_dir, "fig2_rtn_kinematics_distributions.png")
    fig.savefig(fig2_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {fig2_path}")

    # -------------------------------------------------------------
    # FIGURE 3: Mutual Information & Pearson Correlation Heatmap
    # -------------------------------------------------------------
    print("[FIGURES] Generating Figure 3: Mutual Information & Correlations...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=200)

    # 3A: Mutual Information Score for Categorical Variables
    mi_scores = {}
    for cat_col in CATEGORICAL_FEATURES:
        score = mutual_info_score(df[cat_col], df[TARGET_COLUMN])
        mi_scores[cat_col] = score
    mi_df = pd.DataFrame(list(mi_scores.items()), columns=["feature", "mi_score"]).sort_values(by="mi_score", ascending=True)

    axes[0].barh(mi_df["feature"], mi_df["mi_score"], color="#336699", edgecolor="#1A334D")
    axes[0].set_title("Categorical Telemetry Mutual Information vs Target", pad=12)
    axes[0].set_xlabel("Mutual Information Score (nats)")
    axes[0].set_ylabel("Telemetry Feature")
    for p in axes[0].patches:
        width = p.get_width()
        axes[0].annotate(f"{width:.4f}", (width + 0.0005, p.get_y() + p.get_height() / 2.0), va="center", fontsize=9)

    # 3B: Numerical Correlation Matrix (Pearson)
    corr_cols = [
        "radial_separation_abs_m",
        "in_track_separation_abs_m",
        "combined_sigma_radial_m",
        "combined_sigma_in_track_m",
        "chaser_rcs_m2",
        "solar_flux_f10_7_sfu",
        "mahalanobis_rtn_distance",
        "kinetic_lethality_log_index",
        "thermospheric_drag_perturbation",
        TARGET_COLUMN,
    ]
    corr_matrix = df[corr_cols].corr()

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        vmin=-0.6,
        vmax=0.6,
        square=True,
        cbar_kws={"shrink": 0.8},
        ax=axes[1],
    )
    axes[1].set_title("Physical Telemetry Correlation Matrix (Pearson r)", pad=12)
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=35, ha="right", fontsize=8)
    axes[1].set_yticklabels(axes[1].get_yticklabels(), rotation=0, fontsize=8)

    plt.tight_layout()
    fig3_path = os.path.join(output_dir, "fig3_mutual_information_and_correlations.png")
    fig.savefig(fig3_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {fig3_path}")

    # -------------------------------------------------------------
    # FIGURE 4: Hyperparameter Tuning Curves (LR, RF, XGBoost)
    # -------------------------------------------------------------
    print("[FIGURES] Generating Figure 4: Hyperparameter Tuning Curves...")
    from sklearn.model_selection import train_test_split
    df_train, df_val = train_test_split(df, test_size=0.25, random_state=42, stratify=df[TARGET_COLUMN])

    dict_train = df_train[CATEGORICAL_FEATURES + NUMERICAL_RAW_FEATURES + ENGINEERED_FEATURES].to_dict(orient="records")
    dict_val = df_val[CATEGORICAL_FEATURES + NUMERICAL_RAW_FEATURES + ENGINEERED_FEATURES].to_dict(orient="records")
    y_tr, y_va = df_train[TARGET_COLUMN].values, df_val[TARGET_COLUMN].values

    dv = DictVectorizer(sparse=False)
    X_tr = dv.fit_transform(dict_train)
    X_va = dv.transform(dict_val)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), dpi=200)

    # 4A: Logistic Regression C Parameter Sweep
    c_values = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    lr_rocs = []
    for c in c_values:
        m = LogisticRegression(C=c, max_iter=500, random_state=42)
        m.fit(X_tr_s, y_tr)
        lr_rocs.append(roc_auc_score(y_va, m.predict_proba(X_va_s)[:, 1]))

    axes[0].plot(c_values, lr_rocs, marker="o", color="#2B5B84", linewidth=1.8)
    axes[0].set_xscale("log")
    axes[0].set_title("Logistic Regression: L2 Penalty vs ROC-AUC", pad=12)
    axes[0].set_xlabel("Inverse Regularization Strength C")
    axes[0].set_ylabel("Validation ROC-AUC")
    axes[0].axvline(0.1, color="#D9534F", linestyle="--", label="Selected C=0.1")
    axes[0].legend(loc="lower right")

    # 4B: Random Forest Depth vs Estimators
    depths = [4, 6, 8, 10, 12]
    rf_rocs_100 = []
    rf_rocs_150 = []
    for d in depths:
        m100 = RandomForestClassifier(n_estimators=100, max_depth=d, min_samples_leaf=5, random_state=42, n_jobs=-1)
        m100.fit(X_tr, y_tr)
        rf_rocs_100.append(roc_auc_score(y_va, m100.predict_proba(X_va)[:, 1]))

        m150 = RandomForestClassifier(n_estimators=150, max_depth=d, min_samples_leaf=5, random_state=42, n_jobs=-1)
        m150.fit(X_tr, y_tr)
        rf_rocs_150.append(roc_auc_score(y_va, m150.predict_proba(X_va)[:, 1]))

    axes[1].plot(depths, rf_rocs_100, marker="s", color="#5C6B73", linewidth=1.5, label="n_estimators=100")
    axes[1].plot(depths, rf_rocs_150, marker="o", color="#3E8E7E", linewidth=1.8, label="n_estimators=150")
    axes[1].set_title("Random Forest: Tree Depth vs Generalization", pad=12)
    axes[1].set_xlabel("Maximum Tree Depth")
    axes[1].set_ylabel("Validation ROC-AUC")
    axes[1].legend(loc="lower right")

    # 4C: XGBoost Learning Curves (Trees vs ROC-AUC across eta)
    etas = [0.03, 0.08, 0.20]
    n_trees = np.linspace(20, 200, 10, dtype=int)
    colors = ["#4A6FA5", "#D9534F", "#E0A96D"]
    for eta, col in zip(etas, colors):
        curve = []
        for n in n_trees:
            m = xgb.XGBClassifier(n_estimators=n, learning_rate=eta, max_depth=5, min_child_weight=3, random_state=42, n_jobs=-1, eval_metric="logloss")
            m.fit(X_tr, y_tr)
            curve.append(roc_auc_score(y_va, m.predict_proba(X_va)[:, 1]))
        axes[2].plot(n_trees, curve, marker="o", markersize=4, color=col, linewidth=1.8, label=f"eta = {eta}")

    axes[2].set_title("XGBoost: Boosting Rounds across Learning Rates", pad=12)
    axes[2].set_xlabel("Number of Estimators")
    axes[2].set_ylabel("Validation ROC-AUC")
    axes[2].legend(loc="lower right")

    plt.tight_layout()
    fig4_path = os.path.join(output_dir, "fig4_hyperparameter_tuning_curves.png")
    fig.savefig(fig4_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {fig4_path}")

    # -------------------------------------------------------------
    # FIGURE 5: Propellant Cost vs Decision Threshold Optimization
    # -------------------------------------------------------------
    print("[FIGURES] Generating Figure 5: Propellant Cost Optimization...")
    # Train tuned XGBoost model
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

    taus = np.linspace(0.05, 0.95, 91)
    precisions = []
    recalls = []
    f1s = []
    f2s = []
    costs = []

    cost_fn = 25.0
    cost_fp = 1.0

    for tau in taus:
        preds = (probs_val >= tau).astype(int)
        tp = np.sum((y_va == 1) & (preds == 1))
        fp = np.sum((y_va == 0) & (preds == 1))
        fn = np.sum((y_va == 1) & (preds == 0))
        tn = np.sum((y_va == 0) & (preds == 0))

        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-8)
        f2 = 5 * p * r / max(4 * p + r, 1e-8)
        cost_per_k = ((fn * cost_fn + fp * cost_fp) / len(y_va)) * 1000.0

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        f2s.append(f2)
        costs.append(cost_per_k)

    best_idx = int(np.argmin(costs))
    optimal_tau = taus[best_idx]

    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=200)

    ax1.plot(taus, precisions, label="Precision", color="#4A6FA5", linestyle="--", linewidth=1.5)
    ax1.plot(taus, recalls, label="Recall (Safety-Critical)", color="#3E8E7E", linestyle="-", linewidth=1.8)
    ax1.plot(taus, f1s, label="F1-Score", color="#8E7DBE", linestyle=":", linewidth=1.5)
    ax1.plot(taus, f2s, label="F2-Score (Recall-Weighted)", color="#E0A96D", linestyle="-", linewidth=1.8)
    ax1.set_xlabel("Classification Decision Threshold (tau)")
    ax1.set_ylabel("Classification Metric Score", color="#333333")
    ax1.set_ylim(0.0, 1.05)

    ax2 = ax1.twinx()
    ax2.plot(taus, costs, label="Operational Cost per 1,000 CDMs", color="#D9534F", linewidth=2.2)
    ax2.set_ylabel("Expected Operational Cost (Units / 1,000 CDMs)", color="#D9534F")
    ax2.grid(False)

    # Highlight optimal threshold
    ax1.axvline(optimal_tau, color="#D9534F", linestyle="-.", linewidth=1.5, label=f"Cost-Optimal tau* = {optimal_tau:.2f}")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left", frameon=True)
    plt.title("Operational Threshold Calibration: Safety Recall vs Propellant Conservation", pad=14)

    plt.tight_layout()
    fig5_path = os.path.join(output_dir, "fig5_propellant_cost_threshold_optimization.png")
    fig.savefig(fig5_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {fig5_path}")

    print("[SUCCESS] All 5 publication figures rendered to", output_dir)


if __name__ == "__main__":
    generate_all_figures()
