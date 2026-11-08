"""Production Model Training and Export Pipeline for KESSLER-SENTINEL.

Performs stratified splitting, feature engineering, cross-validation,
asymmetric operational cost threshold optimization, and model serialization.
"""

import argparse
import os
import pickle
import sys
import time
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from src.features import (
    compute_astrodynamics_features,
    CATEGORICAL_FEATURES,
    NUMERICAL_RAW_FEATURES,
    ENGINEERED_FEATURES,
    TARGET_COLUMN,
)


def load_and_split_data(
    data_path: str,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """Load raw CDM dataset and perform leak-free stratified 60/20/20 split."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run build_cdm_dataset.py first.")

    df_raw = pd.read_csv(data_path)
    print(f"[DATA] Ingested {len(df_raw):,} raw CDM records from {data_path}")

    # Stratified 60% train, 20% val, 20% test
    df_full_train, df_test = train_test_split(
        df_raw,
        test_size=0.20,
        random_state=random_seed,
        stratify=df_raw[TARGET_COLUMN],
    )

    df_train, df_val = train_test_split(
        df_full_train,
        test_size=0.25,  # 0.25 of 0.80 = 0.20 of total
        random_state=random_seed,
        stratify=df_full_train[TARGET_COLUMN],
    )

    # Compute training set medians strictly on train split to prevent data leakage
    medians: Dict[str, float] = {
        "chaser_rcs_m2": float(df_train["chaser_rcs_m2"].median()),
        "solar_flux_f10_7_sfu": float(df_train["solar_flux_f10_7_sfu"].median()),
    }

    # Compute astrodynamics features for each partition
    df_train_proc = compute_astrodynamics_features(df_train, medians=medians)
    df_val_proc = compute_astrodynamics_features(df_val, medians=medians)
    df_test_proc = compute_astrodynamics_features(df_test, medians=medians)

    print(f"[DATA] Train: {len(df_train_proc):,} | Val: {len(df_val_proc):,} | Test: {len(df_test_proc):,}")
    print(f"[DATA] Positive rate - Train: {df_train_proc[TARGET_COLUMN].mean()*100:.2f}% | Val: {df_val_proc[TARGET_COLUMN].mean()*100:.2f}% | Test: {df_test_proc[TARGET_COLUMN].mean()*100:.2f}%")

    return df_train_proc, df_val_proc, df_test_proc, medians


def prepare_feature_dicts(
    df: pd.DataFrame,
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """Extract categorical and numerical feature dictionaries for DictVectorizer."""
    feature_cols = CATEGORICAL_FEATURES + NUMERICAL_RAW_FEATURES + ENGINEERED_FEATURES
    dict_records = df[feature_cols].to_dict(orient="records")
    y = df[TARGET_COLUMN].values
    return dict_records, y


def optimize_decision_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_fn: float = 25.0,
    cost_fp: float = 1.0,
) -> Tuple[float, float, Dict[str, float]]:
    """Determine asymmetric cost-optimal probability threshold tau* for CAM screening."""
    thresholds = np.linspace(0.05, 0.95, 181)
    best_tau = 0.50
    min_cost = float("inf")
    best_metrics = {}

    n_samples = len(y_true)

    for tau in thresholds:
        y_pred = (y_prob >= tau).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Operational expected cost per 1,000 conjunctions
        cost_per_thousand = ((fn * cost_fn + fp * cost_fp) / n_samples) * 1000.0

        if cost_per_thousand < min_cost:
            min_cost = cost_per_thousand
            best_tau = tau
            best_metrics = {
                "threshold": float(tau),
                "cost_per_thousand": float(cost_per_thousand),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "f2": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "tn": int(tn),
            }

    return best_tau, min_cost, best_metrics


def train_pipeline(data_path: str, output_model: str):
    """Execute end-to-end model benchmarking, cross-validation, and production export."""
    t0 = time.time()
    print("=" * 80)
    print("KESSLER-SENTINEL: ASTRODYNAMICS MODEL TRAINING & SERIALIZATION ENGINE")
    print("=" * 80)

    # 1. Ingest and partition
    df_train, df_val, df_test, medians = load_and_split_data(data_path)

    dict_train, y_train = prepare_feature_dicts(df_train)
    dict_val, y_val = prepare_feature_dicts(df_val)
    dict_test, y_test = prepare_feature_dicts(df_test)

    # 2. Vectorize features
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dict_train)
    X_val = dv.transform(dict_val)
    feature_names = dv.get_feature_names_out().tolist()
    print(f"[VECTORIZER] One-hot and numerical feature space dimensionality: {len(feature_names)}")

    # 3. Benchmark Model Families on Train/Val
    print("\n--- MODEL ARCHITECTURE BENCHMARKING (VALIDATION SPLIT) ---")
    
    # Model 1: Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    lr = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    lr_prob = lr.predict_proba(X_val_scaled)[:, 1]
    lr_roc = roc_auc_score(y_val, lr_prob)
    lr_pr = average_precision_score(y_val, lr_prob)
    print(f"1. Logistic Regression (C=0.1, L2): ROC-AUC = {lr_roc:.4f} | PR-AUC = {lr_pr:.4f}")

    # Model 2: Decision Tree
    dt = DecisionTreeClassifier(max_depth=6, min_samples_leaf=15, random_state=42)
    dt.fit(X_train, y_train)
    dt_prob = dt.predict_proba(X_val)[:, 1]
    dt_roc = roc_auc_score(y_val, dt_prob)
    dt_pr = average_precision_score(y_val, dt_prob)
    print(f"2. Decision Tree (depth=6, min_leaf=15): ROC-AUC = {dt_roc:.4f} | PR-AUC = {dt_pr:.4f}")

    # Model 3: Random Forest
    rf = RandomForestClassifier(n_estimators=150, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_prob = rf.predict_proba(X_val)[:, 1]
    rf_roc = roc_auc_score(y_val, rf_prob)
    rf_pr = average_precision_score(y_val, rf_prob)
    print(f"3. Random Forest (trees=150, depth=10): ROC-AUC = {rf_roc:.4f} | PR-AUC = {rf_pr:.4f}")

    # Model 4: Tuned XGBoost
    xgb_clf = xgb.XGBClassifier(
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
    xgb_clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_prob = xgb_clf.predict_proba(X_val)[:, 1]
    xgb_roc = roc_auc_score(y_val, xgb_prob)
    xgb_pr = average_precision_score(y_val, xgb_prob)
    print(f"4. Tuned XGBoost (lr=0.08, depth=5, trees=180): ROC-AUC = {xgb_roc:.4f} | PR-AUC = {xgb_pr:.4f}")

    # 4. Asymmetric Operational Cost Threshold Optimization
    best_tau, min_val_cost, val_metrics = optimize_decision_threshold(y_val, xgb_prob)
    print(f"\n[DECISION THRESHOLD OPTIMIZATION]")
    print(f"Optimal operational screening threshold tau*: {best_tau:.4f}")
    print(f"Expected operational cost per 1,000 passes: {min_val_cost:.2f} cost units (vs baseline tau=0.50)")
    print(f"Validation Recall at tau*: {val_metrics['recall']*100:.2f}% | F2-Score: {val_metrics['f2']:.4f}")

    # 5. 5-Fold Stratified Cross-Validation on Full Train (Train + Val)
    print("\n--- 5-FOLD STRATIFIED CROSS-VALIDATION (FULL TRAIN) ---")
    df_full_train = pd.concat([df_train, df_val], ignore_index=True)
    dict_full_train, y_full_train = prepare_feature_dicts(df_full_train)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_roc_scores = []
    cv_pr_scores = []
    cv_f2_scores = []

    for fold, (train_idx, oof_idx) in enumerate(skf.split(dict_full_train, y_full_train), 1):
        fold_train_dict = [dict_full_train[i] for i in train_idx]
        fold_oof_dict = [dict_full_train[i] for i in oof_idx]
        y_f_train, y_f_oof = y_full_train[train_idx], y_full_train[oof_idx]

        fold_dv = DictVectorizer(sparse=False)
        X_f_train = fold_dv.fit_transform(fold_train_dict)
        X_f_oof = fold_dv.transform(fold_oof_dict)

        fold_model = xgb.XGBClassifier(
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
        fold_model.fit(X_f_train, y_f_train)
        oof_probs = fold_model.predict_proba(X_f_oof)[:, 1]

        f_roc = roc_auc_score(y_f_oof, oof_probs)
        f_pr = average_precision_score(y_f_oof, oof_probs)
        f_f2 = fbeta_score(y_f_oof, (oof_probs >= best_tau).astype(int), beta=2)

        cv_roc_scores.append(f_roc)
        cv_pr_scores.append(f_pr)
        cv_f2_scores.append(f_f2)
        print(f"  Fold {fold}: ROC-AUC={f_roc:.4f} | PR-AUC={f_pr:.4f} | F2={f_f2:.4f}")

    print(f"Cross-Validation Summary:")
    print(f"  Mean ROC-AUC: {np.mean(cv_roc_scores):.4f} +/- {np.std(cv_roc_scores):.4f}")
    print(f"  Mean PR-AUC:  {np.mean(cv_pr_scores):.4f} +/- {np.std(cv_pr_scores):.4f}")
    print(f"  Mean F2:      {np.mean(cv_f2_scores):.4f} +/- {np.std(cv_f2_scores):.4f}")

    # 6. Retrain Final Model on Full Training Set and Test on Held-out 20%
    print("\n--- FINAL MODEL RETRAINING & OUT-OF-SAMPLE TEST EVALUATION ---")
    final_dv = DictVectorizer(sparse=False)
    X_full_train = final_dv.fit_transform(dict_full_train)
    X_test = final_dv.transform(dict_test)

    final_model = xgb.XGBClassifier(
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
    final_model.fit(X_full_train, y_full_train)

    test_probs = final_model.predict_proba(X_test)[:, 1]
    test_roc = roc_auc_score(y_test, test_probs)
    test_pr = average_precision_score(y_test, test_probs)

    test_preds = (test_probs >= best_tau).astype(int)
    test_precision = precision_score(y_test, test_preds)
    test_recall = recall_score(y_test, test_preds)
    test_f1 = f1_score(y_test, test_preds)
    test_f2 = fbeta_score(y_test, test_preds, beta=2)

    cm_test = confusion_matrix(y_test, test_preds)
    tn, fp, fn, tp = cm_test.ravel()
    test_cost_per_thousand = ((fn * 25.0 + fp * 1.0) / len(y_test)) * 1000.0

    print(f"[TEST EVALUATION METRICS @ tau*={best_tau:.4f}]")
    print(f"  Test ROC-AUC:            {test_roc:.4f}")
    print(f"  Test PR-AUC:             {test_pr:.4f}")
    print(f"  Test Recall:             {test_recall*100:.2f}% (Safety Critical)")
    print(f"  Test Precision:          {test_precision*100:.2f}%")
    print(f"  Test F1-Score:           {test_f1:.4f}")
    print(f"  Test F2-Score:           {test_f2:.4f}")
    print(f"  Operational Cost/1,000:  {test_cost_per_thousand:.2f} cost units")
    print(f"  Confusion Matrix (N={len(y_test):,}): TP={tp}, FP={fp}, FN={fn}, TN={tn}")

    # 7. Serialize Production Artifact
    test_metrics = {
        "roc_auc": float(test_roc),
        "pr_auc": float(test_pr),
        "recall": float(test_recall),
        "precision": float(test_precision),
        "f1": float(test_f1),
        "f2": float(test_f2),
        "cost_per_thousand": float(test_cost_per_thousand),
        "cv_roc_mean": float(np.mean(cv_roc_scores)),
        "cv_roc_std": float(np.std(cv_roc_scores)),
        "cv_pr_mean": float(np.mean(cv_pr_scores)),
        "cv_pr_std": float(np.std(cv_pr_scores)),
        "cv_f2_mean": float(np.mean(cv_f2_scores)),
        "cv_f2_std": float(np.std(cv_f2_scores)),
    }

    artifact = {
        "dv": final_dv,
        "model": final_model,
        "threshold": float(best_tau),
        "medians": medians,
        "feature_names": feature_names,
        "metrics": test_metrics,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    os.makedirs(os.path.dirname(output_model) if os.path.dirname(output_model) else ".", exist_ok=True)
    with open(output_model, "wb") as f:
        pickle.dump(artifact, f)

    file_size_mb = os.path.getsize(output_model) / (1024 * 1024)
    print(f"\n[ARTIFACT] Successfully serialized model to {output_model} ({file_size_mb:.2f} MB)")
    print(f"[TOTAL TIME] Pipeline completed in {time.time() - t0:.2f} seconds.")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="KESSLER-SENTINEL Astrodynamics Model Training Pipeline")
    parser.add_argument("--data-path", type=str, default="data/leo_conjunction_cdm_events.csv", help="Path to input CSV")
    parser.add_argument("--output-model", type=str, default="model_kessler_sentinel_v1.bin", help="Path to output serialized model artifact")
    args = parser.parse_args()

    train_pipeline(data_path=args.data_path, output_model=args.output_model)


if __name__ == "__main__":
    main()
