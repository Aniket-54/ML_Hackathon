"""
train_lightgbm.py - LightGBM Candidate Classifier for PathSnap Map-Matching.
Author: PES2UG24CS064 (Aniket S Patil)
Algorithm Family: Boosting (Gradient Boosted Decision Trees).
"""

from pathlib import Path
import json
import time
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    log_loss,
    confusion_matrix,
)
import joblib

ROOT = Path(__file__).resolve().parents[2]
SHARED_DIR = ROOT / "PES2UG24CS102_RandomForest" / "shared_data"
FALLBACK_DIR = ROOT / "data"
DATA_DIR = SHARED_DIR if SHARED_DIR.exists() else FALLBACK_DIR
OUT_DIR = Path(__file__).resolve().parents[1] / "results"
OUT_DIR.mkdir(exist_ok=True)


def calculate_top1_accuracy(obs_ids: np.ndarray, y_true: np.ndarray, pred_probs: np.ndarray) -> float:
    obs_best = {}
    for i, obs_id in enumerate(obs_ids):
        p = pred_probs[i]
        if obs_id not in obs_best or p > obs_best[obs_id][0]:
            obs_best[obs_id] = (p, y_true[i])
    correct = sum(1 for prob, label in obs_best.values() if label == 1)
    return float(correct / len(obs_best) * 100.0)


def main():
    print(f"Loading data from {DATA_DIR}...")
    train = np.load(DATA_DIR / "train_candidates.npz")
    test = np.load(DATA_DIR / "student_test_candidates.npz")

    X_tr, y_tr = train["X"], train["y"]
    trip_tr, obs_tr = train["trip_id"], train["gps_observation_id"]
    feature_names = [str(f) for f in train["feature_names"]]

    X_te, y_te = test["X"], test["y"]
    obs_te = test["gps_observation_id"]

    print(f"Features: {feature_names}")
    print(f"Train samples: {len(X_tr):,} | Student-test samples: {len(X_te):,}")

    lgb_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "n_estimators": 150,
        "learning_rate": 0.08,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    # 1. 5-Fold GroupKFold Cross-Validation
    print("\nRunning 5-Fold GroupKFold Cross-Validation (by trip_id)...")
    gkf = GroupKFold(n_splits=5)
    cv_f1s, cv_top1s, cv_losses = [], [], []

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_tr, y_tr, groups=trip_tr), 1):
        m = lgb.LGBMClassifier(**lgb_params)
        m.fit(X_tr[tr_idx], y_tr[tr_idx])

        val_probs = m.predict_proba(X_tr[val_idx])[:, 1]
        val_preds = (val_probs >= 0.5).astype(int)

        f1 = float(f1_score(y_tr[val_idx], val_preds, average="macro"))
        ll = float(log_loss(y_tr[val_idx], val_probs))
        top1 = calculate_top1_accuracy(obs_tr[val_idx], y_tr[val_idx], val_probs)

        cv_f1s.append(f1)
        cv_top1s.append(top1)
        cv_losses.append(ll)
        print(f"Fold {fold}: Macro F1={f1:.4f} | Top-1 Acc={top1:.2f}% | Log Loss={ll:.4f}")

    print(f"CV Macro F1:       {np.mean(cv_f1s):.4f} (+/- {np.std(cv_f1s):.4f})")
    print(f"CV Top-1 Accuracy: {np.mean(cv_top1s):.2f}% (+/- {np.std(cv_top1s):.2f}%)")
    print(f"CV Log Loss:       {np.mean(cv_losses):.4f}")

    # 2. Fit Full Model
    print("\nFitting final LightGBM model on full training set...")
    t0 = time.time()
    final_model = lgb.LGBMClassifier(**lgb_params)
    final_model.fit(X_tr, y_tr)
    train_time = time.time() - t0
    print(f"Trained in {train_time:.2f} seconds.")

    # 3. Evaluate on Student-Testing Partition
    print("\nEvaluating on Student-Testing Partition...")
    t1 = time.time()
    test_probs = final_model.predict_proba(X_te)[:, 1]
    infer_time = time.time() - t1
    test_preds = (test_probs >= 0.5).astype(int)

    test_macro_f1 = float(f1_score(y_te, test_preds, average="macro"))
    test_precision = float(precision_score(y_te, test_preds, zero_division=0))
    test_recall = float(recall_score(y_te, test_preds, zero_division=0))
    test_ll = float(log_loss(y_te, test_probs))
    test_top1_acc = calculate_top1_accuracy(obs_te, y_te, test_probs)
    latency_per_1000 = (infer_time / len(X_te)) * 1e6

    print(f"Student-Test Top-1 Accuracy: {test_top1_acc:.2f}%")
    print(f"Student-Test Macro F1:       {test_macro_f1:.4f}")
    print(f"Student-Test Precision:      {test_precision:.4f}")
    print(f"Student-Test Recall:         {test_recall:.4f}")
    print(f"Student-Test Log Loss:       {test_ll:.4f}")
    print(f"Inference Latency:           {latency_per_1000:.3f} ms / 1,000 candidates")

    cm = confusion_matrix(y_te, test_preds)
    print("\nConfusion Matrix:")
    print(f"  TN: {cm[0, 0]:,} | FP: {cm[0, 1]:,}")
    print(f"  FN: {cm[1, 0]:,} | TP: {cm[1, 1]:,}")

    # Feature Importances
    importances = final_model.feature_importances_
    feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    print("\nFeature Importances:")
    for fname, fimp in feat_imp:
        print(f"  {fname:24s} : {fimp:6d}")

    # Save artifacts locally
    model_path = OUT_DIR / "lightgbm_model.joblib"
    joblib.dump(final_model, model_path)

    metrics = {
        "model_name": "LightGBM Classifier",
        "author": "PES2UG24CS064 (Aniket S Patil)",
        "algorithm_family": "Boosting",
        "hyperparameters": lgb_params,
        "training_time_seconds": train_time,
        "inference_latency_ms_per_1000": latency_per_1000,
        "cv_results": {
            "mean_macro_f1": float(np.mean(cv_f1s)),
            "std_macro_f1": float(np.std(cv_f1s)),
            "mean_top1_accuracy": float(np.mean(cv_top1s)),
            "std_top1_accuracy": float(np.std(cv_top1s)),
            "mean_log_loss": float(np.mean(cv_losses)),
        },
        "student_test_results": {
            "top1_accuracy": test_top1_acc,
            "macro_f1": test_macro_f1,
            "precision": test_precision,
            "recall": test_recall,
            "log_loss": test_ll,
            "confusion_matrix": cm.tolist(),
        },
        "feature_importances": [{"feature": f, "importance": int(imp)} for f, imp in feat_imp],
    }
    with open(OUT_DIR / "lightgbm_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nWrote results to {OUT_DIR / 'lightgbm_metrics.json'}")


if __name__ == "__main__":
    main()
