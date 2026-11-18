"""
train_decision_tree.py - Decision Tree Candidate Classifier for PathSnap Map-Matching.
Algorithm Family: Tree-based.
Trained on the standardized team dataset (train_candidates.npz, student_test_candidates.npz).
"""

from pathlib import Path
import json
import time
import numpy as np
from sklearn.tree import DecisionTreeClassifier
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

    dt_params = {
        "max_depth": 12,
        "min_samples_split": 50,
        "min_samples_leaf": 25,
        "criterion": "gini",
        "random_state": 42,
    }

    # 1. 5-Fold GroupKFold Cross-Validation
    print("\nRunning 5-Fold GroupKFold Cross-Validation (by trip_id)...")
    gkf = GroupKFold(n_splits=5)
    cv_f1s, cv_top1s, cv_losses = [], [], []

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_tr, y_tr, groups=trip_tr), 1):
        dt = DecisionTreeClassifier(**dt_params)
        dt.fit(X_tr[tr_idx], y_tr[tr_idx])

        val_probs = dt.predict_proba(X_tr[val_idx])[:, 1]
        val_preds = (val_probs >= 0.5).astype(int)

        f1 = float(f1_score(y_tr[val_idx], val_preds, average="macro"))
        ll = float(log_loss(y_tr[val_idx], np.clip(val_probs, 1e-7, 1 - 1e-7)))
        top1 = calculate_top1_accuracy(obs_tr[val_idx], y_tr[val_idx], val_probs)

        cv_f1s.append(f1)
        cv_top1s.append(top1)
        cv_losses.append(ll)
        print(f"Fold {fold}: Macro F1={f1:.4f} | Top-1 Acc={top1:.2f}% | Log Loss={ll:.4f}")

    print(f"CV Macro F1:       {np.mean(cv_f1s):.4f} (+/- {np.std(cv_f1s):.4f})")
    print(f"CV Top-1 Accuracy: {np.mean(cv_top1s):.2f}% (+/- {np.std(cv_top1s):.2f}%)")

    # 2. Fit Full Model
    print("\nFitting final Decision Tree model on full training set...")
    t0 = time.time()
    final_dt = DecisionTreeClassifier(**dt_params)
    final_dt.fit(X_tr, y_tr)
    train_time = time.time() - t0
    print(f"Trained in {train_time:.2f} seconds.")

    # 3. Evaluate on Student-Testing Partition
    print("\nEvaluating on Student-Testing Partition...")
    t1 = time.time()
    test_probs = final_dt.predict_proba(X_te)[:, 1]
    infer_time = time.time() - t1
    test_preds = (test_probs >= 0.5).astype(int)

    test_macro_f1 = float(f1_score(y_te, test_preds, average="macro"))
    test_precision = float(precision_score(y_te, test_preds, zero_division=0))
    test_recall = float(recall_score(y_te, test_preds, zero_division=0))
    test_ll = float(log_loss(y_te, np.clip(test_probs, 1e-7, 1 - 1e-7)))
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

    imp = sorted(zip(feature_names, final_dt.feature_importances_), key=lambda x: x[1], reverse=True)
    print("\nFeature Importances (Gini Impurity Reduction):")
    for fname, fimp in imp:
        print(f"  {fname:24s} : {fimp:.4f}")

    joblib.dump(final_dt, OUT_DIR / "decision_tree_model.joblib")

    metrics = {
        "model_name": "Decision Tree Classifier",
        "algorithm_family": "Tree-based",
        "hyperparameters": dt_params,
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
        "feature_importances": [{"feature": f, "importance": float(v)} for f, v in imp],
    }
    with open(OUT_DIR / "decision_tree_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
