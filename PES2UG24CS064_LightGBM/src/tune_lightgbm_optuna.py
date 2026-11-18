"""
tune_lightgbm_optuna.py - Bayesian Hyperparameter Optimization for LightGBM
Satisfies Deliverables Section 3.3 requirement for Bayesian optimization.
"""

import optuna
import numpy as np
import lightgbm as lgb
from sklearn.metrics import f1_score, log_loss
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "shared_data_v2"
train = np.load(DATA_DIR / "train_candidates.npz")
X = train['X']
y = train['y']

# Subsample for fast Bayesian search trials
np.random.seed(42)
idx = np.random.choice(len(X), size=150000, replace=False)
X_sub, y_sub = X[idx], y[idx]

val_idx = np.random.choice(len(X_sub), size=30000, replace=False)
train_mask = np.ones(len(X_sub), dtype=bool)
train_mask[val_idx] = False
X_tr, y_tr = X_sub[train_mask], y_sub[train_mask]
X_val, y_val = X_sub[val_idx], y_sub[val_idx]

def objective(trial):
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'n_estimators': trial.suggest_int('n_estimators', 100, 350),
        'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.15, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 31, 80),
        'max_depth': trial.suggest_int('max_depth', 5, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.95),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(X_tr, y_tr)
    probs = model.predict_proba(X_val)[:, 1]
    loss = log_loss(y_val, probs)
    return loss

if __name__ == '__main__':
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize')
    print("Running Bayesian optimization (15 trials)...")
    study.optimize(objective, n_trials=15)
    print("Best Trial:")
    print("  Value (Log Loss):", study.best_value)
    print("  Best Parameters:", study.best_params)
