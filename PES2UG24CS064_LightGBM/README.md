# PES2UG24CS064 — LightGBM First-Review Bundle

This bundle contains the reproducible LightGBM candidate classification model for PathSnap map-matching:

- `01_first_review.ipynb` — Interactive Jupyter notebook with complete step-by-step training, cross-validation, and metric visualisations.
- `FIRST_REVIEW.md` — In-depth technical report answering all compulsory Section 3.2 questions, hyperparameters, and experimental comparisons.
- `src/train_lightgbm.py` — Standalone script running 5-Fold GroupKFold CV and Student-Test evaluation.
- `src/train_logistic_regression.py` — Linear baseline model comparison script.
- `requirements.txt` — Python dependencies.

## Data Source

Reads the standardized candidate feature dataset from:
`PES2UG24CS102_RandomForest/shared_data/` (`train_candidates.npz` and `student_test_candidates.npz`).

## How to Run

From the repository root:

```bash
python PES2UG24CS064_LightGBM/src/train_lightgbm.py
```

Or open `PES2UG24CS064_LightGBM/01_first_review.ipynb` and click **Run All**.
