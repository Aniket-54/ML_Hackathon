# Logistic Regression on the shared preprocessed data

This folder contains a portable notebook for training Logistic Regression on the team's existing PathSnap candidate tables. It loads the preprocessed arrays directly and does not rerun raw-data preprocessing.

## Files

- `03_logistic_regression_same_preprocessed_data.ipynb` — five-fold trip-grouped validation, final fit on all train candidate rows, and student-test evaluation.
- `requirements.txt` — minimal runtime and notebook dependencies.
- `results/logistic_regression_cv.json` and `results/logistic_regression_results.json` — actual outputs from the original full-data run.
- `shared_data/` — place the same preprocessed train/student-test NPZ files and their metadata JSON files here. The dataset files are intentionally not copied into GitHub.

Required files in `shared_data/`:

```text
train_candidates.npz
train_metadata.json
student_test_candidates.npz
student_test_metadata.json
```

The notebook also checks expected matrix dimensions: training `(1,145,304, 6)` and student test `(450,648, 6)`.

## Install and run

From the repository root:

```bash
python3 -m venv .venv-logreg
source .venv-logreg/bin/activate
python -m pip install -r LogisticRegression/requirements.txt
python -m ipykernel install --user --name pathsnap-logreg --display-name "PathSnap Logistic Regression"
```

Open `LogisticRegression/03_logistic_regression_same_preprocessed_data.ipynb`, select **PathSnap Logistic Regression**, then choose **Run All**. The notebook looks for the data in this folder's `shared_data/` or nearby PathSnap workspaces. It writes the fitted model and fresh metrics to `artifacts/` and `results/` beneath this folder.

## Original full-data run

- 5-fold GroupKFold observation top-1: **0.7077 ± 0.0284**
- CV candidate macro-F1: **0.6573**
- Student-test observation top-1 among covered observations: **0.7095**
- Student-test candidate macro-F1: **0.6629**
- Student-test candidate log loss: **0.2651**
- Student-test scoring time: **0.03 s**

These are candidate-ranking metrics on pointwise geometric pseudo-labels, not complete route accuracy. The saved JSON files preserve the actual run; rerunning the notebook creates results for the data files teammates provide.
