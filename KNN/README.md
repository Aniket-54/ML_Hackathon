# KNN on the shared preprocessed data

This folder contains a K-Nearest Neighbors notebook that trains on the already-preprocessed PathSnap candidate tables. It does not run preprocessing again.

## Files

- `02_knn_same_preprocessed_data.ipynb` — loads the candidate NPZ data, runs five-fold trip-grouped validation, fits final KNN on all training candidate rows, evaluates the student-test partition, and saves the model/metrics locally.
- `requirements.txt` — minimal notebook dependencies.
- `results/knn_cv.json` and `results/knn_results.json` — actual results from the original full-data run that produced this notebook.
- `shared_data/` — put the four preprocessed files here: train and student-test `.npz`, plus both metadata `.json` files. These data files are intentionally not duplicated in GitHub.

## Prepare the data

Copy/extract these four existing preprocessed files into this folder's `shared_data/` directory:

```text
shared_data/
├── train_candidates.npz
├── train_metadata.json
├── student_test_candidates.npz
└── student_test_metadata.json
```

Use the same files created by the team's shared preprocessing. Do not regenerate them unless the team intentionally decides to change preprocessing/settings. The notebook also checks the expected candidate matrix dimensions.

## Install and run

From the repository root:

```bash
python3 -m venv .venv-knn
source .venv-knn/bin/activate
python -m pip install -r KNN/requirements.txt
python -m ipykernel install --user --name pathsnap-knn --display-name "PathSnap KNN"
```

Open `KNN/02_knn_same_preprocessed_data.ipynb`, select the **PathSnap KNN** kernel, and choose **Run All**. The notebook looks for the data in this folder's `shared_data/`; it writes `artifacts/knn.joblib`, `results/knn_cv.json`, and `results/knn_results.json` beneath this folder when run.

The full-data KNN validation and student-test inference take a few minutes. KNN is especially slower at prediction time because it searches stored training examples for each new candidate. The notebook scores in batches to limit temporary memory use.

## Original run summary

The included JSON files record the actual run on 1,145,304 training candidate rows and 450,648 student-test candidate rows:

- 5-fold GroupKFold observation top-1: `0.7351 ± 0.0246`
- Student-test observation top-1 among covered points: `0.7401`
- Student-test candidate macro-F1 at 0.5: `0.8419`
- Student-test candidate log loss: `0.1357`
- Candidate miss rate: train `3.91%`, student test `2.00%`
- Final fit: `2.41 s`; student-test scoring: `46.85 s`

These are pointwise candidate-ranking metrics on geometric pseudo-labels, not sequence-level route accuracy. The saved result files and executed notebook outputs are evidence of this run; rerunning the notebook will calculate fresh values on the files provided by the user.
