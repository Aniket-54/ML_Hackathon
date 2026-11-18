# Decision Tree on the shared PathSnap data

This folder contains a teammate-ready notebook for Decision Tree training and evaluation using the team's existing preprocessed candidate data. It does not rerun preprocessing. Results and the fitted model are generated when you run the notebook; no model run is included in this bundle.

## Files

- `04_decision_tree_same_preprocessed_data.ipynb` — five-fold trip-grouped validation, final training on all training rows, and student-test evaluation.
- `requirements.txt` — required Python packages.
- `shared_data/` — put the four shared preprocessed dataset files here. Data files are not stored in GitHub.

Required files:

```text
train_candidates.npz
train_metadata.json
student_test_candidates.npz
student_test_metadata.json
```

The notebook expects training features with shape `(1,145,304, 6)` and student-test features with shape `(450,648, 6)`. If the shared dataset has different dimensions, check that your teammate gave you the same generated dataset version before changing the notebook's checks.

## Install and run

From the repository root:

```bash
python3 -m venv .venv-decision-tree
source .venv-decision-tree/bin/activate
python -m pip install -r DecisionTree/requirements.txt
python -m ipykernel install --user --name pathsnap-decision-tree --display-name "PathSnap Decision Tree"
```

Open `DecisionTree/04_decision_tree_same_preprocessed_data.ipynb`, select **PathSnap Decision Tree**, then choose **Run All**. The notebook searches `DecisionTree/shared_data/` first and also checks the nearby shared PathSnap workspace locations. It writes the model to `DecisionTree/artifacts/decision_tree.joblib` and metrics to `DecisionTree/results/`.

## Model setup and reported results

The notebook uses `max_depth=12`, `min_samples_leaf=5`, `class_weight="balanced"`, and `random_state=42`. It reports observation top-1, candidate macro-F1, candidate log loss, candidate-generation miss rate, and run times. Five-fold GroupKFold keeps trips together between training and validation.

Observation top-1 is conditional on the reference candidate being present among generated candidates. These are pointwise candidate-ranking measurements using geometric pseudo-labels; they are not full sequence-level route accuracy. The reserved evaluator partition should remain untouched.
