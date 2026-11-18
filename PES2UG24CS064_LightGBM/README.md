# LightGBM

Primary classifier and shared candidate-data pipeline for PathSnap.

## Contents

- `src/prepare_high_acc_data.py` generates the shared 18-feature candidate dataset.
- `src/train_lightgbm.py` runs grouped cross-validation and student-test evaluation.
- `src/tune_lightgbm_optuna.py` runs the configured Optuna search.
- `lightgbm_model_evaluation.ipynb` contains the model evaluation and comparison workflow.
- `results/` contains the saved experiment summaries.

## Run

From the repository root, install `PES2UG24CS064_LightGBM/requirements.txt`, then run preprocessing and training as described in the [repository README](../README.md). To run the notebook, start Jupyter from this folder so the local decoder and shared dataset paths resolve.

The raw dataset must be available as `Student folder/` at the repository root or its parent. The pipeline writes generated data to the ignored root-level `shared_data_v2/` directory.
