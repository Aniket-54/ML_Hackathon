# Logistic Regression

Logistic Regression baseline on the shared PathSnap candidate-feature dataset.

## Contents

- `03_logistic_regression_same_preprocessed_data.ipynb` trains, evaluates, and decodes candidate sequences.
- `viterbi_decoder.py` provides route decoding.
- `results/logistic_regression_results.json` contains the saved evaluation summary.

## Run

Generate `shared_data_v2/` using the preprocessing instructions in the [repository README](../README.md). Install the shared model dependencies from `PES2UG24CS064_LightGBM/requirements.txt`, start Jupyter with this directory as the working directory, and run the notebook.
