# Random Forest

Random Forest evaluation on the shared PathSnap candidate-feature dataset.

## Contents

- `PES2UG24CS102_RandomForest.ipynb` trains and evaluates the classifier and applies the Viterbi decoder.
- `src/viterbi_decoder.py` provides route decoding for the notebook.
- `results/random_forest_results.json` records the saved evaluation output.

## Run

Generate `shared_data_v2/` using the preprocessing instructions in the [repository README](../README.md). Install the shared model dependencies from `PES2UG24CS064_LightGBM/requirements.txt`, start Jupyter with this directory as the working directory, and run the notebook.
