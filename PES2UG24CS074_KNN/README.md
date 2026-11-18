# K-Nearest Neighbors

KNN evaluation on the shared PathSnap candidate-feature dataset. The notebook applies feature scaling and uses a bounded training sample for memory-efficient neighbor search.

## Contents

- `02_knn_same_preprocessed_data.ipynb` trains, evaluates, and decodes candidate sequences.
- `viterbi_decoder.py` provides route decoding.
- `results/knn_results.json` contains the saved evaluation summary.

## Run

Generate `shared_data_v2/` using the preprocessing instructions in the [repository README](../README.md). Install the shared model dependencies from `PES2UG24CS064_LightGBM/requirements.txt`, start Jupyter with this directory as the working directory, and run the notebook.
