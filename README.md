# PathSnap: GPS Map Matching and Route Reconstruction

PathSnap maps noisy GPS observations to a directed road network. This repository contains the shared preprocessing pipeline, five model experiments, route-decoding code, evaluation results, a FastAPI backend, and a Streamlit evaluator interface.

## Project structure

```text
PES2UG24CS064_LightGBM/src/   Shared preprocessing, LightGBM training, tuning, and decoder
PES2UG24CS064_LightGBM/       LightGBM notebook, dependencies, and result records
PES2UG24CS102_RandomForest/   Random Forest notebook, decoder, and result record
PES2UG24CS074_KNN/           KNN notebook, decoder, and result record
PES2UG25CS810_LogisticRegression1/  Logistic Regression notebook, decoder, and result record
DecisionTree/                 Decision Tree notebook, decoder, and result record
backend/                      FastAPI inference service and bundled LightGBM model
frontend/                     Streamlit interface and browser map rendering
```

## Data requirements

The dataset is not included in this repository. Obtain the authorized PathSnap data and extract it as `Student folder/`, either at the repository root or beside the repository directory. The preprocessing and backend code search both locations. Each partition must contain trip folders with the corresponding `.track`, `.nodes`, `.arcs`, and `.route` files.

```text
Student folder/
├── train/<trip_id>/<trip_id>.{track,nodes,arcs,route}
└── student_test/<trip_id>/<trip_id>.{track,nodes,arcs,route}
```

Do not commit the raw dataset, credentials, or local generated data. The repository ignores `Student folder/`, generated candidate arrays, model artifacts, and local environments.

## Install the experiment environment

Use Python 3.10 or newer. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r PES2UG24CS064_LightGBM/requirements.txt
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

## Prepare the shared candidate dataset

Run the shared pipeline from the repository root:

```bash
python PES2UG24CS064_LightGBM/src/prepare_high_acc_data.py
```

It writes `shared_data_v2/train_candidates.npz` and `shared_data_v2/student_test_candidates.npz`. Each array contains candidate features, labels, trip/observation identifiers, candidate and reference arc identifiers, and the candidate count. The pipeline uses eight road-arc candidates per GPS observation and the 18 features listed in `FEATURE_NAMES` in the preprocessing script. The data directory is generated locally and is not tracked by Git.

## Train and evaluate models

Run the LightGBM script and Bayesian search from the repository root:

```bash
python PES2UG24CS064_LightGBM/src/train_lightgbm.py
python PES2UG24CS064_LightGBM/src/tune_lightgbm_optuna.py
```

The tuning script reports the best trial in the terminal. The checked-in experiment record is `PES2UG24CS064_LightGBM/results/bayesian_tuning_results.json`.

The five model notebooks use the same `shared_data_v2` arrays. Launch Jupyter from the relevant model folder so local decoder imports resolve. For example, from the repository root:

```bash
cd PES2UG24CS064_LightGBM
jupyter lab lightgbm_model_evaluation.ipynb
```

Use the same pattern for the other notebooks: change to the listed directory, then launch Jupyter with its notebook filename. The shared arrays are found in the repository-level `shared_data_v2/` directory.

| Model | Directory | Notebook |
|---|---|---|
| LightGBM | `PES2UG24CS064_LightGBM/` | `lightgbm_model_evaluation.ipynb` |
| Random Forest | `PES2UG24CS102_RandomForest/` | `PES2UG24CS102_RandomForest.ipynb` |
| KNN | `PES2UG24CS074_KNN/` | `02_knn_same_preprocessed_data.ipynb` |
| Logistic Regression | `PES2UG25CS810_LogisticRegression1/` | `03_logistic_regression_same_preprocessed_data.ipynb` |
| Decision Tree | `DecisionTree/` | `04_decision_tree_same_preprocessed_data.ipynb` |

Run notebooks one at a time using the environment above. The KNN notebook limits its training sample for memory-efficient nearest-neighbor evaluation. Model-specific result records are under each model’s `results/` directory. Notebooks and scripts can regenerate local metrics; compare the outputs with the committed JSON files before changing reported values.

## Run the evaluator system

The backend reads the raw student-test trips and uses the bundled model at `backend/lightgbm_model.joblib`. If that model file is unavailable, it trains from the locally generated training candidate data in `shared_data_v2/`.

Terminal 1, from the repository root:

```bash
python -m pip install -r backend/requirements.txt
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Check `http://localhost:8000/health`; interactive API documentation is at `http://localhost:8000/docs`.

Terminal 2, from the repository root:

```bash
python -m pip install -r frontend/requirements.txt
cd frontend
streamlit run app.py
```

In the interface, set the backend URL to `http://<backend-host-ip>:8000`, select an available student-test trip, and request route reconstruction. For a same-machine run, use `http://127.0.0.1:8000`. The backend exposes `GET /health`, `GET /trips`, `GET /model_info`, and `POST /predict_trip/{trip_id}`. The frontend renders the observed GPS points, reference route when available, and decoded route on a Folium map.

For two-computer deployment, both machines must be on a network that allows the frontend machine to reach the backend machine on port 8000. The raw dataset is needed on the backend machine.

## Evaluation notes

- Candidate labels are generated from the supplied route geometry by the preprocessing pipeline. They are pointwise labels used for candidate ranking.
- Candidate-level Macro F1 and log loss describe binary candidate classification. Observation Top-1 selects the highest-scored candidate per observation.
- HMM/Viterbi metrics describe decoded route sequences and should be reported separately from classifier-only metrics.
- Candidate coverage/miss rate must be considered with accuracy. A top-1 score over covered observations does not count observations for which no positive reference candidate was generated.
- `student_test` is the supplied student-testing partition. Keep any reserved evaluator partition separate from training and tuning.
- Use the result JSONs, notebooks, and source scripts to verify metrics. The JSON records may reflect earlier runs; replace them only with outputs produced from the intended data and configuration.

## Results and implementation files

Each model folder contains its experiment notebook, decoder where used, and result JSON. The primary preprocessing, training, and tuning implementation is under `PES2UG24CS064_LightGBM/src/`. The backend and frontend include their own dependency files and run instructions. Keep the code, configuration, model artifact, and result record in sync when updating the submission.
