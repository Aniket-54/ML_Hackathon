# PathSnap: the first-review guide, from zero to the formal version

This document explains the project in two passes: first in ordinary language, then with the technical terms. It also lists the files and gives the exact dataset counts.

## 1. The challenge in one minute

A phone records GPS dots. The dots are noisy: they can land beside the road, and there may be long gaps between them. The challenge is to work out which road segments the person probably travelled.

A road map is represented as many small connected road pieces. For each GPS dot, our system first finds a shortlist of nearby road pieces. A model scores the pieces. A later part of the full challenge is to use the order of GPS dots and road connections to choose a sensible complete journey.

**Analogy:** GPS dots are clues; nearby roads are possible answers; the model ranks the answers; a route decoder checks which sequence makes sense.

For this first review, we audited the supplied data, made one shared table of GPS-to-road candidate examples, and trained a Random Forest model. I chose Random Forest for this first implementation. This is real work on your Ubuntu laptop, not a Colab run. The notebook is the visible, reproducible walkthrough.

## 2. The numbers your teammate asked for

The word “row” means different things in the raw files and in the processed files. These are the processed dataset counts:

| File | GPS observations considered | GPS observations with candidates | Candidate rows (actual file rows) | Observations with no candidates |
|---|---:|---:|---:|---:|
| `shared_data/train_candidates.npz` | 49,664 sampled from the 80 training trips | 47,721 | **1,145,304** | 1,943 |
| `shared_data/student_test_candidates.npz` | 19,160 (all 10 student-test trips) | 18,777 | **450,648** | 383 |

**One processed row is one possible road segment for one GPS observation.** We generate 24 road candidates for each observation the candidate finder can cover. So, for example, 47,721 covered training GPS observations × 24 candidates = 1,145,304 training rows. An observation with no candidates contributes zero candidate rows.

The teammate-ready summary is: **train = 1,145,304 candidate rows from 49,664 sampled GPS observations across 80 trips; student test = 450,648 candidate rows from all 19,160 GPS observations across 10 trips.** These are stored as compressed NumPy `.npz` files, not CSVs. Metadata alongside them records the counts and column meanings.

The input data also has raw GPS-point counts: **208,417 training GPS points** and **19,160 student-test GPS points**. The training processed set samples at most 640 points per trip to keep the early model run manageable; student test uses every point. The Random Forest training run therefore uses 49,664 sampled observations, not all 208,417 raw training points.

## 3. What the supplied raw files mean

Each trip is described by four matching files. The raw input data should be in the repository-root folder named `Student folder/` (beside `PES2UG24CS102_RandomForest/`). It is supplied separately and is not included in this code bundle.

| Extension | Plain-language meaning | How we use it |
|---|---|---|
| `.track` | The recorded GPS dots: longitude, latitude, and time | These are the observations the system tries to explain. |
| `.nodes` | The map's road-junction points and their coordinates | Road pieces connect these points. |
| `.arcs` | The map's directed road pieces, represented by start/end node IDs | These are the road candidates. The row number identifies an arc. |
| `.route` | The reference journey, written as ordered row IDs into that trip's `.arcs` file | This supplies reference information for development/evaluation. |

**Important:** a `.route` number is an index into `.arcs`, not a node number or a GPS-point number. Do not reorder `.arcs` without updating `.route`. A GPS point and a route arc do not correspond one-to-one: many GPS points may lie along one road arc, and a single noisy point may be near several possible arcs.

### Raw-data audit

| Partition | Trips | GPS points | Map nodes | Arc rows | Reference route arcs |
|---|---:|---:|---:|---:|---:|
| Train | 80 | 208,417 | 1,933,485 | 3,876,904 | 31,336 |
| Student test | 10 | 19,160 | 90,689 | 185,041 | 3,477 |

All expected files were present. The audit found no invalid longitude/latitude values or decreasing timestamps. It found 590 training and 27 student-test time gaps longer than 10 seconds; the largest were 153,064 seconds and 210 seconds. It also found 14 reference-route connection breaks across the 90 trips. These are data characteristics to remember when interpreting results, not rows silently removed from the source files.

The reserved evaluator data mentioned in the challenge is not in the supplied archive. “Student test” here means the supplied student-test partition, not the hidden evaluator set.

## 4. What we did, step by step

1. **Read and audit the trip files.** Counted trips, points, nodes, arcs, route entries, and checked coordinates/timestamps and route connectivity.
2. **Choose GPS observations for the first training run.** Took up to 640 evenly spaced observations per training trip (49,664 total). Kept all 19,160 student-test observations.
3. **Find road candidates.** For each GPS observation, the candidate generator finds up to 24 nearby road arcs. Some GPS points have none within the candidate search area; those are recorded as misses in metadata.
4. **Describe each GPS/road pairing with six features.** These numbers describe how the GPS point relates geometrically and temporally to a candidate road.
5. **Add a development label.** Among the candidate arcs, the arc geometrically closest to the reference route at that point gets label `y = 1`; other candidates get `y = 0`. This is pointwise geometric supervision, not a complete route-sequence label.
6. **Save one shared processed dataset.** Saved candidate rows and a JSON data dictionary for both partitions. Teammates can use the same features rather than independently generating different versions.
7. **Train the Random Forest.** The saved model uses the 49,664 sampled training observations and is evaluated with validation folds grouped by trip.
8. **Save the trained model, metrics, charts, and an example map.** The notebook calls the same project scripts so the process can be shown and rerun.

## 5. What is inside one processed row?

### Direct answer: how many features?

There are **6 input features**. In the saved `.npz`, the feature matrix `X` has shape **(number of candidate rows, 6)**: training `X` is 1,145,304 × 6 and student-test `X` is 450,648 × 6. These six are the only columns the model learns from. The files also store a target (`y`) and identifiers/audit columns; those are not features. So it is accurate to say “6 features,” while the complete saved record has more than six arrays/fields.

### Preprocessing, slowly and in order

Here is what the code actually does to turn the trip files into those rows:

1. **Load one trip.** Read its GPS track, map nodes, directed arcs, and reference route. The arc rows point to node rows; the route contains arc-row IDs.
2. **Choose GPS points.** For training, select up to 640 evenly spaced points per trip. This gives 49,664 points in total across 80 trips. For student test, select every point (19,160 across 10 trips).
3. **Convert coordinates to local metres.** Longitude/latitude values are projected to a simple local, trip-centered metre coordinate system. This lets the code measure distances and lengths in metres without relying on a national map projection.
4. **Prepare road segments.** Look up each arc's start/end node coordinates, form the straight segment between them, and compute its length and midpoint. A spatial KD-tree indexed by segment midpoints makes nearby-candidate lookup fast.
5. **Generate up to 24 road candidates per GPS point.** Query the midpoint index for the 24 nearest map arcs. This is a shortlist, not a guarantee that it includes the true road. A point is counted as covered only when its reference candidate is in that shortlist; otherwise it is a candidate miss and is not included in the candidate rows.
6. **Calculate six numbers for every point/candidate pair.** Project the GPS point onto the candidate segment and calculate its perpendicular distance, position along the segment, segment length, direction difference, estimated GPS speed, and time since the previous point. The same point's speed/time features repeat across its candidate rows because they describe the observation rather than a particular road.
7. **Create a development label for each candidate.** Find the nearest reference-route arc geometrically for that point (from up to eight nearby route arcs, using point-to-segment distance). If that target arc is among the 24 candidates, mark that candidate `y=1` and the other 23 `y=0`. The target is geometric and pointwise; it does not yet check whether the complete arc sequence follows the ordered route correctly.
8. **Attach identifiers and save.** Save `X`, `y`, trip ID, GPS observation ID, candidate arc ID, reference arc ID, and feature names. Save metadata separately with counts and definitions. The `.npz` is a compressed NumPy archive, which is why it does not look like a normal spreadsheet when opened in a text editor.

**A single example:** one covered GPS dot produces 24 candidate rows. Each row has 6 feature values and one `y` label. Exactly one of those 24 rows is positive under the current geometric-label rule. The 24 rows share one `gps_observation_id`, and all rows for that trip share one `trip_id`.

**What this preprocessing does not do:** it does not yet choose a final route through time. It does not use HMM/Viterbi or enforce that adjacent chosen road segments connect. That is a later sequence-decoding stage in the full problem.

The model receives these six numeric features:

| Feature | Beginner explanation |
|---|---|
| `cross_track_m` | How far, in metres, the GPS point is from the candidate road segment. |
| `heading_error_rad` | Difference between the direction of travel and the road's direction, in radians. |
| `along_fraction` | Where the closest point falls along the road segment (roughly 0 at one end, 1 at the other). |
| `segment_length_m` | Length of the candidate road segment in metres. |
| `gps_speed_mps` | Estimated GPS speed in metres per second. |
| `time_gap_s` | Time since the previous GPS point, in seconds. |

Other saved columns are identifiers/labels, not additional model features:

- `y`: binary training target; 1 for the geometrically selected reference candidate, 0 for the other candidates.
- `trip_id`: identifies the trip. We use it to keep all observations from one trip together during cross-validation.
- `gps_observation_id`: identifies which GPS observation produced a set of candidate rows. Used to choose the highest-scoring candidate per observation.
- `candidate_arc_id`: the original row number of the candidate in that trip's `.arcs` file.
- `reference_arc_id`: the reference target arc for auditing/evaluation. **Do not give this column to the model as an input feature**; doing so would reveal the answer.

## 6. What each project file is for

The project folder in this repository is `PES2UG24CS102_RandomForest/`.

### The file you should open and show

- `01_first_review.ipynb` — one Jupyter notebook with the review steps arranged in cells: audit, preprocessing, Random Forest training, and map example. It calls the scripts below. Open this notebook and choose **Run All** to see the steps execute and outputs appear in the notebook. It is a single notebook with multiple cells, not a collection of separate notebooks.
- `FIRST_REVIEW.md` — this guide.
- `README.md` — setup and run instructions for teammates.

Notebook outputs are not guaranteed to be saved just because the notebook file exists. To demonstrate a live run, open it, run all cells, wait for completion, then save the notebook. That makes the run outputs visible in the notebook. The project also already contains a trained model and saved metric files, so the notebook is showing a reproducible rerun, not the only evidence that training happened.

### Source scripts (the code that performs the work)

- `src/eda.py` — audits the raw trip data and writes the trip inventory.
- `src/candidates.py` — candidate-generation and feature/label logic used by preprocessing.
- `src/prepare_shared_data.py` — creates the shared train and student-test candidate datasets and metadata.
- `src/train_random_forest.py` — trains/evaluates the Random Forest run and saves the model/results.
- `src/make_map.py` — makes an interactive HTML map example.

You can read these scripts, but for a first demonstration, the notebook is the clearest entry point because its cells put the steps in order.

### Shared processed data (generated when you run the notebook)

- `shared_data/train_candidates.npz` — the canonical processed training candidate rows (1,145,304 rows).
- `shared_data/student_test_candidates.npz` — the canonical processed student-test candidate rows (450,648 rows).
- `shared_data/train_metadata.json` and `shared_data/student_test_metadata.json` — row counts, sampling description, features, target, IDs, and candidate miss rates.

The processed-data files are generated when the notebook runs; they are not required to be committed in order for teammates to reproduce them.

### Model, results, and map

- `artifacts/random_forest.joblib` — the saved fitted Random Forest model (about 52 MB). It can be loaded for prediction; it is not a notebook and not the dataset.
- `results/random_forest_cv.json` — fold-by-fold/grouped cross-validation details for the larger Random Forest run.
- `results/random_forest_results.json` — summary metrics for that larger run.
- `results/trip_inventory.csv` — per-trip audit counts from the raw data.
The model, metrics, inventory, and map are generated by the notebook; they are outputs, not required inputs to reproduce the workflow.
- `figures/trip_00000001_reference.html` — interactive map example for one trip. Open it in a browser; it is an example of the geographic data, not the final FastAPI demo.

### Project setup

- `requirements.txt` — Python package requirements for the project.
- `.gitignore` — tells Git which generated/local files not to track.
The Python virtual environment is created separately on each computer from `requirements.txt`; do not copy another teammate’s `.venv` folder.

## 7. What the model results actually say

### Random Forest run

| Measurement | Result |
|---|---:|
| Training trips | 80 |
| Sampled training GPS observations | 49,664 (up to 640 per trip) |
| Training candidate rows | 1,145,304 |
| 5-fold cross-validation top-1, grouped by trip | 0.7933 ± 0.0270 |
| Student-test GPS observations | 19,160 (all) |
| Student-test observations covered by candidates | 18,777 |
| Student-test candidate miss rate | 0.0200 (about 2.00%) |
| Student-test top-1 among covered observations | 0.7945 |
| Student-test candidate macro-F1 at probability threshold 0.5 | 0.8480 |
| Full fit time / student-test inference time | 88.5 s / 0.83 s |

Model settings: 60 trees, maximum depth 20, minimum leaf size 4, square-root feature selection, balanced subsampling of classes, four CPU workers, random seed 42. The saved model is `artifacts/random_forest.joblib`.

**Interpretation limit:** top-1 here is point-by-point candidate ranking, and the labels are pointwise geometric labels. It is not the final accuracy of a complete decoded road journey. A candidate miss is also not a model ranking error: the correct road might never have been offered to the model.

## 8. Formal explanation

The task is GPS trajectory map matching. For each GPS observation, a candidate generator retrieves nearby directed road arcs and computes a feature vector for each observation–arc pair. A binary Random Forest classifier estimates whether a candidate arc matches the pointwise reference target. The highest-scoring candidate is selected per observation for the top-1 metric.

The candidate table contains 1,145,304 training rows from 49,664 sampled observations and 450,648 student-test rows from 19,160 observations. Each covered observation contributes 24 candidate rows. Train/test observations without candidates are counted in metadata and excluded from candidate-row scoring. `reference_arc_id` is retained for target/audit use and excluded from the feature matrix to avoid target leakage.

Cross-validation uses five GroupKFold splits with `trip_id` as the group, so observations from the same trip do not appear in both training and validation folds. This reduces overly optimistic scores caused by near-duplicate observations from one trajectory crossing the split. The student-test set is used here for development evaluation because it is the supplied labeled student partition; it is not the reserved hidden evaluator partition.

The full problem statement also calls for sequence-level route reconstruction using an HMM/Viterbi-style decoder: a transition model should reward road arcs that connect and are plausible given travel time/distance, while the observation model scores how well each arc explains each GPS point. That sequence decoder, hyperparameter tuning, FastAPI endpoint, and evaluator-interface integration are **not yet implemented in this first-review package**. The present package demonstrates audited preprocessing, Random Forest candidate scoring/training, and an example map.

## 9. How to run and show it on this laptop

The original run was performed locally on Ubuntu. No Google Colab is required. Follow the setup steps in this folder’s `README.md`, then open `PES2UG24CS102_RandomForest/01_first_review.ipynb` in Jupyter. The notebook works from the repository on another computer and does not use the original author’s absolute path.

In the notebook, select the **Python (PathSnap local)** kernel, then use **Run → Run All**. Wait for the cells to finish, then save the notebook so the outputs are visible. The complete Random Forest cross-validation was measured at about 416 seconds; a full notebook run can take several minutes depending on what it reruns.

The laptop with Ryzen 5 5500U and 8 GB RAM is adequate for this CPU-based first-review pipeline; a dedicated GPU is not needed. Keep other memory-heavy apps closed while running it. The full challenge server and sequence decoder will need their own integration/testing later.

The same stages can be run individually from the bundle directory after installing the requirements:

```bash
python src/eda.py
python src/prepare_shared_data.py
python src/train_random_forest.py
python src/make_map.py
```

If you are only showing the already-saved results, open the notebook and inspect its cells/results; if you want to prove a live rerun, use **Run All**. Running preprocessing again replaces/regenerates the processed files, so tell teammates before changing the shared dataset after they have started using it.

## 10. A short explanation you can say out loud

> “The goal is to match noisy GPS points to roads. We audited the supplied 80 training trips and 10 student-test trips. We made one shared dataset where each row is one possible road for one GPS point, with 24 candidates per covered point and six geometry/time features. That gives 1,145,304 training candidate rows from 49,664 sampled training points and 450,648 student-test candidate rows from all 19,160 test points. I chose Random Forest for the first model, trained it, and evaluated it with folds grouped by trip. Its covered-point top-1 was about 0.794. This is pointwise candidate ranking; sequence decoding with HMM/Viterbi and the FastAPI evaluator integration are the next full-challenge steps.”

## 11. What remains after this first-review work

The official challenge asks for a trained road-candidate model, route reconstruction with HMM/Viterbi, and predictions served through FastAPI for the evaluator interface. This bundle contains the data-audit and preprocessing code, Random Forest training code, and the saved first-review explanation. Running the notebook generates the datasets, model, results, and map. Still to do for the complete submission/demo: hyperparameter tuning, sequence-level HMM/Viterbi decoding, FastAPI prediction interface, connection to the supplied evaluator UI, and two-laptop network demonstration. The reserved evaluator partition is not present in the supplied data, so final unseen-evaluator performance cannot yet be measured here.
