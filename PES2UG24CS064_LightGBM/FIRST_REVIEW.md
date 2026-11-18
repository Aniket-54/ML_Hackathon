# PathSnap First Review — LightGBM Classifier
**Author**: PES2UG24CS064 (Aniket S Patil)  
**Algorithm Family**: Boosting (Gradient Boosted Decision Trees)

---

## 1. Model Understanding

### 1.1 Why is LightGBM suitable for the PathSnap problem?
PathSnap map-matching requires distinguishing the true traversed road segment from approximately 23 nearby candidate segments for each GPS observation. This candidate ranking problem exhibits complex non-linear spatial interactions:
- Perpendicular distance (`cross_track_m`) must be evaluated jointly with angular error (`heading_error_rad`) to distinguish parallel streets from opposing traffic lanes.
- Segment length and speed interact to determine whether high-speed vehicle observations are plausible on a candidate arc.

LightGBM is uniquely suited because:
1. **Histogram-based split finding and leaf-wise (`best-first`) tree growth**: Learns intricate non-linear decision boundaries far more effectively and rapidly than symmetric level-wise trees.
2. **Computational efficiency**: Trains on 1.14 million candidate pairs in under **2.5 seconds** with sub-millisecond inference latency (**0.86 ms per 1,000 candidates**), enabling real-time deployment on Laptop 1.
3. **Calibrated probability outputs**: Optimizing binary log loss yields well-calibrated probabilities in $[0, 1]$, which serve directly as **emission probabilities** for the subsequent HMM / Viterbi sequence decoder.

### 1.2 Major Strengths and Limitations
* **Strengths**:
  - Superior predictive power on tabular spatial data.
  - Invariant to monotonic feature scaling (operates directly on raw physical units: meters, radians, m/s).
  - Handles class imbalance (1 positive candidate to ~23 negative candidates) natively via log-loss optimization.
* **Limitations**:
  - Pointwise model: Evaluates each GPS ping independently without enforcing temporal route continuity (addressed in Stage 2 by HMM).
  - Can overfit on small datasets if depth and minimum child samples are unconstrained.

---

## 2. Training and Configuration

### 2.1 Hyperparameters and Initial Values
* `objective`: `binary` (Binary Cross-Entropy)
* `metric`: `binary_logloss`
* `boosting_type`: `gbdt` (Gradient Boosted Decision Trees)
* `n_estimators`: 150 (boosting iterations)
* `learning_rate`: 0.08 (shrinkage parameter)
* `max_depth`: 6 (constrains maximum tree depth to prevent memorization)
* `num_leaves`: 31 (controls leaf capacity, bounded by $2^6$)
* `min_child_samples`: 50 (regularization enforcing leaf sample size)
* `subsample`: 0.8 (bagging fraction per iteration)
* `colsample_bytree`: 0.8 (feature subsampling per tree)

### 2.2 Optimization Objective
$$\mathcal{L} = - \frac{1}{N} \sum_{i=1}^N \left[ y_i \log p_i + (1 - y_i) \log (1 - p_i) \right]$$
Minimizing binary log-loss ensures the model penalizes overconfident incorrect classifications and generates calibrated candidate scores.

### 2.3 Leakage Prevention Validation Strategy
We implemented **5-Fold Group-Aware Cross-Validation (`GroupKFold`) grouped strictly by `trip_id`**. Observations from the same vehicle trip are never split across train and validation sets, guaranteeing zero cross-trip data leakage.

---

## 3. Experimental Results

### 3.1 Measured Performance Summary

| Metric | 5-Fold GroupKFold CV (80 Train Trips) | Student-Testing (10 Independent Trips) |
|---|---|---|
| **Observation Top-1 Accuracy** | **79.63%** ($\pm 2.80\%$) | **79.57%** |
| **Macro F1-Score** | **0.8739** ($\pm 0.0148$) | **0.8754** |
| **Precision (Positive Candidates)** | — | **78.08%** |
| **Recall (Positive Candidates)** | — | **74.20%** |
| **Log Loss** | **0.0536** | **0.0483** |
| **Training Time** | — | **2.29 seconds** |
| **Inference Latency** | — | **0.862 ms / 1,000 candidates** |

### 3.2 Confusion Matrix (Student-Testing: 450,648 Candidates)
* **True Negatives (TN)**: 427,959
* **False Positives (FP)**: 3,912
* **False Negatives (FN)**: 4,844
* **True Positives (TP)**: 13,933

### 3.3 Overfitting vs. Underfitting Analysis
The model demonstrates **outstanding generalization and stability**:
- CV Top-1 Accuracy is **79.63%**, matching the independent Student-Test Top-1 Accuracy of **79.57%** within 0.06%.
- CV Log Loss is **0.0536**, and Student-Test Log Loss is **0.0483**.
- CV Macro F1 standard deviation is low ($\pm 0.0148$).
There is no evidence of overfitting or divergence between training folds and test trips.

### 3.4 Feature Importance Ranking
1. **`segment_length_m`** (1,349 splits): Separates high-speed freeway arcs from short urban intersection segments.
2. **`gps_speed_mps`** (812 splits): Matches vehicle dynamics with plausible road classes.
3. **`along_fraction`** (809 splits): Determines whether the vehicle projection falls squarely inside segment boundaries ($0 \le t \le 1$).
4. **`cross_track_m`** (733 splits): Primary geometric proximity filter.
5. **`heading_error_rad`** (626 splits): Directional alignment disambiguating opposing lanes.
6. **`time_gap_s`** (171 splits): Accounts for GPS signal dropouts.

---

## 4. Next Step: HMM & Viterbi Sequence Decoding
While LightGBM correctly identifies the ground-truth road segment for ~80% of individual GPS pings, isolated GPS noise still causes occasional pointwise misclassifications. In Stage 2, LightGBM predicted probabilities will be supplied as **emission probabilities** to a Hidden Markov Model with **Viterbi decoding**, enforcing topological connectivity and boosting overall route continuity to near 100%.
