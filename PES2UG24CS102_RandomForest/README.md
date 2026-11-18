# PathSnap: Random Forest Model & Evaluation Bundle
**Author / PRN:** PES2UG24CS102  
**Algorithm Family:** Ensemble (Bagging of Decision Trees)  
**Hackathon:** UE24CS352A - PathSnap: Distributed Map-Matching & Route Reconstruction

---

## 1. Model Overview & Mathematical Foundation
Random Forest is an ensemble learning method that constructs a forest of uncorrelated decision trees using **Bootstrap Aggregating (Bagging)** and **Random Feature Subspace Projection**.

- **Ensemble Hypothesis**: Let $T_1(x), T_2(x), \dots, T_M(x)$ be $M$ decision trees trained on bootstrap samples of the candidate dataset. The ensemble prediction probability is:
  $$\hat{P}(y=1|x) = \frac{1}{M} \sum_{m=1}^M T_m(x)$$
- **Variance Reduction**: Individual deep decision trees have low bias but high variance. By averaging $M=100$ decorrelated trees (each splitting on a random subset of $\sqrt{p} \approx 4$ features), the variance is reduced by approximately $\frac{1}{M}$, preventing overfitting on noisy GPS observations.
- **Handling Non-Linear Interactions**: Random Forest captures complex non-linear spatial interactions (such as the interaction between `cross_track_m` and `heading_error_rad` during sharp turns) without requiring explicit polynomial feature transformations or feature scaling.

---

## 2. Hyperparameter Configuration & Rationale
| Hyperparameter | Value | Rationale |
| :--- | :--- | :--- |
| `n_estimators` | `100` | Sufficient ensemble size to stabilize variance without excessive inference latency. |
| `max_depth` | `16` | Constrains tree depth to prevent memorizing micro-level GPS sensor noise. |
| `max_features` | `'sqrt'` | Decorrelates individual trees by considering $\approx 4$ random features per split. |
| `min_samples_split` | `10` | Prevents splits on isolated outlier GPS candidate pairs. |
| `min_samples_leaf` | `4` | Ensures leaves represent statistically stable spatial density. |
| `class_weight` | `'balanced_subsample'` | Compensates for the severe 1:7 class imbalance in candidate selection. |

---

## 3. Empirical Performance Summary
Evaluated on the standardized **18-feature dataset** (`shared_data_v2`) and decoded via the **Multi-Hop Viterbi HMM Decoder**:

| Evaluation Metric | Pointwise ML (Before HMM) | Sequence Decoded (After HMM) |
| :--- | :---: | :---: |
| **Top-1 Accuracy (Continuous Trips)** | **86.32%** | **93.45%** |
| **Macro F1-Score** | **0.9326** | — |
| **Log Loss** | **0.0806** | — |
| **Route Continuity** | 90.59% | **97.40%** |
| **Training Time** | 17.12 seconds | — |
| **Inference Time** | 0.28 seconds | — |

*Note: On truncated student-test trips (Trips 77 & 65 where ground-truth ends prematurely), overall test Top-1 is 87.78%. On all standard continuous routes, HMM decoded accuracy reaches 93.45%.*

---

## 4. Top Feature Importances (Gini Impurity Reduction)
1. `spatial_emission_prob`: **28.4%** (Gaussian likelihood of orthogonal distance)
2. `cross_track_m`: **21.5%** (Perpendicular distance from GPS fix to segment centerline)
3. `rank_by_dist`: **16.2%** (Relative spatial rank among candidate arcs for the same epoch)
4. `heading_error_rad`: **11.3%** (Angular divergence between vehicle heading and road vector)
5. `dist_gap_to_next`: **8.2%** (Margin between best candidate and runner-up)

---

## 5. Strengths & Limitations (Defense Points)
- **Strengths**: Robust to GPS jitter, does not require feature scaling, produces naturally regularized probabilities, highly parallelizable training.
- **Limitations**: Slower training time (17.1s) compared to LightGBM (9.7s) and Logistic Regression (1.6s); higher memory footprint when storing 100 deep trees.
