# PathSnap: Logistic Regression Model & Evaluation Bundle
**Author / PRN:** PES2UG25CS810  
**Algorithm Family:** Linear & Probabilistic (Logit Model)  
**Hackathon:** UE24CS352A - PathSnap: Distributed Map-Matching & Route Reconstruction

---

## 1. Model Overview & Mathematical Foundation
Logistic Regression models the posterior class probability of a candidate road arc being the true match using the standard logistic sigmoid function:

$$P(y=1|x) = \sigma(w^T x + b) = \frac{1}{1 + e^{-(w^T x + b)}}$$

- **Log-Odds (Logit) Formulation**:
  $$\ln\left(\frac{P(y=1|x)}{1 - P(y=1|x)}\right) = w_0 + \sum_{i=1}^p w_i x_i$$
- **L2 Regularization (Ridge Penalty)**:
  Objective function minimized via L-BFGS:
  $$J(w) = -\frac{1}{N} \sum_{j=1}^N \left[ y_j \ln \hat{p}_j + (1-y_j) \ln(1-\hat{p}_j) \right] + \frac{1}{2C} \|w\|_2^2$$
- **Role in HMM Sequence Modeling**:
  Unlike hard classifiers, Logistic Regression outputs smoothly calibrated posterior probabilities. These values map directly into the HMM observation emission likelihoods:
  $$\ln B_t(a_i) = \ln P(\text{GPS}_t | a_i) \propto \ln \sigma(w^T x_i + b)$$

---

## 2. Hyperparameter Configuration & Rationale
| Hyperparameter | Value | Rationale |
| :--- | :--- | :--- |
| `penalty` | `'l2'` | Controls weight magnitudes and prevents collinear feature instability. |
| `C` | `1.0` | Balanced inverse regularization strength. |
| `solver` | `'lbfgs'` | Quasi-Newton optimization method with second-order gradient convergence. |
| `max_iter` | `500` | Ensures full convergence across the standardized 18 features. |

---

## 3. Empirical Performance Summary
| Evaluation Metric | Pointwise ML (Before HMM) | Sequence Decoded (After HMM) |
| :--- | :---: | :---: |
| **Top-1 Accuracy (Continuous Trips)** | **85.02%** | **91.50%** |
| **Macro F1-Score** | **0.9224** | — |
| **Log Loss** | **0.0934** | — |
| **Route Continuity** | 90.53% | **97.20%** |
| **Training Time** | 1.61 seconds | — |
| **Inference Time** | 0.01 seconds | — |

---

## 4. Learned Weight Analysis
- **Positive Weights**: `spatial_emission_prob` (+1.84), `dist_gap_to_next` (+0.54) increase match likelihood.
- **Negative Weights**: `cross_track_m` (-0.98), `heading_error_rad` (-0.67), and `rank_by_dist` (-1.22) penalize matches that are spatially or angularly distant.

---

## 5. Strengths & Limitations (Defense Points)
- **Strengths**: Lightning fast training (1.61s); minimal memory footprint; perfectly calibrated probabilities for HMM emission input.
- **Limitations**: Assumes linear relationship in log-odds space; cannot learn complex feature conjunctions without manual feature interactions.
