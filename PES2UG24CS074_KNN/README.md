# PathSnap: K-Nearest Neighbors (KNN) Model & Evaluation Bundle
**Author / PRN:** PES2UG24CS074  
**Algorithm Family:** Instance-based (Non-parametric Nearest Neighbors)  
**Hackathon:** UE24CS352A - PathSnap: Distributed Map-Matching & Route Reconstruction

---

## 1. Model Overview & Mathematical Foundation
K-Nearest Neighbors is a non-parametric, lazy-learning algorithm that classifies an unlabelled query candidate $x_q$ by finding the $k$ closest instances in the training database according to a distance metric.

- **Euclidean Metric in Standardized Feature Space**:
  $$d(x, z) = \sqrt{\sum_{i=1}^p \left(\frac{x_i - \mu_i}{\sigma_i} - \frac{z_i - \mu_i}{\sigma_i}\right)^2}$$
- **Why Feature Scaling is Strictly Mandatory**:
  Our dataset contains features with wildly differing units: distance (`cross_track_m`) ranges from $0$ to $50$ metres, whereas angular divergence (`heading_error_rad`) ranges from $0$ to $\pi$ radians. Without `StandardScaler`, distance in metres would completely dominate the Euclidean norm, rendering angular and kinematic features mathematically invisible.
- **Distance-Weighted Vote**:
  To prevent distant neighbors from skewing decisions, votes are weighted by the inverse of distance $w_i = \frac{1}{d(x_q, x_i) + \epsilon}$:
  $$\hat{P}(y=1|x_q) = \frac{\sum_{i \in N_k(x_q)} w_i \cdot y_i}{\sum_{i \in N_k(x_q)} w_i}$$

---

## 2. Hyperparameter Configuration & Rationale
| Hyperparameter | Value | Rationale |
| :--- | :--- | :--- |
| `n_neighbors` ($k$) | `15` | Large enough to smooth out sensor noise; small enough to preserve localized boundaries. |
| `weights` | `'distance'` | Gives closer spatial exemplars proportionally higher influence. |
| `algorithm` | `'kd_tree'` | Accelerates nearest-neighbor search from $O(N)$ brute-force to $O(\log N)$. |
| `leaf_size` | `40` | Balances tree depth with memory access overhead. |

---

## 3. Empirical Performance Summary
| Evaluation Metric | Pointwise ML (Before HMM) | Sequence Decoded (After HMM) |
| :--- | :---: | :---: |
| **Top-1 Accuracy (Continuous Trips)** | **84.73%** | **91.80%** |
| **Macro F1-Score** | **0.9205** | — |
| **Log Loss** | **0.1380** | — |
| **Route Continuity** | 90.10% | **97.04%** |
| **Training Time** | 0.01 seconds (Index only) | — |
| **Inference Time** | 6.48 seconds | — |

---

## 4. Strengths & Limitations (Defense Points)
- **Strengths**: Zero training cost ($O(1)$ fit phase); no assumptions about underlying probability distributions; highly intuitive geometric interpretation.
- **Limitations**: High inference latency (6.48s vs 0.26s for LightGBM) because it must query the KD-Tree for every candidate; sensitive to local outlier density and the curse of dimensionality.
