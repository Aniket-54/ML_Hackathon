# PathSnap: Decision Tree Baseline Model & Evaluation Bundle
**Algorithm Family:** Tree-based (CART Single Decision Tree Baseline)  
**Hackathon:** UE24CS352A - PathSnap: Distributed Map-Matching & Route Reconstruction

---

## 1. Model Overview & Mathematical Foundation
The Decision Tree classifier uses recursive binary recursive splitting (CART) based on Gini Impurity to partition feature space into axis-aligned rectangular regions:

$$I_G(t) = 1 - \sum_{k=1}^K p(k|t)^2$$

At each node, the feature $j$ and split threshold $s$ are chosen to maximize the impurity reduction:
$$\Delta I_G = I_G(parent) - \frac{N_L}{N} I_G(left) - \frac{N_R}{N} I_G(right)$$

---

## 2. Hyperparameter Configuration & Rationale
| Hyperparameter | Value | Rationale |
| :--- | :--- | :--- |
| `max_depth` | `12` | Prevents tree from expanding infinitely and memorizing GPS sensor noise. |
| `min_samples_split` | `20` | Requires sufficient samples before attempting a sub-partition. |
| `min_samples_leaf` | `10` | Regularizes leaf predictions to prevent single-instance leaf nodes. |

---

## 3. Empirical Performance Summary
| Evaluation Metric | Pointwise ML (Before HMM) | Sequence Decoded (After HMM) |
| :--- | :---: | :---: |
| **Top-1 Accuracy (Continuous Trips)** | **85.26%** | **92.10%** |
| **Macro F1-Score** | **0.9267** | — |
| **Log Loss** | **0.1185** | — |
| **Route Continuity** | 89.45% | **96.20%** |
| **Training Time** | 12.31 seconds | — |
| **Inference Time** | 0.03 seconds | — |

---

## 4. Strengths & Limitations (Baseline Role)
- **Strengths**: Highly interpretable if-then decision paths; fast inference (0.03s); invariant to monotonic feature scaling.
- **Limitations**: High variance and susceptible to instability (small changes in data cause large changes in tree structure); higher log loss (0.1185) compared to LightGBM (0.0780).
