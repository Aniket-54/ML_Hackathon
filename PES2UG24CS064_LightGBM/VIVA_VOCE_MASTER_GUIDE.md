# PathSnap: Technical Viva Voce Master Guide & Code Walkthrough
**Event:** UE24CS352A Machine Learning Hackathon ("PathSnap: Distributed Map-Matching & Route Reconstruction")  
**Student PRN:** PES2UG24CS064 (Aniket S Patil)  
**Primary Assigned Model:** Gradient Boosted Decision Trees (LightGBM)  
**Role:** Primary Model Owner, 18-Feature Preprocessing Architect, Multi-Hop Viterbi HMM Lead

---

## Table of Contents
1. [Repository & Code Architecture Walkthrough](#1-repository--code-architecture-walkthrough)
2. [Deep Dive: Primary Model (LightGBM)](#2-deep-dive-primary-model-lightgbm)
3. [Deep Dive: Teammates' Models](#3-deep-dive-teammates-models)
4. [Deep Dive: Preprocessing & The 18 Features](#4-deep-dive-preprocessing--the-18-features)
5. [Deep Dive: HMM & Multi-Hop Viterbi Sequence Decoding](#5-deep-dive-hmm--multi-hop-viterbi-sequence-decoding)
6. [Consolidated 5-Model Benchmark Comparison](#6-consolidated-5-model-benchmark-comparison)
7. [Panel Viva Questions & Defenses (Section 3.2 & Rubrics)](#7-panel-viva-questions--defenses)

---

# 1. Repository & Code Architecture Walkthrough

### Where Everything Lives:
```
ML_Hackathon/
├── PES2UG24CS064_LightGBM/                 # << YOUR FOLDER (Team Lead / Best Model)
│   ├── 01_first_review.ipynb              # Executed primary notebook (EDA, training, 5-model table)
│   ├── README.md                          # Technical report, architecture, and benchmark table
│   ├── results/
│   │   └── lightgbm_results.json          # Empirical metrics (86.43% Pointwise, 94.21% HMM Top-1)
│   └── src/
│       ├── prepare_high_acc_data.py       # 18-feature KDTree pipeline (98.11% candidate recall)
│       ├── viterbi_decoder.py             # Multi-hop HMM Viterbi route sequence decoder
│       ├── tune_lightgbm_optuna.py        # Bayesian hyperparameter optimization (Section 3.3)
│       └── train_lightgbm.py              # Standalone training script
│
├── PES2UG24CS102_RandomForest/             # Teammate 1 (Random Forest Ensemble)
├── PES2UG24CS074_KNN/                      # Teammate 2 (K-Nearest Neighbors Instance-Based)
├── PES2UG25CS810_LogisticRegression/       # Teammate 3 (Logistic Regression Linear / Probabilistic)
├── DecisionTree/                           # Baseline Single CART Decision Tree
└── shared_data_v2/                         # 18-feature dataset (train: 1.66M rows, test: 153K rows)
```

### Detailed File Functions:
1. `prepare_high_acc_data.py`:
   - Extracts road geometries from `.arcs` files.
   - Builds a 3-point segment KDTree (indexing arc Start, Midpoint, and End node coordinates) to ensure long road segments are not missed by single-point searches.
   - Solves the candidate recall bottleneck: raises candidate recall from teammate's 92.7% ($K=24$) to **98.11%** with just $K=8$.
   - Computes 18 discriminative features per candidate (spatial projection, heading cosine errors, velocity, candidate distance rank, and Gaussian emission scores).
   - Tracks true reference progression using dynamic windowing with forward-hop and directional penalties to prevent highway jumping.
2. `viterbi_decoder.py`:
   - Implements a First-Order Hidden Markov Model (HMM).
   - Hidden states: Candidate road arcs at epoch $t$ ($K=8$).
   - Emissions: Calibrated classifier prediction probabilities transformed into log likelihoods.
   - Transitions: Pre-computes directed graph adjacency and runs multi-hop shortest paths (1-hop, 2-hop, 3-hop) across complex intersections.
   - Decodes the globally optimal sequence of arcs via dynamic programming, eliminating GPS zigzag jumps.
3. `tune_lightgbm_optuna.py`:
   - Implements Tree-structured Parzen Estimator (TPE) Bayesian optimization using `optuna`.
   - Explores `learning_rate` (log uniform: 0.03 to 0.15), `num_leaves` (31 to 80), `max_depth` (5 to 10), and subsampling ratios to minimize binary cross-entropy loss.

---

# 2. Deep Dive: Primary Model (LightGBM)

### Mathematical Formulation
LightGBM builds an additive ensemble of $M$ decision trees:
$$\hat{y}_i = \sum_{m=1}^M f_m(x_i), \quad f_m \in \mathcal{F}$$

At step $m$, it minimizes the second-order Taylor approximation of the binary log-loss:
$$\mathcal{L}^{(m)} \approx \sum_{i=1}^N \left[ g_i f_m(x_i) + \frac{1}{2} h_i f_m^2(x_i) \right] + \Omega(f_m)$$
where:
- First-order gradient: $g_i = \frac{\partial \mathcal{L}(y_i, \hat{y}_i^{(m-1)})}{\partial \hat{y}_i^{(m-1)}} = \sigma(\hat{y}_i^{(m-1)}) - y_i$
- Second-order Hessian: $h_i = \frac{\partial^2 \mathcal{L}(y_i, \hat{y}_i^{(m-1)})}{\partial (\hat{y}_i^{(m-1)})^2} = p_i (1 - p_i)$

### Key Architectural Innovations in LightGBM
1. **Histogram-Based Algorithm**:
   - Instead of sorting continuous feature values (which costs $O(N \cdot p)$ like XGBoost), LightGBM discretizes continuous features into 256 discrete integer bins ($O(N)$ memory).
   - Evaluating splits takes $O(K_{\text{bins}})$, which speeds up candidate training by $8\times$ and regularizes against GPS sensor precision noise.
2. **Leaf-Wise (Best-First) Tree Growth**:
   - Standard tree models (Random Forest, XGBoost) split level-wise (depth-wise).
   - LightGBM splits the single leaf with the highest gain (reduction of loss) regardless of tree symmetry. This allows it to learn deep, asymmetric sub-trees that isolate rare topological edge cases (e.g. sharp turns, parallel access roads). Depth is constrained via `max_depth=7` to prevent overfitting.
3. **GOSS (Gradient-based One-Side Sampling)**:
   - Samples with large gradients $|g_i|$ (instances with high prediction error) contribute most to information gain.
   - GOSS retains all top instances with large gradients and randomly samples instances with small gradients (e.g., 10%), scaling small-gradient instances by $\frac{1-a}{b}$ to preserve the original data distribution without retraining on easy candidates.
4. **Calibrated Probabilities for HMM**:
   - Achieved the lowest Log Loss (**0.0780**) among all tested algorithms.
   - Sharp, well-calibrated probabilities prevent ambiguous emission scores, allowing the Viterbi trellis to discriminate between parallel roads effortlessly.

### Hyperparameters Used
- `n_estimators`: 200
- `learning_rate`: 0.08 (moderate shrinkage to ensure smooth gradient descent)
- `num_leaves`: 45 (controls tree complexity; $< 2^{\text{max\_depth}} = 128$)
- `max_depth`: 7 (guards against overfitting to specific GPS coordinates)
- `subsample`: 0.8 & `colsample_bytree`: 0.8 (decorrelates trees)

---

# 3. Deep Dive: Teammates' Models

You must be able to explain all 5 models when the panel asks for comparisons:

### 1. Teammate 1: Random Forest (`PES2UG24CS102`)
- **Family**: Ensemble (Bagging of Decision Trees).
- **Core Mechanism**: Trains 100 deep decision trees on independent bootstrap samples. Each split considers only a random subset of $\sqrt{p} \approx 4$ features.
- **Why it performs well**:
  - Variance reduction: Individual deep trees have high variance and low bias. Bagging averages their predictions, reducing variance by $\approx \frac{1}{M}$ without increasing bias.
  - Scale invariance: Tree splits only depend on ordinal rank, so feature units (metres vs radians) do not distort splits.
- **Performance**: Pointwise Top-1 = 86.32%, Macro F1 = 0.9326, Log Loss = 0.0806, HMM Route Top-1 = 93.45%.
- **Weakness vs LightGBM**: Takes 17.12s to train (vs 9.75s for LightGBM) and uses much more memory; splits level-wise rather than leaf-wise.

### 2. Teammate 2: K-Nearest Neighbors (`PES2UG24CS074`)
- **Family**: Instance-based / Non-parametric ("Lazy Learner").
- **Core Mechanism**: Stores training feature vectors. For each query candidate, computes Euclidean distance $d(x, z) = \sqrt{\sum (x_i - z_i)^2}$, identifies $k=15$ neighbors, and computes an inverse distance-weighted vote.
- **Why `StandardScaler` is Mandatory**:
  - Distance features range from $0$ to $50$ metres; angular features range from $0$ to $3.14$ radians.
  - Without zero-mean unit-variance scaling, a 5-metre GPS offset completely swamps a 1.5-radian heading error!
- **Performance**: Pointwise Top-1 = 84.73%, Macro F1 = 0.9205, Log Loss = 0.1380, HMM Route Top-1 = 91.80%.
- **Weakness**: **Inference Latency**. Training is instantaneous ($0.01$s), but testing takes **6.48 seconds** because it must search KD-Tree neighbors for every candidate row. Poor scaling for real-time map matching.

### 3. Teammate 3: Logistic Regression (`PES2UG25CS810`)
- **Family**: Linear & Probabilistic (Logit Model).
- **Core Mechanism**: Models log-odds as a linear combination of standardized features:
  $$P(y=1|x) = \frac{1}{1 + e^{-(w^T x + b)}}$$
  Optimized with L-BFGS and L2 Ridge regularization ($C=1.0$).
- **Why it fits HMM so well**:
  - Outputs continuous, well-calibrated posterior probabilities. These sigmoid values feed directly into the HMM observation emission likelihoods: $\ln B_t(a_i) = \ln P(\text{GPS}_t | a_i)$.
- **Performance**: Pointwise Top-1 = 85.02%, Macro F1 = 0.9224, Log Loss = 0.0934, HMM Route Top-1 = 91.50%. Fastest to train (1.61s).
- **Weakness**: Linear decision boundary. Cannot capture complex non-linear spatial interactions (e.g. cross-track distance conditioned on turning angles) without explicit feature interactions.

### 4. Baseline: Single Decision Tree (`DecisionTree/`)
- **Family**: Tree-based (CART Baseline).
- **Core Mechanism**: Greedy recursive binary partitioning maximizing Gini Impurity reduction ($\max \Delta I_G$). Constrained to `max_depth=12`, `min_samples_leaf=10`.
- **Performance**: Pointwise Top-1 = 85.26%, Macro F1 = 0.9267, Log Loss = 0.1185, HMM Route Top-1 = 92.10%.
- **Weakness**: High variance and uncalibrated leaf probabilities (step-wise step functions), leading to higher log loss ($0.1185$) and occasional discontinuous HMM transitions.

---

# 4. Deep Dive: Preprocessing & The 18 Features

### The Candidate Generation Bottleneck We Solved:
- **Teammate's old baseline**: Looked only at the start node of segments using KD-Tree. On curved or long highway arcs (e.g. 200m–500m arcs), the start node is far away from the GPS point even when the vehicle is right in the middle of the segment! Result: Candidate recall was capped at 92.7% even with $K=24$.
- **Our 3-Point KD-Tree**:
  - We index 3 points per road arc: Start node, Midpoint, and End node.
  - Queries return candidate arcs if ANY of the three points is close.
  - **Result**: Candidate recall jumped to **98.11%** with just **$K=8$ candidates per GPS fix**. This reduced candidate matrix size from 450K rows to 153K rows (saving memory and accelerating inference by $3\times$ while increasing accuracy!).

### The 18 Engineered Features:
1. `cross_track_m`: Orthogonal perpendicular distance from GPS fix to segment vector.
2. `heading_error_rad`: Angular difference between GPS bearing and segment direction in radians $[0, \pi]$.
3. `along_fraction`: Projection position along the arc $[0, 1]$ (detects if vehicle is within segment bounds).
4. `segment_length_m`: Physical length of the candidate arc in metres.
5. `gps_speed_mps`: Instantaneous speed reported by GPS receiver.
6. `time_gap_s`: Epoch difference $\Delta t = t_i - t_{i-1}$ (detects GPS dropouts).
7. `segment_heading_sin`: Sine of road segment orientation.
8. `segment_heading_cos`: Cosine of road segment orientation.
9. `gps_bearing_sin`: Sine of vehicle movement bearing.
10. `gps_bearing_cos`: Cosine of vehicle movement bearing.
11. `heading_diff_abs`: Absolute heading difference.
12. `dist_to_start_m`: Euclidean distance from GPS fix to segment start node.
13. `dist_to_end_m`: Euclidean distance from GPS fix to segment end node.
14. `spatial_emission_prob`: Gaussian probability density $N(0, \sigma^2)$ based on cross-track distance.
15. `rank_by_dist`: Relative rank of candidate by distance among the $K=8$ candidates for that fix ($0 = \text{closest}$).
16. `rank_by_heading`: Relative rank of candidate by heading alignment among candidates ($0 = \text{best aligned}$).
17. `dist_gap_to_next`: Distance margin between this candidate and the next best candidate.
18. `emission_gap_to_next`: Probability margin between this candidate and the runner-up candidate.

---

# 5. Deep Dive: HMM & Multi-Hop Viterbi Sequence Decoding

### Why Machine Learning Alone is Not Enough
A machine learning classifier evaluates each GPS point **independently in isolation (pointwise)**.
- If a GPS point has sensor multipath noise and jumps 15 metres to a parallel residential alley, the ML classifier will pick the alley because it is spatially closest.
- This results in a zigzag route that violates vehicle physics and road topology (cars cannot jump through buildings).

### The Hidden Markov Model (HMM) Solution
We formulate map-matching as finding the most likely sequence of hidden states (road segments) given the sequence of noisy GPS observations $Z = (z_1, z_2, \dots, z_T)$.

1. **Hidden States**: The set of candidate road segments $C_t = \{a_1, a_2, \dots, a_K\}$ for epoch $t$.
2. **Observation Emission Probability**:
   $$\ln B_t(a_i) = \ln P(z_t | a_i) \propto \ln \hat{P}_{\text{ML}}(y=1|x_i)$$
   The ML classifier provides the emission likelihood that road arc $a_i$ generated the GPS observation $z_t$.
3. **Transition Probability & Multi-Hop Graph Topology**:
   Vehicles must travel along connected road segments.
   - **Same arc**: Vehicle continues on the same segment $\rightarrow$ Transition cost $= 0.0$.
   - **Direct 1-Hop successor**: Arc $a_j$ starts at the end node of arc $a_i$ $\rightarrow$ Transition cost $= -0.1$.
   - **2-Hop connection** (e.g. short turning arc or junction): Arc $a_j$ is reached via 1 intermediate node $\rightarrow$ Transition cost $= -0.3$.
   - **3-Hop connection**: Short path through roundabouts $\rightarrow$ Transition cost $= -0.6$.
   - **GPS Time Gaps ($> 5$s)**: When GPS signals drop out (tunnel/signal loss), topological constraints are relaxed ($0.0$) because intermediate positions were unobserved.
   - **Disconnected Arcs**: Heavy penalty ($-5.0$).

4. **Viterbi Dynamic Programming Algorithm**:
   Computes the globally optimal state sequence $S^* = (s_1^*, s_2^*, \dots, s_T^*)$ maximizing joint sequence probability:
   - **Initialization**: $V_1(i) = \ln B_1(a_i)$
   - **Recursion**: $V_t(j) = \max_i \left[ V_{t-1}(i) + \ln A_{ij} \right] + \ln B_t(j)$
   - **Backtracking**: Trace back the pointer $\psi_t(j) = \arg\max_i [V_{t-1}(i) + \ln A_{ij}]$ from $t=T$ to $t=1$.

### Why HMM Accuracy Hits 93%–95%
- On clean continuous trips, HMM increases Top-1 Accuracy from **86.43% to 94.21%**, and Route Continuity from **90.63% to 97.12%**.
- Isolated GPS noise spikes are overridden by the topological momentum of the surrounding trajectory.

---

# 6. Consolidated 5-Model Benchmark Comparison

This table directly satisfies **Deliverables Section 3.1** and proves Section 3.2 compliance (all evaluated on the exact same 18-feature dataset `shared_data_v2`):

| Model | Algorithm Family | Configuration | Pointwise Top-1 | Macro F1 | Log Loss | Continuous HMM Top-1 | Route Continuity | Train Time | Infer Time |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM (Primary)** | Gradient Boosting | `n_est=200, lr=0.08, leaves=45, depth=7` | **86.43%** | **0.9329** | **0.0780** | **94.21%** | **97.12%** | **9.75s** | **0.26s** |
| **Random Forest** | Ensemble (Bagging) | `n_est=100, max_depth=16, sqrt` | 86.32% | 0.9326 | 0.0806 | 93.45% | 97.40% | 17.12s | 0.28s |
| **Decision Tree** | Tree-based (CART) | `max_depth=12, min_split=20, leaf=10` | 85.26% | 0.9267 | 0.1185 | 92.10% | 96.20% | 12.31s | 0.03s |
| **Logistic Regression** | Linear / Probabilistic | `L2 penalty, C=1.0, lbfgs, scaled` | 85.02% | 0.9224 | 0.0934 | 91.50% | 97.20% | 1.61s | 0.01s |
| **KNN** | Instance-based | `k=15, distance weights, scaled` | 84.73% | 0.9205 | 0.1380 | 91.80% | 97.04% | 0.01s | 6.48s |

---

# 7. Panel Viva Questions & Defenses

### Q1: "Why did your team pick LightGBM as the best model instead of Random Forest?"
**Your Answer:**
> "While Random Forest and LightGBM achieved comparable pointwise Top-1 accuracies (~86.3% vs 86.4%), LightGBM clearly outperformed Random Forest across three decisive operational criteria:
> 1. **Lower Log Loss (0.0780 vs 0.0806)**: LightGBM produces better-calibrated probabilities. In an HMM pipeline, well-calibrated probabilities are critical because they serve directly as emission likelihoods in the Viterbi trellis.
> 2. **Training Speed (9.75s vs 17.12s)**: LightGBM's histogram-based binning avoids sorting feature values, training almost twice as fast on 1.66 million candidate rows.
> 3. **Leaf-Wise Growth**: LightGBM's best-first split strategy allows it to isolate rare, complex road geometries (like sharp turn angles and highway ramps) with asymmetric trees."

### Q2: "Why didn't you just assign each GPS point to the nearest road segment?"
**Your Answer:**
> "Pure geometric distance fails in urban environments. A GPS error of 10–15 metres often places a vehicle closer to a parallel service road, an alleyway, or an overpass than to the actual multi-lane arterial road. Furthermore, nearest distance ignores vehicle heading, velocity, and road connectivity, resulting in discontinuous zigzag routes with broken topology. Our 18-feature classifier combines orthogonal distance, heading alignment, relative candidate ranks, and kinematic continuity."

### Q3: "What are the hidden states and emissions in your HMM?"
**Your Answer:**
> - **Hidden States**: The true road arc IDs from the road network graph. For each GPS fix $t$, our 3-point KDTree generates $K=8$ candidate road segments, forming the active state space at step $t$.
> - **Emissions**: The posterior probabilities predicted by our trained LightGBM model, mapped into log likelihoods: $\ln B_t(a_i) = \ln P(z_t | a_i) \propto \ln \hat{P}_{\text{LGBM}}(y=1|x_i)$.
> - **Transitions**: Multi-hop graph adjacency pre-computed from the `.arcs` topology. If segment $a_j$ directly succeeds $a_i$ or connects via a 2-hop/3-hop junction, it is permitted with a minimal penalty ($-0.1$ to $-0.6$). Disconnected jumps receive a $-5.0$ penalty."

### Q4: "How did you prevent data leakage during cross-validation?"
**Your Answer:**
> "We used **5-Fold `GroupKFold` grouped strictly by `trip_id`**. GPS points belonging to the same trip are temporally and spatially auto-correlated. If observations from the same trip appeared in both training and validation sets, the model would simply memorize specific coordinates, yielding artificially inflated metrics. GroupKFold guarantees that entire trips are held out, measuring true generalization to unseen vehicle trajectories."

### Q5: "Some test trips show 88% overall accuracy, but you report 94% on clean trips. Why?"
**Your Answer:**
> "We conducted a rigorous audit of the student test partition. Trips `00000077` and `00000065` contain truncated reference route files:
> - In Trip 77, the GPS track is 36.2 km long, but the ground-truth `.route` file terminates prematurely at 15.9 km. For the final 20 km of points, reference IDs are missing.
> On all standard continuous evaluation trips (e.g. Trips 40, 7, 52, 0, 76), our HMM decoded Top-1 accuracy consistently sits at **93.78% – 94.21%**, exactly within the expected non-overfitting range."

### Q6: "Why did KNN require feature scaling while Random Forest and LightGBM did not?"
**Your Answer:**
> "KNN computes the Euclidean distance $\sqrt{\sum (x_i - z_i)^2}$ across all feature dimensions. In our dataset, `cross_track_m` spans $0$ to $50$ metres, while `heading_error_rad` spans $0$ to $3.14$ radians. Without `StandardScaler`, a 5-metre distance change would mathematically outweigh a complete 180-degree heading reversal. Decision trees and tree ensembles split each feature independently based on ordinal value ordering, making them invariant to monotonic scaling."
