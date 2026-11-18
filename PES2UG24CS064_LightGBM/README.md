# PathSnap: LightGBM Primary Model & Distributed Map-Matching Pipeline
**Author / PRN:** PES2UG24CS064 (Aniket S Patil)  
**Assigned Model Family:** Gradient Boosted Decision Trees (LightGBM)  
**Hackathon:** UE24CS352A - PathSnap: Distributed Map-Matching & Route Reconstruction

---

## 1. Executive Summary & Architecture
As the primary model and pipeline lead for the PathSnap hackathon, this repository contains:
1. **18-Feature Preprocessing Pipeline (`src/prepare_high_acc_data.py`)**: 3-point segment KDTree candidate generation lifting candidate recall from 92.7% to **98.11%**.
2. **Primary Classifier (LightGBM)**: Gradient-based One-Side Sampling (GOSS) and Exclusive Feature Bundling (EFB) achieving **86.43% Pointwise Top-1** accuracy with lowest log loss (**0.0780**).
3. **Multi-Hop Viterbi HMM Decoder (`src/viterbi_decoder.py`)**: Resolves topological road transitions across multi-hop intersections and GPS gaps, achieving **94.21% Top-1 Route Accuracy** and **97.12% Route Continuity**.
4. **Bayesian Hyperparameter Optimization (`src/tune_lightgbm_optuna.py`)**: Optuna-driven search optimizing tree depth, learning rate, and regularization.

---

## 2. Complete 5-Model Benchmark Comparison (Section 3.1)
All 5 model families evaluated on the **exact same 18-feature standardized dataset**:

| Model Family | Member / PRN | Pointwise Top-1 | Macro F1 | Log Loss | Continuous HMM Top-1 | Route Continuity | Train Time | Infer Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM (Primary)** | **PES2UG24CS064** | **86.43%** | **0.9329** | **0.0780** | **94.21%** | **97.12%** | **9.75s** | **0.26s** |
| **Random Forest** | PES2UG24CS102 | 86.32% | 0.9326 | 0.0806 | 93.45% | 97.40% | 17.12s | 0.28s |
| **Decision Tree** | Baseline | 85.26% | 0.9267 | 0.1185 | 92.10% | 96.20% | 12.31s | 0.03s |
| **KNN** | PES2UG24CS074 | 84.73% | 0.9205 | 0.1380 | 91.80% | 97.04% | 0.01s | 6.48s |
| **Logistic Regression** | PES2UG25CS810 | 85.02% | 0.9224 | 0.0934 | 91.50% | 97.20% | 1.61s | 0.01s |

---

## 3. Why LightGBM Outperformed Other Models
1. **Histogram-Based Splitting**: Bins continuous spatial features (e.g., `cross_track_m`, `heading_error_rad`) into discrete bins, dramatically accelerating training while avoiding overfitting to sensor quantization noise.
2. **Leaf-Wise (Best-First) Tree Growth**: LightGBM expands the leaf with maximum loss reduction rather than growing depth-wise. This creates asymmetric trees that isolate rare spatial patterns (e.g. sharp U-turns, parallel service roads).
3. **Calibrated Probabilities for HMM**: LightGBM achieved the lowest log-loss ($0.0780$), producing sharp, calibrated emission probabilities that allow Viterbi decoding to navigate complex junctions confidently.

---

## 4. Repository Structure
```
PES2UG24CS064_LightGBM/
├── 01_first_review.ipynb          # Fully executed Jupyter notebook with all metrics
├── README.md                      # Comprehensive technical documentation & benchmark
├── requirements.txt               # Dependencies (lightgbm, optuna, scikit-learn, etc.)
├── results/
│   └── lightgbm_results.json      # Structured benchmark results
└── src/
    ├── prepare_high_acc_data.py   # 18-feature preprocessing generator (98.11% recall)
    ├── viterbi_decoder.py         # Multi-hop HMM Viterbi route decoder
    ├── tune_lightgbm_optuna.py    # Optuna Bayesian hyperparameter optimization
    └── train_lightgbm.py          # Standalone training script
```
