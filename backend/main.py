"""
main.py - PathSnap Production Backend Server (FastAPI)
Two-Laptop Deployment Architecture: Laptop 1 (Model and API Server)
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Optional, List
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import lightgbm as lgb

# Setup paths
BACKEND_DIR = Path(__file__).resolve().parent
REPO_DIR = BACKEND_DIR.parent
BASE_DIR = REPO_DIR.parent
STUDENT_FOLDER = BASE_DIR / "Student folder"
USER_DIR = REPO_DIR / "PES2UG24CS064_LightGBM"
SHARED_DATA = REPO_DIR / "shared_data_v2"

sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(USER_DIR / "src"))
from viterbi_decoder import PathSnapViterbiDecoder
from prepare_high_acc_data import process_trip, FEATURE_NAMES

app = FastAPI(
    title="PathSnap Distributed Map-Matching API",
    description="Backend API hosting primary LightGBM model and Multi-Hop Viterbi HMM Decoder",
    version="2.0.0"
)

# Enable CORS for cross-laptop queries
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model and cache state
MODEL = None
TRIPS_CACHE = {}

def load_or_train_model():
    global MODEL
    model_path = BACKEND_DIR / "lightgbm_model.joblib"
    if model_path.exists():
        print(f"[SERVER STARTUP] Loading pre-trained LightGBM model from {model_path.name}...")
        t0 = time.time()
        import joblib
        MODEL = joblib.load(model_path)
        print(f"[SERVER STARTUP] LightGBM model loaded in {time.time()-t0:.3f}s.")
        return

    print("[SERVER STARTUP] Loading training dataset to fit LightGBM model...")
    train_path = SHARED_DATA / "train_candidates.npz"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing train dataset: {train_path}")
        
    t0 = time.time()
    train_data = np.load(train_path)
    X_tr = train_data['X']
    y_tr = train_data['y']
    
    # Best Bayesian-tuned parameters
    MODEL = lgb.LGBMClassifier(
        n_estimators=103,
        learning_rate=0.0506,
        num_leaves=47,
        max_depth=7,
        subsample=0.8856,
        colsample_bytree=0.7099,
        min_child_samples=51,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    MODEL.fit(X_tr, y_tr)
    print(f"[SERVER STARTUP] LightGBM model loaded in {time.time()-t0:.2f}s ({len(X_tr):,} training rows).")

@app.on_event("startup")
def startup_event():
    load_or_train_model()
    test_dir = STUDENT_FOLDER / "student_test"
    if test_dir.exists():
        for d in sorted(test_dir.iterdir()):
            if d.is_dir() and (d / f"{d.name}.track").exists():
                TRIPS_CACHE[d.name] = d
    print(f"[SERVER STARTUP] Discovered {len(TRIPS_CACHE)} evaluation trips: {list(TRIPS_CACHE.keys())}")
    print("[SERVER READY] Listening for evaluator requests on 0.0.0.0:8000...\n")

@app.get("/")
@app.get("/health")
def health():
    return {
        "status": "online",
        "service": "PathSnap Map-Matching & Route Reconstruction API",
        "model": "LightGBM + Multi-Hop Viterbi HMM",
        "author": "PES2UG24CS064 (Aniket S Patil)",
        "available_trips_count": len(TRIPS_CACHE),
        "available_trips": list(TRIPS_CACHE.keys())
    }

@app.get("/trips")
def get_trips():
    """Returns available student-test evaluation trips with summary metadata."""
    summaries = []
    for tid, tdir in sorted(TRIPS_CACHE.items()):
        track_file = tdir / f"{tid}.track"
        track = np.loadtxt(track_file)
        n_pts = len(track)
        duration_s = float(track[-1, 2] - track[0, 2])
        # Approximate distance
        dlon = np.radians(np.diff(track[:, 0]))
        dlat = np.radians(np.diff(track[:, 1]))
        alat = np.radians((track[:-1, 1] + track[1:, 1]) / 2)
        dist_m = float(np.sum(np.sqrt((dlat * 6371000)**2 + (dlon * 6371000 * np.cos(alat))**2)))
        
        has_ref = (tdir / f"{tid}.route").exists()
        summaries.append({
            "trip_id": tid,
            "gps_points": n_pts,
            "duration_seconds": round(duration_s, 1),
            "distance_km": round(dist_m / 1000, 2),
            "has_reference_route": has_ref
        })
    return {"trips": summaries}

@app.get("/model_info")
def model_info():
    """Returns model architecture, parameters, and 5-model benchmark metrics."""
    return {
        "primary_model": "LightGBM (Gradient Boosted Decision Trees)",
        "optimization": "Bayesian Optimization (Optuna / TPE)",
        "parameters": {
            "n_estimators": 103,
            "learning_rate": 0.0506,
            "num_leaves": 47,
            "max_depth": 7,
            "subsample": 0.8856,
            "colsample_bytree": 0.7099,
            "min_child_samples": 51
        },
        "features": FEATURE_NAMES,
        "benchmark_summary": [
            {"model": "LightGBM (Primary)", "family": "Gradient Boosting", "pointwise_top1": "86.43%", "macro_f1": 0.9329, "log_loss": 0.0780, "hmm_top1": "94.21%", "route_continuity": "97.12%", "infer_time": "0.26s"},
            {"model": "Random Forest", "family": "Ensemble", "pointwise_top1": "86.32%", "macro_f1": 0.9326, "log_loss": 0.0806, "hmm_top1": "93.45%", "route_continuity": "97.40%", "infer_time": "0.28s"},
            {"model": "Decision Tree", "family": "Tree-based", "pointwise_top1": "85.26%", "macro_f1": 0.9267, "log_loss": 0.1185, "hmm_top1": "92.10%", "route_continuity": "96.20%", "infer_time": "0.03s"},
            {"model": "KNN", "family": "Instance-based", "pointwise_top1": "84.73%", "macro_f1": 0.9205, "log_loss": 0.1380, "hmm_top1": "91.80%", "route_continuity": "97.04%", "infer_time": "6.48s"},
            {"model": "Logistic Regression", "family": "Linear / Proba", "pointwise_top1": "85.02%", "macro_f1": 0.9224, "log_loss": 0.0934, "hmm_top1": "91.50%", "route_continuity": "97.20%", "infer_time": "0.01s"}
        ]
    }

@app.post("/predict_trip/{trip_id}")
def predict_trip(trip_id: str, k_candidates: int = Query(8, ge=4, le=16)):
    """
    Core Map-Matching and Route Reconstruction Endpoint.
    Matches noisy GPS trajectory to road network and decodes continuous route using Viterbi HMM.
    """
    t_start = time.time()
    if trip_id not in TRIPS_CACHE:
        raise HTTPException(status_code=404, detail=f"Trip '{trip_id}' not found.")
        
    tdir = TRIPS_CACHE[trip_id]
    track_file = tdir / f"{trip_id}.track"
    nodes_file = tdir / f"{trip_id}.nodes"
    arcs_file = tdir / f"{trip_id}.arcs"
    route_file = tdir / f"{trip_id}.route"
    
    print("=" * 70)
    print(f"[API REQUEST] Incoming request from Evaluator UI for Trip: {trip_id}")
    print("=" * 70)
    
    # 1. Process candidate features using high-recall 3-point KDTree pipeline
    t_proc = time.time()
    X_t, y_t, g_t, o_t, c_t, r_t, matched_true = process_trip(
        tdir, trip_id, is_train=False, k_cands=k_candidates
    )
    proc_time = time.time() - t_proc
    
    n_pts = len(y_t) // k_candidates
    print(f"  Processed {n_pts:,} observations ({len(y_t):,} candidate rows) in {proc_time:.2f}s.")
    
    # 2. Model Inference (LightGBM)
    t_ml = time.time()
    probs = MODEL.predict_proba(X_t)[:, 1]
    ml_time = time.time() - t_ml
    
    # Pointwise Top-1
    probs_mat = probs.reshape(n_pts, k_candidates)
    cands_mat = c_t.reshape(n_pts, k_candidates)
    best_cand_idx = np.argmax(probs_mat, axis=1)
    ml_pred_arcs = cands_mat[np.arange(n_pts), best_cand_idx]
    
    # 3. Multi-Hop Viterbi HMM Route Decoding
    t_hmm = time.time()
    arcs = np.loadtxt(arcs_file, dtype=np.int64, ndmin=2)
    decoder = PathSnapViterbiDecoder(arcs)
    hmm_pred_arcs = decoder.decode_trip(cands_mat, probs_mat)
    hmm_time = time.time() - t_hmm
    
    # 4. Continuity & Accuracies
    ml_continuity = float(decoder.compute_route_continuity(ml_pred_arcs))
    hmm_continuity = float(decoder.compute_route_continuity(hmm_pred_arcs))
    
    pointwise_top1 = None
    hmm_top1 = None
    
    true_arcs = r_t[::k_candidates]
    valid_mask = (true_arcs != -1)
    
    if np.sum(valid_mask) > 0:
        pointwise_top1 = float(np.mean(ml_pred_arcs[valid_mask] == true_arcs[valid_mask]) * 100)
        hmm_top1 = float(np.mean(np.array(hmm_pred_arcs)[valid_mask] == true_arcs[valid_mask]) * 100)
        
    total_latency = time.time() - t_start
    
    # 5. Extract Map Polyline Coordinates for Leaflet/Folium ([lat, lon])
    track = np.loadtxt(track_file)
    nodes = np.loadtxt(nodes_file)
    
    # Raw GPS points: track[:, 0] is lon, track[:, 1] is lat -> [lat, lon]
    raw_gps = [[float(track[j, 1]), float(track[j, 0])] for j in range(0, n_pts, max(1, n_pts // 250))]
    
    # Predicted route coordinates: stitch arc u -> v nodes
    pred_coords = []
    for arc_id in hmm_pred_arcs:
        u_node = arcs[arc_id, 0]
        v_node = arcs[arc_id, 1]
        pred_coords.append([float(nodes[u_node, 1]), float(nodes[u_node, 0])])
    if len(hmm_pred_arcs) > 0:
        last_v = arcs[hmm_pred_arcs[-1], 1]
        pred_coords.append([float(nodes[last_v, 1]), float(nodes[last_v, 0])])
        
    # Reference route coordinates
    ref_coords = []
    if np.sum(valid_mask) > 0:
        for arc_id in true_arcs[valid_mask]:
            if arc_id < len(arcs):
                u_node = arcs[arc_id, 0]
                ref_coords.append([float(nodes[u_node, 1]), float(nodes[u_node, 0])])
                
    center = [float(np.mean(track[:, 1])), float(np.mean(track[:, 0]))]
    
    # Terminal Log for Screenshot requirement (Section 7)
    print(f"  --- RESULTS FOR TRIP {trip_id} ---")
    print(f"  Pointwise Top-1 (Classifier only): {pointwise_top1:.2f}%" if pointwise_top1 else "  Pointwise Top-1: N/A")
    print(f"  >>> HMM DECODED TOP-1 ACCURACY:    {hmm_top1:.2f}% <<<" if hmm_top1 else "  HMM Decoded Top-1: N/A")
    print(f"  Route Continuity (HMM):            {hmm_continuity:.2f}%")
    print(f"  ML Inference Latency:              {ml_time:.3f}s")
    print(f"  Viterbi Decoding Latency:          {hmm_time:.3f}s")
    print(f"  Total API Response Time:           {total_latency:.2f}s")
    print("=" * 70 + "\n")
    
    return {
        "trip_id": trip_id,
        "gps_points_count": n_pts,
        "status": "success",
        "metrics": {
            "pointwise_top1_accuracy": f"{pointwise_top1:.2f}%" if pointwise_top1 is not None else "N/A",
            "hmm_decoded_top1_accuracy": f"{hmm_top1:.2f}%" if hmm_top1 is not None else "N/A",
            "pointwise_route_continuity": f"{ml_continuity:.2f}%",
            "hmm_route_continuity": f"{hmm_continuity:.2f}%",
            "ml_inference_seconds": round(ml_time, 3),
            "viterbi_decoding_seconds": round(hmm_time, 3),
            "total_execution_seconds": round(total_latency, 2)
        },
        "predicted_arc_ids": [int(a) for a in hmm_pred_arcs],
        "coordinates": {
            "center": center,
            "raw_gps": raw_gps,
            "predicted_route": pred_coords,
            "reference_route": ref_coords
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
