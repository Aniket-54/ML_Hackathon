"""
viterbi_decoder.py - Hidden Markov Model (HMM) & Viterbi Sequential Route Decoder (V2)
Author: Aniket S Patil (PES2UG24CS064)
Project: PathSnap: Distributed Map-Matching & Route Reconstruction

Combines supervised candidate emission probabilities with road-network topological
transitions to decode globally coherent, physically valid vehicle trajectories.
"""

from pathlib import Path
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Tuple, Any
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DATA_ROOT = PROJECT_ROOT / "Student folder"


class PathSnapViterbiDecoder:
    """
    Decodes the most probable sequence of road segments using Viterbi dynamic programming.
    
    States: Candidate road arcs at each time step.
    Emissions: Supervised candidate probabilities from trained ML models combined with
               spatial distance priors.
    Transitions: Road-network adjacency from .arcs, with high probability for segment continuity,
                 graph transitions, and exponential penalty for graph disconnection.
    """

    def __init__(
        self,
        arcs: np.ndarray,
        p_same: float = 0.88,
        p_connected: float = 0.11,
        p_disconnected: float = 1e-4,
        gap_threshold_s: float = 5.0,
    ):
        self.arcs = arcs
        self.p_same = p_same
        self.p_connected = p_connected
        self.p_disconnected = p_disconnected
        self.gap_threshold_s = gap_threshold_s

        # Build adjacency graph
        self.node_out = defaultdict(set)
        for aid, (u, v) in enumerate(arcs):
            self.node_out[u].add(aid)

        # Precompute log transitions
        self.log_same = 0.0          # reference score for staying on segment
        self.log_conn = -0.3         # mild cost for moving to connected downstream arc
        self.log_disc = -5.0         # penalty for disconnected graph transitions

    def decode_trip(
        self,
        cand_ids_per_step: np.ndarray,
        probs_per_step: np.ndarray,
        timestamps: np.ndarray = None,
        dists_per_step: np.ndarray = None,
    ) -> List[int]:
        """
        Executes Viterbi sequence decoding for a single trip.
        """
        n_steps, k_cands = cand_ids_per_step.shape
        if n_steps == 0:
            return []

        # Calibrated log emission: ML probability + optional Gaussian distance penalty
        if dists_per_step is not None:
            log_em = np.log(np.maximum(probs_per_step, 1e-4)) - 0.5 * (dists_per_step / 15.0) ** 2
        else:
            log_em = np.log(np.maximum(probs_per_step, 1e-4))

        dp = log_em[0].copy()
        backpointers = np.zeros((n_steps, k_cands), dtype=int)

        for t in range(1, n_steps):
            curr_cands = cand_ids_per_step[t]
            prev_cands = cand_ids_per_step[t - 1]

            # GPS time gap handling: reset transition to neutral prior across gaps
            is_gap = False
            if timestamps is not None and (timestamps[t] - timestamps[t - 1]) > self.gap_threshold_s:
                is_gap = True

            if is_gap:
                log_trans = np.zeros((k_cands, k_cands))
            else:
                log_trans = np.full((k_cands, k_cands), self.log_disc)
                for j in range(k_cands):
                    prev_a = prev_cands[j]
                    prev_to = self.arcs[prev_a, 1]
                    out_arcs = self.node_out[prev_to]

                    for k in range(k_cands):
                        curr_a = curr_cands[k]
                        if curr_a == prev_a:
                            log_trans[j, k] = self.log_same
                        elif curr_a in out_arcs:
                            log_trans[j, k] = self.log_conn

            # Total score matrix: dp[j] + log_trans[j, k] + log_em[t, k]
            scores = dp[:, None] + log_trans + log_em[t][None, :]
            best_prev = np.argmax(scores, axis=0)
            dp = np.max(scores, axis=0)
            backpointers[t] = best_prev

        # Backtrack optimal state path
        best_last = np.argmax(dp)
        curr = best_last
        path = [curr]
        for t in range(n_steps - 1, 0, -1):
            curr = backpointers[t, curr]
            path.append(curr)
        path.reverse()

        return [cand_ids_per_step[t, path[t]] for t in range(n_steps)]

    def compute_route_continuity(self, arc_sequence: List[int]) -> float:
        """
        Calculates Route Continuity:
        Percentage of consecutive transitions (a_{t-1}, a_t) that are physically connected.
        """
        if len(arc_sequence) <= 1:
            return 100.0

        connected = 0
        for i in range(len(arc_sequence) - 1):
            a_prev = arc_sequence[i]
            a_curr = arc_sequence[i + 1]
            if a_prev == a_curr or (a_curr in self.node_out[self.arcs[a_prev, 1]]):
                connected += 1

        return float(connected / (len(arc_sequence) - 1) * 100.0)
