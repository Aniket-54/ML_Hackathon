"""Build a reproducible candidate-feature dataset that teammates can reuse."""
import json
from pathlib import Path
import numpy as np
from candidates import build,FEATURES
BASE=Path(__file__).resolve().parents[1];OUT=BASE/'shared_data';OUT.mkdir(exist_ok=True)
for part,limit in [('train',640),('student_test',20000)]:
 print(f'Preparing {part}...',flush=True)
 X,y,trip,obs,arc_id,reference_arc_id,miss,points=build(part,max_points=limit,include_ids=True)
 path=OUT/f'{part}_candidates.npz'
 np.savez_compressed(path,X=X.astype(np.float32),y=y.astype(np.uint8),trip_id=trip.astype(str),gps_observation_id=obs.astype(str),candidate_arc_id=arc_id.astype(np.int32),reference_arc_id=reference_arc_id.astype(np.int32),feature_names=np.asarray(FEATURES))
 info={'partition':part,'rows':int(len(y)),'sampled_or_total_gps_points':int(points),'covered_gps_points':int(points-miss),'candidate_miss_rate':float(miss/max(points,1)),'feature_names':FEATURES,'target':'y: 1 if candidate arc is closest reference-route arc for that GPS point; otherwise 0','group':'trip_id; use this for group-wise CV','observation_id':'gps_observation_id; group candidate rows to compute observation-level top-1','candidate_arc_id':'row index into that trip’s .arcs file','reference_arc_id':'pointwise geometric target arc ID; keep for audit/evaluation, not as a model feature','sampling':'train: up to 640 evenly spaced GPS observations per trip; student_test: all observations'}
 (OUT/f'{part}_metadata.json').write_text(json.dumps(info,indent=2));print(info,flush=True)
