"""Fit Random Forest on the shared processed dataset and evaluate on student_test."""
import json,time
from pathlib import Path
import numpy as np
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score,log_loss
BASE=Path(__file__).resolve().parents[1];DATA=BASE/'shared_data';OUT=BASE/'results';ART=BASE/'artifacts';OUT.mkdir(exist_ok=True);ART.mkdir(exist_ok=True)
PARAMS=dict(n_estimators=60,max_depth=20,min_samples_leaf=4,max_features='sqrt',class_weight='balanced_subsample',n_jobs=4,random_state=42)
def load(part):
 z=np.load(DATA/f'{part}_candidates.npz',allow_pickle=False)
 return {k:z[k] for k in z.files}
def top1(y,p,obs):
 ids,inv=np.unique(obs,return_inverse=True);best=np.full(len(ids),-1.);pred=np.zeros(len(ids),dtype=np.int8);truth=np.zeros(len(ids),dtype=np.int8)
 for j,s in enumerate(p):
  if s>best[inv[j]]:best[inv[j]]=s;pred[inv[j]]=int(y[j])
  if y[j]:truth[inv[j]]=1
 return float(np.mean(pred==truth)),len(ids)
train=load('train');test=load('student_test');X,y,g,o=train['X'],train['y'],train['trip_id'],train['gps_observation_id'];Xt,yt,ot=test['X'],test['y'],test['gps_observation_id']
features=train['feature_names'].astype(str).tolist();train_meta=json.loads((DATA/'train_metadata.json').read_text());test_meta=json.loads((DATA/'student_test_metadata.json').read_text())
print('Shared train matrix',X.shape,'student-test matrix',Xt.shape,flush=True)
folds=[];cv=GroupKFold(n_splits=5);started=time.perf_counter()
for fold,(tr,va) in enumerate(cv.split(X,y,g),1):
 m=RandomForestClassifier(**PARAMS);t=time.perf_counter();m.fit(X[tr],y[tr]);p=m.predict_proba(X[va])[:,1];acc,nobs=top1(y[va],p,o[va])
 row={'fold':fold,'observation_top1':acc,'observations':nobs,'candidate_macro_f1_at_0.5':float(f1_score(y[va],p>=.5,average='macro')),'fit_seconds':time.perf_counter()-t};folds.append(row);print(row,flush=True)
 (OUT/'random_forest_cv.json').write_text(json.dumps({'parameters':PARAMS,'features':features,'folds':folds,'train_metadata':train_meta},indent=2))
print('Fit final model on all shared training rows...',flush=True);model=RandomForestClassifier(**PARAMS);t=time.perf_counter();model.fit(X,y);fit_s=time.perf_counter()-t;dump(model,ART/'random_forest.joblib')
t=time.perf_counter();p=model.predict_proba(Xt)[:,1];infer_s=time.perf_counter()-t;acc,nobs=top1(yt,p,ot)
result={'model':'RandomForestClassifier','parameters':PARAMS,'features':features,'train_candidate_rows':int(len(y)),'train_gps_points':train_meta['sampled_or_total_gps_points'],'train_candidate_miss_rate':train_meta['candidate_miss_rate'],'grouped_5fold_cv_observation_top1_mean':float(np.mean([r['observation_top1'] for r in folds])),'grouped_5fold_cv_observation_top1_std':float(np.std([r['observation_top1'] for r in folds])),'grouped_5fold_cv_candidate_macro_f1_mean':float(np.mean([r['candidate_macro_f1_at_0.5'] for r in folds])),'folds':folds,'student_test_candidate_rows':int(len(yt)),'student_test_gps_points':test_meta['sampled_or_total_gps_points'],'student_test_covered_points':nobs,'student_test_candidate_miss_rate':test_meta['candidate_miss_rate'],'student_test_observation_top1':acc,'student_test_candidate_macro_f1_at_0.5':float(f1_score(yt,p>=.5,average='macro')),'student_test_candidate_log_loss':float(log_loss(yt,np.c_[1-p,p],labels=[0,1])),'final_fit_seconds':fit_s,'student_test_inference_seconds':infer_s,'total_cv_seconds':time.perf_counter()-started,'artifact':'artifacts/random_forest.joblib','evaluation_note':'Top-1 is conditional on a positive reference candidate being generated; candidate miss rate is reported separately. Pointwise nearest-route geometric labels are weak supervision, not exact timestamp-to-route alignment.'}
(OUT/'random_forest_results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
