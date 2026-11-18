"""Candidate feature extraction with per-trip local metric coordinates.
Labels are geometry-derived pseudo-labels and must be audited against ordered route progression.
"""
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[2]/'Student folder'
FEATURES=['cross_track_m','heading_error_rad','along_fraction','segment_length_m','gps_speed_mps','time_gap_s']

def load(p): return np.loadtxt(p, dtype=float, ndmin=2)
def make_trip(part,trip,max_points=320,k=24):
    d=ROOT/part/trip
    gps=load(d/f'{trip}.track'); nodes=load(d/f'{trip}.nodes'); arcs=load(d/f'{trip}.arcs').astype(np.int64); route=load(d/f'{trip}.route').astype(np.int64).ravel()
    # Local tangent-plane projection around this trip; no global UTM assumption.
    lat0=np.deg2rad(np.mean(gps[:,1])); scale_x=111320*np.cos(lat0); scale_y=110540
    xy=np.c_[gps[:,0]*scale_x,gps[:,1]*scale_y]
    nxy=np.c_[nodes[:,0]*scale_x,nodes[:,1]*scale_y]
    a=nxy[arcs[:,0]]; b=nxy[arcs[:,1]]; vec=b-a; lens=np.linalg.norm(vec,axis=1)
    valid=lens>0
    mids=(a+b)/2; tree=cKDTree(mids)
    # Route arc IDs refer to original row positions; duplicates and row order stay intact.
    route=route[(route>=0)&(route<len(arcs))]
    rt=route[lens[route]>0]
    if not len(rt): return None
    rtree=cKDTree(mids[rt])
    ix=np.unique(np.linspace(0,len(gps)-1,min(len(gps),max_points)).astype(int))
    cand_dist,cand=tree.query(xy[ix],k=min(k,len(arcs)),workers=1)
    if cand.ndim==1: cand=cand[:,None]
    _,rnear=rtree.query(xy[ix],k=min(8,len(rt)),workers=1)
    if rnear.ndim==1:rnear=rnear[:,None]
    # Closest point on candidate segment; nearest route geometry supplies weak labels.
    aa=a[cand]; vv=vec[cand]; vv2=np.sum(vv*vv,axis=2)
    rel=xy[ix,None,:]-aa
    t=np.clip(np.divide(np.sum(rel*vv,axis=2),vv2,out=np.zeros_like(vv2),where=vv2>0),0,1)
    residual=rel-t[:,:,None]*vv
    cd=np.linalg.norm(residual,axis=2)
    heading=np.arctan2(vv[:,:,0],vv[:,:,1])
    if len(ix)>1:
        prev=np.maximum(ix-1,0); delta=xy[ix]-xy[prev]; dt=np.maximum(gps[ix,2]-gps[prev,2],1e-3)
        course=np.arctan2(delta[:,0],delta[:,1]); speed=np.linalg.norm(delta,axis=1)/dt; gaps=gps[ix,2]-gps[prev,2]
    else: course=np.zeros(1);speed=np.zeros(1);gaps=np.zeros(1)
    herr=np.abs((heading-course[:,None]+np.pi)%(2*np.pi)-np.pi)
    # Determine the nearest of up to eight route segments by exact point-to-segment distance.
    rr=rt[rnear]; ra=a[rr]; rv=vec[rr]; rv2=np.sum(rv*rv,axis=2)
    rrel=xy[ix,None,:]-ra; rtfrac=np.clip(np.divide(np.sum(rrel*rv,axis=2),rv2,out=np.zeros_like(rv2),where=rv2>0),0,1)
    rd=np.linalg.norm(rrel-rtfrac[:,:,None]*rv,axis=2); target=rr[np.arange(len(ix)),np.argmin(rd,axis=1)]
    hit=cand==target[:,None]; keep=hit.any(axis=1)
    # Include only points whose route target was generated, recording misses separately.
    nmiss=int((~keep).sum())
    if not keep.any(): return {'X':np.empty((0,len(FEATURES))),'y':np.empty(0,int),'groups':np.empty(0,object),'obs':np.empty(0,object),'candidate_arc_ids':np.empty(0,int),'reference_arc_ids':np.empty(0,int),'miss':nmiss,'points':len(ix)}
    feats=np.stack([cd,herr,t,np.broadcast_to(lens[cand],cd.shape),np.broadcast_to(speed[:,None],cd.shape),np.broadcast_to(gaps[:,None],cd.shape)],axis=2)
    y=hit.astype(np.int8)
    return {'X':feats[keep].reshape(-1,len(FEATURES)),'y':y[keep].reshape(-1),'groups':np.repeat(trip,int(keep.sum())*cand.shape[1]),'obs':np.repeat([f'{trip}:{j}' for j in np.flatnonzero(keep)],cand.shape[1]),'candidate_arc_ids':cand[keep].reshape(-1),'reference_arc_ids':np.repeat(target[keep],cand.shape[1]),'miss':nmiss,'points':len(ix)}

def build(part,max_points=320,include_ids=False):
    xs=[];ys=[];gs=[];os=[];arcids=[];targetids=[];miss=pts=0
    for d in sorted((ROOT/part).iterdir()):
        if not d.is_dir():continue
        q=make_trip(part,d.name,max_points=max_points)
        if q is None:continue
        xs.append(q['X']);ys.append(q['y']);gs.append(q['groups']);os.append(q['obs']);arcids.append(q['candidate_arc_ids']);targetids.append(q['reference_arc_ids']);miss+=q['miss'];pts+=q['points']
    base=(np.concatenate(xs),np.concatenate(ys),np.concatenate(gs),np.concatenate(os))
    if include_ids: return (*base,np.concatenate(arcids),np.concatenate(targetids),miss,pts)
    return (*base,miss,pts)

def predict_candidates(part,trip,k=24):
    """Return all GPS samples, candidate arc IDs, feature matrix, and local graph geometry."""
    d=ROOT/part/trip
    gps=load(d/f'{trip}.track');nodes=load(d/f'{trip}.nodes');arcs=load(d/f'{trip}.arcs').astype(np.int64)
    lat0=np.deg2rad(np.mean(gps[:,1])); sx=111320*np.cos(lat0); sy=110540
    xy=np.c_[gps[:,0]*sx,gps[:,1]*sy]; nxy=np.c_[nodes[:,0]*sx,nodes[:,1]*sy]
    a=nxy[arcs[:,0]];b=nxy[arcs[:,1]];v=b-a;lens=np.linalg.norm(v,axis=1);tree=cKDTree((a+b)/2)
    _,cand=tree.query(xy,k=min(k,len(arcs)),workers=1)
    if cand.ndim==1:cand=cand[:,None]
    aa=a[cand];vv=v[cand];den=np.sum(vv*vv,axis=2);rel=xy[:,None,:]-aa
    t=np.clip(np.divide(np.sum(rel*vv,axis=2),den,out=np.zeros_like(den),where=den>0),0,1)
    cross=np.linalg.norm(rel-t[:,:,None]*vv,axis=2)
    heading=np.arctan2(vv[:,:,0],vv[:,:,1])
    prev=np.maximum(np.arange(len(gps))-1,0);delta=xy-xy[prev];dt=np.maximum(gps[:,2]-gps[prev,2],1e-3)
    course=np.arctan2(delta[:,0],delta[:,1]);speed=np.linalg.norm(delta,axis=1)/dt;gap=gps[:,2]-gps[prev,2]
    herr=np.abs((heading-course[:,None]+np.pi)%(2*np.pi)-np.pi)
    feat=np.stack([cross,herr,t,np.broadcast_to(lens[cand],cross.shape),np.broadcast_to(speed[:,None],cross.shape),np.broadcast_to(gap[:,None],cross.shape)],axis=2)
    return gps,nodes,arcs,xy,cand,feat
