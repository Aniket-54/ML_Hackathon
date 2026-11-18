"""Create a layered Folium inspection map for one training trip."""
from pathlib import Path
import numpy as np, folium
ROOT=Path(__file__).resolve().parents[2]/'Student folder'
OUT=Path(__file__).resolve().parents[1]/'figures';OUT.mkdir(exist_ok=True)
trip='00000001';d=ROOT/'train'/trip
track=np.loadtxt(d/f'{trip}.track',ndmin=2);nodes=np.loadtxt(d/f'{trip}.nodes',ndmin=2);arcs=np.loadtxt(d/f'{trip}.arcs',dtype=int,ndmin=2);route=np.loadtxt(d/f'{trip}.route',dtype=int,ndmin=1)
center=[float(track[:,1].mean()),float(track[:,0].mean())]
m=folium.Map(center,zoom_start=14,tiles='OpenStreetMap')
folium.PolyLine(track[:,[1,0]].tolist(),color='#2471A3',weight=3,opacity=.85,tooltip='Raw GPS').add_to(folium.FeatureGroup(name='GPS observations').add_to(m))
coords=[]
for aid in route:
 if 0<=aid<len(arcs):
  u,v=arcs[aid];coords.append([float(nodes[u,1]),float(nodes[u,0])]);coords.append([float(nodes[v,1]),float(nodes[v,0])])
folium.PolyLine(coords,color='#229954',weight=5,opacity=.8,tooltip='Reference route from .route arc IDs').add_to(folium.FeatureGroup(name='Reference route').add_to(m))
folium.LayerControl().add_to(m);m.save(OUT/'trip_00000001_reference.html');print(OUT/'trip_00000001_reference.html')
