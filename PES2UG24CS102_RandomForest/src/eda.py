from pathlib import Path
import csv, math
ROOT=Path(__file__).resolve().parents[2]/'Student folder'
OUT=Path(__file__).resolve().parents[1]/'results'

def rows(p):
    with p.open() as f:
        for line in f:
            if line.strip(): yield [float(x) for x in line.split()]

out=[]
for part in ('train','student_test'):
    for d in sorted((ROOT/part).iterdir()):
        if not d.is_dir(): continue
        tid=d.name
        track=list(rows(d/f'{tid}.track')); nodes_count=sum(1 for _ in rows(d/f'{tid}.nodes'))
        arc_count=sum(1 for _ in rows(d/f'{tid}.arcs'))
        route=[int(r[0]) for r in rows(d/f'{tid}.route')]; arcs=[(int(r[0]),int(r[1])) for r in rows(d/f'{tid}.arcs')]
        ts=[r[2] for r in track]
        gaps=[b-a for a,b in zip(ts,ts[1:])]
        out.append(dict(partition=part,trip=tid,gps_points=len(track),nodes=nodes_count,arcs=arc_count,route_arcs=len(route),missing_files=sum(not (d/f'{tid}.{ext}').exists() for ext in ('track','nodes','arcs','route')),invalid_gps=sum(not(-180<=r[0]<=180 and -90<=r[1]<=90) for r in track),decreasing_timestamps=sum(x<0 for x in gaps),gaps_gt_10s=sum(x>10 for x in gaps),max_gap_s=max(gaps,default=0),one_sec_fraction=round(sum(abs(x-1)<1e-8 for x in gaps)/max(1,len(gaps)),4),route_discontinuities=sum(arcs[x][1]!=arcs[y][0] for x,y in zip(route,route[1:]) if x<len(arcs) and y<len(arcs))))
OUT.mkdir(exist_ok=True)
with (OUT/'trip_inventory.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=out[0].keys());w.writeheader();w.writerows(out)
for part in ('train','student_test'):
    a=[r for r in out if r['partition']==part]
    print(f'{part}: {len(a)} trips, {sum(r["gps_points"] for r in a)} GPS points, {sum(r["nodes"] for r in a)} nodes, {sum(r["arcs"] for r in a)} arcs, {sum(r["route_arcs"] for r in a)} route arcs')
    print('  missing files:',sum(r['missing_files'] for r in a),'invalid GPS:',sum(r['invalid_gps'] for r in a),'decreasing timestamp gaps:',sum(r['decreasing_timestamps'] for r in a),'gaps >10s:',sum(r['gaps_gt_10s'] for r in a),'max gap:',max(r['max_gap_s'] for r in a))
    print('  route discontinuities:',sum(r['route_discontinuities'] for r in a))
print('Wrote',OUT/'trip_inventory.csv')
