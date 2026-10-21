import argparse, json, os, glob, csv
import cv2, numpy as np
from scipy.ndimage import median_filter
from lane_geometry import clean_mask, split_lane_candidates, x_at_y

def load_H(path): return np.array(json.load(open(path))['H_image_to_XZ'],dtype=np.float64)

def project_curve_to_xz(curve,H,h):
    ys=np.linspace(max(0.45*h,curve['ymin']),min(h-1,curve['ymax']),80)
    xs=np.array([x_at_y(curve,y) for y in ys])
    pts=np.stack([xs,ys,np.ones_like(xs)],axis=1)
    q=(H@pts.T).T; q=q[:,:2]/q[:,2:3]
    return q

def x_at_z(q,z):
    q=q[np.isfinite(q).all(axis=1)]
    if len(q)<3: return None
    q=q[np.argsort(q[:,1])]
    if z<q[:,1].min() or z>q[:,1].max(): return None
    return float(np.interp(z,q[:,1],q[:,0]))

def depth_ground_consistency(depth, mask, percentile=50):
    # Robustly compare depth spread inside a candidate lane mask.
    vals=depth[mask>0]
    vals=vals[np.isfinite(vals) & (vals>0)]
    if len(vals)<50: return 0.0
    med=np.median(vals); mad=np.median(np.abs(vals-med))+1e-6
    score=float(np.exp(-mad/(0.15*max(med,1e-3))))
    return max(0.0,min(1.0,score))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--frames',default='data/test_frames'); ap.add_argument('--yolop',default='data/outputs/yolop'); ap.add_argument('--homography',default='calibration/homography.json'); ap.add_argument('--output',default='data/outputs/final'); ap.add_argument('--distance',type=float,default=15.0); ap.add_argument('--depth-npz',default=''); a=ap.parse_args()
    os.makedirs(a.output,exist_ok=True); H=load_H(a.homography)
    paths=sorted(glob.glob(os.path.join(a.frames,'*.jpg'))+glob.glob(os.path.join(a.frames,'*.png')))
    depths=None
    if a.depth_npz and os.path.exists(a.depth_npz):
        z=np.load(a.depth_npz); depths=z['depths']
        if len(depths)!=len(paths): print('Warning: depth/frame count mismatch; depth consistency disabled.')
    rows=[]
    for idx,p in enumerate(paths):
        stem=os.path.splitext(os.path.basename(p))[0]; img=cv2.imread(p); h,w=img.shape[:2]
        lane=cv2.imread(os.path.join(a.yolop,'lane',stem+'.png'),0); drive=cv2.imread(os.path.join(a.yolop,'drive',stem+'.png'),0)
        if lane is None: rows.append({'frame':stem,'width_m':'','confidence':0,'status':'MISSING_LANE_MASK'}); continue
        lane=clean_mask(lane,drive); curves=split_lane_candidates(lane,h)
        if len(curves)<2: rows.append({'frame':stem,'width_m':'','confidence':0,'status':'NO_RELIABLE_LANE_MARKINGS'}); continue
        projected=[(c,project_curve_to_xz(c,H,h)) for c in curves]; vals=[]
        for i in range(len(projected)):
            xi=x_at_z(projected[i][1],a.distance)
            if xi is None: continue
            for j in range(i+1,len(projected)):
                xj=x_at_z(projected[j][1],a.distance)
                if xj is None: continue
                width=abs(xj-xi)
                if 1.5<=width<=6.0: vals.append((width,xi,xj,projected[i][0],projected[j][0]))
        if not vals: rows.append({'frame':stem,'width_m':'','confidence':0,'status':'NO_VALID_PAIR'}); continue
        width,xi,xj,cl,cr=min(vals,key=lambda v:abs(v[0]-3.5))
        lane_strength=min(1.0,0.35+0.65*min(cl['area'],cr['area'])/(h*w*0.02))
        depth_score=1.0
        if depths is not None and idx<len(depths): depth_score=depth_ground_consistency(depths[idx],lane)
        conf=0.65*lane_strength+0.35*depth_score
        status='OK' if conf>=0.45 else 'LOW_CONFIDENCE'
        rows.append({'frame':stem,'width_m':round(width,4) if status=='OK' else round(width,4),'left_x_m':round(min(xi,xj),4),'right_x_m':round(max(xi,xj),4),'confidence':round(conf,3),'status':status})
    valid=[r for r in rows if r.get('width_m') not in ('',None) and r.get('status')=='OK']
    if valid:
        arr=np.array([float(r['width_m']) for r in valid]); sm=median_filter(arr,size=5,mode='nearest'); vi=0
        for r in rows:
            if r in valid: r['width_m_smooth']=round(float(sm[vi]),4); vi+=1
    cols=sorted({k for r in rows for k in r})
    with open(os.path.join(a.output,'lane_width.csv'),'w',newline='') as f: wri=csv.DictWriter(f,fieldnames=cols); wri.writeheader(); wri.writerows(rows)
    ok=[float(r['width_m_smooth']) for r in rows if r.get('width_m_smooth') not in ('',None)]
    summary={'frames':len(rows),'valid_frames':len(ok),'median_width_m':float(np.median(ok)) if ok else None,'p10_width_m':float(np.percentile(ok,10)) if ok else None,'p90_width_m':float(np.percentile(ok,90)) if ok else None}
    json.dump(summary,open(os.path.join(a.output,'summary.json'),'w'),indent=2)
    print('Saved',os.path.join(a.output,'lane_width.csv')); print(summary)
if __name__=='__main__': main()
