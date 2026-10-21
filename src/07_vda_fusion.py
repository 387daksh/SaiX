import argparse, csv, json, os, sys, time
import cv2, numpy as np, torch

ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
VDA_REPO=os.path.join(ROOT,'third_party','Video-Depth-Anything')
sys.path.insert(0,VDA_REPO)
from video_depth_anything.video_depth_stream import VideoDepthAnything

def load_H(path): return np.array(json.load(open(path))['H_image_to_XZ'],dtype=np.float64)
def x_at_y(c,y): return float(np.polyval(np.asarray(c['coef']),y))
def curves(mask):
    h,w=mask.shape; n,lab,stats,_=cv2.connectedComponentsWithStats(mask,8); out=[]
    for i in range(1,n):
        area=stats[i,cv2.CC_STAT_AREA]
        if area<80: continue
        ys,xs=np.where(lab==i); keep=ys>h*0.35; xs=xs[keep].astype(float); ys=ys[keep].astype(float)
        if len(xs)<30: continue
        out.append({'coef':np.polyfit(ys,xs,2).tolist(),'area':float(area),'n':len(xs),'ymin':float(ys.min()),'ymax':float(ys.max()),'mask':(lab==i).astype(np.uint8)*255})
    return out
def proj_curve(c,H,h):
    ys=np.linspace(max(0.45*h,c['ymin']),min(h-1,c['ymax']),80); xs=np.array([x_at_y(c,y) for y in ys]); pts=np.c_[xs,ys,np.ones_like(xs)]; q=(H@pts.T).T; return q[:,:2]/q[:,2:3]
def xz_at(q,z):
    q=q[np.isfinite(q).all(axis=1)]; q=q[np.argsort(q[:,1])]
    if len(q)<3 or z<q[:,1].min() or z>q[:,1].max(): return None
    return float(np.interp(z,q[:,1],q[:,0]))
def smooth(values,window=5):
    if not values: return []
    a=np.asarray(values,float); out=[]; r=window//2
    for i in range(len(a)): out.append(float(np.median(a[max(0,i-r):min(len(a),i+r+1)])))
    return out
def depth_score(depth,mask):
    vals=depth[mask>0]; vals=vals[np.isfinite(vals)&(vals>0)]
    if len(vals)<50: return 0.0
    med=float(np.median(vals)); mad=float(np.median(np.abs(vals-med)))
    return float(np.clip(np.exp(-mad/(0.15*max(med,1e-3))),0,1))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--video',default='data/videos/dashcam.mp4'); ap.add_argument('--yolop',default='data/outputs/yolop'); ap.add_argument('--homography',default='calibration/homography.json'); ap.add_argument('--output',default='data/outputs/final'); ap.add_argument('--fps',type=float,default=5); ap.add_argument('--distance',type=float,default=15); ap.add_argument('--input-size',type=int,default=518); ap.add_argument('--max-res',type=int,default=960); a=ap.parse_args()
    if not os.path.exists(a.homography): raise SystemExit('Missing calibration/homography.json. Run src/03_calibrate_homography.py first.')
    os.makedirs(a.output,exist_ok=True); H=load_H(a.homography)
    cfg={'vits':{'encoder':'vits','features':64,'out_channels':[48,96,192,384]},'vitb':{'encoder':'vitb','features':128,'out_channels':[96,192,384,768]},'vitl':{'encoder':'vitl','features':256,'out_channels':[256,512,1024,1024]}}
    device='cuda' if torch.cuda.is_available() else 'cpu'; model=VideoDepthAnything(**cfg['vits'],metric=True); ckpt=os.path.join(VDA_REPO,'checkpoints','metric_video_depth_anything_vits.pth'); model.load_state_dict(torch.load(ckpt,map_location='cpu'),strict=True); model=model.to(device).eval(); print('VDA device:',device)
    cap=cv2.VideoCapture(a.video); srcfps=cap.get(cv2.CAP_PROP_FPS); stride=max(round(srcfps/a.fps),1); actual=srcfps/stride; total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); rows=[]; sample=0; frame_no=0; start=time.time()
    with torch.no_grad():
      while True:
        ok,frame=cap.read()
        if not ok: break
        if frame_no%stride==0:
          stem=f'{sample:06d}'; lane_path=os.path.join(a.yolop,'lane',stem+'.png'); drive_path=os.path.join(a.yolop,'drive',stem+'.png'); lane=cv2.imread(lane_path,0); drive=cv2.imread(drive_path,0)
          rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB); h0,w0=frame.shape[:2]
          if a.max_res>0 and max(h0,w0)>a.max_res:
            scale=a.max_res/max(h0,w0); rgb=cv2.resize(rgb,(round(w0*scale),round(h0*scale)))
          depth=model.infer_video_depth_one(rgb,input_size=a.input_size,device=device,fp32=False)
          dscore=0.0; width=None; status='NO_RELIABLE_LANE_MARKINGS'; conf=0.0
          if lane is not None:
            if drive is not None: lane=cv2.bitwise_and(lane,drive)
            lane=(lane>0).astype(np.uint8)*255; cs=curves(lane)
            vals=[]
            for i in range(len(cs)):
              qi=proj_curve(cs[i],H,h0); xi=xz_at(qi,a.distance)
              if xi is None: continue
              for j in range(i+1,len(cs)):
                qj=proj_curve(cs[j],H,h0); xj=xz_at(qj,a.distance)
                if xj is None: continue
                ww=abs(xj-xi)
                if 1.5<=ww<=6.0: vals.append((abs(ww-3.5),ww,xi,xj,cs[i],cs[j]))
            if vals:
              _,width,xi,xj,ci,cj=min(vals,key=lambda v:v[0]);
              # Map lane-mask pixels to the depth resolution for a diagnostic consistency score.
              lm=cv2.resize(lane,(depth.shape[1],depth.shape[0]),interpolation=cv2.INTER_NEAREST); dscore=depth_score(depth,lm); lane_strength=min(1.0,0.35+0.65*min(ci['area'],cj['area'])/(h0*w0*0.02)); conf=0.65*lane_strength+0.35*dscore; status='OK' if conf>=0.45 else 'LOW_CONFIDENCE'
          rows.append({'sample':sample,'source_frame':frame_no,'time_s':frame_no/srcfps,'width_m':'' if width is None else round(float(width),4),'confidence':round(float(conf),3),'depth_consistency':round(float(dscore),3),'status':status}); sample+=1
          if sample%25==0: print(f'samples {sample} / ~{int(total/stride)}')
        frame_no+=1
    cap.release()
    vals=[r['width_m'] for r in rows if r['status']=='OK' and r['width_m']!='']; sm=smooth(vals,5); vi=0
    for r in rows:
      if r['status']=='OK' and r['width_m']!='': r['width_m_smooth']=round(sm[vi],4); vi+=1
    outcsv=os.path.join(a.output,'lane_width.csv'); cols=sorted({k for r in rows for k in r})
    with open(outcsv,'w',newline='') as f: w=csv.DictWriter(f,fieldnames=cols); w.writeheader(); w.writerows(rows)
    ok=[float(r['width_m_smooth']) for r in rows if r.get('width_m_smooth') not in ('',None)]
    summary={'source_fps':srcfps,'processed_fps':actual,'processed_frames':len(rows),'valid_frames':len(ok),'median_width_m':float(np.median(ok)) if ok else None,'p10_width_m':float(np.percentile(ok,10)) if ok else None,'p90_width_m':float(np.percentile(ok,90)) if ok else None,'elapsed_s':time.time()-start}
    json.dump(summary,open(os.path.join(a.output,'summary.json'),'w'),indent=2); print(summary); print('Saved',outcsv)
if __name__=='__main__': main()
