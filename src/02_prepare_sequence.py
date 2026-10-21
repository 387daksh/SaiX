import argparse, os, cv2, numpy as np, json

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--video',default='data/videos/dashcam.mp4'); ap.add_argument('--output',default='data/sequence'); ap.add_argument('--fps',type=float,default=5); a=ap.parse_args()
    os.makedirs(a.output,exist_ok=True); cap=cv2.VideoCapture(a.video)
    if not cap.isOpened(): raise SystemExit('Cannot open video')
    srcfps=cap.get(cv2.CAP_PROP_FPS); n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); duration=n/srcfps
    target=max(0.1,a.fps); step=max(1,int(round(srcfps/target))); actual=srcfps/step
    rows=[]; i=0; k=0
    while True:
        ok,frame=cap.read()
        if not ok: break
        if i%step==0:
            path=os.path.join(a.output,f'{k:06d}.jpg'); cv2.imwrite(path,frame,[cv2.IMWRITE_JPEG_QUALITY,92]); rows.append({'sample':k,'source_frame':i,'time_s':i/srcfps,'path':path}); k+=1
        i+=1
    cap.release(); json.dump({'source_fps':srcfps,'target_fps':actual,'frames':rows},open(os.path.join(a.output,'index.json'),'w'),indent=2)
    print(f'Source FPS {srcfps:.3f}; sampled {k} frames at {actual:.3f} FPS over {duration/60:.2f} min')
if __name__=='__main__': main()
