import argparse, os, cv2, numpy as np

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--video",default="data/videos/dashcam.mp4")
    ap.add_argument("--output",default="data/test_frames")
    ap.add_argument("--count",type=int,default=200)
    a=ap.parse_args(); os.makedirs(a.output,exist_ok=True)
    cap=cv2.VideoCapture(a.video)
    if not cap.isOpened(): raise SystemExit(f"Cannot open {a.video}")
    n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs=np.linspace(0,max(0,n-1),a.count).astype(int)
    wanted=set(idxs.tolist()); i=0; saved=0
    while True:
        ok,frame=cap.read()
        if not ok: break
        if i in wanted:
            cv2.imwrite(os.path.join(a.output,f"{saved:04d}.jpg"),frame,[cv2.IMWRITE_JPEG_QUALITY,95]); saved+=1
        i+=1
    cap.release(); print(f"Saved {saved} frames to {a.output}")
if __name__=="__main__": main()
