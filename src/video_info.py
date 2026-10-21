import argparse
import cv2

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--video",default="data/videos/dashcam.mp4")
    a=ap.parse_args()
    cap=cv2.VideoCapture(a.video)
    if not cap.isOpened(): raise SystemExit(f"Cannot open {a.video}")
    w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps=cap.get(cv2.CAP_PROP_FPS); n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Resolution : {w} x {h}")
    print(f"FPS        : {fps:.3f}")
    print(f"Frames     : {n}")
    print(f"Duration   : {n/fps/60:.2f} minutes")
    cap.release()
if __name__=="__main__": main()
