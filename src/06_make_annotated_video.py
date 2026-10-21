import argparse,csv,os,cv2,glob
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--frames',default='data/test_frames'); ap.add_argument('--csv',default='data/outputs/final/lane_width.csv'); ap.add_argument('--output',default='data/outputs/final/lane_width_annotated.mp4'); ap.add_argument('--fps',type=float,default=5); a=ap.parse_args()
    rows=list(csv.DictReader(open(a.csv))); mp={r['frame']:r for r in rows}; paths=sorted(glob.glob(os.path.join(a.frames,'*.jpg'))+glob.glob(os.path.join(a.frames,'*.png')))
    first=cv2.imread(paths[0]); h,w=first.shape[:2]; os.makedirs(os.path.dirname(a.output),exist_ok=True); out=cv2.VideoWriter(a.output,cv2.VideoWriter_fourcc(*'mp4v'),a.fps,(w,h))
    for p in paths:
        stem=os.path.splitext(os.path.basename(p))[0]; img=cv2.imread(p); r=mp.get(stem,{})
        txt='Lane width: '+(r.get('width_m_smooth') or r.get('width_m') or 'N/A')+' m' if r.get('status')=='OK' else r.get('status','UNKNOWN')
        cv2.rectangle(img,(15,15),(610,70),(0,0,0),-1); cv2.putText(img,txt,(30,52),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0) if r.get('status')=='OK' else (0,200,255),2,cv2.LINE_AA); out.write(img)
    out.release(); print('Saved',a.output)
if __name__=='__main__': main()
