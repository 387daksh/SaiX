import argparse, json, os
import cv2, numpy as np

points_img=[]

def cb(event,x,y,flags,param):
    if event==cv2.EVENT_LBUTTONDOWN:
        points_img.append((x,y)); print(f'clicked image point {len(points_img)}: ({x},{y})')

def main():
    ap=argparse.ArgumentParser(description='Click road-plane points and enter known X,Z metres.'); ap.add_argument('--image',default='data/test_frames/0000.jpg'); ap.add_argument('--output',default='calibration/homography.json'); a=ap.parse_args()
    img=cv2.imread(a.image)
    if img is None: raise SystemExit(f'Cannot read {a.image}')
    cv2.namedWindow('calibrate'); cv2.setMouseCallback('calibrate',cb)
    while True:
        vis=img.copy()
        for i,(x,y) in enumerate(points_img): cv2.circle(vis,(x,y),7,(0,255,255),-1); cv2.putText(vis,str(i+1),(x+8,y-8),0,0.7,(0,255,255),2)
        cv2.putText(vis,'Click >=4 ROAD points; ENTER when done; ESC abort',(20,35),0,0.8,(255,255,255),2)
        cv2.imshow('calibrate',vis); key=cv2.waitKey(30)&0xff
        if key==27: cv2.destroyAllWindows(); return
        if key in (13,10) and len(points_img)>=4: break
    cv2.destroyAllWindows()
    world=[]
    print('For each clicked point enter X and Z in metres. X right-positive, Z forward-positive.')
    for i,p in enumerate(points_img):
        while True:
            s=input(f'Point {i+1} image={p}, enter X Z: ').strip().split()
            if len(s)==2:
                try: world.append((float(s[0]),float(s[1]))); break
                except: pass
            print('Please enter two numbers, e.g. -1.75 10')
    src=np.float32(points_img); dst=np.float32(world)
    H,mask=cv2.findHomography(src,dst,cv2.RANSAC,3.0)
    if H is None: raise SystemExit('Homography estimation failed')
    os.makedirs(os.path.dirname(a.output) or '.',exist_ok=True)
    json.dump({'image_points':points_img,'world_XZ':world,'H_image_to_XZ':H.tolist(),'note':'World coordinates are metres. X right-positive, Z forward-positive.'},open(a.output,'w'),indent=2)
    print('Saved',a.output); print(H)
if __name__=='__main__': main()
