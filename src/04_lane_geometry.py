import cv2, numpy as np

def clean_mask(mask, drive=None):
    m=(mask>0).astype(np.uint8)*255
    if drive is not None: m=cv2.bitwise_and(m,drive)
    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)); m=cv2.morphologyEx(m,cv2.MORPH_OPEN,k); m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,k)
    return m

def split_lane_candidates(mask, h, min_area=80):
    n,labels,stats,_=cv2.connectedComponentsWithStats(mask,8); comps=[]
    for i in range(1,n):
        area=stats[i,cv2.CC_STAT_AREA]
        if area<min_area: continue
        ys,xs=np.where(labels==i)
        if len(xs)<20: continue
        comps.append((xs.astype(float),ys.astype(float),area))
    # Convert each component into a robust image-space polynomial x(y).
    curves=[]
    for xs,ys,area in comps:
        keep=ys>h*0.35
        xs,ys=xs[keep],ys[keep]
        if len(xs)<20: continue
        deg=2 if len(xs)>=30 else 1
        coef=np.polyfit(ys,xs,deg)
        curves.append({'coef':coef.tolist(),'area':float(area),'n':int(len(xs)),'ymin':float(ys.min()),'ymax':float(ys.max())})
    return curves

def x_at_y(curve,y): return float(np.polyval(np.array(curve['coef']),y))

def choose_pair(curves,width):
    if len(curves)<2: return None
    y=0.85*1080 if width>=1900 else 0.85*720
    vals=sorted([(x_at_y(c,y),c) for c in curves],key=lambda z:z[0])
    best=None
    for i in range(len(vals)):
        for j in range(i+1,len(vals)):
            gap=vals[j][0]-vals[i][0]
            if gap<width*0.08 or gap>width*0.9: continue
            score=vals[i][1]['area']+vals[j][1]['area']-abs(gap-width*0.35)*10
            if best is None or score>best[0]: best=(score,vals[i][1],vals[j][1])
    return None if best is None else (best[1],best[2])
