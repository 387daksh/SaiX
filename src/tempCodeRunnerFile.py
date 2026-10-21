
import argparse, csv, json, os, time, warnings
import cv2, numpy as np

warnings.filterwarnings('ignore', message='.*Polyfit may be poorly conditioned.*')
try:
    warnings.simplefilter('ignore', np.exceptions.RankWarning)
except AttributeError:
    pass

# ────────────────────────────────────────────────────────────────
# 1.  YOLOP  (unchanged, proven working)
