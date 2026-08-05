# Paste the full contents of your local preprocessing.py here
import numpy as np
import cv2
from skimage.morphology import skeletonize

def imread_unicode(path):
    img_array = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)

def get_clean_ink_skeleton(img):
    if img is None:
        return None, None
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    if np.mean(binary) > 127:
        binary = cv2.bitwise_not(binary)
        
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary_clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    skel_bool = skeletonize(binary_clean > 0)
    
    return binary_clean, skel_bool.astype(np.uint8)
