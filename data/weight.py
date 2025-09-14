import os
import glob
import cv2
import numpy as np

def compute_class_weights(data_path):
    class_weight = {}
    total = 0.0
    
   
    images = glob.glob(os.path.join(data_path, "*", "*"))
    

    for m in filter(lambda x: "mask" in x, images):
        mask = cv2.imread(m, cv2.IMREAD_GRAYSCALE)
        mask = mask / 255
        mask = mask.astype(np.int64)
        
       
        unique, counts = np.unique(mask, return_counts=True)
        for c, count in zip(unique.tolist(), counts.tolist()):
            class_weight[c] = class_weight.get(c, 0.0) + float(count)
            total += float(count)
    
   
    for c in class_weight.keys():
        class_weight[c] = total / class_weight[c]
    
    return class_weight
