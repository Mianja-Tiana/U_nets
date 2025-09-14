import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2


class BiologyDataset(Dataset):
    def __init__(self, root: str, class_weight={}, transforms=None):
        self.root = root
        self.transforms   = transforms
        self.class_weight = class_weight

        pair = {}
        for img in glob.glob(os.path.join(root, "*", "*")):        
            if (m:=re.search(r'([^/]+?)(?:_mask.*)?\.png$', img)):
                img_id = m.group(1)
                if img_id in pair:
                    if "mask" in img:
                        pair[img_id][1].append(img)
                else:
                    if "mask" in img:
                        path = os.path.join(os.path.dirname(img), img_id + ".png")
                        if os.path.exists(path):
                            pair[img_id] = (path, [img])
                    else:
                        pair[img_id] = (img, [])

        self.images   = list(pair.values())
        self.category = [os.path.dirname(p).split("/")[-1] for p, _ in self.images]

    def __getitem__(self, idx):
        img_path, mask_paths = self.images[idx]
        
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        masks = []
        for m in mask_paths:
            mask = cv2.imread(m, cv2.IMREAD_GRAYSCALE)
            masks.append(np.array(mask, dtype=np.uint8))
        mask = np.max(masks, axis=0)
            
        if self.transforms:
            transformed = self.transforms(image=img, mask=mask)
            img  = transformed["image"]
            mask = transformed["mask"]

        weights = torch.empty(mask.size(), dtype=torch.float64)
        for c in np.unique(mask):
            weights[mask == c] = self.class_weight.get(c, 1.0)
        
        return img, mask, weights

    def __len__(self):
        return len(self.images)