"""dataset"""
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import glob
from torchvision import transforms

class Br35HDataset(Dataset):
    def __init__(self, root, train=True, category=None, transform=None, gt_target_transform=None):
        super(Br35HDataset, self).__init__()
        self.img_root = os.path.join(root, 'Br35H')
        self.transform = transform
        self.train = train
        self.preprocess()  
        self.categories = ['br35h']
        self.dataset_name = "br35h"
        self.category = "br35h"
        
    def preprocess(self):
        normal_img_paths = glob.glob(os.path.join(self.img_root, 'no') + "/*.jpg")
        anomaly_img_paths = glob.glob(os.path.join(self.img_root, 'yes') + "/*.jpg")
        normal_img_paths.sort()
        anomaly_img_paths.sort()
        self.img_paths = normal_img_paths + anomaly_img_paths
        self.labels = np.concatenate([np.zeros(len(normal_img_paths)), np.ones(len(anomaly_img_paths))])
    
    def update(self, category):
        pass
             
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        category = "br35h"
        gt = 0
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label, gt, category, img_path
    