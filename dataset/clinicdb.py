"""dataset"""
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import glob
import random

class ClinicDBDataset(Dataset):
    def __init__(self, root, train=True, category=None, transform=None, gt_target_transform=None):
        super(ClinicDBDataset, self).__init__()
        self.train = train
        self.root = os.path.join(root, 'CVC-ClinicDB')
        self.transform = transform
        self.gt_target_transform = gt_target_transform
        self.preprocess()  
        self.categories = ['clinic']
        self.dataset_name = "clinic"
        self.category = "clinic"
    
    def preprocess(self):
        img_paths = glob.glob(os.path.join(self.root, "images", "*.png"))
        gt_paths = glob.glob(os.path.join(self.root, "masks", "*.png"))
        img_paths.sort()
        gt_paths.sort()
        self.img_paths = img_paths
        self.gt_paths = gt_paths
        self.labels = [1] * len(self.img_paths)   
    
    def update(self, category):
        pass                 
    
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        category="clinic"
        img_path = self.img_paths[idx]
        gt_path = self.gt_paths[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert('RGB')
        if gt_path is not None:
            gt = np.array(Image.open(gt_path))
        else:
            gt = np.zeros((img.size[1], img.size[0]), dtype=np.uint8)
        gt = Image.fromarray(gt)
        if self.transform is not None:
            img = self.transform(img)
        if self.gt_target_transform is not None:
            gt = self.gt_target_transform(gt)
        return img, label, gt, category, img_path
    
    