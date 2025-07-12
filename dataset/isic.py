"""dataset"""
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import glob
import random

class ISICDataset(Dataset):
    def __init__(self, root, train=True, category=None, transform=None, gt_target_transform=None):
        super(ISICDataset, self).__init__()
        self.train = train
        self.root = os.path.join(root, 'ISIC2016')
        self.transform = transform
        self.gt_target_transform = gt_target_transform
        self.preprocess()  
        self.categories = ["isic"]
        self.dataset_name = "isic"
        self.category = "isic"
    
    def preprocess(self):
        img_paths = glob.glob(os.path.join(self.root, "ISBI2016_ISIC_Part1_Test_Data", "*.jpg"))
        gt_paths = glob.glob(os.path.join(self.root, "ISBI2016_ISIC_Part1_Test_GroundTruth", "*.png"))
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
        category="ISIC"
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
    

    