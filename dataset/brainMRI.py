"""dataset"""
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import glob
from torchvision import transforms

class BrainMRIDataset(Dataset):
    def __init__(self, root, train=True, category=None, transform=None, gt_target_transform=None):
        super(BrainMRIDataset, self).__init__()
        self.img_root = os.path.join(root, 'BrainMRI')
        self.train = train
        self.transform = transform
        self.preprocess()
        self.categories = ['brainmri']
        self.dataset_name = "brainmri"
        self.category = "brainmri"
        
        
    def preprocess(self):
        normal_img_paths, anomaly_img_paths = [], []
        for t in  ['jpeg', 'jpg', 'JPG', 'JEPG', 'png', 'PNG']:
            normal_img_paths.extend(glob.glob(os.path.join(self.img_root, 'no') + "/*." + str(t)))
            anomaly_img_paths.extend(glob.glob(os.path.join(self.img_root, 'yes') + "/*." + str(t)))
        anomaly_img_paths.sort()
        normal_img_paths.sort()
        self.img_paths = normal_img_paths + anomaly_img_paths
        self.labels = np.concatenate([np.zeros(len(normal_img_paths)), np.ones(len(anomaly_img_paths))])
    
    def update(self, category):
        pass
             
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        category = "brainmri"
        gt = 0
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label, gt, category, img_path
    