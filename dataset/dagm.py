"""dataset"""
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import glob

class DAGMDataset(Dataset):
    def __init__(self, root, train=True, category=None, transform=None, gt_target_transform=None):
        super(DAGMDataset, self).__init__()
        self.categories = ['Class1', 'Class2', 'Class3', 'Class4', 'Class5',
                           'Class6', 'Class7', 'Class8', 'Class9', 'Class10']
        self.train = train
        self.category = category
        self.root = os.path.join(root, 'DAGM_KaggleUpload')
        self.transform = transform
        self.gt_target_transform = gt_target_transform
        print("self.category: ", self.category)
        self.preprocess()  
        self.dataset_name = 'DAGM'
      
    def preprocess(self):
        self.img_paths = {'Train': {category : [] for category in self.categories}, 'Test': {category : [] for category in self.categories}}
        self.gt_paths = {'Train': {category : [] for category in self.categories}, 'Test': {category : [] for category in self.categories}}
        self.labels = {'Train': {category : [] for category in self.categories}, 'Test': {category : [] for category in self.categories}}
        
        for phase in ['Train', 'Test']:
            for category in self.categories:
                img_dir = os.path.join(self.root, category)
                img_paths = glob.glob(os.path.join(img_dir, phase) + "/*.PNG")
                img_paths.sort()
                gt_paths = []
                labels = []
                for img_path in img_paths:
                    idx_path = img_path.split('/')[-1]
                    idx = idx_path.split('.')[-2]
                    gt_path = os.path.join(img_dir, phase, "Label" , idx + "_label.PNG")
                    label = 1
                    if not os.path.exists(gt_path):
                        gt_path = None
                        label = 0
                    gt_paths.append(gt_path)
                    labels.append(label)
                self.img_paths[phase][category] = img_paths
                self.gt_paths[phase][category] = gt_paths
                self.labels[phase][category] = labels
                    
                    
        
    
    def update(self, category=None):
        self.category = category
        self.cur_img_paths, self.cur_gt_paths, self.cur_img_labels, self.cur_img_categories = [], [], [], []
        if self.train:
            phase = 'Train'
        else:
            phase = 'Test'
        if self.category is not None:
            self.cur_img_paths = self.img_paths[phase][self.category]
            self.cur_gt_paths = self.gt_paths[phase][self.category]
            self.cur_img_labels = self.labels[phase][self.category]
            self.cur_img_categories = [self.category] * len(self.cur_img_paths)
        else:
            for category in self.categories:
                self.cur_img_paths.extend(self.img_paths[phase][category])
                self.cur_gt_paths.extend(self.gt_paths[phase][category])
                self.cur_img_labels.extend(self.labels[phase][category])
                self.cur_img_categories.extend([category] * len(self.img_paths[phase][category]))
        assert len(self.cur_img_paths) == len(self.cur_img_labels)
        assert len(self.cur_img_paths) == len(self.cur_img_categories)
        assert len(self.cur_img_paths) == len(self.cur_gt_paths)
    
    def __len__(self):
        return len(self.cur_img_paths)

    def __getitem__(self, idx):
        category = self.cur_img_categories[idx]
        img_path = self.cur_img_paths[idx]
        label = self.cur_img_labels[idx]
        img = Image.open(img_path).convert('RGB')
        if self.cur_gt_paths[idx] is not None:
            gt = np.array(Image.open(self.cur_gt_paths[idx]))
        else:
            gt = np.zeros((img.size[1], img.size[0]), dtype=np.uint8)
        gt = Image.fromarray(gt)
        if self.transform is not None:
            img = self.transform(img)
        if self.gt_target_transform is not None:
            gt = self.gt_target_transform(gt)
        return img, label, gt, category, img_path
    
    