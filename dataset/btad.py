"""dataset"""
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import glob

class BTADDataset(Dataset):
    def __init__(self, root, train=True, category=None, transform=None, gt_target_transform=None):
        super(BTADDataset, self).__init__()
        self.categories = ['01', '02', '03']
        self.train = train
        self.category = category
        self.root = os.path.join(root, 'btad')
        self.transform = transform
        self.gt_target_transform = gt_target_transform
        print("self.category: ", self.category)
        self.preprocess()  
        self.update(self.category)
        assert len(self.cur_img_paths) == len(self.cur_img_labels)
        assert len(self.cur_img_paths) == len(self.cur_img_categories)
        assert len(self.cur_img_paths) == len(self.cur_gt_paths)
        self.dataset_name = "btad"

        
    def preprocess(self):
        self.img_paths = {'train': {category : [] for category in self.categories}, 'test': {category : [] for category in self.categories}}
        self.gt_paths = {'train': {category : [] for category in self.categories}, 'test': {category : [] for category in self.categories}}
        self.labels = {'train': {category : [] for category in self.categories}, 'test': {category : [] for category in self.categories}}
        
        for phase in ['train', 'test']:
            for category in self.categories:
                img_dir = os.path.join(self.root, category)
                defect_types = os.listdir(os.path.join(img_dir, phase))
                for defect_type in defect_types:
                    if defect_type == 'ok':
                        img_paths = glob.glob(os.path.join(img_dir, phase, defect_type) + "/*.bmp") + glob.glob(os.path.join(img_dir, phase, defect_type) + "/*.png")
                        self.img_paths[phase][category].extend(img_paths)
                        self.gt_paths[phase][category].extend([None] * len(img_paths))
                        self.labels[phase][category].extend([0] * len(img_paths))
                    else:
                        img_paths = glob.glob(os.path.join(img_dir, phase, defect_type) + "/*.bmp") + glob.glob(os.path.join(img_dir, phase, defect_type) + "/*.png")
                        gt_paths = glob.glob(os.path.join(img_dir, 'ground_truth', defect_type) + "/*.bmp") + glob.glob(os.path.join(img_dir, 'ground_truth', defect_type) + "/*.png")
                        img_paths.sort()
                        gt_paths.sort()
                        self.img_paths[phase][category].extend(img_paths)
                        self.gt_paths[phase][category].extend(gt_paths)
                        self.labels[phase][category].extend([1] * len(img_paths))
        
    
    def update(self, category=None):
        self.category = category
        self.cur_img_paths, self.cur_gt_paths, self.cur_img_labels, self.cur_img_categories = [], [], [], []
        if self.train:
            phase = 'train'
        else:
            phase = 'test'
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
    
    
    