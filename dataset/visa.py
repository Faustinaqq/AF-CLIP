"""dataset"""
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import glob
import pandas as pd


class VisaDataset(Dataset):
    def __init__(self, root, train=True, category=None, fewshot=0, transform=None, gt_target_transform=None):
        super(VisaDataset, self).__init__()
        self.categories = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
                            'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3',
                           'pcb4', 'pipe_fryum']
        self.train = train
        self.category = category
        self.fewshot = fewshot
        self.root = os.path.join(root, 'visa')
        self.transform = transform
        self.gt_target_transform = gt_target_transform
        self.preprocess()
        self.update(category=category) 
        self.dataset_name = "visa"
       
        
    def preprocess(self):
        self.csv_data = pd.read_csv(os.path.join(self.root, 'split_csv/1cls.csv'), header=0)
        columns = self.csv_data.columns
        images_paths = {'train': {category : [] for category in self.categories}, 'test': {category : [] for category in self.categories}}
        gt_paths = {'train': {category : [] for category in self.categories}, 'test': {category : [] for category in self.categories}}
        labels = {'train': {category : [] for category in self.categories}, 'test': {category : [] for category in self.categories}}
        for category in self.categories:
            cls_data = self.csv_data[self.csv_data[columns[0]] == category]
            for phase in ['train', 'test']:
                cls_data_phase = cls_data[cls_data[columns[1]] == phase]
                for _, row in cls_data_phase.iterrows():
                    img_path = row[columns[3]]
                    label = 1 if row[columns[2]] == 'anomaly' else 0
                    mask_path = row[columns[4]] if row[columns[2]] == 'anomaly' else None
                    images_paths[phase][category].append(img_path)
                    gt_paths[phase][category].append(mask_path)
                    labels[phase][category].append(label)
        self.img_paths = images_paths
        self.gt_paths = gt_paths
        self.labels = labels
        
    def update(self, category=None):
        self.category = category
        tot_img_paths, tot_gt_paths, tot_img_classes, tot_img_labels = [], [], [], []
        if self.train:
            phase = 'train'
        else:
            phase = 'test'
        if self.category is not None:
            tot_img_paths = self.img_paths[phase][self.category]
            tot_gt_paths = self.gt_paths[phase][self.category]
            tot_img_classes = [self.category] * len(tot_img_paths)
            tot_img_labels = self.labels[phase][self.category]
        else:
            for category in self.categories:
                tot_img_paths.extend(self.img_paths[phase][category])
                tot_gt_paths.extend(self.gt_paths[phase][category])
                tot_img_classes.extend([category] * len(self.img_paths[phase][category]))
                tot_img_labels.extend(self.labels[phase][category])
        
        self.cur_img_paths = tot_img_paths 
        self.cur_gt_paths = tot_gt_paths
        self.cur_img_categories = tot_img_classes
        self.cur_img_labels = tot_img_labels
                
        if self.fewshot != 0:
           
            randidx = np.random.choice(len(self.cur_img_paths), size=self.fewshot, replace=False)
            self.cur_img_paths = [self.cur_img_paths[idx] for idx in randidx]
            self.cur_gt_paths = [self.cur_gt_paths[idx] for idx in randidx]
            self.cur_img_labels = [self.cur_img_labels[idx] for idx in randidx]
            self.cur_img_categories = [self.cur_img_categories[idx] for idx in randidx]
    
    def __len__(self):
        return len(self.cur_img_paths)

    def __getitem__(self, idx):
        category = self.cur_img_categories[idx]
        img_path = os.path.join(self.root, self.cur_img_paths[idx])
        label = self.cur_img_labels[idx]
        img = Image.open(img_path).convert('RGB')
        if self.cur_gt_paths[idx] is not None:
            mask_path = os.path.join(self.root, self.cur_gt_paths[idx])
            gt = np.array(Image.open(mask_path))
            gt[gt != 0] = 255
        else:
            gt = np.zeros((img.size[1], img.size[0]), dtype=np.uint8)
        gt = Image.fromarray(gt)
        if self.transform is not None:
            img = self.transform(img)
        if self.gt_target_transform is not None:
            gt = self.gt_target_transform(gt)
        return img, label, gt, category, img_path
    
    
    