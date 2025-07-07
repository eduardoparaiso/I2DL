# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 16:53:31 2025

@author: Eduardo
"""
import os
import numpy as np
from torch.utils.data import Dataset




class Diagset_20x(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.samples_labels = []
        self.transform = transform

        for class_folder in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_folder)
            if not os.path.isdir(class_path):
                continue
    
        for npy_file in os.listdir(class_path):
            if npy_file.endswith('.npy'):
                file_path = os.path.join(class_path, npy_file)
                self.samples.append(file_path)
                self.samples_labels.append(class_folder)
    
        self.data = []
        self.label = []
        for sample_path, class_label in zip(self.samples, self.samples_labels):
            array = np.load(sample_path)  # [N, H, W, C]
            for patch in array:
                self.data.append(patch)
                self.label.append(class_label)
    
    
    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self, idx):
        patch = self.data[idx] #[H,W,C]
        label = self.label[idx]
        
        if self.transform:
            patch = self.transform(patch)    
            
        return patch, label