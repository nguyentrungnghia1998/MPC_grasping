# I want to create a dataset with torch Dataset and DataLoader
 
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import cv2
import pickle
 
class MyDataset(Dataset):
    def __init__(self, base_dir, add_dir):
        self.base_dir = base_dir
        self.add_dir = add_dir
        self.dataset = os.listdir(os.path.join(add_dir, "positive_grasp"))
        self.length = len(self.dataset)
 
    def __getitem__(self, index):
       
        # Name image
        name_pose = self.dataset[index]
        # Remove .pt from name pose
        id_object = name_pose[:-3]
        id, object, _ = id_object.split('_')
        # Load pose
        poses = torch.load(os.path.join(self.add_dir, "positive_grasp", name_pose))
        # Load image
        name_image = id + '.jpg'
        image = cv2.imread(os.path.join(self.base_dir, "image", name_image))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224,224))
        image = np.transpose(image,(2, 1, 0))
        image = image/255.0
        # Load label
        with open(os.path.join(self.base_dir, "prompt", id + '.pkl'), 'rb') as f:
            label = pickle.load(f)  
        queries = label[1]
        query = queries[int(object)]
        pose = poses[0][1:]
        pose[:-1] = pose[:-1]/416.0
        pose[-1] = pose[-1]/180.0
 
        return torch.FloatTensor(image), query, torch.FloatTensor(pose)
 
 
 
 
    def __len__(self):
        return len(self.dataset)
