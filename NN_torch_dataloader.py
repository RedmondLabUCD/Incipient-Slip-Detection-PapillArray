#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 04:02:57 2022

@author: qiang
"""
import torch
from torch.utils.data import Dataset
import numpy as np

def loader(args
           ):
    input_data_train = np.load(args.trainset_path,allow_pickle=True)
    if args.mode == 'direct':
        train_dataset = data_handler_direct(input_data_train,args,transform=None)
    else:
        raise RuntimeError("Invalid input mode")
    print(f"{len(train_dataset)} items are in the trainset")
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size,shuffle=args.shuffle)
    return train_loader

class data_handler_direct(Dataset):
    def __init__(self,input_data, args, transform=None): 
        self.input_data = input_data
        self.transform = transform 
        self.args = args
        self.sample_len = input_data.shape[1]
        
    def __len__(self):
        return self.input_data.shape[0]
    
    def __getitem__(self, index):
        features = self.input_data[index][...,0:int(self.args.input_dim/self.args.stack_samples)]
        label = self.input_data[index][...,int(self.args.input_dim/self.args.stack_samples):]
        if self.transform:
            raise RuntimeError('No transform required here')
            
        features = features.reshape(int(self.sample_len/self.args.stack_samples), int(self.args.input_dim))
        label = label.reshape(int(self.sample_len/self.args.stack_samples), int(self.args.categories * self.args.stack_samples))[...,-self.args.categories:]

        data = [features, label]
        return data

