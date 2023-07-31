# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 16:48:21 2023

@author: amber
"""
import torch 
import os 
import pandas as pd
from torch.utils.data import Dataset,DataLoader
from utils import load_pt

class SeqDataset(Dataset):
    def __init__(self,gene_list,raw_data_path):
        super(SeqDataset,self).__init__()
        self.gene_list = gene_list 
        self.raw_data_path = raw_data_path
       
       
    def __getitem__(self,ind):
        g_item = self.gene_list[ind]
        g_path = os.path.join(self.raw_data_path,g_item+'.pt')
        g_data = load_pt(g_path)
        g_feature = g_data['feature_representation'].mean(0)   # tensor
        g_target = g_data['target']
        return g_feature, g_target

    def __len__(self):
        return len(self.gene_list)



def collate_fn(batch):
   feature,label = list(zip(*batch))      # feature has already been tensor format.
   label = torch.LongTensor(label) 
   return feature,label







