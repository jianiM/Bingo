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
import pickle
from seq2id import tokenizer, Word2Sequence
# gene_list: given train gene list or test gene list under each fold 
# raw_data_path: dir for storing the raw data 

class SeqDataset(Dataset):
    def __init__(self,gene_list,mapping_path):
        super(SeqDataset,self).__init__()
        self.gene_list = gene_list 
        self.mapping_path = mapping_path
        self.all_mapping_info = load_pt(self.mapping_path)  
        
    def __getitem__(self,ind):
        g_item = self.gene_list[ind]
        map_info = self.all_mapping_info[g_item]
        fasta = map_info[0]
        label = map_info[1] 
        return fasta, label

        
    def __len__(self):
        return len(self.gene_list)



def collate_fn(batch):
    feature,label = list(zip(*batch))      # feature has already been tensor format.
    feature = torch.LongTensor(feature)
    label = torch.LongTensor(label) 
    return feature,label







