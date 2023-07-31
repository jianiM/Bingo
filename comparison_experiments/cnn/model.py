# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 11:09:35 2022
@author: Jiani Ma
"""
import torch 
import torch.nn as nn
import math 
torch.manual_seed(1223)
import re 

class CNNBlock(nn.Module):
    def __init__(self,dict_len,embed_dim,pad_idx,n_output,channels,kernel,drop_prob,fc_dims):
        super(CNNBlock,self).__init__()
        self.embed_layer = nn.Embedding(num_embeddings=dict_len, embedding_dim=embed_dim, padding_idx=pad_idx) 
        self.cnn_layerone = nn.Conv1d(in_channels=embed_dim, out_channels=channels[0], kernel_size=kernel, stride=2)
        self.cnn_layertwo = nn.Conv1d(in_channels=channels[0], out_channels=channels[1], kernel_size=kernel, stride=2)
        self.cnn_layerthree = nn.Conv1d(in_channels=channels[1], out_channels=channels[2], kernel_size=kernel, stride=2)
        self.fc_layerone = nn.Linear(fc_dims[0],fc_dims[1])
        self.fc_layertwo = nn.Linear(fc_dims[1],n_output)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=kernel, stride=2)
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self,b_x): 
        x = self.embed_layer(b_x)
        x = x.permute(0,2,1)              #[512,512,1000]

        x = self.cnn_layerone(x)          #[512,128,498]
        x = self.relu(x)
  
        x = self.maxpool(x)               #[512,128,247]
        x = self.dropout(x)

        x = self.cnn_layertwo(x)          #[512,64,122]
        x = self.relu(x)
        x = self.maxpool(x)               #[512,64,59]
        x = self.dropout(x)
        
        x = self.cnn_layerthree(x)        #[512,16,28]
        x = self.relu(x)
        x = self.maxpool(x)               #[512,16,12]
        x = self.dropout(x) 

        x = x.view(x.size(0),x.size(1)*x.size(2))                  #[512, 16*12]
        x = self.fc_layerone(x)           
        x = self.relu(x)
        x = self.dropout(x)               #[512,16]

        out = self.fc_layertwo(x)         #[512,1]
        return out






        









