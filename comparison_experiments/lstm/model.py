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

class BiLSTM(nn.Module):
    def __init__(self,dict_len,embed_size,lstm_hidden_size,pad_idx,drop_prob,fc_dim,n_output):
        super(BiLSTM,self).__init__()
        self.dict_len = dict_len
        self.embed_size = embed_size
        self.lstm_hidden_size = lstm_hidden_size
        self.pad_idx = pad_idx 
        self.drop_prob = drop_prob
        self.fc_dim = fc_dim
        self.n_output =  n_output 
        self.embed_layer = nn.Embedding(num_embeddings = self.dict_len,embedding_dim = self.embed_size,padding_idx=self.pad_idx)
        self.lstm_layer = nn.LSTM(input_size=self.embed_size,hidden_size=self.lstm_hidden_size,num_layers=2, dropout=self.drop_prob, bidirectional=True)
        self.fc1 = nn.Sequential(nn.Linear(2*self.lstm_hidden_size,self.fc_dim),nn.ReLU(),nn.Dropout(p=self.drop_prob))
        self.fc2 = nn.Linear(self.fc_dim,self.n_output)
    
    def forward(self,b_x):
        embed_out = self.embed_layer(b_x)
        lstm_input = embed_out.transpose(0,1)
        _,(h_n,c_n) = self.lstm_layer(lstm_input)
        lstm_out = torch.cat((h_n[-2,:,:],h_n[-1,:,:]),dim=-1)
        fc_out = self.fc1(lstm_out)
        out = self.fc2(fc_out)
        return out 












        









