# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 11:10:05 2022
@author: Jiani Ma
"""
import re 
import os 
import pandas as pd
import numpy as np 
from sklearn import metrics
import torch  
import torch.nn as nn 
from model import BiLSTM 
from seq2id import Word2Sequence,tokenizer 
from torch.utils.data import DataLoader 
import torch.optim as optim
import pickle
import torch.nn.functional as F
from utils import *
from config_init import get_config
from data_loader import SeqDataset,collate_fn 
torch.cuda.manual_seed(1029)
torch.manual_seed(1029)
import pickle

def training(model,train_loader,device,optimizer,loss_fn):            
    epoch_loss = 0.0  
    train_num = 0.0 
    model.train()
    for idx, (b_x,b_y) in enumerate(train_loader):
        b_x = b_x.to(device)
        b_y = b_y.to(device).float().view(-1)
        optimizer.zero_grad()
        out = model(b_x).view(-1)
        loss = loss_fn(out,b_y) 
        loss.backward()        
        optimizer.step()  
        train_num += b_x.size(0)
        epoch_loss += loss.item() * b_x.size(0)
    return epoch_loss/train_num


def predicting(test_loader,model,device):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    with torch.no_grad():
        for idx,(batch_test_data,batch_test_label) in enumerate(test_loader):
            batch_test_data = batch_test_data.to(device)
            batch_test_label = batch_test_label.float().to(device)
            output = model(batch_test_data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, batch_test_label.view(-1, 1).cpu()), 0)
        total_labels_arr = total_labels.numpy().flatten()
        total_preds_arr = total_preds.numpy().flatten() 
    TP,FP,FN,TN,fpr,tpr,auc, aupr,f1_score, accuracy, recall, specificity, precision = get_metric(total_labels_arr, total_preds_arr)
    return TP,FP,FN,TN,fpr,tpr,auc, aupr,f1_score, accuracy, recall, specificity, precision


if __name__ == "__main__": 
    # setting the hyperparameters
    config = get_config()
    species = config.species_topofallfeature 
    n_splits = config.n_splits_topofallfeature  
    kfold_root_path = config.kfold_root_path_topofallfeature  
    
    # parameters for CNN
    embed_size = config.embed_size_topofallfeature  
    pad_idx = config.pad_idx_topofallfeature   
    lstm_hidden_size = config.lstm_hidden_size_topofallfeature  
    fc_dim = config.fc_dim_topofallfeature   
    
    # parameters for main
    max_len = config.max_len_topofallfeature
    lr = config.lr_topofallfeature   
    drop_prob = config.drop_prob_topofallfeature
    train_batch_size = config.train_batch_size_topofallfeature
    test_batch_size = config.test_batch_size_topofallfeature 
    cuda_name = config.cuda_name_topofallfeature
    n_output = config.n_output_topofallfeature
    num_epoches = config.num_epoches_topofallfeature
    weight_decay = config.weight_decay_topofallfeature
    model_saving_path = config.model_saving_path_topofallfeature
    amino_dict_path = config.amino_dict_path_topofallfeature 
    mapping_path = config.mapping_path_topofallfeature

    # define 10-fold CV, providing the train gene list and test gene list 
    test_auc = np.zeros(n_splits)
    test_aupr = np.zeros(n_splits)
    test_f1_score = np.zeros(n_splits)
    test_accuracy = np.zeros(n_splits)
    test_recall = np.zeros(n_splits)
    test_specificity = np.zeros(n_splits)
    test_precision = np.zeros(n_splits)
     
    # build dictionary 
    # gene_list_path = "/home/amber/BioGen/data/celegans/orig_sample_list/gene_list.txt"    
    # sentences = pd.read_csv(gene_list_path,sep='\t')['Fasta'].tolist()
    # tokenized_sentences = [tokenizer(seq) for seq in sentences]
    # ws = Word2Sequence()
    # for ts in tokenized_sentences:
    #     ws.fit(ts)
    # ws.build_vocab(max_features=1000)
    # pickle.dump(ws,open("/home/amber/BioGen/experiments/transformer/amino_acid_dic.pkl","wb")) 

    with open(amino_dict_path,'rb') as fr: 
        ws = torch.load(fr)                 
        dict_len = len(ws.dic)
    print('dict_len:',dict_len)

    for i in range(n_splits):    
        print("This is fold:",i)
        print('-'*20)
        #form train_loader and test_loader for each fold
        fold_path = os.path.join(kfold_root_path,'fold'+str(i))
        model_file_name = 'model_dict_for_fold_{}.pkl'.format(i)
        model_file_path = os.path.join(model_saving_path,model_file_name)       
        train_data_path = os.path.join(fold_path,'train_data.txt')
        test_data_path = os.path.join(fold_path,'test_data.txt')
        
        train_list = pd.read_csv(train_data_path,sep='\t',index_col=False)['GeneSymbol'].tolist()
        test_list = pd.read_csv(test_data_path,sep='\t',index_col=False)['GeneSymbol'].tolist()                
        
        pre_train_data = SeqDataset(gene_list=train_list,mapping_path=mapping_path)
        pre_test_data = SeqDataset(gene_list=test_list,mapping_path=mapping_path) 
        
        # prepare the tokenized train data 
        tokenized_train_seqs = [tokenizer(pair[0]) for pair in pre_train_data]   
        train_content = [ws.transform(ts,max_len=max_len) for ts in tokenized_train_seqs]
        labels = [pair[1] for pair in pre_train_data]
        train_data = list(zip(train_content,labels))   #zip ok 

        # prepare the tokenized test data 
        tokenized_test_seqs = [tokenizer(pair[0]) for pair in pre_test_data]   
        test_content = [ws.transform(ts,max_len=max_len) for ts in tokenized_test_seqs]
        test_labels = [pair[1] for pair in pre_test_data]
        test_data = list(zip(test_content,test_labels))   #zip ok 
        
        # prepare for the train loader, valid loader and test loader 
        train_size = int(0.8 * len(train_data))
        valid_size = len(train_data) - train_size
        train_data, valid_data = torch.utils.data.random_split(train_data, [train_size, valid_size])        
    
        train_loader = DataLoader(dataset=train_data,batch_size=train_batch_size,shuffle=True,collate_fn=collate_fn)
        valid_loader = DataLoader(dataset=valid_data,batch_size=test_batch_size,shuffle=False,collate_fn=collate_fn)
        test_loader = DataLoader(dataset=test_data,batch_size=test_batch_size,shuffle=False,collate_fn=collate_fn)
         
        device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")   
        loss_fn = nn.BCEWithLogitsLoss().to(device)   
        
        lstm_model = BiLSTM(dict_len=dict_len,embed_size=embed_size,lstm_hidden_size=lstm_hidden_size,pad_idx=pad_idx,drop_prob=drop_prob,fc_dim=fc_dim,n_output=n_output).to(device)
        optimizer = torch.optim.Adam(lstm_model.parameters(), lr=lr,weight_decay=weight_decay) 
       
        best_val_aupr = 0.0
        for epoch in range(num_epoches):
            print('Epoch{}/{}'.format(epoch,(num_epoches-1)))
            print('*'*10)
           # adversarial training function  
            epoch_loss = training(model=lstm_model,train_loader=train_loader,device=device,optimizer=optimizer,loss_fn=loss_fn)
            print('epoch_loss:',epoch_loss)
            val_TP,val_FP,val_FN,val_TN,val_fpr,val_tpr,val_auc, val_aupr,val_f1_score, val_accuracy, val_recall, val_specificity, val_precision = predicting(test_loader=valid_loader,model=lstm_model,device=device) 
            if val_aupr > best_val_aupr: 
                print("val_auc:",val_auc)
                print('val_aupr:',val_aupr)
                torch.save(lstm_model, model_file_path)
                best_val_aupr = val_aupr 
   
   
        checkpoint = torch.load(model_file_path)
        TP,FP,FN,TN,fpr,tpr,auc,aupr,f1_score, accuracy, recall, specificity, precision = predicting(test_loader=test_loader,model=checkpoint,device=device)       
        test_auc[i] = auc
        test_aupr[i] = aupr
        test_f1_score[i] = f1_score
        test_accuracy[i] = accuracy
        test_recall[i] = recall
        test_specificity[i] = specificity
        test_precision[i] = precision    
        print('TP:',TP)
        print('FP:',FP)
        print('FN:',FN)
        print('TN:',TN)
        print('fpr:',fpr)
        print('tpr:',tpr)
        print('test_auc:',auc)
        print('test_aupr:',aupr)
        print('f1_score:',f1_score)
        print('accuracy:',accuracy)
        print('recall:',recall)
        print('specificity:',specificity)
        print('precision:',precision)
        
    mean_auroc = np.mean(test_auc)
    mean_aupr = np.mean(test_aupr)
    mean_f1 = np.mean(test_f1_score)
    mean_acc = np.mean(test_accuracy)  
    mean_recall = np.mean(test_recall)
    mean_specificity = np.mean(test_specificity)
    mean_precision = np.mean(test_precision)
    print('mean_auroc:',mean_auroc)
    print('mean_aupr:',mean_aupr)
    print('mean_f1:',mean_f1)
    print('mean_acc:',mean_acc)
    print('mean_recall:',mean_recall)
    print('mean_specificity:',mean_specificity)
    print('mean_precision:',mean_precision)
    std_auc = np.std(test_auc)
    std_aupr = np.std(test_aupr)
    std_f1 = np.std(test_f1_score)
    std_acc = np.std(test_accuracy)
    std_recall = np.std(test_recall)
    std_specificity = np.std(test_specificity)
    std_precision = np.std(test_precision)
    print('std_auc:',std_auc)
    print('std_aupr:',std_aupr)
    print('std_f1:',std_f1)
    print('std_acc:',std_acc)
    print('std_recall:',std_recall)
    print('std_specificity:',std_specificity)
    print('std_precision:',std_precision)




        
        
        
        
        
    
    
    
    
    
    
    
   


    
    
    
    
    
    
    
    
    
    
    
 
