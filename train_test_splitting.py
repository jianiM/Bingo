# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 11:32:18 2023
@author: amber
"""
from config_init import get_config
from sklearn.model_selection import KFold
import numpy as np 
import pandas as pd 
import os


def kfoldSplit(n_splits,samples):
   
    '''
    Usage
    ----------
    store and write the train data and test data in each fold
    train_index and test_index are the index of samples, rather than the elements.
    samples should be the numpy form
    '''

    all_splitted_train_samples=[]
    all_splitted_test_samples=[] 
    kf = KFold(n_splits=n_splits,shuffle=True) 
    for i,(train_index,test_index) in enumerate(kf.split(samples)):
        all_splitted_train_samples.append(samples[train_index])
        all_splitted_test_samples.append(samples[test_index])
    return all_splitted_train_samples,all_splitted_test_samples
    

def generate_fold_dataset(n_splits,pos_samples_path,neg_samples_path,kfold_root_path):
    '''
    Usage
    ----------
    gnerating train_data.txt and test_data.txt for each fold
    '''
    pos_samples = pd.read_excel(pos_samples_path)['Ensembl'].values
    neg_samples = pd.read_excel(neg_samples_path)['Ensembl'].values
    splitted_pos_train_samples_bag, splitted_pos_test_samples_bag = kfoldSplit(n_splits=n_splits,samples=pos_samples)
    splitted_neg_train_samples_bag, splitted_neg_test_samples_bag = kfoldSplit(n_splits=n_splits,samples=neg_samples)    
    for i in range(n_splits):
        batch_pos_train_samples = splitted_pos_train_samples_bag[i].reshape(-1,1)
        batch_neg_train_samples = splitted_neg_train_samples_bag[i].reshape(-1,1)        
        batch_train_samples = np.vstack((batch_pos_train_samples,batch_neg_train_samples))
        batch_pos_test_samples = splitted_pos_test_samples_bag[i].reshape(-1,1)
        batch_neg_test_samples = splitted_neg_test_samples_bag[i].reshape(-1,1)
        batch_test_samples = np.vstack((batch_pos_test_samples,batch_neg_test_samples))
        
        fold_dir_path = os.path.join(kfold_root_path,'fold'+str(i))
        if os.path.exists(fold_dir_path):
            print('fold_{}, exist ...'.format(i))
            
        else:
            print('fold_{} not found, making it...'.format(i))
            os.makedirs(fold_dir_path)
        fold_train_path = os.path.join(fold_dir_path,'train_data.txt')
        fold_test_path = os.path.join(fold_dir_path,'test_data.txt') 
        fold_train_data_df = pd.DataFrame(batch_train_samples,columns=['Ensembl'])
        fold_test_data_df = pd.DataFrame(batch_test_samples,columns=['Ensembl']) 
        fold_train_data_df.to_csv(fold_train_path,sep='\t',index=False)
        fold_test_data_df.to_csv(fold_test_path,sep='\t',index=False)



if __name__ == "__main__":
    config = get_config()
    n_splits = config.n_splits_topofallfeature
    pos_samples_path = config.pos_samples_path_topofallfeature
    neg_samples_path = config.neg_samples_path_topofallfeature
    kfold_root_path = config.kfold_root_path_topofallfeature
    generate_fold_dataset(n_splits,pos_samples_path,neg_samples_path,kfold_root_path)
    
    