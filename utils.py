"""
@author: amber
"""
import torch
import pandas as pd 
import numpy as np
import networkx as nx 
import matplotlib.pyplot as plt 
#import matplotlib as mlp
#mlp.use('Agg')
import seaborn as sns 
torch.manual_seed(1029)

def flatten(l):
    return [item for sublist in l for item in sublist]


def load_pt(file_path):
    data = torch.load(file_path)
    return data 


def cmap2graph(all_features,contact_map,ratio):
    cmap = contact_map.numpy()
    cmap_arr  = cmap.flatten()
    threshold = np.quantile(cmap_arr, ratio)
    binary_cmap = np.zeros_like(cmap)
    binary_cmap[cmap<threshold] = 1 
    binary_cmap[np.diag_indices_from(cmap)] = 0 
    row_sums = np.sum(binary_cmap,axis=1)
    nonzero_ids = np.where(row_sums!=0)[0]
    node_features = all_features[nonzero_ids]
    zero_ids = np.where(row_sums==0)[0]
    if len(zero_ids)!=0:
        pre_final_binary_cmap = np.delete(binary_cmap,zero_ids,axis=0)
        final_binary_cmap = np.delete(pre_final_binary_cmap,zero_ids,axis=1)
    else: 
        final_binary_cmap = binary_cmap
    row_indices, col_indices = np.where(final_binary_cmap==1)
    edges = list(zip(row_indices,col_indices))
    edge_index = []
    g = nx.Graph(edges) 
    for e1, e2 in g.edges:
        edge_index.append([e1,e2])
    return node_features, edge_index 
        
def get_metric(real_score, predict_score):
    sorted_predict_score = np.array(
        sorted(list(set(np.array(predict_score).flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[np.int32(
        sorted_predict_score_num*np.arange(1, 1000)/1000)]
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]
    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1
    TP = predict_score_matrix.dot(real_score.T)
    FP = predict_score_matrix.sum(axis=1)-TP
    FN = real_score.sum()-TP
    TN = len(real_score.T)-TP-FP-FN
    fpr = FP/(FP+TN)
    tpr = TP/(TP+FN)
    ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
    ROC_dot_matrix.T[0] = [0, 0]
    ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]
    x_ROC = ROC_dot_matrix[0].T
    y_ROC = ROC_dot_matrix[1].T
    auc = 0.5*(x_ROC[1:]-x_ROC[:-1]).T*(y_ROC[:-1]+y_ROC[1:])
    recall_list = tpr
    precision_list = TP/(TP+FP)
    PR_dot_matrix = np.mat(sorted(np.column_stack(
        (recall_list, precision_list)).tolist())).T
    PR_dot_matrix.T[0] = [0, 1]
    PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]
    x_PR = PR_dot_matrix[0].T
    y_PR = PR_dot_matrix[1].T
    aupr = 0.5*(x_PR[1:]-x_PR[:-1]).T*(y_PR[:-1]+y_PR[1:])
    f1_score_list = 2*TP/(len(real_score.T)+TP-TN)
    accuracy_list = (TP+TN)/len(real_score.T)
    specificity_list = TN/(TN+FP)
    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    specificity = specificity_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]
    return TP,FP,FN,TN,fpr,tpr,auc[0, 0], aupr[0, 0],f1_score, accuracy, recall, specificity, precision    
