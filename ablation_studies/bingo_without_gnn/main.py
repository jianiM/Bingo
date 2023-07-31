"""
@author: amber
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from utils import * 
import os 
from data_loader import SeqDataset, collate_fn
from torch.utils.data import DataLoader 
from model import fc_layer,FGM
from config_init import get_config
torch.cuda.manual_seed(1029)
torch.manual_seed(1029)

# transfer list of tensors into one combined tensor 
def transfer_list_tensor(b_x):
    batch_tensor = torch.zeros((len(b_x),b_x[0].size(0)))  # [batch_size,1280]
    for row_id in range(len(b_x)):
        batch_tensor[row_id] = b_x[row_id]
    return batch_tensor


def adversarial_training(model,train_loader,device,optimizer,loss_fn,fgm_model):            
    epoch_loss = 0.0 
    epoch_loss_adv = 0.0 
    train_num = 0.0 
    model.train()
    for idx,(b_x,batch_train_label) in enumerate(train_loader):        
        batch_train_data = transfer_list_tensor(b_x).to(device)   
        batch_y = batch_train_label.view(-1).float().to(device)          
        optimizer.zero_grad() 
        out = model(batch_train_data).view(-1)             
        loss = loss_fn(out,batch_y) 
        loss.backward()
        fgm_model.attack() 
        out_attack = model(batch_train_data).view(-1) 
        loss_adv = loss_fn(out_attack,batch_y)  
        loss_adv.backward()
        fgm_model.restore()
        optimizer.step()  
        train_num += batch_train_data.size(0)
        epoch_loss += loss.item() * batch_train_data.size(0)
        epoch_loss_adv += loss_adv.item() * batch_train_data.size(0)
    return epoch_loss/train_num, epoch_loss_adv/train_num 
                 
                       
def predicting(test_loader,gnn_model,device):
    gnn_model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    with torch.no_grad():
        for idx,(batch_test_data,batch_test_label) in enumerate(test_loader):
            batch_test_data = transfer_list_tensor(batch_test_data).to(device)
            output = gnn_model(batch_test_data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, batch_test_label.view(-1, 1).cpu()), 0)
        total_labels_arr = total_labels.numpy().flatten()
        total_preds_arr = total_preds.numpy().flatten() 
    TP,FP,FN,TN,fpr,tpr,auc, aupr,f1_score, accuracy, recall, specificity, precision = get_metric(total_labels_arr, total_preds_arr)
    return TP,FP,FN,TN,fpr,tpr,auc, aupr,f1_score, accuracy, recall, specificity, precision


if __name__ == "__main__":
    # get the hyperparameters 
    
    config = get_config()
    raw_data_path = config.raw_data_path_topofallfeature
    n_splits = config.n_splits_topofallfeature 
    kfold_root_path = config.kfold_root_path_topofallfeature
    model_saving_path = config.model_saving_path_topofallfeature
    train_batch_size = config.train_batch_size_topofallfeature
    test_batch_size = config.test_batch_size_topofallfeature
    num_epoches = config.num_epoches_topofallfeature
    cuda_name = config.cuda_name_topofallfeature
    drop_prob = config.drop_prob_topofallfeature
    n_output = config.n_output_topofallfeature
    lr = config.lr_topofallfeature 
    weight_decay = config.weight_decay_topofallfeature 
         
    #10-fold CV scheme    
    test_auc = np.zeros(n_splits)
    test_aupr = np.zeros(n_splits)
    test_f1_score = np.zeros(n_splits)
    test_accuracy = np.zeros(n_splits)
    test_recall = np.zeros(n_splits)
    test_specificity = np.zeros(n_splits)
    test_precision = np.zeros(n_splits)
    
    #10-fold cross validation
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
        train_data = SeqDataset(gene_list=train_list,raw_data_path=raw_data_path) 
        test_data = SeqDataset(gene_list=test_list,raw_data_path=raw_data_path)  
        
        # split the train, validation and test loader
        train_size = int(0.8 * len(train_data))
        valid_size = len(train_data) - train_size
        train_data, valid_data = torch.utils.data.random_split(train_data, [train_size, valid_size])        
       
        train_loader = DataLoader(dataset=train_data,batch_size=train_batch_size,shuffle=True,collate_fn=collate_fn)
        valid_loader = DataLoader(dataset=valid_data,batch_size=test_batch_size,shuffle=False,collate_fn=collate_fn)
        test_loader = DataLoader(dataset=test_data,batch_size=test_batch_size,shuffle=False,collate_fn=collate_fn)
            
        # initialize      
        device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")   
        loss_fn = nn.BCEWithLogitsLoss().to(device)   
        fc_model = fc_layer(input_dim=1280,emb_dim_one=256,emb_dim_two=16,n_output=n_output,drop_prob=drop_prob).to(device)
        fgm_model = FGM(model=fc_model,fc_one_weight="fc_g1.weight",fc_one_bias="fc_g1.bias",fc_two_weight="fc_g2.weight",fc_two_bias="fc_g2.bias")
        optimizer = torch.optim.Adam(fc_model.parameters(), lr=lr,weight_decay=weight_decay)
        
        best_val_aupr = 0.0
        for epoch in range(num_epoches):
            print('Epoch{}/{}'.format(epoch,(num_epoches-1)))
            print('*'*10)
            # adversarial training function  
            epoch_loss, epoch_loss_adv = adversarial_training(model=fc_model,train_loader=train_loader,device=device,optimizer=optimizer,loss_fn=loss_fn,fgm_model=fgm_model)
            print('epoch_loss:',epoch_loss)
            print('epoch_loss_adv:',epoch_loss_adv)
            val_TP,val_FP,val_FN,val_TN,val_fpr,val_tpr,val_auc, val_aupr,val_f1_score, val_accuracy, val_recall, val_specificity, val_precision = predicting(test_loader,fc_model,device)
            if val_aupr > best_val_aupr: 
                print("val_auc:",val_auc)
                print('val_aupr:',val_aupr)
                torch.save(fc_model, model_file_path)
                best_val_aupr = val_aupr
                
        # test procedure           
        checkpoint = torch.load(model_file_path)
        TP,FP,FN,TN,fpr,tpr,auc,aupr,f1_score, accuracy, recall, specificity, precision = predicting(test_loader,checkpoint,device)        
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
        print('test_aupr：',aupr)
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



