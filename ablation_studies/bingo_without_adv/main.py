"""
@author: amber
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from models.gcn import GCNNet
from models.gat import GATNet 
#from models.fgm import FGM_GCN, FGM_GAT
from utils import * 
import os 
from protein2graph import ProteinGraphDataset
from torch_geometric.loader import DataLoader
from config_init import get_config
torch.cuda.manual_seed(1029)
torch.manual_seed(1029)

                
def training(gnn_model,train_loader,device,optimizer,loss_fn):            
    epoch_loss = 0.0  
    train_num = 0.0 
    gnn_model.train()
    for train_idx, batch_train_data in enumerate(train_loader):
        batch_train_data = batch_train_data.to(device)
        batch_y = batch_train_data.y.view(-1).float()
        optimizer.zero_grad() 
        out = gnn_model(batch_train_data).view(-1)
        loss = loss_fn(out,batch_y) 
        loss.backward()        
        optimizer.step()  
        train_num += batch_train_data.size(0)
        epoch_loss += loss.item() * batch_train_data.size(0)
    return epoch_loss/train_num
       
                
def predicting(test_loader,gnn_model,device):
    gnn_model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    with torch.no_grad():
        for test_idx, batch_test_data in enumerate(test_loader):
            batch_test_data = batch_test_data.to(device) 
            output = gnn_model(batch_test_data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, batch_test_data.y.view(-1, 1).cpu()), 0)
        total_labels_arr = total_labels.numpy().flatten()
        total_preds_arr = total_preds.numpy().flatten() 
    TP,FP,FN,TN,fpr,tpr,auc, aupr,f1_score, accuracy, recall, specificity, precision = get_metric(total_labels_arr, total_preds_arr)
    return TP,FP,FN,TN,fpr,tpr,auc, aupr,f1_score, accuracy, recall, specificity, precision


if __name__ == "__main__":
    # get the hyperparameters 
    config = get_config()
    n_splits = config.n_splits_topofallfeature
    train_batch_size = config.train_batch_size_topofallfeature
    test_batch_size = config.test_batch_size_topofallfeature
    cuda_name = config.cuda_name_topofallfeature
    drop_prob = config.drop_prob_topofallfeature
    n_output = config.n_output_topofallfeature
    lr = config.lr_topofallfeature
    num_epoches = config.num_epoches_topofallfeature
    weight_decay = config.weight_decay_topofallfeature
    kfold_root_path = config.kfold_root_path_topofallfeature            
    ratio = config.ratio_topofallfeature
    raw_data_path = config.raw_data_path_topofallfeature
    modelling = config.modelling_topofallfeature
    model_saving_path = config.model_saving_path_topofallfeature
    print('model:',modelling)
    
    #10-fold CV scheme    
    test_auc = np.zeros(n_splits)
    test_aupr = np.zeros(n_splits)
    test_f1_score = np.zeros(n_splits)
    test_accuracy = np.zeros(n_splits)
    test_recall = np.zeros(n_splits)
    test_specificity = np.zeros(n_splits)
    test_precision = np.zeros(n_splits)
    
    for i in range(n_splits):
        print("This is fold:",i)
        print('-'*20)
        fold_path = os.path.join(kfold_root_path,'fold'+str(i))
        processed_data_file_train = os.path.join(fold_path,'processed','train.pt')
        processed_data_file_test = os.path.join(fold_path,'processed','test.pt')
        model_file_name = 'model_dict_for_fold_{}.pkl'.format(i)
        model_file_path = os.path.join(model_saving_path,model_file_name)
        if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
            print('please run protein2graph.py to prepare data in pytorch format!')
        else:
            # train-validation-test scheme
            train_data_path = os.path.join(fold_path,'train_data.txt')
            test_data_path = os.path.join(fold_path,'test_data.txt')
            train_list = pd.read_csv(train_data_path,sep='\t',index_col=False)['GeneSymbol'].tolist()
            test_list = pd.read_csv(test_data_path,sep='\t',index_col=False)['GeneSymbol'].tolist()

            train_data = ProteinGraphDataset(root=fold_path,gene_list=train_list,mode="train",ratio=ratio,raw_data_path=raw_data_path)
            test_data = ProteinGraphDataset(root=fold_path,gene_list=test_list,mode="test",ratio=ratio,raw_data_path=raw_data_path)

            train_size = int(0.8 * len(train_data))
            valid_size = len(train_data) - train_size
            train_data, valid_data = torch.utils.data.random_split(train_data, [train_size, valid_size])        
    
            train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)  # the dataloader derived from torch geometric
            valid_loader = DataLoader(valid_data, batch_size=test_batch_size, shuffle=False)            
            test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False)  
            
            # adversarial training 
            device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")    
           
            # adversarial attack            
            if modelling == "gcn":                
                gnn_model = GCNNet(esm_embeds=1280,drop_prob=drop_prob,n_output=n_output).to(device)            
            else:
                gnn_model = GATNet(esm_embeds=1280,n_heads=2,drop_prob=drop_prob,n_output=n_output).to(device)
   
            optimizer = torch.optim.Adam(gnn_model.parameters(), lr=lr,weight_decay=weight_decay)
            loss_fn = nn.BCEWithLogitsLoss().to(device)   
            best_val_aupr = 0.0
            
            for epoch in range(num_epoches):
                print('Epoch{}/{}'.format(epoch,(num_epoches-1)))
                print('*'*10)
                ####  training procedure 
                epoch_loss = training(gnn_model,train_loader,device,optimizer,loss_fn)
                print("epoch_loss:",epoch_loss)               
                #### validating the model     
                val_TP,val_FP,val_FN,val_TN,val_fpr,val_tpr,val_auc, val_aupr,val_f1_score, val_accuracy, val_recall, val_specificity, val_precision = predicting(valid_loader,gnn_model,device)
                #taking the aupr as golden standard, save the model parameter for loading in test
                if val_aupr > best_val_aupr: 
                    #save the model parameter for loading in test
                    print("val_auc:",val_auc)
                    print('val_aupr:',val_aupr)
                    torch.save(gnn_model, model_file_path)
                    best_val_aupr = val_aupr
        
        #### test procedure            
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



