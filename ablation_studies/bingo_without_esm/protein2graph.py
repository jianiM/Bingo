"""
@author: amber
"""
import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1"
import pandas as pd 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch_geometric.data import Dataset, Data,InMemoryDataset
from utils import * 
from config_init import get_config
import pickle
import re 


def tokenizer(fasta): 
    seq = re.findall(r'.{1}', fasta)
    return seq 


def one_hot_mat(y, num_dim):
    y_onehot = torch.zeros(y.shape[0], num_dim)
    y_onehot.scatter_(1, y.view(-1,1), 1)
    return y_onehot


class ProteinGraphDataset(InMemoryDataset):
    def __init__(self,root='/tmp',transform=None,pre_transform=None,gene_list=None,mode=None,ratio=None,raw_data_path=None,gene_fasta_dict_path=None,amino_acid_dict=None,max_len=None):
        super(ProteinGraphDataset,self).__init__(root,transform,pre_transform)
        self.mode = mode
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            
            self.process(gene_list,ratio,raw_data_path,gene_fasta_dict_path,amino_acid_dict,max_len)
            self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property 
    def raw_file_names(self):
        pass
    
    @property
    def processed_file_names(self):
        return [self.mode+'.pt']

    def download(self):
        pass 
    
    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
    
    def process(self,gene_list,ratio,raw_data_path,gene_fasta_dict_path,amino_acid_dict,max_len):
        data_list = [] 
        data_len = len(gene_list)
        for i in range(data_len):
            print('Converting gene to graph: {}/{}'.format(i+1, data_len))
            g_item = gene_list[i]
            node_features, edge_index,target= self._get_geometric_input(g_item,ratio,raw_data_path,gene_fasta_dict_path,amino_acid_dict,max_len)
            GCNData = Data(x=node_features, edge_index=torch.LongTensor(edge_index).transpose(1,0),y=torch.FloatTensor([target]))
            data_list.append(GCNData)
        print("Graph construction done. Saving to file.")
        print("data list for pytorch geometric:",data_list)         
        #if self.pre_filter is not None:
        #    data_list = [data for data in data_list if self.pre_filter(data)]
        #if self.pre_transform is not None:
        #    data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def _get_geometric_input(self,g_item,ratio,raw_data_path,gene_fasta_dict_path,amino_acid_dict,max_len):
        g_path = g_item + '.pt'
        raw_gene_path = os.path.join(raw_data_path,g_path)  
        raw_data = load_pt(raw_gene_path)
        # print(raw_data)
        ensembl = raw_data['gene_symbol']
        gene_fasta_dict = load_pt(gene_fasta_dict_path)
        fasta = gene_fasta_dict[ensembl][0]
        tokenized_fasta = tokenizer(fasta)
        encoded_fasta = [amino_acid_dict.get(word,0) for word in tokenized_fasta]
        encoded_fasta = torch.LongTensor(encoded_fasta)       
        # one_hot has some problems
        all_onehot_features = one_hot_mat(encoded_fasta,len(amino_acid_dict)+1)
        all_onehot_features = all_onehot_features.float()
        contact_map = raw_data['cmap']
        target = raw_data['target']
        node_features, edge_index = cmap2graph(all_onehot_features,contact_map,ratio=ratio)
        return node_features, edge_index, target
       
                
if __name__ == "__main__":
    config = get_config()
    n_splits = config.n_splits_topofallfeature
    root_path = config.kfold_root_path_topofallfeature            #/home/amber/BioESMGNN_v2/data/celegans/kfold_splitted_data/
    raw_data_path = config.raw_data_path_topofallfeature          #/home/amber/BioESMGNN_v2/data/celegans/raw/
    ratio = config.ratio_topofallfeature 
    gene_fasta_dict_path = config.gene_fasta_dict_path_topofallfeature    # "/home/amber/BioGen/data/celegans/orig_sample_list/gene_fasta_dict.pkl"
    max_len = config.max_len_topofallfeature   
    
    
    amino_acid_dict = {'L': 1, 'S': 2, 'I': 3, 'E': 4, 'V': 5, 'A': 6, 'K': 7, 'T': 8, 'G': 9, 'D': 10, 'F': 11, 'R': 12, 'N': 13, 'P': 14, 'Q': 15, 'Y': 16, 'M': 17, 'H': 18, 'C': 19, 'W': 20, 'U': 21}

    
    for i in range(n_splits):
        print('fold:',i)
        fold_root_path = os.path.join(root_path,'fold'+str(i))
        train_data_path = os.path.join(fold_root_path,'train_data.txt')
        test_data_path = os.path.join(fold_root_path,'test_data.txt')
       
        train_list = pd.read_csv(train_data_path,sep='\t',index_col=False)['GeneSymbol'].tolist()
        test_list = pd.read_csv(test_data_path,sep='\t',index_col=False)['GeneSymbol'].tolist()

        train_data = ProteinGraphDataset(root=fold_root_path,gene_list=train_list,mode="train",ratio=ratio,raw_data_path=raw_data_path,gene_fasta_dict_path=gene_fasta_dict_path,amino_acid_dict=amino_acid_dict,max_len=max_len)
        test_data = ProteinGraphDataset(root=fold_root_path,gene_list=test_list,mode="test",ratio=ratio,raw_data_path=raw_data_path,gene_fasta_dict_path=gene_fasta_dict_path,amino_acid_dict=amino_acid_dict,max_len=max_len)







 
