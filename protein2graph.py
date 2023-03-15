"""
@author: amber
"""
import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1"
import pandas as pd 
import torch 
from torch_geometric.data import Dataset, Data,InMemoryDataset
from utils import * 
from config_init import get_config


class ProteinGraphDataset(InMemoryDataset):
    def __init__(self,root='/tmp',transform=None,pre_transform=None,gene_list=None,mode=None,ratio=None,raw_data_path=None):
        super(ProteinGraphDataset,self).__init__(root,transform,pre_transform)
        self.mode = mode
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(gene_list,ratio,raw_data_path)
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
   
    def process(self,gene_list,ratio,raw_data_path):
        data_list = [] 
        data_len = len(gene_list)
        for i in range(data_len):
            print('Converting gene to graph: {}/{}'.format(i+1, data_len))
            g_item = gene_list[i]
            node_features, edge_index,target= self._get_geometric_input(g_item,ratio,raw_data_path)
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

    def _get_geometric_input(self,g_item,ratio,raw_data_path):
        g_path = g_item + '.pt'
        raw_gene_path = os.path.join(raw_data_path,g_path)  
        raw_data = load_pt(raw_gene_path)
        all_features = raw_data['feature_representation']
        contact_map = raw_data['cmap']
        target = raw_data['target']
        node_features, edge_index = cmap2graph(all_features,contact_map,ratio=ratio)
        return node_features, edge_index, target
       
                
if __name__ == "__main__":
    config = get_config()
    n_splits = config.n_splits_topofallfeature
    root_path = config.kfold_root_path_topofallfeature            #/home/amber/BioESMGNN_v2/data/celegans/kfold_splitted_data/
    raw_data_path = config.raw_data_path_topofallfeature          #/home/amber/BioESMGNN_v2/data/celegans/raw/
    ratio = config.ratio_topofallfeature 
    
    for i in range(n_splits):
        print('fold:',i)
        fold_root_path = os.path.join(root_path,'fold'+str(i))
        train_data_path = os.path.join(fold_root_path,'train_data.txt')
        test_data_path = os.path.join(fold_root_path,'test_data.txt')

        train_list = pd.read_csv(train_data_path,sep='\t',index_col=False)['Ensembl'].tolist()
        test_list = pd.read_csv(test_data_path,sep='\t',index_col=False)['Ensembl'].tolist()

        train_data = ProteinGraphDataset(root=fold_root_path,gene_list=train_list,mode="train",ratio=ratio,raw_data_path=raw_data_path)
        test_data = ProteinGraphDataset(root=fold_root_path,gene_list=test_list,mode="test",ratio=ratio,raw_data_path=raw_data_path)






 
