import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool as gmp

class GCNNet(nn.Module):
    def __init__(self,esm_embeds,drop_prob,n_output):
        super(GCNNet, self).__init__()
        self.n_output = n_output
        self.gcnconv1 = GCNConv(in_channels=esm_embeds,out_channels=esm_embeds)
        self.gcnconv2 = GCNConv(in_channels=esm_embeds,out_channels=esm_embeds)
        self.gcnconv3 = GCNConv(in_channels=esm_embeds,out_channels=esm_embeds)
        
        self.fc1 = nn.Linear(in_features=esm_embeds,out_features=256)
        self.fc2 = nn.Linear(in_features=256,out_features=16)
        self.fc3 = nn.Linear(in_features=16,out_features=n_output)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop_prob)
        
    def forward(self,data):
        x, edge_index,batch = data.x, data.edge_index,data.batch
        
        x = self.gcnconv1(x,edge_index)
        x = self.relu(x)

        x = self.gcnconv2(x,edge_index)
        x = self.relu(x) 
        
        x = self.gcnconv3(x,edge_index)
        x = self.relu(x) 
        
        x = gmp(x, batch) 
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)

        out = self.fc3(x)
        return out 





        

