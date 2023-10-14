import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.nn import Sequential,Linear,ReLU,BatchNorm1d 
from torch_geometric.nn import GINConv,global_add_pool 


class GINConvNet(nn.Module): 
    def __init__(self,input_dim,hidden_dim,n_output,dropout):
        super(GINConvNet,self).__init__()
        self.dropout = nn.Dropout(dropout) 
        self.relu = ReLU()
        self.n_output = n_output
        # GIN Convolutional layers
        
        nn1 = Sequential(Linear(input_dim,hidden_dim[0]),ReLU(),Linear(hidden_dim[0],hidden_dim[0])) 
        self.conv1 = GINConv(nn1)
        self.bn1 = BatchNorm1d(hidden_dim[0]) 
        
        nn2 = Sequential(Linear(hidden_dim[0],hidden_dim[1]),ReLU(),Linear(hidden_dim[1],hidden_dim[1])) 
        self.conv2 = GINConv(nn2)
        self.bn2 = BatchNorm1d(hidden_dim[1]) 
        
        nn3 = Sequential(Linear(hidden_dim[1],hidden_dim[2]),ReLU(),Linear(hidden_dim[2],hidden_dim[2])) 
        self.conv3 = GINConv(nn3)
        self.bn3 = BatchNorm1d(hidden_dim[2]) 
        
        nn4 = Sequential(Linear(hidden_dim[2],hidden_dim[3]),ReLU(),Linear(hidden_dim[3],hidden_dim[3])) 
        self.conv4 = GINConv(nn4)
        self.bn4 = BatchNorm1d(hidden_dim[3]) 
        
        self.fc = Linear(hidden_dim[3],self.n_output)


    def forward(self,data): 
        x, edge_index,batch = data.x, data.edge_index,data.batch   
        x = F.relu(self.conv1(x,edge_index))
        x = self.bn1(x) 
        x = self.dropout(x)

        x = F.relu(self.conv2(x,edge_index))
        x = self.bn2(x) 
        x = self.dropout(x) 

        x = F.relu(self.conv3(x,edge_index))
        x = self.bn3(x) 
        x = self.dropout(x) 
    
        
        x = F.relu(self.conv4(x,edge_index))
        x = self.bn4(x) 
        x = self.dropout(x) 

        x = global_add_pool(x,batch)
        
        out = self.fc(x)   
        return out 
         



