"""
@author: amber
"""
import torch 
import torch.nn as nn



class fc_layer(nn.Module):
    def __init__(self,input_dim,emb_dim_one,emb_dim_two,n_output,drop_prob):
        super(fc_layer,self).__init__()
        self.fc_g1 = nn.Linear(input_dim,emb_dim_one)
        self.fc_g2 = nn.Linear(emb_dim_one,emb_dim_two)
        self.fc_g3 = nn.Linear(emb_dim_two,n_output)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop_prob)
    
    def forward(self,data):
        x = self.fc_g1(data)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc_g2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        out = self.fc_g3(x)
        return out 


class FGM():
    def __init__(self,model,fc_one_weight,fc_one_bias,fc_two_weight,fc_two_bias):
        self.model = model
        self.backup = {}
        self.fc_one_weight = fc_one_weight
        self.fc_one_bias = fc_one_bias
        self.fc_two_weight = fc_two_weight
        self.fc_two_bias = fc_two_bias
        
        
    def attack(self,epsilon=1.):
        for name, param in self.model.named_parameters():
            if param.requires_grad and (self.fc_one_weight in name) and (self.fc_one_bias in name) and (self.fc_two_weight in name) and (self.fc_two_bias in name):
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)     # norm2 in default
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and (self.fc_one_weight in name) and (self.fc_one_bias in name) and (self.fc_two_weight in name) and (self.fc_two_bias in name):
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
        
