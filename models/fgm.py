"""
@author: amber
"""
import torch 


class FGM_GCN():
    def __init__(self,model,bias_one="gcnconv1.bias",weight_one="gcnconv1.lin.weight",bias_two="gcnconv2.bias",weight_two="gcnconv2.lin.weight"):
        self.model = model
        self.backup = {}
        self.bias_one = bias_one
        self.weight_one = weight_one
        self.bias_two = bias_two
        self.weight_two = weight_two
        
    def attack(self,epsilon=1.):
        for name, param in self.model.named_parameters():
            if param.requires_grad and (self.bias_one in name) and (self.weight_one in name) and (self.bias_two in name) and (self.weight_two in name):
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)     # norm2 in default
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and (self.bias_one in name) and (self.weight_one in name) and (self.bias_two in name) and (self.weight_two in name):
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
        
        
class FGM_GAT():
    def __init__(self,model,bias_one="gcn1.bias",weight_one="gcn1.lin_src.weight",bias_two="gcn2.bias",weight_two="gcn2.lin_src.weight"):
        self.model = model
        self.backup = {}
        self.bias_one = bias_one
        self.weight_one = weight_one
        self.bias_two = bias_two
        self.weight_two = weight_two
        
    def attack(self,epsilon=1.):
        for name, param in self.model.named_parameters():
            if param.requires_grad and (self.bias_one in name) and (self.weight_one in name) and (self.bias_two in name) and (self.weight_two in name):
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)     # norm2 in default
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and (self.bias_one in name) and (self.weight_one in name) and (self.bias_two in name) and (self.weight_two in name):
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}