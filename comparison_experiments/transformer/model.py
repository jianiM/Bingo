# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 11:09:35 2022
@author: Jiani Ma
"""
import torch 
import torch.nn as nn
import math 
torch.manual_seed(1223)
import re 

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform_(m.weight.data)

class TokenEmbedding(nn.Embedding):
    """
    Token Embedding using torch.nn
    they will dense representation of word using weighted matrix
    """
    def __init__(self, vocab_size, d_model):
        """
        class for token embedding that included positional information
        :param vocab_size: size of vocabulary
        :param d_model: dimensions of model
        """
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)

class PostionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """
    def __init__(self, d_model, max_len, device):
        """
        constructor of sinusoid encoding class
        :param d_model: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        """
        super(PostionalEncoding, self).__init__()
        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False  # we don't need to compute gradient
        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position
        _2i = torch.arange(0, d_model, step=2, device=device).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # compute positional encoding to consider positional information of words
    def forward(self, x):
        batch_size, seq_len = x.size()
        return self.encoding[:seq_len, :]
        # it will add with tok_emb : [128, 30, 512]
class TransformerEmbedding(nn.Module):
    """
    token embedding + positional encoding (sinusoid)
    positional encoding can give positional information to network
    """
    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        """
        class for word embedding that included positional information
        :param vocab_size: size of vocabulary
        :param d_model: dimensions of model
        """
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PostionalEncoding(d_model, max_len, device)
        self.drop_out = nn.Dropout(p=drop_prob)
    def forward(self, x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        return self.drop_out(tok_emb + pos_emb)
class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention
    """
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, q, k, v, mask=None, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()
        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(-1, -2)  # transpose
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product
        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)
        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)
        # 4. multiply with Value
        v = score @ v
        return v, score
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)
    def forward(self, input_q, input_k, input_v, mask=None):
        # 1. dot product with weight matrices
        # input_q, input_k, input_v: [batch_size, sequence_length, d_model]
        #residue = input_q
        q, k, v = self.w_q(input_q), self.w_k(input_k), self.w_v(input_v)
        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)
        # 3. do scale dot product to compute similarity
        out, attention = self.attention(q, k, v, mask=mask)
        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out)       
        #out = nn.LayerNorm(self.d_model)(out + residue)
        return out
    def split(self, tensor):
        """
        split tensor by number of head
        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()
        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)
        return tensor
    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)
        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor
        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.3):
        super(PositionwiseFeedForward, self).__init__()
        self.d_model = d_model
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)
    def forward(self, x):
        #residue = x 
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        out = self.linear2(x)        
        #out = nn.LayerNorm(self.d_model)(out + residue)
        return out
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        # '-1' means last dimension. 
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out
class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)
    def forward(self, x, s_mask):
        # 1. compute self attention
        _x = x
        x = self.attention(x, x, x, s_mask)
        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)
        # 3. positionwise feed forward network
        _x = x
        x = self.ffn(x)
        # 4. add and norm
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x
def make_pad_mask(q, k,src_pad_idx):
    len_q, len_k = q.size(1), k.size(1)
    # batch_size x 1 x 1 x len_k
    k = k.ne(src_pad_idx).unsqueeze(1).unsqueeze(2)
    # batch_size x 1 x len_q x len_k
    k = k.repeat(1, 1, len_q, 1)
    # batch_size x 1 x len_q x 1
    q = q.ne(src_pad_idx).unsqueeze(1).unsqueeze(3)
    # batch_size x 1 x len_q x len_k
    q = q.repeat(1, 1, 1, len_k)
    mask = k & q
    return mask
class Encoder(nn.Module):
    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        self.emb = TransformerEmbedding(d_model=d_model,
                                        max_len=max_len,
                                        vocab_size=enc_voc_size,
                                        drop_prob=drop_prob,
                                        device=device)
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])
        self.fc1 = nn.Sequential(nn.Linear(in_features = max_len*d_model, out_features=32),nn.ReLU())
        self.fc3 = nn.Linear(32,1)  
        self.dropout = nn.Dropout(p=drop_prob)
    
    def forward(self,src,s_mask):        
        x = self.emb(src)
        for layer in self.layers:
            x = layer(x, s_mask)                      
        out = x.contiguous().view(src.size(0),-1)    #[batch_size,max_len*d_model]
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out
   