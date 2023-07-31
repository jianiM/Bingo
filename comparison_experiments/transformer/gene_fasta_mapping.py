
import torch 
import torch.nn as nn
import os 
from utils import load_pt
import pandas as pd 



gene_list_path  = "/home/amber/Bingo_balanced/data/human/orig_sample_list/gene_list.txt"
gene_info = pd.read_csv(gene_list_path,sep='\t')
gene_ensembls = gene_info['Ensembl']
gene_fasta = gene_info['Fasta']
gene_target = gene_info['Target']
df = pd.concat([gene_ensembls,gene_fasta,gene_target],axis=1)
ensembl_fasta_mapping = df.set_index('Ensembl').T.to_dict('list')
torch.save(ensembl_fasta_mapping,'/home/amber/Bingo_balanced/data/human/orig_sample_list/gene_fasta_dict.pkl')

#data = torch.load('/home/amber/BioGen/experiments/transformer/gene_fasta_dict.pkl')
#print(len(data))





