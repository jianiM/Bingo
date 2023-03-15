"""
@author: amber
"""
#import torch
#model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")

import os 
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import torch
import esm
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd 
from utils import *
mpl.use('Agg')
from config_init import get_config

"""
generate gene dict for each gene, with the format like:
gene_dict: {'gene_ensembl:'str,'feature_representation:'tensor,'cmap:'tensor,'target：',target}
root_path: dir for saving those pt files 
data_path: data list we want to transform
"""

def trimming_fasta(fasta_seqs,trimmed_thresh):
    trimmed_fastas = []
    for i in range(len(fasta_seqs)):
        fasta = fasta_seqs[i]
        seq_len = len(fasta)
        if seq_len > trimmed_thresh: 
            trimmed_fasta = fasta[0:trimmed_thresh]
        else: 
            trimmed_fasta = fasta 
        trimmed_fastas.append(trimmed_fasta)    
    return trimmed_fastas  


if __name__ == "__main__": 
    config = get_config()
    #get the parameters 
    species = config.species_topofallfeature
    root_path = config.root_path_topofallfeature  
    trimmed_thresh = config.trim_thresh_topofallfeature
    #get the directory and path
    raw_path = os.path.join(root_path,species,'raw')
    gene_list_path = os.path.join(root_path,species,"orig_sample_list/gene_list.txt")
    
    orig_data = pd.read_csv(gene_list_path,sep='\t')
    geneEnsembls = list(orig_data['Ensembl'].values)
    fasta_seqs = orig_data['Fasta'].values
    targets = list(orig_data['Target'].values)
    trimmed_fastas = trimming_fasta(fasta_seqs,trimmed_thresh)
    print("There exists {} samples".format(len(geneEnsembls))) 
    
    esm_data_packages = list(zip(geneEnsembls,trimmed_fastas))
    # Load ESM-2 model
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()                  # disables dropout for deterministic results
    for i in range(len(esm_data_packages)):        
        item = esm_data_packages[i]
        data = [] 
        data.append(item)
        gene_dict = {}
        gene, fasta, token = batch_converter(data)   # gene: list; fasta:list, token:tensor
        with torch.no_grad():
            results = model(token, repr_layers=[33], return_contacts=True)
        representations = results['representations'][33].squeeze(0)[1:-1,:]
        contact_map = results['contacts'].squeeze(0)   
        # feed those items into the gene dict
        gene_dict['gene_ensembl'] = gene[0] 
        gene_dict['feature_representation'] = representations
        gene_dict['cmap'] = contact_map 
        gene_dict['target'] = targets[i]
        file_name = gene[0] + ".pt"
        file_path = os.path.join(raw_path,file_name) 
        torch.save(gene_dict,file_path)
        print('Transforming---'+ gene[0] + '---is over!')
    
