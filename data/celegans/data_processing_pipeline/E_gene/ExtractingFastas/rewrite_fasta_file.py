# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 16:46:16 2023

@author: amber
"""

import os 
import re 
import pandas as pd 
import numpy as np 


pattern_title = re.compile(r'^>.*', re.M)
pattern_n = re.compile(r'\n')

root_path = "/home/amber/Species_Datasets/Celegans/E_gene/Ensembl2Uniprot/"
suffix_protein_path = "mapped_EGenes.xlsx" 
protein_path = os.path.join(root_path,suffix_protein_path)
protein_list = list(pd.read_excel(protein_path)["Uniprot_ID"])


for i in range(len(protein_list)):
    uid = protein_list[i]
    suffix_fasta_name = "mapping_genes/"+uid + ".fasta"
    fasta_path = os.path.join(root_path,suffix_fasta_name)
    with open(fasta_path,'r',encoding='utf-8') as f: 
        text = f.read()
    data = re.split(pattern_title, text)
    data2 = [re.sub(pattern_n, '', i) for i in data]
    fasta_sequence = data2[1]
    print(fasta_sequence)






