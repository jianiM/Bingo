# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 10:03:42 2022
@author: Jiani Ma
"""
#import wget
import pandas as pd 
from urllib.parse import urljoin

root_url = "http://www.uniprot.org/uniprot/" 
suffix = ".fasta"


file_name_path = "/home/amber/Species_Datasets/Celegans/NE_gene/Ensembl2Uniprot/mapped_NEGenes.xlsx"
file = pd.read_excel(file_name_path)
uniprot_ids = list(file['Uniprot_ID'])
uniprot_ids = [uid + suffix for uid in uniprot_ids]

for uid in uniprot_ids: 
    url = urljoin(root_url,uid)
    print(url)
