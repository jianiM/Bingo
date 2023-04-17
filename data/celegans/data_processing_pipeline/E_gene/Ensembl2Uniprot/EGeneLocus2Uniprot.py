# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 17:32:33 2023

@author: amber
"""
import os 
import mygene 
import pandas as pd 

def Typeretain(inputt):
    if type(inputt) == str: 
        li = [] 
        li.append(inputt)
    else: 
        li = inputt 
    return li     

def uniprot_info(uniprot_dict):   
    keys = list(uniprot_dict.keys()) 
    if keys == ['Swiss-Prot']: 
        protein_val = Typeretain(uniprot_dict['Swiss-Prot'])
        row_num = len(protein_val)
        source_li = ['Swiss-Prot'] * row_num
    elif keys == ['TrEMBL']:
        protein_val = Typeretain(uniprot_dict['TrEMBL']) 
        row_num = len(protein_val)
        source_li = ['TrEMBL'] * row_num
    elif keys == ['Swiss-Prot','TrEMBL']: 
        swiss_prot_val = Typeretain(uniprot_dict['Swiss-Prot']) 
        swiss_len = len(swiss_prot_val) 
        trEMBL_prot_val = Typeretain(uniprot_dict['TrEMBL']) 
        trEMBL_len = len(trEMBL_prot_val) 
        protein_val = swiss_prot_val + trEMBL_prot_val
        row_num = len(protein_val)
        source_li = ['Swiss-Prot'] * swiss_len + ['TrEMBL'] * trEMBL_len
    return protein_val, source_li, row_num 
        
def generate_list(str_object,row_num):
    li = []
    li.append(str_object)
    final_li = li * row_num
    return final_li    

def generate_gene_card(obj,species,rootpath):
    data = mg.getgenes(ids = obj, fields = ["name","uniprot"],species = species,returnall=True)[0] 
    data_key = list(data.keys())
    if ('notfound' in data_key):
        print(obj + ' ' + "not Exists")
    elif ('uniprot' not in data_key):
        print('There is no uniprot ID')
    else:      
        entrez_id = data['_id'] 
        full_name = data['name'] 
        uniprot_dict = data['uniprot'] 
        uniprot_id_li, source_li, row_num  = uniprot_info(uniprot_dict)    
        esembl_li = Typeretain(obj) * row_num 
        entrez_li = Typeretain(entrez_id) * row_num 
        full_name_li = Typeretain(full_name) * row_num  
        gene_uniprot_card = pd.DataFrame()
        gene_uniprot_card['Ensembl'] = esembl_li 
        gene_uniprot_card['Entrez'] = entrez_li
        gene_uniprot_card['Full_name'] = full_name_li 
        gene_uniprot_card['Uniprot_ID'] = uniprot_id_li  
        gene_uniprot_card['Uniprot_Source'] = source_li
        suffix_path = obj + '.txt'
        file_path = os.path.join(rootpath,suffix_path)
        gene_uniprot_card.to_csv(file_path,sep='\t')  


if __name__ == "__main__":
    mg = mygene.MyGeneInfo()
    species = "6239"
    rootpath = "/home/amber/EGP/Celegans/E_gene/Ensembl2Uniprot/mapping_genes"
    raw_file_path = "/home/amber/EGP/Celegans/E_gene/SouceData/OGEE_Celegans_EGenes.xlsx" 
    gene_data = pd.read_excel(raw_file_path)
    all_esembls = list(gene_data['locus']) 
    for obj in all_esembls: 
        generate_gene_card(obj,species,rootpath)    
        
        
        
        
        
        
        
        
        
        
        
        
    

    
    
    
    

