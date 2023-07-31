# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 16:15:47 2023
@author: amber
"""

import argparse
def get_config():
    parse = argparse.ArgumentParser(description='common train config')
    # hyperparameters for overall setting
    parse.add_argument('-species', '--species_topofallfeature', type=str, nargs='?', default="human",help="setting the species: Celegans or DM")
    parse.add_argument('-n_splits', '--n_splits_topofallfeature', type=int, nargs='?', default= 10,help="k fold")
    parse.add_argument('-kfold_root_path', '--kfold_root_path_topofallfeature', type=str, nargs='?', default= "/home/amber/Bingo_balanced/data/human/kfold_splitted_data/",help="k fold data store dir")

    # hyperparameters for transformer model 
    parse.add_argument('-max_len', '--max_len_topofallfeature', type=int, nargs='?', default=1000,help='trimming length of the fasta sequences.')
    parse.add_argument('-d_model', '--d_model_topofallfeature', type=int, nargs='?', default=128,help='embedding size of the transformer')
    parse.add_argument('-ffn_hidden', '--ffn_hidden_topofallfeature', type=int, nargs='?', default=128,help='size of the hidden layer of the transformer')
    parse.add_argument('-n_head', '--n_head_topofallfeature', type=int, nargs='?', default=2,help='head of the transformer')   # d_model//n_head
    parse.add_argument('-n_layers', '--n_layers_topofallfeature', type=int, nargs='?', default=2,help='layer of the transformer')
    parse.add_argument('-lr', '--lr_topofallfeature', type=float, nargs='?', default=1e-5,help='learning rate')
    parse.add_argument('-drop_prob', '--drop_prob_topofallfeature', type=float, nargs='?', default=0.7,help='drop rate')
    # parameter setting for main.py 
    
    parse.add_argument('-train_batch_size', '--train_batch_size_topofallfeature', type=int, nargs='?', default=64,help="batch size for train data")
    parse.add_argument('-test_batch_size', '--test_batch_size_topofallfeature', type=int, nargs='?', default=64,help="batch size for test data")
    parse.add_argument('-cuda_name', '--cuda_name_topofallfeature', type=str, nargs='?', default= "cuda:0",help="cuda")  
    parse.add_argument('-n_output', '--n_output_topofallfeature', type=int, nargs='?', default=1,help="output dimension of the gnn model")
    parse.add_argument('-num_epoches', '--num_epoches_topofallfeature', type=int, nargs='?', default=100,help="epoch number") 
    parse.add_argument('-weight_decay', '--weight_decay_topofallfeature', type=float, nargs='?', default= 5e-4, help="weight decay for adam optimizer") 
    parse.add_argument('-model_saving_path', '--model_saving_path_topofallfeature', type=str, nargs='?', default= "/home/amber/Bingo_balanced/human_experiments/transformer/kfold_model_saving/", help="path for saving the models") 
    parse.add_argument('-src_pad_idx', '--src_pad_idx_topofallfeature', type=int, nargs='?', default=1, help="padding index") 
    parse.add_argument('-amino_dict_path', '--amino_dict_path_topofallfeature', type=str, nargs='?', default= "/home/amber/Bingo_balanced/data/human/orig_sample_list/amino_acid_dic.pkl", help="path for amino acid dictionary") 
    parse.add_argument('-mapping_path', '--mapping_path_topofallfeature', type=str, nargs='?', default= "/home/amber/Bingo_balanced/data/human/orig_sample_list/gene_fasta_dict.pkl", help="path for ensembl fasta mapping") 
    config = parse.parse_args()
    return config




    
    
