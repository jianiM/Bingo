"""
@author: amber
"""
import argparse
def get_config():
    parse = argparse.ArgumentParser(description='common train config')
    # parameter setting for create_data.py
    parse.add_argument('-species', '--species_topofallfeature', type=str, nargs='?', default="human",help="setting the species: Celegans or DM")
    parse.add_argument('-root_path', '--root_path_topofallfeature', type=str, nargs='?', default="/home/amber/Bingo_balanced/human_experiments/biogen_without_esm/kfold_splitted_data/",help="root dataset path")

    # parameter setting for train-test-split.py
    parse.add_argument('-n_splits', '--n_splits_topofallfeature', type=int, nargs='?', default= 10,help="k fold")
    parse.add_argument('-pos_samples_path', '--pos_samples_path_topofallfeature', type=str, nargs='?', default= "/home/amber/Bingo_balanced/data/human/orig_sample_list/HEGP2_EGenes.xlsx",help="essential gene path")
    parse.add_argument('-neg_samples_path', '--neg_samples_path_topofallfeature', type=str, nargs='?', default= "/home/amber/Bingo_balanced/data/human/orig_sample_list/sampled_nonessential_genes.xlsx",help="non-essential gene path")
    parse.add_argument('-kfold_root_path', '--kfold_root_path_topofallfeature', type=str, nargs='?', default= "/home/amber/Bingo_balanced/human_experiments/biogen_without_esm/kfold_splitted_data/",help="k fold data store dir")
    
    # parameter setting for protein2graph.py
    parse.add_argument('-raw_data_path', '--raw_data_path_topofallfeature', type=str, nargs='?', default= "/home/amber/Bingo_balanced/data/human/raw/",help="raw data package")
    parse.add_argument('-gene_fasta_dict_path', '--gene_fasta_dict_path_topofallfeature', type=str, nargs='?', default= "/home/amber/Bingo_balanced/data/human/orig_sample_list/gene_fasta_dict.pkl",help="gene fasta dictionary mapping")
    parse.add_argument('-ratio', '--ratio_topofallfeature', type=float, nargs='?', default=0.2,help="ratio of cmap to generate graph")
    parse.add_argument('-max_len', '--max_len_topofallfeature', type=int, nargs='?', default=1000,help="max length for trimming the protein sequence")
    
    # parameter setting for main.py 
    parse.add_argument('-train_batch_size', '--train_batch_size_topofallfeature', type=int, nargs='?', default=16,help="batch size for train data")
    parse.add_argument('-test_batch_size', '--test_batch_size_topofallfeature', type=int, nargs='?', default=8,help="batch size for test data")
    parse.add_argument('-cuda_name', '--cuda_name_topofallfeature', type=str, nargs='?', default= "cuda:0",help="cuda")  
    parse.add_argument('-drop_prob', '--drop_prob_topofallfeature', type=float, nargs='?', default= 0.3,help="drop out probability") 
    parse.add_argument('-n_output', '--n_output_topofallfeature', type=int, nargs='?', default=1,help="output dimension of the gnn model")
    parse.add_argument('-lr', '--lr_topofallfeature', type=float, nargs='?', default= 1e-7, help="learning rate") 
    parse.add_argument('-num_epoches', '--num_epoches_topofallfeature', type=int, nargs='?', default=80, help="epoch number") 
    parse.add_argument('-weight_decay', '--weight_decay_topofallfeature', type=float, nargs='?', default= 5e-4, help="weight decay for adam optimizer") 
    parse.add_argument('-modelling', '--modelling_topofallfeature', type=str, nargs='?', default= "gat", help="gnn model choice") 
    parse.add_argument('-model_saving_path', '--model_saving_path_topofallfeature', type=str, nargs='?', default= "/home/amber/Bingo_balanced/human_experiments/biogen_without_esm/kfold_model_saving/", help="path for saving the models") 
    config = parse.parse_args()
    return config
