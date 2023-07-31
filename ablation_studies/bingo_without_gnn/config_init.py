"""
@author: amber
"""

import argparse
def get_config():
    parse = argparse.ArgumentParser(description='common train config')

    # parameter setting for data_loader.py 
    parse.add_argument('-raw_data_path', '--raw_data_path_topofallfeature', type=str, nargs='?', default= "/home/amber/Bingo_balanced/data/human/raw/",help="raw data package")
    parse.add_argument('-n_splits', '--n_splits_topofallfeature', type=int, nargs='?', default= 10,help="k fold")
    parse.add_argument('-kfold_root_path', '--kfold_root_path_topofallfeature', type=str, nargs='?', default= "/home/amber/Bingo_balanced/data/human/kfold_splitted_data/",help="k fold data store dir")

    # parameter setting for main.py 
    parse.add_argument('-train_batch_size', '--train_batch_size_topofallfeature', type=int, nargs='?', default=256,help="batch size for train data")
    parse.add_argument('-test_batch_size', '--test_batch_size_topofallfeature', type=int, nargs='?', default=16,help="batch size for test data")
    parse.add_argument('-cuda_name', '--cuda_name_topofallfeature', type=str, nargs='?', default= "cuda:0",help="cuda")  
    parse.add_argument('-drop_prob', '--drop_prob_topofallfeature', type=float, nargs='?', default= 0.5,help="drop out probability") 
    parse.add_argument('-n_output', '--n_output_topofallfeature', type=int, nargs='?', default=1,help="output dimension of the gnn model")
    parse.add_argument('-lr', '--lr_topofallfeature', type=float, nargs='?', default= 1e-5, help="learning rate") 
    parse.add_argument('-num_epoches', '--num_epoches_topofallfeature', type=int, nargs='?', default=80, help="epoch number") 
    parse.add_argument('-weight_decay', '--weight_decay_topofallfeature', type=float, nargs='?', default= 5e-4, help="weight decay for adam optimizer")  
    parse.add_argument('-model_saving_path', '--model_saving_path_topofallfeature', type=str, nargs='?', default= "/home/amber/Bingo_balanced/human_experiments/biogen_without_gnn/kfold_model_saving/", help="path for saving the models") 
    config = parse.parse_args()
    return config
