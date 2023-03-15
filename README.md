# Resources:

+ README.md: this file.
+ list all files and packages of this project......
....
....

# Datasets and Data Processing 
The source data was downloaded from the Online GEne Essentiality (OGEE) database. After the data processing pipeline, we ultimately got the following datasets:

## statistics of Caenorhabditis elegans (C.elegans)

Celegans_dataset: 578 ssential genes(positive samples)       
                  13104 non-essnetial genes(negative samples)    

## data processing pipeline 
Taking essential genes of C.elegans as example, 

### SourceData package
s1. source data downloading: download the source data from OGEE, named as OGEE_Celegans_Egenes.xlsx, including "dataset","taxaID","locus"."gene","essentiality","pmid" and "Ref_db"; 

---------> OGEE_Celegans_Egenes.xlsx

### EGeneLocus2Uniprot package
s2. protein mapping: map ensembl id to uniprot id, and to simultaneously generate a "gene card" for one gene which includes "Ensembl", "Entrez", "Full_name","Uniprot_ID" and "Uniprot_Source". All gene cards were stored in the "mapping gene" packages; 
    Running: 
    python EGeneLocus2Uniprot.py

s3. data combination and removing useless titles, and finally got the complete mapped essential genes
    Running: 
    cat WB*.txt > Celegans_Egenes.txt; 
    sed -i '/Ensembl/d' Celegans_Egenes.txt > mapped_Egenes.txt

s4. removing redundant mapping proteins: as for mapped_Egenes.txt, we remove duplicates and retain the first protein that each gene maps, got mapped_EGenes.xlsx 

---------> mapped_EGenes.xlsx

### ExtractingFasta package
s5. preparing the urls for all fastas of proteins, and downloading the fasta sequences according to the urls to obatin the fasta sequence for each uniprot id in the "ProteinFastas" package
    Running:
    python generate_urls.py > fasta_urls.txt
    bash generate_fasta_file.sh 
 
s6: rewrite the fasta sequences in a more compact form and map to mapped_Egenes.txt,  and combine all_fasta_files to
"Celegans_Essential_Genes.xlsx" 
    Running:
    python rewrite_fasta_file.py > all_fasta_seqs.txt
   
(s7: for readers' convenience, we also form a package "GeneFastas" where each fasta was named by gene ensembl.
    Running:
    python gene_fasta_writing.py )

### ProcessedData
---------> Celegans_Essential_Genes.txt && Celegans_Essential_Genes.xlsx 

the same data processing pipeline was also applied on the non-essential genes and other species.  

## Step-by-step running for bioGen 



### 0 Prepare conda enviroment and install Python libraries needed
conda create -n bio python=3.9 
source activate bio 
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv torch_geometric -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
pip install torch-geometric
pip install git+https://github.com/facebookresearch/esm.git
python -c "import torch; model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")"

dependencies: 
    python == 3.9.16 
    torch == 1.13.1+cu116
    torch-geometric == 2.2.0 
    torch-scatter == 2.1.0+pt113cu116
    torch-sparse == 0.6.16+pt113cu116
    torchvision == 0.14.1+cu116
    fair-esm == 2.0.1
    numpy == 1.24.2 
    pandas == 1.5.3
    scikit-learn == 1.2.2 
    scipy == 1.10.1
    networkx == 3.0
    matplotlib == 3.7.1 

### Usage 

s1: setting the species, root_path and trimmed sequence length in config_init.py, generate residue-level features and contact map for each gene(protein) with well-pretrained Evolutionary Scale Modeling(esm) method
    Running: 
    python create_data.py 
    Return: 
    a bunch of gene information dictionaries for every gene: {'gene_ensembl':.., 'feature_representation':...,'cmap':...,'target':...} which are stored in the raw package.

s2: setting the n_splits, pos_samples_path, neg_samples_path and kfold_root_path in config_init.py, split the genes into training set and test set with kfold scheme.
    Running:
    python train_test_splitted_data.py  
    Return(if n_split was set as 10): 
    fold0,fold1,...,fold9 packages where each consist of idenpendent training set and test set. 

s3: setting the ratio of contact map in config_init.py, transform the protein contact map into graph, and prepare the data format to meet the need of torch geometric.
    python protein2graph.py
    Return: 
    train.pt and test.pt for each fold , and those train.pt and test.pt meet the demand of torch geometric.


s4: setting the GNN, GAT or GCN, in config_init.py, then train and test model in the train-validation-test scheme under kfold cross validation
    Running:
    python main.py  










