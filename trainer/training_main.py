import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import csv
import random
import time

from utils.read_gene_sequences import read_gene_sequences
from utils.read_metabolite_smiles import read_metabolite_smiles
from utils.read_gene_metabolite_relationships import read_gene_metabolite_relationships
from utils.create_gene_metabolite_dataset import create_gene_metabolite_dataset
from utils.evaluations import *
from model.initialize_model import initialize_model 
from trainer.trainer import train_model
from trainer.save_model_with_confirmation import save_model_with_confirmation
###################################################

#CBM: e_coli_core OR iMM904 OR iML1515
CBM = 'iML1515'

# Set the bound for fixed_gene exclusion
bound = 0.9
test_data_size = 0.9

# Hyperparameters
vocab_embedding_dim = 32  #The size of the embedding vector for each token word
lstm_hidden_dim = 64      #The number of units in the LSTM hidden layer
batch_size = 8
epochs = 100

# Set the random seed for reproducibility
random_seed = 1001
random.seed(random_seed)           # For Python's built-in random module
np.random.seed(random_seed)        # For numpy-based randomness (e.g., train_test_split)

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
# Define paths relative to the script's location
gene_folder_path = os.path.join(script_dir, 'Data', CBM, 'Gene_AA')
metabolite_smiles_path = os.path.join(script_dir, 'Data', CBM, 'Target metabolites_smile.csv')
relationship_folder_path = os.path.join(script_dir, 'Data', CBM, 'Label')
output_csv_path = os.path.join(script_dir, 'Data', CBM, 'Results', 'all_metabolites_predictions_temp.csv')
model_save_path = os.path.join(script_dir, 'train_disc', f'DeepGdel_{CBM}_trained.sav')
###################################################

# Reading data
gene_names, gene_sequences = read_gene_sequences(gene_folder_path)
metabolite_names, smiles_features = read_metabolite_smiles(metabolite_smiles_path)
relationships = read_gene_metabolite_relationships(relationship_folder_path)

# Extract unique metabolites
unique_metabolites = list(relationships.keys())

# Split metabolites into train and validation sets
train_metabolites, val_metabolites = train_test_split(unique_metabolites, test_size=test_data_size, random_state=None)

# Print which metabolites are used for training and validation
print("Metabolites used for training:")
print(train_metabolites)
print("\nMetabolites used for validation:")
print(val_metabolites)

# Initialize counters for gene relationships (for all data)
gene_count_all = {}
gene_total_all = {}

# Calculate gene deletion statistics for all data (train + val)
for meta_name in unique_metabolites:  # Loop through all metabolites, not just train
    gene_names, labels = relationships[meta_name]
    for gene_name, label in zip(gene_names, labels):
        if gene_name not in gene_count_all:
            gene_count_all[gene_name] = {'0': 0, '1': 0}
            gene_total_all[gene_name] = 0
        gene_count_all[gene_name][str(label)] += 1
        gene_total_all[gene_name] += 1

# Calculate percentages for each gene
gene_percentages = {}
genes_close_to_100_percent_0 = []
genes_close_to_100_percent_1 = []
genes_100_percent_0 = []
genes_100_percent_1 = []
genes_to_exclude = []

for gene in gene_count_all:
    total = gene_total_all[gene]
    percentage_0 = gene_count_all[gene]['0'] / total if total > 0 else 0.0
    percentage_1 = gene_count_all[gene]['1'] / total if total > 0 else 0.0
    
    gene_percentages[gene] = {
        '0': percentage_0,
        '1': percentage_1
    }
    
    if percentage_0 == 1.0:
        genes_100_percent_0.append(gene)
        genes_to_exclude.append(gene)
    elif percentage_0 >= bound:
        genes_close_to_100_percent_0.append(gene)
        genes_to_exclude.append(gene)
        
    if percentage_1 == 1.0:
        genes_100_percent_1.append(gene)
        genes_to_exclude.append(gene)
    elif percentage_1 >= bound:
        genes_close_to_100_percent_1.append(gene)
        genes_to_exclude.append(gene)

# Print genes to exclude
print(f"\nGenes to exclude during ACU calculation: {len(genes_to_exclude)}")
print(f"Number of genes with 100% deletion (0): {len(genes_100_percent_0)}")
print(f"Number of genes with nearly 100% deletion (0): {len(genes_close_to_100_percent_0)}")
print(f"Number of genes with 100% non-deletion (1): {len(genes_100_percent_1)}")
print(f"Number of genes with nearly 100% non-deletion (1): {len(genes_close_to_100_percent_1)}")

# Filter data based on metabolite split for training
X_genes_train = []
X_smiles_train = []
y_train = []

for meta_name in train_metabolites:
    gene_names_train, labels_train = relationships[meta_name]
    gene_names_train_list = gene_names_train.tolist()  # Convert to list
    for gene_name, label in zip(gene_names_train, labels_train):
        gene_index = gene_names_train_list.index(gene_name)
        meta_index = np.where(metabolite_names == meta_name)[0][0] # Get index from numpy array
        X_genes_train.append(gene_sequences[gene_index])
        X_smiles_train.append(smiles_features[meta_index])
        y_train.append(label)

# Convert to numpy arrays if necessary
X_genes_train = np.array(X_genes_train)
X_smiles_train = np.array(X_smiles_train)
y_train = np.array(y_train)

# Filter data based on metabolite split for validation
X_genes_val = []
X_smiles_val = []
y_val = []

for meta_name in val_metabolites:
    gene_names_val, labels_val = relationships[meta_name]
    gene_names_val_list = gene_names_val.tolist()  # Convert to list
    for gene_name, label in zip(gene_names_val, labels_val):
        gene_index = gene_names_val_list.index(gene_name)
        meta_index = np.where(metabolite_names == meta_name)[0][0]  # Get index from numpy array
        X_genes_val.append(gene_sequences[gene_index])
        X_smiles_val.append(smiles_features[meta_index])
        y_val.append(label)

# Convert to numpy arrays if necessary
X_genes_val = np.array(X_genes_val)
X_smiles_val = np.array(X_smiles_val)
y_val = np.array(y_val)

# Creating DataLoader
train_dataset = create_gene_metabolite_dataset(X_genes_train, X_smiles_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_dataset = create_gene_metabolite_dataset(X_genes_val, X_smiles_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
###################################################

# Tokenizing/padding gene sequences
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(gene_sequences)
gene_vocab_size = len(tokenizer.word_index) + 1   #Size of the vocabulary
max_len = max(len(seq) for seq in gene_sequences)
gene_sequences = pad_sequences(tokenizer.texts_to_sequences(gene_sequences), maxlen=max_len)

# Tokenizing/padding meta sequences
smiles_tokenizer = Tokenizer(char_level=True)
smiles_tokenizer.fit_on_texts(smiles_features)
smiles_vocab_size = len(smiles_tokenizer.word_index) + 1   #Size of the vocabulary
smiles_max_len = max(len(smiles) for smiles in smiles_features)
smiles_features = pad_sequences(smiles_tokenizer.texts_to_sequences(smiles_features), maxlen=smiles_max_len)

# Initialize the model
model, device = initialize_model(gene_vocab_size, smiles_vocab_size, vocab_embedding_dim, lstm_hidden_dim, freeze_gene=False, freeze_smiles=False)
# Load the model weights from the current directory
model.load_state_dict(torch.load(f'train_disc/DeepGdel_{CBM}.sav',map_location=torch.device(device)))
model.to(device)
print("Model loaded!")
# Print the model architecture
print("\nModel Architecture:\n")
print(model)

# Model tuning and training
trained_model, train_losses, val_losses, val_accuracies = train_model(
        model, train_loader, val_loader, epochs=epochs, lr=0.0001, device=device
    )

#Save trained model disc
save_model_with_confirmation(model_save_path)      
###################################################

#Evaluation metrics report
calculate_metrics_for_val_metabolites(trained_model, val_metabolites, metabolite_names, gene_sequences, smiles_features, relationships, device, genes_to_exclude)