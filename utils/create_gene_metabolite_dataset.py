import torch
from torch.utils.data import Dataset
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

def create_gene_metabolite_dataset(metabolites, relationships, gene_sequences, smiles_features, filtered_genes, graph_data, device):
    """
    Function to create a dataset for gene sequences, SMILES features, and graph data.
    Data is already tokenized like the source code.

    Parameters:
    - metabolites (list): List of metabolite names
    - relationships (dict): Dictionary mapping metabolites to (gene_names, labels)
    - gene_sequences (numpy array): Already tokenized and padded gene sequences
    - smiles_features (numpy array): Already tokenized and padded SMILES sequences
    - filtered_genes (list): List of filtered gene names
    - graph_data: PyTorch Geometric Data object
    - device: Device to place tensors on

    Returns:
    - dataset (Dataset): A PyTorch Dataset containing gene sequences, SMILES features, labels, and graph data
    """

    class GraphGdelDataset(Dataset):
        def __init__(self, metabolites, relationships, gene_sequences, smiles_features, filtered_genes, graph_data, device):
            self.metabolites = metabolites
            self.relationships = relationships
            self.gene_sequences = gene_sequences
            self.smiles_features = smiles_features
            self.filtered_genes = filtered_genes
            self.graph_data = graph_data
            self.device = device
            
            # Create data samples
            self.samples = []
            for metabolite in metabolites:
                if metabolite in relationships:
                    gene_names, labels = relationships[metabolite]
                    for gene_name, label in zip(gene_names, labels):
                        if gene_name in filtered_genes:
                            self.samples.append((metabolite, gene_name, label))

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            metabolite, gene_name, label = self.samples[idx]
            
            # Get gene sequence (use first available sequence for now)
            gene_seq = self.gene_sequences[0] if len(self.gene_sequences) > 0 else np.zeros(100)
            
            # Get SMILES sequence (use first available sequence for now)
            smiles_seq = self.smiles_features[0] if len(self.smiles_features) > 0 else np.zeros(100)
            
            # Create node index (simplified - you may need to adjust based on your graph structure)
            node_index = hash(metabolite) % self.graph_data.num_nodes
            
            return (torch.tensor(gene_seq, dtype=torch.long),
                    torch.tensor(smiles_seq, dtype=torch.long),
                    torch.tensor(label, dtype=torch.float),
                    torch.tensor(node_index, dtype=torch.long),
                    self.graph_data)

    # Return the dataset
    return GraphGdelDataset(metabolites, relationships, gene_sequences, smiles_features, filtered_genes, graph_data, device)
