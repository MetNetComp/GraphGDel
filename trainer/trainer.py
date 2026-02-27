import torch
import torch.optim as optim
import time
from torch.utils.data import DataLoader

from model import initialize_model
from utils import read_gene_sequences, read_metabolite_smiles, read_gene_metabolite_relationships, create_gene_metabolite_dataset

def train_model(model, train_loader, val_loader, epochs=10, lr=0.001, device='cpu'):
    """
    Train the model with provided data loaders and parameters.

    Parameters:
    - model (nn.Module): The model to train.
    - train_loader (DataLoader): DataLoader for the training set.
    - val_loader (DataLoader): DataLoader for the validation set.
    - epochs (int): Number of training epochs.
    - lr (float): Learning rate for the optimizer.
    - device (str or torch.device): The device for training (cpu or cuda).
    """

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Loss functions
    reconstruction_loss_fn = torch.nn.CrossEntropyLoss()  # For categorical reconstruction tasks
    criterion = torch.nn.BCELoss()  # For binary classification task

    # Lists to store metrics
    train_losses = []
    val_losses = []
    val_accuracies = []

    # Record the total training time
    total_start_time = time.time()

    # Training loop
    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0.0

        for gene_seq, smiles_feat, labels in train_loader:
            gene_seq, smiles_feat, labels = gene_seq.to(device), smiles_feat.to(device), labels.to(device, dtype=torch.float)

            optimizer.zero_grad()

            # Forward pass
            output, reconstructed_gene, reconstructed_smiles = model(gene_seq, smiles_feat)
            
            # Classification loss 
            output = output.squeeze(-1) 
            loss = criterion(output, labels)

            # Reconstruction losses
            gene_reconstruction_loss = reconstruction_loss_fn(reconstructed_gene.float().view(-1), gene_seq.view(-1))
            smiles_reconstruction_loss = reconstruction_loss_fn(reconstructed_smiles.float().view(-1), smiles_feat.view(-1))

            # Combine losses
            total_loss = loss + gene_reconstruction_loss + smiles_reconstruction_loss

            # Backward pass
            total_loss.backward()
            optimizer.step()

            # Update training loss
            train_loss += total_loss.item() * gene_seq.size(0)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation loop
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for gene_seq, smiles_feat, labels in val_loader:
                gene_seq, smiles_feat, labels = gene_seq.to(device), smiles_feat.to(device), labels.to(device, dtype=torch.float)

                # Forward pass
                output, reconstructed_gene, reconstructed_smiles = model(gene_seq, smiles_feat)

                # Classification loss
                output = output.squeeze(-1)  # Ensure output is squeezed for binary classification
                loss = criterion(output, labels)

                # Reconstruction losses
                gene_reconstruction_loss = reconstruction_loss_fn(reconstructed_gene.float().view(-1), gene_seq.view(-1))
                smiles_reconstruction_loss = reconstruction_loss_fn(reconstructed_smiles.float().view(-1), smiles_feat.view(-1))

                # Combine losses
                total_loss = loss + gene_reconstruction_loss + smiles_reconstruction_loss

                # Update validation loss
                val_loss += total_loss.item() * gene_seq.size(0)

                # Compute accuracy
                preds = output.round()  # For binary classification, use round for prediction threshold
                correct += (preds.squeeze() == labels).sum().item()
                total += labels.size(0)

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        val_accuracy = correct / total
        val_accuracies.append(val_accuracy)

        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Epoch Time: {epoch_time:.2f} seconds")

    # Summary of training time
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    avg_epoch_time = total_time / epochs
    print(f"\nTotal Training Time: {total_time:.2f} seconds")
    print(f"Average Epoch Time: {avg_epoch_time:.2f} seconds")

    return model, train_losses, val_losses, val_accuracies