import numpy as np
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import csv
from collections import defaultdict
import time
import os
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

def predict_gene_deletions_for_metabolite(model, metabolite_name, metabolite_names, gene_sequences, smiles_features, relationships, device, genes_to_exclude):
    # Find the index of the metabolite
    meta_index = metabolite_names.tolist().index(metabolite_name)
    fingerprint_feature = smiles_features[meta_index]  # shape: (F,)

    model.eval()
    with torch.no_grad():
        fingerprint_tensor = torch.tensor(fingerprint_feature, dtype=torch.float32).to(device)
        gene_seqs_tensor = torch.tensor(gene_sequences, dtype=torch.long).to(device)

        # Repeat fingerprint for each gene
        fingerprint_feat = fingerprint_tensor.expand(len(gene_sequences), -1).contiguous()

        # Model prediction
        output = model(gene_seqs_tensor, fingerprint_feat)
        prediction = output[0] if isinstance(output, tuple) else output
        predicted_probs = torch.sigmoid(prediction).view(-1).cpu().numpy()

    # Extract true labels
    if metabolite_name in relationships:
        gene_names, true_labels = relationships[metabolite_name]
        true_labels = np.array(true_labels)
    else:
        print(f"No relationship data found for metabolite '{metabolite_name}'")
        return [], [], predicted_probs, []

    # Filter out excluded genes
    filtered_indices = [i for i, gene in enumerate(gene_names) if gene not in genes_to_exclude]
    gene_names = [gene_names[i] for i in filtered_indices]
    predicted_probs = predicted_probs[filtered_indices]
    true_labels = true_labels[filtered_indices]

    # Automatically find best threshold from ROC
    if len(np.unique(true_labels)) > 1:
        fpr, tpr, thresholds = roc_curve(true_labels, predicted_probs)
        optimal_idx = np.argmax(tpr - fpr)
        best_threshold = thresholds[optimal_idx]
    else:
        best_threshold = 0.5  # fallback if only one class exists

    predicted_deletions = (predicted_probs >= best_threshold).astype(int)

    return gene_names, predicted_deletions, predicted_probs, true_labels


def calculate_metrics_for_val_metabolites(model, val_metabolites, metabolite_names, gene_sequences, smiles_features, relationships, device, genes_to_exclude):
    total_correct = 0
    total_genes = 0
    total_true_positive = [0, 0]
    total_false_positive = [0, 0]
    total_true_negative = [0, 0]
    total_false_negative = [0, 0]

    metabolite_metrics = []

    for metabolite_name in val_metabolites:
        gene_names, predicted_deletions, predicted_probs, true_labels = predict_gene_deletions_for_metabolite(
            model, metabolite_name, metabolite_names, gene_sequences, smiles_features, relationships, device, genes_to_exclude
        )
        
        if len(gene_names) > 0:
            num_correct = np.sum(predicted_deletions == true_labels)
            accuracy = (num_correct / len(gene_names)) * 100
            total_correct += num_correct
            total_genes += len(gene_names)

            true_positive = [0, 0]
            false_positive = [0, 0]
            true_negative = [0, 0]
            false_negative = [0, 0]
            
            for label in [0, 1]:
                true_positive[label] = np.sum((predicted_deletions == label) & (true_labels == label))
                false_positive[label] = np.sum((predicted_deletions == label) & (true_labels != label))
                true_negative[label] = np.sum((predicted_deletions != label) & (true_labels != label))
                false_negative[label] = np.sum((predicted_deletions != label) & (true_labels == label))

                total_true_positive[label] += true_positive[label]
                total_false_positive[label] += false_positive[label]
                total_true_negative[label] += true_negative[label]
                total_false_negative[label] += false_negative[label]

            precision, recall, f1_score = [], [], []
            for label in [0, 1]:
                precision_val = (true_positive[label] / (true_positive[label] + false_positive[label]) * 100) if (true_positive[label] + false_positive[label]) > 0 else 0.0
                recall_val = (true_positive[label] / (true_positive[label] + false_negative[label]) * 100) if (true_positive[label] + false_negative[label]) > 0 else 0.0
                f1_score_val = (2 * precision_val * recall_val) / (precision_val + recall_val) if (precision_val + recall_val) > 0 else 0.0
                precision.append(precision_val)
                recall.append(recall_val)
                f1_score.append(f1_score_val)

            avg_precision = np.mean(precision)
            avg_recall = np.mean(recall)
            avg_f1_score = np.mean(f1_score)

            # Compute AUC
            try:
                auc = roc_auc_score(true_labels, predicted_probs) * 100  # Prob of class 1
            except ValueError:
                auc = 0.0  # If only one class present in true_labels

            metabolite_metrics.append({
                'accuracy': accuracy,
                'precision': avg_precision,
                'recall': avg_recall,
                'f1_score': avg_f1_score,
                'auc': auc
            })

            print(f"Metabolite: {metabolite_name}")
            print(f"  First 10 True Gene Status: {true_labels[:10]}")
            print(f"  First 10 Predicted Gene Status: {predicted_deletions[:10]}")
            print(f"  Accuracy: {accuracy:.2f}%")
            print(f"  Precision: {avg_precision:.2f}% (Macro-Averaged)")
            print(f"  Recall: {avg_recall:.2f}% (Macro-Averaged)")
            print(f"  F1 Score: {avg_f1_score:.2f}% (Macro-Averaged)")
            print(f"  AUC: {auc:.2f}%")
            print()

    overall_accuracy = (total_correct / total_genes) * 100 if total_genes > 0 else 0.0

    total_tp_sum = np.sum(total_true_positive)
    total_fp_sum = np.sum(total_false_positive)
    total_fn_sum = np.sum(total_false_negative)

    micro_precision = (total_tp_sum / (total_tp_sum + total_fp_sum) * 100) if (total_tp_sum + total_fp_sum) > 0 else 0.0
    micro_recall = (total_tp_sum / (total_tp_sum + total_fn_sum) * 100) if (total_tp_sum + total_fn_sum) > 0 else 0.0
    micro_f1_score = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0

    average_precision = np.mean([metrics['precision'] for metrics in metabolite_metrics])
    average_recall = np.mean([metrics['recall'] for metrics in metabolite_metrics])
    average_f1_score = np.mean([metrics['f1_score'] for metrics in metabolite_metrics])
    average_auc = np.mean([metrics['auc'] for metrics in metabolite_metrics])

    print()
    print("====================== DNN baseline Report ======================")
    print(f"Overall Accuracy: {overall_accuracy:.2f}%")
    print(f"Macro-Averaged Precision: {average_precision:.2f}%")
    print(f"Macro-Averaged Recall: {average_recall:.2f}%")
    print(f"Macro-Averaged F1 Score: {average_f1_score:.2f}%")
    print(f"Macro-Averaged AUC: {average_auc:.2f}%")
    print()
