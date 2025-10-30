# GraphGDel: Graph Neural Network-based Gene Deletion Prediction Framework for Growth-Coupled Production in Genome-Scale Metabolic Models

## About GraphGDel
GraphGDel is an advanced framework for predicting gene deletion strategies for growth-coupled production in genome-scale metabolic models.
It extends the DeepGDel framework by incorporating Graph Neural Networks (GNNs) to better capture the complex relationships between genes and metabolites in metabolic networks.

GraphGDel consists of four neural network-based modules: (1) **Meta-M**, which sets up a metabolite representation learning task using LSTM autoencoders to learn the characteristics of metabolites; (2) **Gene-M**, which sets up a gene representation learning task using LSTM autoencoders to learn the characteristics of genes; (3) **Graph-M**, which uses Graph Convolutional Networks (GCN) to learn representations of metabolites based on graph-structured metabolic model; and (4) **Pred-M**, which integrates the latent representations from all three modules to predict gene deletion states and outputs the final gene deletion strategy.

GraphGDel Framework Overview|
:-------------------------:|
| <img width="1000" alt="image" src="https://github.com/yangziwei96/GraphGDel/blob/main/ov.png">

## Necessary Environments

To use GraphGDel, you need the following core environment setup with the recommended versions:

- [Python](https://www.python.org/) (Version 3.12)
- [Pytorch](https://pytorch.org/) (Version 2.2.2)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) (Version 2.4.0)
- [Tensorflow](https://www.tensorflow.org/) (Version 2.17.0)

Additional auxiliary Python packages (e.g., pandas, numpy, networkx) are specified in the source code.

### Installation Options

**Option 1: Install dependencies only**
```
pip install -r requirements.txt
```

**Option 2: Install as a Python package (recommended)**
```
pip install -e .
```
This installs GraphGDel as a development package, making it available as a Python module and providing the `graphgdel` command-line tool.

If you're using other than CUDA 11.7, you may need to install PyTorch for the proper version of CUDA. See [instructions](https://pytorch.org/get-started/locally/) for more details.

## Datasets and Data Download

The maximal gene deletion strategy data (specifying a metabolic model and a target metabolite) can be downloaded from [MetNetComp](https://metnetcomp.github.io/database1/indexFiles/index.html) via the following URL:

```
https://metnetcomp.github.io/database1/csv/<model>&<target metabolite>&core.csv
```
For example, to download the maximal gene deletion strategy of succ_e in the e_coli_core model, change the last part of the above link to '/e_coli_core&succ_e.csv'.

The gene deletion strategy data for each metabolite in the three studied models, used as ground-truth labels, are available in the following file: `Data/<model>/Label`.

The amino acid data corresponding to each gene in the three studied metabolic models downloaded from the [KEGG](https://www.genome.jp/kegg/) database, used as framework input (Gene-M module), are available in the following file: `Data/<model>/Gene_AA`.

The SMILES data corresponding to each metabolite in the three studied metabolic models downloaded from the [MetaNetX](https://www.metanetx.org/) database, used as framework input (Meta-M module), are available in the following file: `Data/<model>/Target metabolites_smile.csv`.

The graph structure data for the metabolic networks, used as framework input (Graph-M module), are available in the following file: `Data/<model>/SMILES_node_feature_final.csv`.

Note: Please unzip the individual CSV file `Data/<model>/Label/<metabolite_name>.csv` from `Data/<model>/Label/Label.zip` using the following command:

```
unzip Data/<model>/Label/Label.zip -d Data/<model>/Label/;
```

Similarly, unzip the individual CSV file for amino acid data `Data/<model>/Gene_AA/<gene_name>.csv`, from `Data/<model>/Gene_AA/Gene_AA.zip` using the following command:

```
unzip Data/<model>/Gene_AA/Gene_AA.zip -d Data/<model>/Gene_AA/;
```

## Predicting Gene Deletion Strategies with GraphGDel

The main file `GraphGDel_main.py` provides the full implementation of the GraphGDel framework. It calls the module components `GeneLSTM.py`, `MetaLSTM.py`, `GCN.py`, and `CombinedModel.py` in the `model` directory to implement the designed four modules.

1. Function `GeneLSTM(nn.Module)` and `MetaLSTM(nn.Module)`

These two functions serve as the learning models in **Gene-M** and **Meta-M** components in the GraphGDel framework. 
They are responsible for encoding input genes and metabolites' sequential data using an **LSTM-based autoencoder**.  
They consist of the following key components:

- Embedding Layer: Converts input sequence indices into dense vector representations of size `vocab_embedding_dim`.
- LSTM Encoder: Processes the embedded sequences and captures temporal dependencies; Outputs hidden states for each time step.
- Layer Normalization: Normalizes LSTM outputs to stabilize training and improve generalization.
- Mean Pooling & Fully Connected Layer: Summarizes sequence features by taking the mean over time steps; Maps the summarized features to a fixed-size hidden representation.
- LSTM Decoder: Reconstructs the original input sequence from the learned hidden representations.
- Output Projection: Maps the decoded LSTM outputs back to the original vocabulary size; Uses `argmax` to predict the most likely token at each time step.

2. Function `GCN(nn.Module)`

This function serves as the **Graph-M** component in the GraphGDel framework.
It is responsible for learning graph-structured representations of the metabolic network using Graph Convolutional Networks.

It consists of the following key components:

- Graph Convolutional Layers: Process graph-structured data to learn node representations
- Link Prediction: Uses negative sampling to train the model on link prediction tasks
- Graph Encoding: Encodes the graph structure into node embeddings
- Graph Decoding: Decodes node embeddings to predict edge existence

3. Function `CombinedModel(nn.Module)`

This function serves as the **Pred-M** integrates GeneLSTM, MetaLSTM, and GCN for joint feature learning in the GraphGDel framework. 
It is responsible for extracting sequence representations and performing gene deletion status prediction.

It consists of the following key components:

- Feature Fusion: Combines learned representations from all three modules via element-wise multiplication.
- Fully Connected Layer: Maps the fused feature representation to a single output neuron.
- Binary Classification Output: Uses a sigmoid activation function to make the final prediction.
- Multi-task Learning: Combines classification, reconstruction, and link prediction losses.

For more detailed information, please refer to the **comments within the source code**.

## Output Details

GraphGDel saves the output gene deletion strategies as a CSV file in the following directory: `Data/<model>/Results/all_metabolites_predictions.csv`.

The output CSV represents a **Metabolite × Gene** matrix, where:

- The first column lists the target metabolites.
- The remaining columns represent a binary (0/1) vector indicating which genes should be deleted:
  - `0` → Gene **to be deleted**.
  - `1` → Gene **to remain**.

## Example Code for Quick Run

### Quick Run on the e_coli_core Model

We provide scripts to run example tests on the e_coli_core model with GraphGDel and baseline methods:

-  `quick_run_GraphGDel.py`: predicting gene deletion strategies with GraphGDel on e_coli_core model.
-  `quick_run_baseline_DNN.py`: predicting gene deletion strategies with the baseline method (DNN) on e_coli_core model.
-  `quick_run_baseline_DeepGDel.py`: predicting gene deletion strategies with the baseline method (DeepGDel) on e_coli_core model.

You can run the test script (which default to running computations on the CPU) using the following command:

```
python3 quick_run_GraphGDel.py;
```
```
python3 quick_run_baseline_DNN.py;
```
```
python3 quick_run_baseline_DeepGDel.py;
```

Note: Please unzip the e_coli_core data before testing using the following commands:

```
unzip Data/e_coli_core/Gene_AA/Gene_AA.zip -d Data/e_coli_core/Gene_AA/;
unzip Data/e_coli_core/Label/Label.zip -d Data/e_coli_core/Label/;
```

### Quick Run Reports and Outputs

The quick run scripts generate performance reports, i.e., **GraphGDel Report**, **DNN Baseline Report**, and **DeepGDel Baseline Report**, including five performance metrics: Overall Accuracy, Macro-Averaged Precision, Macro-Averaged Recall, Macro-Averaged F1-score, and AUC.
The test scripts additionally save the resulting gene deletion strategies for e_coli_core as CSV files in the following directory: `Data/e_coli_core/Results/all_metabolites_predictions_temp.csv`.

## Visualization of Feature Representations Using PCA and t-SNE

We provide visualization of the learned feature representations from the GraphGDel framework using dimensionality reduction techniques (PCA and t-SNE). This helps understand how the model learns to separate different classes of gene-metabolite relationships.

![Feature Representations](feature.png)

*Figure 1: Visualization of feature representations learned by GraphGDel using PCA (up) and t-SNE (down). The plot shows how the model learns to separate different gene-metabolite relationship patterns in the learned embedding space.*

## Precision-Recall Curve

We also provide precision-recall curves to evaluate the performance of GraphGDel across different threshold settings, demonstrating the model's ability to balance precision and recall for gene deletion prediction.

![Precision-Recall Curve](PR.png)

*Figure 2: Precision-Recall curve for GraphGDel on the e_coli_core model. The curve shows the trade-off between precision and recall at different classification thresholds, providing insights into the model's predictive performance.*



## Complementary: Learning Model Training and Ablations

### (1) Learning Model Training

We provide functions for **training learning models within the GraphGDel framework**, making it easy to apply and extend GraphGDel to additional models or datasets.
The main script, `training_main.py`, and its dependency functions are in the `trainer` folder.

After training, the model's state dictionary, which includes all trainable parameters (such as weights and biases), is stored in the `train_disc` folder as a `.sav` file. 
This file is generated by the `torch.save` function, which is provided by PyTorch, and is used to serialize and save the model's learned parameters for future use.

1. Function `train_model(model, train_loader, val_loader, epochs, lr, device)`:

Trains the given model using the specified training and validation datasets.

- model (nn.Module): The model to train.
- train_loader (DataLoader): DataLoader for the training set.
- val_loader (DataLoader): DataLoader for the validation set.
- epochs (int): Number of training epochs.
- lr (float): Learning rate for the optimizer.
- device (str or torch.device): The device for training (cpu or cuda).

2. Function `save_model_with_confirmation(model, CBM)`

Save a PyTorch model only after receiving explicit confirmation. 

- model (torch.nn.Module): The PyTorch model to be saved.
- CBM (str): A unique identifier used to name the saved model file.

### (2) Learning Model Ablations

We provide functions for **learning models ablations within the GraphGDel framework** to explore the further impact on different learning models in the designed modules.
The functions are built-in by default with the above `training_main.py` script, and its dependency functions are in the `model` folder.

1. Function `freeze_model(model)`:

Prevents a PyTorch model's parameters from being updated during training by freezing all layers, also working with pre-trained models.

- model (nn.Module): The PyTorch model whose parameters will be frozen.

2. Function `is_model_frozen(model)`:

Checks whether all parameters in a PyTorch model are frozen, meaning none of them require gradient updates.

- model (nn.Module): The PyTorch model to check.

## License

Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
