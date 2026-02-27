import pandas as pd

def read_metabolite_smiles(file_path):
    """Read metabolite SMILES from CSV file."""
    try:
        df = pd.read_csv(file_path)
        smiles_features = df['Smiles'].tolist()  # Note: column name is 'Smiles' not 'SMILES'
        metabolite_names = df['Metabolite'].tolist()
        return metabolite_names, smiles_features
    except Exception as e:
        print(f"Error reading metabolite SMILES: {e}")
        return [], []