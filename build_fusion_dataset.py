"""
Builds the "Early Fusion" dataset from the raw DeepChem CSVs.

This script performs all preprocessing:
1.  Loads the raw CSV data (train, valid, test).
2.  Generates three feature sets:
    - Graph Features (Nodes, Edges)
    - Fingerprint Features (ECFP4)
    - Descriptor Features (RDKit 2D)
3.  Normalizes the Node features using a StandardScaler fit *only* on the training set.
4.  Handles and skips invalid SMILES.
5.  Saves the final, processed data lists (for PyTorch Geometric) to disk.
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.preprocessing import StandardScaler
import joblib
import torch
from torch_geometric.data import Data

# --- Configuration ---
RAW_DATA_DIR = "dataset"
PROCESSED_DATA_DIR = "processed_fusion_data"
SCALER_PATH = os.path.join(PROCESSED_DATA_DIR, "node_feature_scaler.joblib")

# Column names from the CSV
SMILES_COLUMN = 'ids'
LABEL_COLUMNS = [
    'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
    'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
]
WEIGHT_COLUMNS = [
    'w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w7', 'w8', 'w9', 'w10', 'w11', 'w12'
]

# --- RDKit Descriptors ---
# List of 2D descriptors to calculate
DESCRIPTOR_FUNCTIONS = {
    name: func for name, func in Descriptors.descList
    if not any(prefix in name for prefix in ['Ipc', 'Kappa', 'Chi']) # Exclude some complex ones
    and '3D' not in name # Exclude 3D descriptors
}
# Sort to ensure consistent order
DESCRIPTOR_NAMES = sorted(DESCRIPTOR_FUNCTIONS.keys())
print(f"Calculating {len(DESCRIPTOR_NAMES)} RDKit descriptors.")


# --- 1. Graph Featurization Functions ---

def get_atom_features(atom):
    """ Generates a feature vector for a single atom. """
    # One-hot encodings for categorical features
    symbol = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'I', 'other']
    degree = [0, 1, 2, 3, 4, 5, 6]
    hybridization = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        'other'
    ]
    formal_charge = [-2, -1, 0, 1, 2]
    num_hydrogens = [0, 1, 2, 3, 4]

    # Get features
    atom_symbol = atom.GetSymbol()
    atom_degree = atom.GetDegree()
    atom_hybrid = atom.GetHybridization()
    atom_charge = atom.GetFormalCharge()
    atom_hs = atom.GetTotalNumHs()

    features = []
    # Symbol
    features.extend([int(atom_symbol == s) for s in symbol[:-1]])
    if sum(features[-len(symbol)+1:]) == 0: features.append(1) # 'other'
    else: features.append(0)
    # Degree
    features.extend([int(atom_degree == d) for d in degree])
    # Hybridization
    features.extend([int(atom_hybrid == h) for h in hybridization[:-1]])
    if sum(features[-len(hybridization)+1:]) == 0: features.append(1) # 'other'
    else: features.append(0)
    # Formal Charge
    features.extend([int(atom_charge == c) for c in formal_charge])
    # Num Hydrogens
    features.extend([int(atom_hs == h) for h in num_hydrogens])
    # Boolean features
    features.append(int(atom.GetIsAromatic()))
    features.append(int(atom.IsInRing()))

    # --- New features from your script (Bug fixed) ---
    features.append(atom.GetAtomicNum())
    features.append(atom.GetMass())

    chirality = atom.GetChiralTag()
    features.append(int(chirality == Chem.rdchem.ChiralType.CHI_UNSPECIFIED))
    features.append(int(chirality == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW))
    features.append(int(chirality == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW))

    return features

def mol_to_graph_data_obj(mol):
    """ Converts an RDKit Mol object into a PyG Data object. """
    if mol is None:
        return None

    # Node features
    node_features = [get_atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(node_features, dtype=torch.float)

    # Edge index and edge features
    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices.extend([[i, j], [j, i]])

        # Edge features
        bond_type = bond.GetBondType()
        bt_features = [
            int(bond_type == Chem.rdchem.BondType.SINGLE),
            int(bond_type == Chem.rdchem.BondType.DOUBLE),
            int(bond_type == Chem.rdchem.BondType.TRIPLE),
            int(bond_type == Chem.rdchem.BondType.AROMATIC),
        ]
        stereo = bond.GetStereo()
        stereo_feats = [
            int(stereo == Chem.rdchem.BondStereo.STEREONONE),
            int(stereo == Chem.rdchem.BondStereo.STEREOZ),
            int(stereo == Chem.rdchem.BondStereo.STEREOE),
            int(stereo == Chem.rdchem.BondStereo.STEREOANY),
        ]
        
        attr = bt_features + [int(bond.GetIsConjugated()), int(bond.IsInRing())] + stereo_feats
        edge_attrs.extend([attr, attr]) # Add for both directions

    if len(edge_indices) > 0:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, len(edge_attrs[0]) if edge_attrs else 10), dtype=torch.float) # Match dim

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


# --- 2. Fingerprint Featurization Function ---

def get_fingerprint_features(mol):
    """
    Generates ECFP4 fingerprint.
    RDKit's 'MorganFingerprint' with radius=2 is equivalent to ECFP4.
    """
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    return fp.ToBitString()


# --- 3. Descriptor Featurization Function ---

def get_descriptor_features(mol):
    """ Generates RDKit 2D descriptor features. """
    if mol is None:
        return [np.nan] * len(DESCRIPTOR_NAMES)
    
    # Calculate all descriptors
    mol_descriptors = {}
    for name, func in DESCRIPTOR_FUNCTIONS.items():
        try:
            mol_descriptors[name] = func(mol)
        except Exception:
            mol_descriptors[name] = np.nan # Handle calculation errors

    # Return in consistent order
    return [mol_descriptors[name] for name in DESCRIPTOR_NAMES]


# --- Main Data Creation Function ---

def create_data_list(df, scaler=None):
    """
    Processes a DataFrame and creates a list of PyG Data objects,
    each populated with all three feature sets.
    
    Args:
        df (pd.DataFrame): The raw data.
        scaler (StandardScaler, optional): A *fitted* scaler to apply to node features.
                                          If None, no scaling is done.
    
    Returns:
        list: A list of processed torch_geometric.data.Data objects.
    """
    data_list = []
    
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Featurizing molecules"):
        smi = row[SMILES_COLUMN]
        mol = Chem.MolFromSmiles(smi)
        
        # Add explicit H's
        if mol is not None:
            mol = Chem.AddHs(mol)

        # 1. Graph Features
        graph_data = mol_to_graph_data_obj(mol)
        if graph_data is None:
            print(f"Warning: Skipping invalid SMILES: {smi}")
            continue # --- THIS IS THE FIX FOR THE 'NoneType' ERROR ---
        
        # --- THIS IS THE FIX FOR THE LOW 0.76 AUC ---
        # Apply normalization if scaler is provided
        if scaler is not None:
            try:
                graph_data.x = torch.tensor(scaler.transform(graph_data.x.numpy()), dtype=torch.float)
            except Exception as e:
                print(f"Warning: Failed to scale features for {smi}. Skipping. Error: {e}")
                continue
        # --- END FIX ---
        
        # 2. Fingerprint Features
        fp_str = get_fingerprint_features(mol)
        if fp_str is None:
            print(f"Warning: Skipping molecule with failed FP: {smi}")
            continue
        fp_features = [int(b) for b in fp_str]
        graph_data.fp_features = torch.tensor(fp_features, dtype=torch.float).reshape(1, -1)

        # 3. Descriptor Features
        desc_features = get_descriptor_features(mol)
        graph_data.desc_features = torch.tensor(desc_features, dtype=torch.float).reshape(1, -1)
        
        # 4. Get Labels
        labels = list(row[LABEL_COLUMNS])
        graph_data.y = torch.tensor(labels, dtype=torch.float).reshape(1, 12)

        # 5. Get Weights (Fix for Bug 1)
        weights = list(row[WEIGHT_COLUMNS])
        graph_data.w = torch.tensor(weights, dtype=torch.float).reshape(1, 12)
        
        data_list.append(graph_data)
        
    return data_list


# --- Main Execution ---

if __name__ == "__main__":
    
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    # Load raw data
    try:
        train_df = pd.read_csv(os.path.join(RAW_DATA_DIR, "train_raw.csv"))
        valid_df = pd.read_csv(os.path.join(RAW_DATA_DIR, "valid_raw.csv"))
        test_df = pd.read_csv(os.path.join(RAW_DATA_DIR, "test_raw.csv"))
    except FileNotFoundError:
        print("Error: train.csv, valid.csv, or test.csv not found.")
        print("Please run the DeepChem loader script first to generate these files.")
        exit()

    # --- 1. Process TRAINING data (to fit scaler) ---
    print("\n--- Processing Training Data (Pass 1) ---")
    # We create the train list *without* scaling first
    train_data_list = create_data_list(train_df, scaler=None)

    # --- 2. Fit and Apply Scaler (The 0.76 AUC Fix) ---
    print("\n--- Fitting Node Feature Scaler ---")
    # Stack all node features from the training set
    all_train_node_features = np.concatenate(
        [data.x.numpy() for data in train_data_list],
        axis=0
    )
    print(f"Fitting scaler on {all_train_node_features.shape[0]} atoms...")
    
    # Fit the scaler
    scaler = StandardScaler()
    scaler.fit(all_train_node_features)
    
    # Save the scaler
    joblib.dump(scaler, SCALER_PATH)
    print(f"Node feature scaler saved to {SCALER_PATH}")
    
    # Apply the fitted scaler to the training data (in-place)
    print("Applying scaler to training data (Pass 2)...")
    for data in tqdm(train_data_list, desc="Scaling train nodes"):
        data.x = torch.tensor(scaler.transform(data.x.numpy()), dtype=torch.float)
    
    # Save the normalized training data
    torch.save(train_data_list, os.path.join(PROCESSED_DATA_DIR, "train_data.pt"))
    print(f"Processed training data saved! ({len(train_data_list)} graphs)")

    # --- 3. Process VALIDATION data (with fitted scaler) ---
    print(f"\n--- Processing Validation Data ---")
    valid_data_list = create_data_list(valid_df, scaler=scaler)
    torch.save(valid_data_list, os.path.join(PROCESSED_DATA_DIR, "valid_data.pt"))
    print(f"Processed validation data saved! ({len(valid_data_list)} graphs)")

    # --- 4. Process TEST data (with fitted scaler) ---
    print(f"\n--- Processing Test Data ---")
    test_data_list = create_data_list(test_df, scaler=scaler)
    torch.save(test_data_list, os.path.join(PROCESSED_DATA_DIR, "test_data.pt"))
    print(f"Processed test data saved! ({len(test_data_list)} graphs)")

    print("\n--- Preprocessing Complete! ---")
    print(f"All processed data saved to '{PROCESSED_DATA_DIR}'")


# import os
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# from rdkit import Chem
# from rdkit.Chem import AllChem, Descriptors
# from rdkit.DataStructs import ConvertToNumpyArray
# import torch
# from torch_geometric.data import Data
# import deepchem as dc
# from sklearn.preprocessing import StandardScaler
# from sklearn.impute import SimpleImputer

# # --- RDKit Featurization Functions ---

# def one_hot_encode(value, allowed_set):
#     if value not in allowed_set:
#         value = allowed_set[-1]
#     return [int(value == s) for s in allowed_set]

# def get_atom_features(atom):
#     """More robust one-hot atom features."""
#     ATOM_SYMBOLS = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'other']
#     DEGREES = [0, 1, 2, 3, 4, 5, 'other']
#     HYBRIDIZATIONS = [
#         Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
#         Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
#         Chem.rdchem.HybridizationType.SP3D2, 'other'
#     ]
#     FORMAL_CHARGES = [-2, -1, 0, 1, 2, 'other']
#     TOTAL_HS = [0, 1, 2, 3, 4, 'other']
    
#     features = (
#         one_hot_encode(atom.GetSymbol(), ATOM_SYMBOLS) +
#         one_hot_encode(atom.GetDegree(), DEGREES) +
#         one_hot_encode(atom.GetHybridization(), HYBRIDIZATIONS) +
#         one_hot_encode(atom.GetFormalCharge(), FORMAL_CHARGES) +
#         one_hot_encode(atom.GetTotalNumHs(), TOTAL_HS) +
#         [int(atom.GetIsAromatic())] +
#         [int(atom.IsInRing())]
#     )
#     return features

# def featurize_graph(mol):
#     """Converts an RDKit Mol object into graph node/edge features."""
#     try:
#         mol = Chem.AddHs(mol)
#         node_features = [get_atom_features(atom) for atom in mol.GetAtoms()]
#         if not node_features:
#             return None
        
#         edge_indices = []
#         for bond in mol.GetBonds():
#             i = bond.GetBeginAtomIdx()
#             j = bond.GetEndAtomIdx()
#             edge_indices.append((i, j))
#             edge_indices.append((j, i))

#         node_features_np = np.array(node_features, dtype=np.float32)
        
#         if edge_indices:
#             edge_index_np = np.array(edge_indices, dtype=np.int64).T
#         else:
#             edge_index_np = np.array([], dtype=np.int64).reshape(2, 0)
            
#         return {
#             'node_features': torch.tensor(node_features_np, dtype=torch.float),
#             'edge_index': torch.tensor(edge_index_np, dtype=torch.long)
#         }
#     except Exception as e:
#         # print(f"Warning: RDKit error featurizing graph: {e}")
#         return None

# def get_fingerprint_features(mol):
#     """Generates Morgan (ECFP) fingerprints."""
#     try:
#         fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
#         fp_np = np.zeros((1,), dtype=np.int8)
#         ConvertToNumpyArray(fp, fp_np)
#         return torch.tensor(fp_np.astype(np.float32), dtype=torch.float)
#     except Exception as e:
#         return None

# def get_descriptor_features(mol):
#     """Generates a panel of RDKit descriptors."""
#     descriptor_names = [
#         'MolWt', 'MolLogP', 'NumHDonors', 'NumHAcceptors',
#         'TPSA', 'NumRotatableBonds', 'FpDensityMorgan1', 'FractionCSP3'
#     ]
#     features = []
#     for name in descriptor_names:
#         try:
#             desc_val = getattr(Descriptors, name)(mol)
#             features.append(desc_val)
#         except:
#             features.append(np.nan)
#     return features # Return a list for now, we'll numpy/tensor it later

# def add_descriptors(df):
#     desc_data = []
#     for smiles in tqdm(df[SMILES_COLUMN], desc="Calculating descriptors"):
#         mol = Chem.MolFromSmiles(smiles)
#         if mol:
#             desc_data.append(get_descriptor_features(mol))
#         else:
#             desc_data.append([np.nan] * len(DESCRIPTOR_COLUMNS))
#     return pd.concat([df, pd.DataFrame(desc_data, columns=DESCRIPTOR_COLUMNS, index=df.index)], axis=1)

# # --- Main Featurization Loop ---

# def create_data_list(df):
#     """
#     Processes a DataFrame and returns a list of PyG Data objects.
#     This is now fast and efficient.
#     """
#     data_list = []
    
#     # Use iterrows() to handle column names with special characters ('-')
#     for _, row in tqdm(df.iterrows(), total=len(df), desc="Creating Data objects"):
#         smiles = row[SMILES_COLUMN]
#         mol = Chem.MolFromSmiles(smiles)
        
#         if mol is None:
#             continue
        
#         # 1. Get Graph Features
#         graph_features = featurize_graph(mol)
#         if graph_features is None:
#             continue
            
#         # 2. Get Fingerprint Features
#         fp_features = get_fingerprint_features(mol)
#         if fp_features is None:
#             continue
            
#         # 3. Get Pre-scaled Descriptor Features
#         # Access by list/string name, not attribute
#         desc_features = list(row[DESCRIPTOR_COLUMNS])
#         desc_tensor = torch.tensor(desc_features, dtype=torch.float)

#         # 4. Get Labels
#         # Access by list/string name, not attribute
#         labels = list(row[LABEL_COLUMNS])
#         labels_tensor = torch.tensor(labels, dtype=torch.float).reshape(1, 12)

#         # 5. Get Weights (Fix for Bug 1)
#         # Access by list/string name, not attribute
#         weights = list(row[WEIGHT_COLUMNS])
#         weights_tensor = torch.tensor(weights, dtype=torch.float).reshape(1, 12)

#         # 6. Create the "Early Fusion" Data Object
#         data = Data(
#             x=graph_features['node_features'],
#             edge_index=graph_features['edge_index'],
#             y=labels_tensor,
#             w=weights_tensor,  # Store the 12 weights
#             fp=fp_features.reshape(1, -1), # Store the fingerprint
#             desc=desc_tensor.reshape(1, -1) # Store the descriptors
#         )
#         data_list.append(data)
        
#     return data_list

# if __name__ == '__main__':
#     # --- Main Data Loading and Processing ---

#     print("📥 Loading Tox21 dataset from DeepChem...")
#     train_df = pd.read_csv('tox21_processed/train_raw.csv')
#     valid_df = pd.read_csv('tox21_processed/valid_raw.csv')
#     test_df = pd.read_csv('tox21_processed/test_raw.csv')

#     print(f"Original sizes: Train={len(train_df)}, Valid={len(valid_df)}, Test={len(test_df)}")

#     # Define all column names
#     LABEL_COLUMNS = [
#         'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 
#         'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 
#         'SR-HSE', 'SR-MMP', 'SR-p53'
#     ]
#     # This assumes the 'w' columns are named w1, w2, etc. in the dataframe
#     # Let's check the columns and create the list dynamically
#     weight_columns = [col for col in train_df.columns if col.startswith('w') and col[1:].isdigit()]
#     if len(weight_columns) != 12:
#         print(f"Warning: Expected 12 weight columns ('w1'...'w12'), found {len(weight_columns)}. Using them anyway.")
#     WEIGHT_COLUMNS = weight_columns

#     SMILES_COLUMN = 'ids'
#     DESCRIPTOR_COLUMNS = [
#         'MolWt', 'MolLogP', 'NumHDonors', 'NumHAcceptors',
#         'TPSA', 'NumRotatableBonds', 'FpDensityMorgan1', 'FractionCSP3'
#     ]

#     # --- 3. Pre-calculate Descriptors and Handle Scaling ---
#     # We do this *before* the main loop to handle scaling and imputing correctly
#     print("🔬 Pre-calculating descriptors for scaling...")

#     train_df = add_descriptors(train_df)
#     valid_df = add_descriptors(valid_df)
#     test_df = add_descriptors(test_df)

#     # Impute and Scale Descriptors
#     print("⚖️ Imputing and Scaling descriptors...")
#     imputer = SimpleImputer(strategy='mean')
#     scaler = StandardScaler()

#     # Fit on training data ONLY
#     imputer.fit(train_df[DESCRIPTOR_COLUMNS])
#     train_df[DESCRIPTOR_COLUMNS] = imputer.transform(train_df[DESCRIPTOR_COLUMNS])

#     scaler.fit(train_df[DESCRIPTOR_COLUMNS])
#     train_df[DESCRIPTOR_COLUMNS] = scaler.transform(train_df[DESCRIPTOR_COLUMNS])

#     # Transform valid and test data
#     valid_df[DESCRIPTOR_COLUMNS] = imputer.transform(valid_df[DESCRIPTOR_COLUMNS])
#     valid_df[DESCRIPTOR_COLUMNS] = scaler.transform(valid_df[DESCRIPTOR_COLUMNS])

#     test_df[DESCRIPTOR_COLUMNS] = imputer.transform(test_df[DESCRIPTOR_COLUMNS])
#     test_df[DESCRIPTOR_COLUMNS] = scaler.transform(test_df[DESCRIPTOR_COLUMNS])

#     print("✅ Scaling and Imputing complete.")

#     # Process all datasets
#     print("\n--- Processing Training Data ---")
#     train_data_list = create_data_list(train_df)
#     print(f"\n--- Processing Validation Data ---")
#     valid_data_list = create_data_list(valid_df)
#     print(f"\n--- Processing Test Data ---")
#     test_data_list = create_data_list(test_df)

#     print("\n--- Featurization Summary ---")
#     print(f"Processed Train Graphs: {len(train_data_list)}")
#     print(f"Processed Valid Graphs: {len(valid_data_list)}")
#     print(f"Processed Test Graphs:  {len(test_data_list)}")

#     print("\nSample Data Object (from training set):")
#     if train_data_list:
#         print(train_data_list[0])
#         print("Notice it now contains 'y', 'w', 'fp', and 'desc' keys!")

#     # --- 5. Save the Processed Data ---
#     OUTPUT_DIR = "processed_fusion_data"
#     os.makedirs(OUTPUT_DIR, exist_ok=True)

#     print(f"\n💾 Saving processed data to '{OUTPUT_DIR}'...")
#     torch.save(train_data_list, os.path.join(OUTPUT_DIR, "train_data.pt"))
#     torch.save(valid_data_list, os.path.join(OUTPUT_DIR, "valid_data.pt"))
#     torch.save(test_data_list, os.path.join(OUTPUT_DIR, "test_data.pt"))

#     print("✅ All processing complete and saved!")