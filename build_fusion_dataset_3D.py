import torch
from torch_geometric.data import Data
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from tqdm import tqdm
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# --- 1. Configuration (Constants) ---
PROCESSED_DATA_DIR = "processed_fusion_data"
RAW_DATA_FILES = {
    "train": "train_raw.csv",
    "valid": "valid_raw.csv",
    "test": "test_raw.csv"
}
LABEL_COLUMNS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
    "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
]
WEIGHT_COLUMNS = [
    "w1", "w2", "w3", "w4", "w5", "w6", "w7", "w8", "w9", "w10", "w11", "w12"
]
SMILES_COLUMN = "ids"
N_TASKS = 12

# Scaler and Imputer save paths
NODE_SCALER_PATH = os.path.join(PROCESSED_DATA_DIR, "node_feature_scaler.joblib")
DESC_IMPUTER_PATH = os.path.join(PROCESSED_DATA_DIR, "desc_feature_imputer.joblib")
DESC_SCALER_PATH = os.path.join(PROCESSED_DATA_DIR, "desc_feature_scaler.joblib")

# --- 2. Feature Engineering Definitions ---

# --- 2a. Descriptor Features ---
DESCRIPTOR_NAMES_2D = [
    "MolWt", "NumHDonors", "NumHAcceptors", "MolLogP", "TPSA", "NumRotatableBonds",
    "NumAliphaticRings", "NumAromaticRings", "NumHeteroatoms", "qed"
]
DESCRIPTOR_NAMES_3D = [
    "Asphericity", "NPR1", "NPR2", "PMI1", "PMI2", "PMI3", "SpherocityIndex"
]
DESCRIPTOR_NAMES = DESCRIPTOR_NAMES_2D + DESCRIPTOR_NAMES_3D
N_DESC_FEATURES = len(DESCRIPTOR_NAMES)

def get_descriptor_features(mol_2d, mol_3d, conformer_failed):
    """
    Generates RDKit 2D and 3D descriptor features.
    Handles 3D generation failures by returning 0.0 for 3D descriptors.
    """
    features = []
    
    # Calculate 2D Descriptors
    desc_dict_2d = Descriptors.CalcMolDescriptors(mol_2d, missingVal=np.nan)
    for name in DESCRIPTOR_NAMES_2D:
        features.append(desc_dict_2d.get(name, np.nan))
    
    # Calculate 3D Descriptors
    if conformer_failed:
        # If 3D gen failed (e.g., metals), append 0.0 for all 3D features.
        features.extend([0.0] * len(DESCRIPTOR_NAMES_3D))
    else:
        # Otherwise, calculate 3D descriptors safely
        try: d_asphericity = rdMolDescriptors.CalcAsphericity(mol_3d)
        except: d_asphericity = 0.0
        try: d_npr1 = rdMolDescriptors.CalcNPR1(mol_3d)
        except: d_npr1 = 0.0
        try: d_npr2 = rdMolDescriptors.CalcNPR2(mol_3d)
        except: d_npr2 = 0.0
        try: d_pmi1 = rdMolDescriptors.CalcPMI1(mol_3d)
        except: d_pmi1 = 0.0
        try: d_pmi2 = rdMolDescriptors.CalcPMI2(mol_3d)
        except: d_pmi2 = 0.0
        try: d_pmi3 = rdMolDescriptors.CalcPMI3(mol_3d)
        except: d_pmi3 = 0.0
        try: d_spherocity = rdMolDescriptors.CalcSpherocityIndex(mol_3d)
        except: d_spherocity = 0.0
        features.extend([d_asphericity, d_npr1, d_npr2, d_pmi1, d_pmi2, d_pmi3, d_spherocity])

    return features

# --- 2b. Fingerprint Features ---
def get_fingerprint_features(mol):
    """ Generates ECFP4 (Morgan) fingerprint features. """
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    return np.array(fp, dtype=np.float32)

# --- 2c. Graph Node Features ---
def get_atom_features(atom):
    features = []
    atom_num = atom.GetAtomicNum()
    atom_symbols = [5, 6, 7, 8, 9, 15, 16, 17, 35, 53] # B, C, N, O, F, P, S, Cl, Br, I
    atom_one_hot = [0] * (len(atom_symbols) + 1)
    if atom_num in atom_symbols:
        atom_one_hot[atom_symbols.index(atom_num)] = 1
    else:
        atom_one_hot[-1] = 1 # 'Other'
    features.extend(atom_one_hot)
    
    degree = atom.GetDegree()
    degree_one_hot = [0] * 7
    degree_one_hot[min(degree, 6)] = 1
    features.extend(degree_one_hot)
    
    hybrid = atom.GetHybridization()
    hybrid_types = [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2, 
                    Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D, 
                    Chem.rdchem.HybridizationType.SP3D2]
    hybrid_one_hot = [0] * (len(hybrid_types) + 1)
    if hybrid in hybrid_types:
        hybrid_one_hot[hybrid_types.index(hybrid)] = 1
    else:
        hybrid_one_hot[-1] = 1
    features.extend(hybrid_one_hot)

    hydrogens = atom.GetTotalNumHs()
    hydrogen_one_hot = [0] * 6
    hydrogen_one_hot[min(hydrogens, 5)] = 1
    features.extend(hydrogen_one_hot)

    charge = atom.GetFormalCharge()
    charge_map = {-2: 0, -1: 1, 0: 2, 1: 3, 2: 4}
    charge_one_hot = [0] * 6
    charge_one_hot[charge_map.get(charge, 5)] = 1
    features.extend(charge_one_hot)
    
    features.append(atom.GetIsAromatic())
    features.append(atom.IsInRing())
    
    return features

# --- 2d. Graph Edge Features (NOW INCLUDES 3D BOND LENGTH!) ---
def get_bond_features(bond, conformer):
    """ Gets 2D bond features + 3D bond length. """
    features = []
    # 1. Bond Type (one-hot)
    bond_type = bond.GetBondType()
    bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, 
                  Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
    bond_one_hot = [0] * (len(bond_types) + 1)
    if bond_type in bond_types:
        bond_one_hot[bond_types.index(bond_type)] = 1
    else:
        bond_one_hot[-1] = 1 # 'Other'
    features.extend(bond_one_hot)
    
    # 2. Boolean features
    features.append(bond.GetIsConjugated())
    features.append(bond.IsInRing())
    
    # 3. Stereo (one-hot)
    stereo = bond.GetStereo()
    stereo_types = [Chem.rdchem.BondStereo.STEREONONE, Chem.rdchem.BondStereo.STEREOE, 
                    Chem.rdchem.BondStereo.STEREOZ, Chem.rdchem.BondStereo.STEREOANY]
    stereo_one_hot = [0] * (len(stereo_types) + 1)
    if stereo in stereo_types:
        stereo_one_hot[stereo_types.index(stereo)] = 1
    else:
        stereo_one_hot[-1] = 1 # 'Other'
    features.extend(stereo_one_hot)
    
    # --- 4. 3D BOND LENGTH (NEW!) ---
    bond_len = 0.0
    if conformer is not None:
        try:
            pt1 = conformer.GetAtomPosition(bond.GetBeginAtomIdx())
            pt2 = conformer.GetAtomPosition(bond.GetEndAtomIdx())
            bond_len = pt1.Distance(pt2)
        except Exception:
            bond_len = 0.0 # Failed to get distance
            
    features.append(bond_len)
    
    return features

# --- 3. Main Data Processing Function ---

def create_data_list(df, desc_imputer=None, desc_scaler=None, node_scaler=None):
    """
    Processes a DataFrame into a list of PyTorch Geometric Data objects.
    Now includes 3D conformer generation and 3D bond lengths.
    """
    data_list = []
    all_desc_features = []
    all_node_features = []
    
    print(f"Processing {len(df)} SMILES strings...")
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Featurizing molecules"):
        smi = row[SMILES_COLUMN]
        
        # --- 2D Mol Generation ---
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print(f"Warning: RDKit failed to parse SMILES: {smi}. Skipping.")
            continue
            
        # --- 3D Conformer Generation ---
        conformer_failed = True
        mol_3d_with_hs = None
        conformer = None
        try:
            mol_3d_with_hs = Chem.AddHs(mol)
            if AllChem.EmbedMolecule(mol_3d_with_hs, randomSeed=42) == 0:
                AllChem.MMFFOptimizeMolecule(mol_3d_with_hs)
                conformer = mol_3d_with_hs.GetConformer(0)
                conformer_failed = False
            else:
                if AllChem.EmbedMolecule(mol_3d_with_hs, useRandomCoords=True, randomSeed=42) == 0:
                    AllChem.MMFFOptimizeMolecule(mol_3d_with_hs)
                    conformer = mol_3d_with_hs.GetConformer(0)
                    conformer_failed = False
        except Exception as e:
            conformer_failed = True
            mol_3d_with_hs = None
            conformer = None

        # --- Feature Extraction ---
        # 1. Graph Features (Nodes)
        node_features = [get_atom_features(atom) for atom in mol.GetAtoms()]
        if not node_features:
            print(f"Warning: Mol with no atoms: {smi}. Skipping.")
            continue
        x = torch.tensor(node_features, dtype=torch.float)
        
        # 2. Graph Features (Edges) - NOW PASSES CONFORMER
        edge_index = []
        edge_attr = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            # Pass the conformer (or None) to get bond length
            bond_features = get_bond_features(bond, conformer) 
            edge_index.extend([[i, j], [j, i]])
            edge_attr.extend([bond_features, bond_features])
            
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # Handle molecules with no bonds
        if not edge_attr:
            # We must define the shape of the edge_attr tensor
            # Our bond features has 10 (2D) + 1 (3D) = 11 features
            edge_attr = torch.empty((0, 11), dtype=torch.float)
        else:
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        # 3. Fingerprint Features
        fp_features = get_fingerprint_features(mol)
        
        # 4. Descriptor Features
        desc_features = get_descriptor_features(mol, mol_3d_with_hs, conformer_failed)

        # 5. Get Labels
        labels = list(row[LABEL_COLUMNS])
        labels_tensor = torch.tensor(labels, dtype=torch.float).reshape(1, N_TASKS)
        
        # 6. Get Weights
        weights = list(row[WEIGHT_COLUMNS])
        weights_tensor = torch.tensor(weights, dtype=torch.float).reshape(1, N_TASKS)

        # Store for scaling
        all_desc_features.append(desc_features)
        all_node_features.append(x)

        # Create the Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=labels_tensor,
            w=weights_tensor,
            fp_features=torch.tensor(fp_features, dtype=torch.float).reshape(1, -1),
            desc_features=torch.tensor(desc_features, dtype=torch.float).reshape(1, -1),
            smiles=smi
        )
        data_list.append(data)
        
    print(f"Successfully processed {len(data_list)} molecules.")

    # --- Feature Scaling & Imputation ---
    
    # 1. Descriptors (Impute NaNs, then Scale)
    print("Processing descriptor features...")
    all_desc_features = np.array(all_desc_features, dtype=float)
    if desc_imputer is None:
        print("Fitting descriptor imputer...")
        desc_imputer = SimpleImputer(strategy='mean')
        all_desc_features = desc_imputer.fit_transform(all_desc_features)
    else:
        print("Transforming with existing descriptor imputer...")
        all_desc_features = desc_imputer.transform(all_desc_features)
    if desc_scaler is None:
        print("Fitting descriptor scaler...")
        desc_scaler = StandardScaler()
        all_desc_features = desc_scaler.fit_transform(all_desc_features)
    else:
        print("Transforming with existing descriptor scaler...")
        all_desc_features = desc_scaler.transform(all_desc_features)
    
    # 2. Node Features (Scale only)
    print("Processing node features...")
    all_node_features_stacked = torch.cat(all_node_features, dim=0).numpy()
    if node_scaler is None:
        print("Fitting node feature scaler...")
        node_scaler = StandardScaler()
        all_node_features_stacked = node_scaler.fit_transform(all_node_features_stacked)
    else:
        print("Transforming with existing node feature scaler...")
        all_node_features_stacked = node_scaler.transform(all_node_features_stacked)
        
    # 3. Re-insert scaled features back into Data objects
    print("Re-inserting scaled features...")
    current_atom_idx = 0
    for data in data_list:
        num_atoms = data.x.shape[0]
        scaled_x = all_node_features_stacked[current_atom_idx : current_atom_idx + num_atoms]
        data.x = torch.tensor(scaled_x, dtype=torch.float)
        current_atom_idx += num_atoms
    for i, data in enumerate(data_list):
        data.desc_features = torch.tensor(all_desc_features[i], dtype=torch.float).reshape(1, -1)
        
    return data_list, desc_imputer, desc_scaler, node_scaler

# --- 4. Main Execution ---
if __name__ == "__main__":
    
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)
        
    print("\n--- Processing Training Data (and fitting scalers) ---")
    train_df = pd.read_csv(RAW_DATA_FILES["train"])
    train_data, desc_imputer, desc_scaler, node_scaler = create_data_list(train_df)
    joblib.dump(desc_imputer, DESC_IMPUTER_PATH)
    joblib.dump(desc_scaler, DESC_SCALER_PATH)
    joblib.dump(node_scaler, NODE_SCALER_PATH)
    print(f"Scalers and imputer saved to {PROCESSED_DATA_DIR}")
    torch.save(train_data, os.path.join(PROCESSED_DATA_DIR, "train_data.pt"))
    print("Training data saved.")

    print("\n--- Processing Validation Data (using existing scalers) ---")
    valid_df = pd.read_csv(RAW_DATA_FILES["valid"])
    valid_data, _, _, _ = create_data_list(valid_df, desc_imputer, desc_scaler, node_scaler)
    torch.save(valid_data, os.path.join(PROCESSED_DATA_DIR, "valid_data.pt"))
    print("Validation data saved.")

    print("\n--- Processing Test Data (using existing scalers) ---")
    test_df = pd.read_csv(RAW_DATA_FILES["test"])
    test_data, _, _, _ = create_data_list(test_df, desc_imputer, desc_scaler, node_scaler)
    torch.save(test_data, os.path.join(PROCESSED_DATA_DIR, "test_data.pt"))
    print("Test data saved.")
    
    print("\n--- All processing complete! ---")