"""
Unified preprocessing pipeline for Tox21:
- Loads Tox21 dataset (SMILES + labels)
- Creates scaffold-based train/val/test splits
- Exports:
    - Graph datasets (for GNNs, PyTorch Geometric)
    - Molecular-level descriptors/fingerprints (for baseline ML)
    - Split metadata (so both use identical molecules)
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, Descriptors
import torch
from torch_geometric.data import Data
import deepchem as dc

# ---------------------------------------
# 0. Setup
# ---------------------------------------
np.random.seed(42)
torch.manual_seed(42)
os.makedirs("tox21_processed", exist_ok=True)

# ---------------------------------------
# 1. Load Tox21 dataset (all splits)
# ---------------------------------------
print("📥 Loading Tox21 dataset...")
tasks, datasets, transformers = dc.molnet.load_tox21(featurizer="Raw", data_dir=".", save_dir=".")
train_dataset, valid_dataset, test_dataset = datasets

print(type(train_dataset))

train_df = train_dataset.to_dataframe()
#print(train_df.head())
valid_df = valid_dataset.to_dataframe()
test_df = test_dataset.to_dataframe()

# print the shape of each dataset
print(f"Train set shape: {train_df.shape}")
print(f"Validation set shape: {valid_df.shape}")
print(f"Test set shape: {test_df.shape}")

print("Columns:", train_df.columns.tolist())
print("Tasks:", tasks)

# Combine all splits with an indicator column
train_df["split"] = "train"
valid_df["split"] = "valid"
test_df["split"] = "test"
df = pd.concat([train_df, valid_df, test_df], ignore_index=True)
tox_columns = tasks  # list of label columns
print(f"Total dataset shape: {df.shape}")
print(df.head())

# ---------------------------------------
# 2. Save the split results for consistency
# ---------------------------------------
# save the train/valid/test indices for reference
train_indices = df.index[df["split"] == "train"].tolist()
valid_indices = df.index[df["split"] == "valid"].tolist()
test_indices = df.index[df["split"] == "test"].tolist()
print(f"Train indices: {train_indices[:5]}... ({len(train_indices)} total)")
print(f"Valid indices: {valid_indices[:5]}... ({len(valid_indices)} total)")
print(f"Test indices: {test_indices[:5]}... ({len(test_indices)} total)")

# save them as np arrays
np.save("tox21_processed/train_indices.npy", np.array(train_indices))
np.save("tox21_processed/valid_indices.npy", np.array(valid_indices))
np.save("tox21_processed/test_indices.npy", np.array(test_indices))
print("✅ Saved original split indices.")
# ---------------------------------------

# save the SMILES strings for reference seperately
df.iloc[train_indices]["ids"].to_csv("tox21_processed/train_smiles.txt", index=False)
df.iloc[valid_indices]["ids"].to_csv("tox21_processed/valid_smiles.txt", index=False)
df.iloc[test_indices]["ids"].to_csv("tox21_processed/test_smiles.txt", index=False)
print("✅ Saved consistent scaffold splits.")

# ---------------------------------------
# 3. Define featurization functions
# ---------------------------------------
def atom_features(atom):
    return torch.tensor(
        [
            atom.GetAtomicNum(),
            atom.GetTotalDegree(),
            atom.GetFormalCharge(),
            atom.GetTotalNumHs(),
            atom.GetIsAromatic(),
        ],
        dtype=torch.float,
    )
def bond_features(bond):
    return torch.tensor(
        [bond.GetBondTypeAsDouble(), bond.GetIsConjugated(), bond.IsInRing()],
        dtype=torch.float,
    )

def mol_to_graph(mol):
    x = torch.stack([atom_features(a) for a in mol.GetAtoms()])
    edge_index, edge_attr = [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bf = bond_features(b)
        edge_index += [[i, j], [j, i]]
        edge_attr += [bf, bf]
    if len(edge_index) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 3), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.stack(edge_attr)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

def mol_features(smiles):
    """Calculate molecular features from SMILES"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [np.nan] * 8
    
    mol_wt = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    rotatable_bonds = Descriptors.NumRotatableBonds(mol)
    aromatic_rings = Descriptors.NumAromaticRings(mol)
    tpsa = Descriptors.TPSA(mol)
    num_atoms = mol.GetNumAtoms()
    
    return [mol_wt, logp, hbd, hba, rotatable_bonds, aromatic_rings, tpsa, num_atoms]

def get_morgan_fp(mol, radius=2, n_bits=1024):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    return np.array(fp, dtype=int)

def get_maccs_fp(mol):
    fp = MACCSkeys.GenMACCSKeys(mol)
    return np.array(fp, dtype=int)

def get_rdkit_descriptors(mol):
    vals = []
    for _, func in Descriptors._descList:
        try:
            vals.append(func(mol))
        except Exception:
            vals.append(np.nan)
    return np.array(vals)
desc_names = [d[0] for d in Descriptors._descList]

# ---------------------------------------
# 4. Featurize molecules

# Compute molecular features for all molecules
feature_names = ["MolWt", "LogP", "HBD", "HBA", "RotBonds", "AromaticRings", "TPSA", "NumAtoms"]
df[feature_names] = df['ids'].apply(lambda x: pd.Series(mol_features(x)))

# add morgan, maccs, and rdkit descriptors to the feature set
df['morgan'] = df['ids'].apply(lambda x: get_morgan_fp(Chem.MolFromSmiles(x)))
df['maccs'] = df['ids'].apply(lambda x: get_maccs_fp(Chem.MolFromSmiles(x)))
df['rdkit_desc'] = df['ids'].apply(lambda x: get_rdkit_descriptors(Chem.MolFromSmiles(x)))

print("Shape after adding features:", df.shape)
print(df.head())
# save the dataframe with features for inspection
df.to_csv("tox21_processed/tox21_with_features.csv", index=False)
print("✅ Saved intermediate dataframe with molecular features.")

exit()

# ---------------------------------------
graph_list, basefeat_list, valid_indices = [], [], []
print("🔬 Converting molecules to graphs & features...")
for i, row in tqdm(df.iterrows(), total=len(df), desc="Featurizing molecules"):
    smi = row["ids"]
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        print(f"⚠️ Invalid SMILES at index {i}: {smi}")
        continue
    img = Chem.Draw.MolToImage(mol)
    img.save(f"tox21_processed/molecule_{i}.png")
    # --- Graph ---
    try:
        data = mol_to_graph(mol)
        y = torch.tensor(row[tox_columns].values, dtype=torch.float)
        y[torch.isnan(y)] = -1  # missing labels
        data.y = y
        data.smiles = smi
        graph_list.append(data)
    except Exception as e:
        print(f"⚠️ Graph failed for {smi}: {e}")
        continue
    # --- Baseline features ---
    try:
        morgan = get_morgan_fp(mol)
        maccs = get_maccs_fp(mol)
        desc = get_rdkit_descriptors(mol)
        features = np.concatenate([morgan, maccs, desc])
        basefeat_list.append(features)
        valid_indices.append(i)
    except Exception as e:
        print(f"⚠️ Feature extraction failed for {smi}: {e}")
print(f"✅ Molecule conversion complete: {len(valid_indices)} valid molecules.")
# ---------------------------------------
# 5. Save graph datasets according to split
def subset(lst, idx):
    n = len(lst)
    return [lst[i] for i in idx if i < n and lst[i] is not None]
torch.save(subset(graph_list, train_indices), "tox21_processed/tox21_train.pt")
torch.save(subset(graph_list, valid_indices), "tox21_processed/tox21_valid.pt")
torch.save(subset(graph_list, test_indices), "tox21_processed/tox21_test.pt")
print("💾 Saved graph datasets (.pt)")
# ---------------------------------------
# 6. Save baseline feature table
df_valid = df.iloc[valid_indices].reset_index(drop=True)
feature_matrix = np.vstack([basefeat_list[i] for i in range(len(basefeat_list
))])
feature_cols = (
    [f"morgan_{i}" for i in range(1024)]
    + [f"maccs_{i}" for i in range(167)]
    + desc_names
)
feat_df = pd.DataFrame(feature_matrix, columns=feature_cols)
final_df = pd.concat([df_valid[["ids"] + tox_columns], feat_df], axis=1)
final_df.to_csv("tox21_processed/tox21_features.csv", index=False)
print("💾 Saved baseline features tox21_processed/tox21_features.csv")
print("\n🎉 Unified Tox21 preprocessing complete!")


# def atom_features(atom):
#     return torch.tensor(
#         [
#             atom.GetAtomicNum(),
#             atom.GetTotalDegree(),
#             atom.GetFormalCharge(),
#             atom.GetTotalNumHs(),
#             atom.GetIsAromatic(),
#         ],
#         dtype=torch.float,
#     )


# def bond_features(bond):
#     return torch.tensor(
#         [bond.GetBondTypeAsDouble(), bond.GetIsConjugated(), bond.IsInRing()],
#         dtype=torch.float,
#     )


# def mol_to_graph(mol):
#     x = torch.stack([atom_features(a) for a in mol.GetAtoms()])
#     edge_index, edge_attr = [], []
#     for b in mol.GetBonds():
#         i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
#         bf = bond_features(b)
#         edge_index += [[i, j], [j, i]]
#         edge_attr += [bf, bf]
#     if len(edge_index) == 0:
#         edge_index = torch.empty((2, 0), dtype=torch.long)
#         edge_attr = torch.empty((0, 3), dtype=torch.float)
#     else:
#         edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
#         edge_attr = torch.stack(edge_attr)
#     return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


# def get_morgan_fp(mol, radius=2, n_bits=1024):
#     fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
#     return np.array(fp, dtype=int)


# def get_maccs_fp(mol):
#     fp = MACCSkeys.GenMACCSKeys(mol)
#     return np.array(fp, dtype=int)


# def get_rdkit_descriptors(mol):
#     vals = []
#     for _, func in Descriptors._descList:
#         try:
#             vals.append(func(mol))
#         except Exception:
#             vals.append(np.nan)
#     return np.array(vals)


# desc_names = [d[0] for d in Descriptors._descList]

# # ---------------------------------------
# # 4. Featurize molecules
# # ---------------------------------------
# graph_list, basefeat_list, valid_indices = [], [], []

# print("🔬 Converting molecules to graphs & features...")
# for i, row in tqdm(df.iterrows(), total=len(df), desc="Featurizing molecules"):
#     smi = row["ids"]
#     mol = Chem.MolFromSmiles(smi)
#     if mol is None:
#         print(f"⚠️ Invalid SMILES at index {i}: {smi}")
#         continue

#     # --- Graph ---
#     try:
#         data = mol_to_graph(mol)
#         y = torch.tensor(row[tox_columns].values, dtype=torch.float)
#         y[torch.isnan(y)] = -1  # missing labels
#         data.y = y
#         data.smiles = smi
#         graph_list.append(data)
#     except Exception as e:
#         print(f"⚠️ Graph failed for {smi}: {e}")
#         continue

#     # --- Baseline features ---
#     try:
#         morgan = get_morgan_fp(mol)
#         maccs = get_maccs_fp(mol)
#         desc = get_rdkit_descriptors(mol)
#         features = np.concatenate([morgan, maccs, desc])
#         basefeat_list.append(features)
#         valid_indices.append(i)
#     except Exception as e:
#         print(f"⚠️ Feature extraction failed for {smi}: {e}")

# print(f"✅ Molecule conversion complete: {len(valid_indices)} valid molecules.")

# # ---------------------------------------
# # 5. Save graph datasets according to split
# # ---------------------------------------
# def subset(lst, idx):
#     n = len(lst)
#     return [lst[i] for i in idx if i < n and lst[i] is not None]


# torch.save(subset(graph_list, train_indices), "tox21_processed/tox21_train.pt")
# torch.save(subset(graph_list, valid_indices), "tox21_processed/tox21_valid.pt")
# torch.save(subset(graph_list, test_indices), "tox21_processed/tox21_test.pt")
# print("💾 Saved graph datasets (.pt)")

# # ---------------------------------------
# # 6. Save baseline feature table
# # ---------------------------------------
# df_valid = df.iloc[valid_indices].reset_index(drop=True)
# feature_matrix = np.vstack([basefeat_list[i] for i in range(len(basefeat_list))])
# feature_cols = (
#     [f"morgan_{i}" for i in range(1024)]
#     + [f"maccs_{i}" for i in range(167)]
#     + desc_names
# )
# feat_df = pd.DataFrame(feature_matrix, columns=feature_cols)
# final_df = pd.concat([df_valid[["ids"] + tox_columns], feat_df], axis=1)
# final_df.to_csv("tox21_processed/tox21_features.csv", index=False)
# print("💾 Saved baseline features tox21_processed/tox21_features.csv")

# print("\n🎉 Unified Tox21 preprocessing complete!")
# print("Files created in ./tox21_processed/:")
# print(" - tox21_train.pt, tox21_valid.pt, tox21_test.pt (for GNNs)")
# print(" - tox21_features.csv (for ML baselines)")
# print(" - train_smiles.txt / valid_smiles.txt / test_smiles.txt")
