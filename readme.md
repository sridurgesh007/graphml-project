# CS5284 Project
## Molecular Toxicity Prediction with Graph Neural Networks

To recreate the environment run in the PowerShell:
`conda env create -f tox21_gnn_env.yaml
conda activate tox21_gnn`

Dataset can be extracted by unzipping the 'tox21_preprocessed.zip' file.

The molecular features are standerdized with StandarScaler. No scaling is yet done on the node features of the graph data.

📂 Data Access & Preparation
Due to GitHub file size limits, the preprocessed datasets, graph objects, and model checkpoints are hosted externally.

Step 1: Download Data
Please download the required files from the following Google Drive link:

👉 [https://drive.google.com/drive/folders/1VFI8eS-SUUkvcUi4scdOVY5Ijq5J8boB?usp=sharing]

File structre to run Text-GNN models :
```text
GNN-TEXT/
│
├── 📁 graphs/                  # PyTorch Geometric Graph Objects (.pt)
│   ├── train_2d.pt             # Serialized graph list for training set
│   ├── val_2d.pt               # Serialized graph list for validation set
│   └── test_2d.pt              # Serialized graph list for test set
│
├── 📁 processed/               # Tabular Data & Feature Vectors
    ├── train_clean.csv         # Cleaned SMILES, Labels (12 tasks), and Weights
    ├── val_clean.csv
    ├── test_clean.csv
    │
    ├── train_ecfp4.npz         # Compressed NumPy archive: ECFP4 Fingerprints (2048-bit)
    ├── val_ecfp4.npz
    ├── test_ecfp4.npz
    │
    ├── train_rdkit_desc.npz    # Compressed NumPy archive: Standardized RDKit Descriptors
    ├── val_rdkit_desc.npz
    └── test_rdkit_desc.npz





