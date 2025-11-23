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

File structre to run below mentioned cells:
1. GINE Assay Multi-Head Cross Attention
2. PNA Cross Attention
3. Text Enhanced Weighted Assay GINE (MLP Projection)
4. Graph Transformer with Positional Encoding (Text+Graph)
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
```
This structure reflects the contents of your graphml_llm folder, including the inference scripts, training logs, and the critical checkpoint files required to run the Glass-Box model.

📂 GNN-LLM Directory Structure
To run the Glass-Box Multimodal LLM, ensure your graphml_llm directory matches the structure below. This module contains the Python scripts for training/inference and the LoRA checkpoints for the DeepSeek-R1 backbone.

```text
graphml_llm/
│
├── 📁 python_files/            # Core Scripts
│   ├── llm_train.py            # Main training loop (LoRA + Multi-Head Loss)
│   └── inference.py            # Inference script for generating explanations
│
├── 📁 sample/                  # Sample Outputs
│   └── glassbox_final_val_results_graphhead.json  # Validation results from Graph Head
│
├── 📁 log files/               # Training Logs
│   ├── fusion_train_289517.log
│   ├── fusion_train_289774.log
│   └── fusion_train_289876.log
│
├── 📁 checkpoints_labelaux_fixed/
  └── 📁 checkpoint_best/     #  Best Model Artifacts (Required for Inference)
       ├── adapter_config.json # LoRA configuration
       ├── graph_head.pt       # Saved weights for the Graph Prediction Head
       ├── llm_head.pt         # Saved weights for the LLM Prediction Head
       ├── special_tokens_map.json
       ├── tokenizer.json      # Custom tokenizer with <GRAPH> tokens
       ├── tokenizer_config.json
       └── README.md           # Model specific documentation
For checkpoint safe tensor file and projector.pt please check google drive link
Google Drive/
│
├── 📁 graph_ml/
   │
   └── 📁 Safe tensor file/        # Critical LLM Checkpoints
       ├── adapter_model.safetensors  # LoRA Adapter weights for DeepSeek-R1
       └── projector.pt               # Trained Graph-to-LLM Projector weights
```


