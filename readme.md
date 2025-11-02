# CS5284 Project
## Molecular Toxicity Prediction with Graph Neural Networks

To recreate the environment run in the PowerShell:
`conda env create -f tox21_gnn_env.yaml
conda activate tox21_gnn`

Dataset can be extracted by unzipping the 'tox21_preprocessed.zip' file.

The molecular features are standerdized with StandarScaler. No scaling is yet done on the node features of the graph data.
