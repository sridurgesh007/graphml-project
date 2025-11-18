import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, GINConv, global_max_pool

class EarlyFusionModel(nn.Module):
    """
    A GNN-based model that fuses three types of input features.
    This version is modified to accept hyperparameters for tuning.
    """
    def __init__(self, 
                 graph_in_dim, graph_edge_dim, 
                 fp_in_dim, 
                 desc_in_dim, 
                 num_tasks=12,
                 
                 # --- NEW GNN Hyperparameters ---
                 gnn_type='gat',      # <-- NEW: 'gat' or 'gin'
                 graph_hidden_dim=128,
                 graph_out_dim=64,
                 gat_heads=4,
                 
                 # --- Common Hyperparameters ---
                 fp_out_dim=256,
                 desc_out_dim=64,
                 gnn_dropout=0.1,
                 classifier_dropout_1=0.5,
                 classifier_dropout_2=0.25
                 ):
        """
        Initialize the model components.
        
        Args:
            graph_in_dim (int): Dimensionality of node features
            graph_edge_dim (int): Dimensionality of edge features
            fp_in_dim (int): Dimensionality of fingerprint
            desc_in_dim (int): Dimensionality of descriptors
            graph_hidden_dim (int): Hidden size for GNN layers
            graph_out_dim (int): Output size of the GNN branch
            fp_out_dim (int): Output size of the fingerprint branch
            desc_out_dim (int): Output size of the descriptor branch
            num_tasks (int): Number of output tasks (12 for Tox21)
            gat_heads (int): Number of attention heads for GATv2
            gnn_dropout (float): Dropout rate in GNN layers
            classifier_dropout_1 (float): Dropout rate in classifier (first layer)
            classifier_dropout_2 (float): Dropout rate in classifier (second layer)
        """
        
        super(EarlyFusionModel, self).__init__()
        self.gnn_type = gnn_type

        # --- 1. Graph Branch ---
        if self.gnn_type == 'gat':
            self.conv1 = GATv2Conv(graph_in_dim, graph_hidden_dim, heads=gat_heads, dropout=gnn_dropout, edge_dim=graph_edge_dim)
            self.conv2 = GATv2Conv(graph_hidden_dim * gat_heads, graph_hidden_dim, heads=gat_heads, dropout=gnn_dropout, edge_dim=graph_edge_dim)
            self.graph_out_linear = nn.Linear(graph_hidden_dim * gat_heads, graph_out_dim)
        
        elif self.gnn_type == 'gin':
            # GIN requires an MLP for its "apply" function. A simple 2-layer MLP is standard.
            gin_mlp1 = nn.Sequential(
                nn.Linear(graph_in_dim, graph_hidden_dim),
                nn.ReLU(),
                nn.Linear(graph_hidden_dim, graph_hidden_dim),
                nn.BatchNorm1d(graph_hidden_dim) # BatchNorm is crucial for GIN
            )
            gin_mlp2 = nn.Sequential(
                nn.Linear(graph_hidden_dim, graph_hidden_dim),
                nn.ReLU(),
                nn.Linear(graph_hidden_dim, graph_hidden_dim),
                nn.BatchNorm1d(graph_hidden_dim)
            )
            self.conv1 = GINConv(gin_mlp1, train_eps=True)
            self.conv2 = GINConv(gin_mlp2, train_eps=True)
            self.graph_out_linear = nn.Linear(graph_hidden_dim, graph_out_dim) # No heads for GIN
        
        else:
            raise ValueError(f"Unknown GNN type: {self.gnn_type}")
            
        self.graph_bn = nn.BatchNorm1d(graph_out_dim)

        # --- 2. Fingerprint Branch ---
# ... existing code ...
        self.fp_linear1 = nn.Linear(fp_in_dim, fp_out_dim)
        self.fp_bn = nn.BatchNorm1d(fp_out_dim)
        
        # --- 3. Descriptor Branch ---
        self.desc_linear1 = nn.Linear(desc_in_dim, desc_out_dim)
# ... existing code ...
        self.desc_bn = nn.BatchNorm1d(desc_out_dim)

        # --- 4. Classifier Head ---
        total_in_dim = graph_out_dim + fp_out_dim + desc_out_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(total_in_dim, total_in_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(total_in_dim // 2),
            nn.Dropout(classifier_dropout_1),
            
            nn.Linear(total_in_dim // 2, total_in_dim // 4),
            nn.ReLU(),
            nn.BatchNorm1d(total_in_dim // 4),
            nn.Dropout(classifier_dropout_2),
            
            nn.Linear(total_in_dim // 4, num_tasks)
        )

    def forward(self, graph_data):
        # 1. Process Graph
        x, edge_index, edge_attr, batch = graph_data.x, graph_data.edge_index, graph_data.edge_attr, graph_data.batch
        
        if self.gnn_type == 'gat':
            x = F.relu(self.conv1(x, edge_index, edge_attr))
            x = F.relu(self.conv2(x, edge_index, edge_attr))
        elif self.gnn_type == 'gin':
            # GIN does not use edge_attr in this simple form
            x = F.relu(self.conv1(x, edge_index))
            x = F.relu(self.conv2(x, edge_index))
            
        x_graph = global_max_pool(x, batch)
        x_graph = F.relu(self.graph_out_linear(x_graph))
        x_graph = self.graph_bn(x_graph)

        # 2. Process Fingerprint
# ... existing code ...
        x_fp = F.relu(self.fp_linear1(graph_data.fp))
        x_fp = self.fp_bn(x_fp)
        
        # 3. Process Descriptors
        x_desc = F.relu(self.desc_linear1(graph_data.desc))
# ... existing code ...
        x_desc = self.desc_bn(x_desc)
        
        # 4. Concatenate and Classify
        x_combined = torch.cat([x_graph, x_fp, x_desc], dim=1)
        out = self.classifier(x_combined)
        return out


# --- Example Usage (for testing the file) ---
if __name__ == '__main__':
    from build_fusion_dataset import get_atom_features
    from rdkit import Chem
    from torch_geometric.data import Data, Batch
    
    # 1. Get feature dimensions
    dummy_mol = Chem.MolFromSmiles('C')
    dummy_atom = dummy_mol.GetAtomWithIdx(0)
    NODE_DIM = len(get_atom_features(dummy_atom))
    FP_DIM = 2048
    DESC_DIM = 8 
    NUM_TASKS = 12
    EDGE_DIM = 10

    print(f"Model Configuration:")
    print(f"  Node features:    {NODE_DIM}")
    print(f"  Fingerprint dim:  {FP_DIM}")
    print(f"  Descriptor dim:   {DESC_DIM}")
    print(f"  Output tasks:     {NUM_TASKS}")

    # 2. Instantiate the model (now requires new args)
    model = EarlyFusionModel(
        graph_in_dim=NODE_DIM,
        graph_edge_dim=EDGE_DIM,
        fp_in_dim=FP_DIM,
        desc_in_dim=DESC_DIM,
        num_tasks=NUM_TASKS,
        gat_heads=4,
        gnn_dropout=0.2,
        classifier_dropout_1=0.5,
        classifier_dropout_2=0.3
    )
    
    print("\nGAT Model architecture:")
    print(model)
    
    # 3. Test with a dummy Batch object
    data1 = Data(
        x=torch.randn(3, NODE_DIM), edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t(),
        fp=torch.randn(1, FP_DIM), desc=torch.randn(1, DESC_DIM)
    )
    data2 = Data(
        x=torch.randn(4, NODE_DIM), edge_index=torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.long).t(),
        fp=torch.randn(1, FP_DIM), desc=torch.randn(1, DESC_DIM)
    )
    batch = Batch.from_data_list([data1, data2])
    
    print(f"\nTesting GAT model with dummy batch:")
    output_logits = model(batch)
    
    print(f"Output logits shape: {output_logits.shape}")
    assert output_logits.shape == (2, NUM_TASKS)

    # --- Test GIN ---
    print("\nTesting GIN Model...")
    GIN_MODEL = EarlyFusionModel(
        graph_in_dim=NODE_DIM,
        graph_edge_dim=EDGE_DIM,
        fp_in_dim=FP_DIM,
        desc_in_dim=DESC_DIM,
        num_tasks=NUM_TASKS,
        gnn_type='gin',
        graph_hidden_dim=128,
        graph_out_dim=64,
        fp_out_dim=256,
        desc_out_dim=64,
        gnn_dropout=0.2,
        classifier_dropout_1=0.5,
        classifier_dropout_2=0.3
    )
    print("\nGIN Model architecture:")
    print(GIN_MODEL)

    print(f"Dummy Batch (GIN): {batch}")
    out_gin = GIN_MODEL(batch)
    print(f"Output shape (GIN): {out_gin.shape}")

    assert out_gin.shape == (2, NUM_TASKS)
    print("✅ Forward pass successful!")