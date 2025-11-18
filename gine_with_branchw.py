import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GINEConv,
    global_mean_pool, 
    global_max_pool
)

class EarlyFusionModel(nn.Module):
    # ... (init function signature is the same) ...
    def __init__(self, 
                 node_feature_dim, 
                 edge_feature_dim, 
                 fp_feature_dim,
                 desc_feature_dim,
                 n_tasks,
                 graph_hidden_dim=128,
                 graph_out_dim=64,
                 gnn_dropout=0.2,
                 fp_out_dim=256,
                 classifier_dropout_1=0.5,
                 classifier_dropout_2=0.3
                ):
        
        super(EarlyFusionModel, self).__init__()

        # --- 1. GNN Branch (Using GINEConv) ---
        # ... (GNN branch code is unchanged) ...
        nn1 = nn.Sequential(
            nn.Linear(node_feature_dim, graph_hidden_dim),
            nn.ReLU(),
            nn.Linear(graph_hidden_dim, graph_hidden_dim)
        )
        self.gnn_conv1 = GINEConv(nn1, edge_dim=edge_feature_dim)
        nn2 = nn.Sequential(
            nn.Linear(graph_hidden_dim, graph_hidden_dim),
            nn.ReLU(),
            nn.Linear(graph_hidden_dim, graph_hidden_dim)
        )
        self.gnn_conv2 = GINEConv(nn2, edge_dim=edge_feature_dim)
        self.gnn_batch_norm1 = nn.BatchNorm1d(graph_hidden_dim)
        self.gnn_batch_norm2 = nn.BatchNorm1d(graph_hidden_dim)

        self.gnn_dropout = gnn_dropout

        gnn_mlp_in_dim = graph_hidden_dim * 2 
        self.graph_fc = nn.Sequential(
            nn.Linear(gnn_mlp_in_dim, graph_out_dim),
            nn.ReLU(),
            nn.Dropout(gnn_dropout)
        )

        # --- 2. Fingerprint (FP) Branch ---
        # ... (FP branch code is unchanged) ...
        self.fp_fc = nn.Sequential(
            nn.Linear(fp_feature_dim, fp_out_dim),
            nn.ReLU(),
            nn.Dropout(classifier_dropout_1)
        )

        # --- 3. Descriptor (Desc) Branch ---
        desc_output_size = desc_feature_dim

        # --- 4. Learnable Branch Weights (NEW!) ---
        # We create a learnable parameter vector with 3 weights, one for each branch.
        # We initialize them all to 1.0.
        self.branch_weights = nn.Parameter(torch.ones(3))

        # --- 5. Final Classifier Head (Fusion) ---
        classifier_in_dim = graph_out_dim + fp_out_dim + desc_output_size
        
        # ... (Classifier code is unchanged) ...
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in_dim, 128),
            nn.ReLU(),
            nn.Dropout(classifier_dropout_2),
            nn.Linear(128, n_tasks)
        )

    # ... (forward_graph helper function is unchanged) ...
    def forward_graph(self, x, edge_index, edge_attr, batch):
        # GNN Conv 1
        x = self.gnn_conv1(x, edge_index, edge_attr)
        x = self.gnn_batch_norm1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.gnn_dropout, training=self.training)
        # GNN Conv 2
        x = self.gnn_conv2(x, edge_index, edge_attr)
        x = self.gnn_batch_norm2(x)
        x = F.elu(x)
        # Readout/Pooling
        mean_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)
        # Concatenate the two pooling results
        graph_out = torch.cat([mean_pool, max_pool], dim=1)
        # Pass through GNN's MLP
        graph_out = self.graph_fc(graph_out)
        return graph_out

    def forward(self, graph_data):
        
        x, edge_index, edge_attr, fp_features, desc_features, batch = graph_data.x, graph_data.edge_index, graph_data.edge_attr, graph_data.fp_features, graph_data.desc_features, graph_data.batch
        
        # 1. GNN Branch
        graph_out = self.forward_graph(x, edge_index, edge_attr, batch)
        
        # 2. FP Branch
        fp_out = self.fp_fc(fp_features)
        
        # 3. Descriptor Branch
        desc_out = desc_features
        
        # 4. Fusion (NEW: Apply learned weights!)
        # We multiply each branch output by its learned weight before concatenating.
        # This allows the model to scale the "importance" of each branch.
        fused_vector = torch.cat([
            graph_out * self.branch_weights[0],
            fp_out * self.branch_weights[1],
            desc_out * self.branch_weights[2]
        ], dim=1)
        
        # 5. Final Classification
        out = self.classifier(fused_vector)
        return out

# =============================================================================
# --- SELF-TEST BLOCK ---
# To run this test: `python model.py`
# =============================================================================
if __name__ == "__main__":
    
    print("--- Running Model Self-Test ---")
    
    # --- 1. Define Mock Dimensions ---
    B = 4  # Batch size
    N_TASKS = 12
    
    # Feature dimensions (must match build_fusion_dataset.py)
    NODE_DIM = 41
    EDGE_DIM = 11
    FP_DIM = 2048
    DESC_DIM = 200
    
    # Model hyperparameters
    GRAPH_HIDDEN = 128
    GRAPH_OUT = 64
    FP_OUT = 256

    # --- 2. Create a Dummy Batch (simulating PyG DataLoader) ---
    # We'll create 4 "molecules" of different sizes
    from torch_geometric.data import Data, Batch
    # Mol 1: 3 nodes, 2 edges
    d1 = Data(
        x=torch.rand(3, NODE_DIM),
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t().contiguous(),
        edge_attr=torch.rand(2, EDGE_DIM),
        fp=torch.rand(1, FP_DIM),
        desc=torch.rand(1, DESC_DIM),
        y=torch.rand(1, N_TASKS), # Not used in forward, but good to have
        w=torch.rand(1, N_TASKS)  # Not used in forward
    )
    
    # Mol 2: 5 nodes, 4 edges
    d2 = Data(
        x=torch.rand(5, NODE_DIM),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 2, 3, 4]], dtype=torch.long).t().contiguous(),
        edge_attr=torch.rand(4, EDGE_DIM),
        fp=torch.rand(1, FP_DIM),
        desc=torch.rand(1, DESC_DIM),
        y=torch.rand(1, N_TASKS),
        w=torch.rand(1, N_TASKS)
    )
    
    # Mol 3: 2 nodes, 1 edge
    d3 = Data(
        x=torch.rand(2, NODE_DIM),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long).t().contiguous(),
        edge_attr=torch.rand(1, EDGE_DIM),
        fp=torch.rand(1, FP_DIM),
        desc=torch.rand(1, DESC_DIM),
        y=torch.rand(1, N_TASKS),
        w=torch.rand(1, N_TASKS)
    )

    # Mol 4: 4 nodes, 3 edges
    d4 = Data(
        x=torch.rand(4, NODE_DIM),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long).t().contiguous(),
        edge_attr=torch.rand(3, EDGE_DIM),
        fp=torch.rand(1, FP_DIM),
        desc=torch.rand(1, DESC_DIM),
        y=torch.rand(1, N_TASKS),
        w=torch.rand(1, N_TASKS)
    )
    
    # Create a PyG Batch from this list
    data_list = [d1, d2, d3, d4]
    data_batch = Batch.from_data_list(data_list)
    
    print(f"Created a dummy batch of {B} graphs.")
    print(f"  Batch.x shape (total nodes):         {data_batch.x.shape}")
    print(f"  Batch.edge_index shape:              {data_batch.edge_index.shape}")
    print(f"  Batch.edge_attr shape (total edges): {data_batch.edge_attr.shape}")
    print(f"  Batch.fp_features shape:             {data_batch.fp_features.shape}")
    print(f"  Batch.desc_features shape:           {data_batch.desc_features.shape}")
    print(f"  Batch.batch vector shape:            {data_batch.batch.shape}")

    # --- 3. Instantiate Model ---
    model = EarlyFusionModel(
        node_feature_dim=NODE_DIM,
        edge_feature_dim=EDGE_DIM,
        fp_feature_dim=FP_DIM,
        desc_feature_dim=DESC_DIM,
        n_tasks=N_TASKS,
        graph_hidden_dim=GRAPH_HIDDEN,
        graph_out_dim=GRAPH_OUT,
        fp_out_dim=FP_OUT
    )
    
    model.train() # Set to training mode
    print("\nModel instantiated successfully.")

    # --- 4. Run Forward Pass ---
    try:
        out = model(data_batch)
        
        print("\n--- TEST SUCCESSFUL ---")
        print(f"Forward pass ran without errors.")
        print(f"Input batch size:  {B}")
        print(f"Output shape:      {out.shape}")
        
        # Check if the output shape is correct
        assert out.shape == (B, N_TASKS)
        print("Output shape is correct! (Batch Size, Num Tasks)")
        
    except Exception as e:
        print("\n--- TEST FAILED ---")
        print(f"An error occurred during the forward pass:")
        print(e)
        import traceback
        traceback.print_exc()