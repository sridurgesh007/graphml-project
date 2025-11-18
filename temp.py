# gatv2conv with max and mean pool
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import (
#     GATv2Conv, 
#     global_mean_pool, 
#     global_max_pool  # <-- 1. Import global_max_pool
# )

# class EarlyFusionModel(nn.Module):
#     """
#     An "Early Fusion" model that processes three branches of features:
#     1. A GNN branch for graph-level features.
#     2. An MLP branch for molecular fingerprints (ECFP).
#     3. An MLP branch for molecular descriptors (RDKit 2D/3D).
    
#     The outputs of these branches are concatenated and passed through a
#     final classifier.
    
#     This version is "tunable" via constructor arguments.
#     """
#     def __init__(self, 
#                  # Dimensionality
#                  node_feature_dim, 
#                  edge_feature_dim, 
#                  fp_feature_dim,
#                  desc_feature_dim,
#                  n_tasks,
#                  # GNN Hyperparams
#                  graph_hidden_dim=128,
#                  graph_out_dim=64,
#                  gat_heads=4,
#                  gnn_dropout=0.2,
#                  # FP MLP Hyperparams
#                  fp_out_dim=256,
#                  # Classifier Hyperparams
#                  classifier_dropout_1=0.5,
#                  classifier_dropout_2=0.3
#                 ):
        
#         super(EarlyFusionModel, self).__init__()

#         # --- 1. GNN Branch ---
        
#         # First GNN layer
#         self.gnn_conv1 = GATv2Conv(
#             node_feature_dim, 
#             graph_hidden_dim, 
#             heads=gat_heads, 
#             dropout=gnn_dropout
#         )
        
#         # Second GNN layer
#         # Input dim is hidden_dim * heads from the previous layer
#         self.gnn_conv2 = GATv2Conv(
#             graph_hidden_dim * gat_heads, 
#             graph_hidden_dim, 
#             heads=gat_heads, 
#             dropout=gnn_dropout
#         )
        
#         # GNN's readout MLP
        
#         # <-- 2. UPDATE: Input dim is now 2x due to Mean + Max pooling
#         gnn_mlp_in_dim = (graph_hidden_dim * gat_heads) * 2 
        
#         self.graph_fc = nn.Sequential(
#             nn.Linear(gnn_mlp_in_dim, graph_out_dim),
#             nn.ReLU(),
#             nn.Dropout(gnn_dropout)
#         )

#         # --- 2. Fingerprint (FP) Branch ---
#         self.fp_fc = nn.Sequential(
#             nn.Linear(fp_feature_dim, fp_out_dim),
#             nn.ReLU(),
#             nn.Dropout(classifier_dropout_1)
#         )

#         # --- 3. Descriptor (Desc) Branch ---
#         # Descriptors are often lower-dim, so we use a simpler/no MLP
#         # We'll just use the raw descriptor features.
#         # If you want to process them, uncomment the following:
#         # self.desc_fc = nn.Sequential(
#         #     nn.Linear(desc_feature_dim, desc_out_dim),
#         #     nn.ReLU(),
#         #     nn.Dropout(classifier_dropout_1)
#         # )
#         # desc_output_size = desc_out_dim
        
#         # For now, we'll just use the raw features
#         desc_output_size = desc_feature_dim

#         # --- 4. Final Classifier Head (Fusion) ---
        
#         # This is the "Early Fusion" step
#         classifier_in_dim = graph_out_dim + fp_out_dim + desc_output_size
        
#         self.classifier = nn.Sequential(
#             nn.Linear(classifier_in_dim, classifier_in_dim // 2),
#             nn.ReLU(),
#             nn.Dropout(classifier_dropout_2),
#             nn.Linear(classifier_in_dim // 2, n_tasks)
#         )

#     def forward_graph(self, x, edge_index, edge_attr, batch):
#         """Helper function for the GNN branch."""
#         # 1. GNN Conv 1
#         x = self.gnn_conv1(x, edge_index)
#         x = F.elu(x)
#         x = F.dropout(x, p=self.gnn_dropout, training=self.training)
        
#         # 2. GNN Conv 2
#         x = self.gnn_conv2(x, edge_index)
#         x = F.elu(x)
        
#         # 3. Readout (Pooling)
        
#         # <-- 3. UPDATE: Apply both mean and max pooling
#         mean_pool = global_mean_pool(x, batch) # [batch_size, graph_hidden_dim * gat_heads]
#         max_pool = global_max_pool(x, batch)   # [batch_size, graph_hidden_dim * gat_heads]
        
#         # Concatenate the two pooling results
#         graph_out = torch.cat([mean_pool, max_pool], dim=1) # [batch_size, 2 * (graph_hidden_dim * gat_heads)]
        
#         # 4. Pass through GNN's MLP
#         graph_out = self.graph_fc(graph_out)
#         return graph_out

#     def forward(self, x, edge_index, edge_attr, fp_features, desc_features, batch):
#         """Full forward pass of the fusion model."""
        
#         # 1. GNN Branch
#         graph_out = self.forward_graph(x, edge_index, edge_attr, batch)
        
#         # 2. FP Branch
#         fp_out = self.fp_fc(fp_features)
        
#         # 3. Descriptor Branch
#         # If you added an MLP, you would pass it through:
#         # desc_out = self.desc_fc(desc_features)
        
#         # For now, we just use the raw features:
#         desc_out = desc_features
        
#         # 4. Fusion
#         # Concatenate all feature vectors
#         fused_vector = torch.cat([graph_out, fp_out, desc_out], dim=1)
        
#         # 5. Final Classification
#         out = self.classifier(fused_vector)
#         return out

# # --- Self-Test Block ---
# if __name__ == "__main__":
#     # This block allows you to run `python model.py` to test it
#     print("--- Running Model Self-Test ---")

#     # Mock dimensions
#     N_NODES_AVG = 30
#     BATCH_SIZE = 16
#     N_TASKS = 12
    
#     # Feature dims (must match build_fusion_dataset.py)
#     NODE_DIM = 40  # (Update this if get_atom_features changes)
#     EDGE_DIM = 10
#     FP_DIM = 2048
#     DESC_DIM = 200 # (Update this based on your descriptor list)

#     # Mock Hyperparameters
#     params = {
#         "node_feature_dim": NODE_DIM,
#         "edge_feature_dim": EDGE_DIM,
#         "fp_feature_dim": FP_DIM,
#         "desc_feature_dim": DESC_DIM,
#         "n_tasks": N_TASKS,
#         "graph_hidden_dim": 64,
#         "graph_out_dim": 32,
#         "gat_heads": 2,
#         "gnn_dropout": 0.1,
#         "fp_out_dim": 128,
#         "classifier_dropout_1": 0.3,
#         "classifier_dropout_2": 0.2
#     }

#     try:
#         model = EarlyFusionModel(**params)
#         print("Model initialized successfully.")
        
#         # Create mock data (simulating a batch from DataLoader)
#         # Note: In a real batch, n_nodes and n_edges are combined
#         n_nodes_total = N_NODES_AVG * BATCH_SIZE
        
#         x = torch.randn(n_nodes_total, NODE_DIM)
#         # Create a simple, fully-connected-like edge index
#         edge_index = torch.randint(0, n_nodes_total, (2, n_nodes_total * 2), dtype=torch.long)
#         edge_attr = torch.randn(n_nodes_total * 2, EDGE_DIM)
        
#         fp_features = torch.randn(BATCH_SIZE, FP_DIM)
#         desc_features = torch.randn(BATCH_SIZE, DESC_DIM)
        
#         # Create the batch vector
#         batch = torch.tensor([i for i in range(BATCH_SIZE) for _ in range(N_NODES_AVG)], dtype=torch.long)

#         print(f"Mock Data Shapes (simulating batch size {BATCH_SIZE}):")
#         print(f"  x (nodes):        {x.shape}")
#         print(f"  edge_index:       {edge_index.shape}")
#         print(f"  edge_attr:        {edge_attr.shape}")
#         print(f"  fp_features:      {fp_features.shape}")
#         print(f"  desc_features:    {desc_features.shape}")
#         print(f"  batch:            {batch.shape}")

#         # Test forward pass
#         with torch.no_grad():
#             output = model(x, edge_index, edge_attr, fp_features, desc_features, batch)
        
#         print("\nForward pass successful.")
#         print(f"Output shape: {output.shape} (Expected: [{BATCH_SIZE}, {N_TASKS}])")
        
#         assert output.shape == (BATCH_SIZE, N_TASKS)
        
#         print("\n--- Model Self-Test Passed! ---")
        
#     except Exception as e:
#         print(f"\n--- Model Self-Test FAILED ---")
#         print(f"Error: {e}")
#         import traceback
#         traceback.print_exc()

# gineconv with max and mean pool
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GINEConv,  # <-- 1. Import GINEConv
    global_mean_pool, 
    global_max_pool
)

class EarlyFusionModel(nn.Module):
    """
    An "Early Fusion" model that processes three branches of features:
    1. A GNN branch (using GINEConv) for graph-level features.
    2. An MLP branch for molecular fingerprints (ECFP).
    3. An MLP branch for molecular descriptors (RDKit 2D).
    
    The outputs of these branches are concatenated and passed through a
    final classifier.
    
    This version is "tunable" via constructor arguments.
    """
    def __init__(self, 
                 # Dimensionality
                 node_feature_dim, 
                 edge_feature_dim, 
                 fp_feature_dim,
                 desc_feature_dim,
                 n_tasks,
                 # GNN Hyperparams
                 graph_hidden_dim=128,
                 graph_out_dim=64,
                 # gat_heads is no longer needed for GINEConv
                 gnn_dropout=0.2,
                 # FP MLP Hyperparams
                 fp_out_dim=256,
                 # Classifier Hyperparams
                 classifier_dropout_1=0.5,
                 classifier_dropout_2=0.3
                ):
        
        super(EarlyFusionModel, self).__init__()

        # --- 1. GNN Branch (Using GINEConv) ---
        
        # GINEConv requires a small MLP to process the aggregated features.
        # This is how it learns to combine node + edge features.
        
        # MLP for first GINEConv layer
        nn1 = nn.Sequential(
            nn.Linear(node_feature_dim, graph_hidden_dim),
            nn.ReLU(),
            nn.Linear(graph_hidden_dim, graph_hidden_dim)
        )
        self.gnn_conv1 = GINEConv(nn1, edge_dim=edge_feature_dim)
        
        # MLP for second GINEConv layer
        nn2 = nn.Sequential(
            nn.Linear(graph_hidden_dim, graph_hidden_dim),
            nn.ReLU(),
            nn.Linear(graph_hidden_dim, graph_hidden_dim)
        )
        self.gnn_conv2 = GINEConv(nn2, edge_dim=edge_feature_dim)
        
        self.gnn_batch_norm1 = nn.BatchNorm1d(graph_hidden_dim)
        self.gnn_batch_norm2 = nn.BatchNorm1d(graph_hidden_dim)

        self.gnn_dropout = gnn_dropout
        # GNN's readout MLP
        
        # Input dim is 2x due to Mean + Max pooling
        # The output of the final GINEConv is graph_hidden_dim
        gnn_mlp_in_dim = graph_hidden_dim * 2 
        
        self.graph_fc = nn.Sequential(
            nn.Linear(gnn_mlp_in_dim, graph_out_dim),
            nn.ReLU(),
            nn.Dropout(gnn_dropout)
        )

        # --- 2. Fingerprint (FP) Branch ---
        self.fp_fc = nn.Sequential(
            nn.Linear(fp_feature_dim, fp_out_dim),
            nn.ReLU(),
            nn.Dropout(classifier_dropout_1)
        )

        # --- 3. Descriptor (Desc) Branch ---
        # For now, we'll just use the raw features
        desc_output_size = desc_feature_dim

        # --- 4. Final Classifier Head (Fusion) ---
        
        # This is the "Early Fusion" step
        classifier_in_dim = graph_out_dim + fp_out_dim + desc_output_size
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in_dim, classifier_in_dim // 2),
            nn.ReLU(),
            nn.Dropout(classifier_dropout_2),
            nn.Linear(classifier_in_dim // 2, n_tasks)
        )

    def forward_graph(self, x, edge_index, edge_attr, batch):
        """Helper function for the GNN branch."""
        
        # 1. GNN Conv 1
        x = self.gnn_conv1(x, edge_index, edge_attr)
        x = self.gnn_batch_norm1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.gnn_dropout, training=self.training)
        
        # 2. GNN Conv 2
        x = self.gnn_conv2(x, edge_index, edge_attr)
        x = self.gnn_batch_norm2(x)
        x = F.elu(x)
        
        # 3. Readout (Pooling)
        mean_pool = global_mean_pool(x, batch) # [batch_size, graph_hidden_dim]
        max_pool = global_max_pool(x, batch)   # [batch_size, graph_hidden_dim]
        
        # Concatenate the two pooling results
        graph_out = torch.cat([mean_pool, max_pool], dim=1) # [batch_size, 2 * graph_hidden_dim]
        
        # 4. Pass through GNN's MLP
        graph_out = self.graph_fc(graph_out)
        return graph_out

    def forward(self, graph_data):
        """Full forward pass of the fusion model."""
        x, edge_index, edge_attr, fp_features, desc_features, batch = graph_data.x, graph_data.edge_index, graph_data.edge_attr, graph_data.fp, graph_data.desc, graph_data.batch

        # 1. GNN Branch
        graph_out = self.forward_graph(x, edge_index, edge_attr, batch)
        
        # 2. FP Branch
        fp_out = self.fp_fc(fp_features)
        
        # 3. Descriptor Branch
        desc_out = desc_features
        
        # 4. Fusion
        fused_vector = torch.cat([graph_out, fp_out, desc_out], dim=1)
        
        # 5. Final Classification
        out = self.classifier(fused_vector)
        return out

# --- Self-Test Block ---
if __name__ == "__main__":
    print("--- Running Model Self-Test (with GINEConv) ---")

    # Mock dimensions
    N_NODES_AVG = 30
    BATCH_SIZE = 16
    N_TASKS = 12
    
    # Feature dims (must match build_fusion_dataset.py)
    NODE_DIM = 40  # From our atom featurizer
    EDGE_DIM = 10  # From our bond featurizer
    FP_DIM = 2048
    DESC_DIM = 200 # From our 2D descriptor list

    # Mock Hyperparameters
    params = {
        "node_feature_dim": NODE_DIM,
        "edge_feature_dim": EDGE_DIM,
        "fp_feature_dim": FP_DIM,
        "desc_feature_dim": DESC_DIM,
        "n_tasks": N_TASKS,
        "graph_hidden_dim": 64,
        "graph_out_dim": 32,
        "gnn_dropout": 0.1,
        "fp_out_dim": 128,
        "classifier_dropout_1": 0.3,
        "classifier_dropout_2": 0.2
    }

    try:
        model = EarlyFusionModel(**params)
        print("Model initialized successfully.")
        
        # Create mock data
        n_nodes_total = N_NODES_AVG * BATCH_SIZE
        x = torch.randn(n_nodes_total, NODE_DIM)
        edge_index = torch.randint(0, n_nodes_total, (2, n_nodes_total * 2), dtype=torch.long)
        edge_attr = torch.randn(n_nodes_total * 2, EDGE_DIM)
        fp_features = torch.randn(BATCH_SIZE, FP_DIM)
        desc_features = torch.randn(BATCH_SIZE, DESC_DIM)
        batch = torch.tensor([i for i in range(BATCH_SIZE) for _ in range(N_NODES_AVG)], dtype=torch.long)

        print(f"Mock Data Shapes (simulating batch size {BATCH_SIZE}):")
        print(f"  x (nodes):        {x.shape}")
        print(f"  edge_index:       {edge_index.shape}")
        print(f"  edge_attr:        {edge_attr.shape}")
        print(f"  fp_features:      {fp_features.shape}")
        print(f"  desc_features:    {desc_features.shape}")
        print(f"  batch:            {batch.shape}")

        # Test forward pass
        with torch.no_grad():
            output = model(x, edge_index, edge_attr, fp_features, desc_features, batch)
        
        print("\nForward pass successful.")
        print(f"Output shape: {output.shape} (Expected: [{BATCH_SIZE}, {N_TASKS}])")
        
        assert output.shape == (BATCH_SIZE, N_TASKS)
        
        print("\n--- Model Self-Test Passed! ---")
        
    except Exception as e:
        print(f"\n--- Model Self-Test FAILED ---")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()