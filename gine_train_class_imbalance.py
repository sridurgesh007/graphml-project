import torch
import torch.nn as nn
import torch.optim as optim
# ... (other imports are the same) ...
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score
import numpy as np
import os
import joblib
from gine_with_branchw import EarlyFusionModel

# --- Configuration ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PROCESSED_DATA_DIR = "processed_fusion_data_3d"
N_TASKS = 12
MODEL_SAVE_PATH = "test_run_best_model_class_imbalance_3d.pth"
N_EPOCHS = 200
EARLY_STOP_PATIENCE = 15
TASK_NAMES = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
    "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
]
DEFAULT_PARAMS = {
    'lr': 1e-4, 'weight_decay': 1e-5, 'batch_size': 64, 'gnn_dropout': 0.3,
    'classifier_dropout_1': 0.2, 'classifier_dropout_2': 0.5,
    'graph_hidden_dim': 128, 'graph_out_dim': 64, 'fp_out_dim': 256
}

# ... --- Load Scalers (for model dimensions) ...
try:
    node_scaler = joblib.load(os.path.join(PROCESSED_DATA_DIR, "node_feature_scaler.joblib"))
    desc_imputer = joblib.load(os.path.join(PROCESSED_DATA_DIR, "desc_feature_imputer.joblib"))
    desc_scaler = joblib.load(os.path.join(PROCESSED_DATA_DIR, "desc_feature_scaler.joblib"))
except FileNotFoundError:
    print("Error: Scaler/imputer files not found. Please run 'build_fusion_dataset_3d.py' first.")
    exit()

# --- Determine Feature Dimensions ---
try:
    _temp_data_list = torch.load(os.path.join(PROCESSED_DATA_DIR, "train_data.pt"), weights_only=False)
    if not _temp_data_list: exit()
    _temp_data = _temp_data_list[0]
    NODE_FEATURE_DIM = _temp_data.x.shape[1]
    EDGE_FEATURE_DIM = _temp_data.edge_attr.shape[1]
    FP_FEATURE_DIM = _temp_data.fp_features.shape[1]
    DESC_FEATURE_DIM = _temp_data.desc_features.shape[1]
    print(f"--- Feature Dimensions Detected ---")
    print(f"Node Features:   {NODE_FEATURE_DIM}")
    print(f"Edge Features:   {EDGE_FEATURE_DIM}")
    print(f"FP Features:     {FP_FEATURE_DIM}")
    print(f"Desc Features:   {DESC_FEATURE_DIM}")
    print(f"---------------------------------")
    train_data_list = _temp_data_list
    valid_data_list = torch.load(os.path.join(PROCESSED_DATA_DIR, "valid_data.pt"), weights_only=False)
    test_data_list = torch.load(os.path.join(PROCESSED_DATA_DIR, "test_data.pt"), weights_only=False)
    del _temp_data, _temp_data_list
except Exception as e:
    print(f"Error loading data: {e}")
    exit()
except Exception as e:
    print(f"Error loading data for dimension check: {e}")
    exit()


# --- 1. Calculate pos_weight for Class Imbalance (NEW!) ---
def calculate_pos_weights(data_list):
    num_pos = torch.zeros(N_TASKS)
    num_neg = torch.zeros(N_TASKS)
    
    for data in data_list:
        labels = data.y.squeeze() # Shape [12]
        weights = data.w.squeeze() # Shape [12]
        is_valid = (weights > 0) & (~torch.isnan(labels))
        
        pos_mask = (labels == 1) & is_valid
        neg_mask = (labels == 0) & is_valid
        
        # --- THIS IS THE FIX ---
        # We add the boolean mask (shape [12]) directly.
        # This correctly adds 1s and 0s element-wise.
        # The old code had `.sum(dim=0)`, which was a bug.
        num_pos += pos_mask
        num_neg += neg_mask
        # --- END OF FIX ---

    pos_weight = num_neg / (num_pos + 1e-6)
    
    # We clip the weights to be at most 15. This stops runaway gradients.
    # pos_weight = torch.clamp(pos_weight, min=1.0, max=15.0)

    print("--- Class Imbalance (unCLIPPED pos_weight) Calculated ---")
    for name, weight in zip(TASK_NAMES, pos_weight):
        print(f"  {name:<16}: {weight:.2f}") # Now these will all be different!
    print("-----------------------------------------------")
    
    return pos_weight.to(DEVICE)

# Calculate weights *only* from the training set
pos_weight_tensor = calculate_pos_weights(train_data_list)


# --- 2. Update Loss Function to use pos_weight (NEW!) ---
def weighted_bce_loss(y_pred, y_true, weights, pos_weight):
    """
    Our full, robust loss function.
    - `weights` handles MISSING labels (w=0).
    - `pos_weight` handles CLASS IMBALANCE (rare positives).
    """
    # pos_weight is shape [12], we give it to BCEWithLogitsLoss
    loss_fn = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)
    
    raw_loss = loss_fn(y_pred, y_true)
    
    # Mask out NaN labels
    is_valid = ~torch.isnan(y_true)
    raw_loss = torch.where(is_valid, raw_loss, torch.zeros_like(raw_loss))
    
    # Apply the missing-label weights
    weighted_loss = raw_loss * weights
    
    # Normalize by the sum of weights
    total_weight = weights.sum()
    final_loss = weighted_loss.sum() / (total_weight + 1e-8)
    return final_loss

# ... (eval_model is unchanged) ...
@torch.no_grad()
def eval_model(model, loader, print_scores=False):
    model.eval()
    all_preds, all_labels, all_weights = [], [], []
    for data in loader:
        data = data.to(DEVICE)
        out = model(data)
        all_preds.append(out.cpu())
        all_labels.append(data.y.cpu())
        all_weights.append(data.w.cpu())
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_weights = torch.cat(all_weights, dim=0)

    val_loss = weighted_bce_loss(all_preds.to(DEVICE), all_labels.to(DEVICE), all_weights.to(DEVICE), pos_weight_tensor)
    # print(f"Validation Loss: {val_loss.item():.4f}")
    task_aucs = []
    valid_labels_mask = ~torch.isnan(all_labels) & (all_weights > 0)
    for i in range(N_TASKS):
        task_labels = all_labels[valid_labels_mask[:, i], i]
        task_preds = all_preds[valid_labels_mask[:, i], i]
        if len(task_labels) > 1 and len(torch.unique(task_labels)) > 1:
            try: task_aucs.append(roc_auc_score(task_labels.numpy(), torch.sigmoid(task_preds).numpy()))
            except ValueError: task_aucs.append(np.nan)
        else: task_aucs.append(np.nan)
    if print_scores:
        print("\n--- Per-Task ROC-AUC Scores ---")
        for name, auc in zip(TASK_NAMES, task_aucs):
            if not np.isnan(auc): print(f"  {name:<16}: {auc:.4f}")
            else: print(f"  {name:<16}: N/A (not enough samples)")
        print("---------------------------------")
    mean_auc = np.nanmean(task_aucs)
    return val_loss, mean_auc

# --- Main Training Function ---
def main():
    print("--- Starting Test Run (with Imbalance Fix) ---")
    print(f"Using device: {DEVICE}")

    # ... (Data loader creation is unchanged) ...
    print(f"Loading {len(train_data_list)} train, {len(valid_data_list)} valid, {len(test_data_list)} test samples.")
    train_loader = DataLoader(train_data_list, batch_size=8, shuffle=True)
    valid_loader = DataLoader(valid_data_list, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_data_list, batch_size=8, shuffle=False)

    # --- 3. Initialize Model (now uses branch_weights) ---
    model = EarlyFusionModel(
        node_feature_dim=NODE_FEATURE_DIM,
        edge_feature_dim=EDGE_FEATURE_DIM,
        fp_feature_dim=FP_FEATURE_DIM,
        desc_feature_dim=DESC_FEATURE_DIM,
        n_tasks=N_TASKS,
        graph_hidden_dim=256,
        graph_out_dim=64,
        fp_out_dim=64,
        gnn_dropout=0.317,
        classifier_dropout_1=0.572,
        classifier_dropout_2=0.432
    ).to(DEVICE)

    print(f"Model Architecture:{model}")
    
    optimizer = optim.Adam(model.parameters(), lr=0.00024, weight_decay=3.948e-06)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-7)
    print(f"Optimizer:{optimizer}")
    print(f"Scheduler:{scheduler}")

    # --- 4. Training Loop (Pass pos_weight_tensor to loss) ---
    best_valid_auc = 0.0
    epochs_no_improve = 0
    
    for epoch in range(N_EPOCHS):
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(DEVICE)
            optimizer.zero_grad()
            
            out = model(data)
            
            # --- 3. Pass pos_weight_tensor to loss (NEW!) ---
            loss = weighted_bce_loss(out, data.y, data.w, pos_weight_tensor)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs

        avg_train_loss = total_loss / len(train_loader.dataset)
        val_loss, valid_auc = eval_model(model, valid_loader, print_scores=False) 
        print(f"Epoch {epoch+1:02d}/{N_EPOCHS:02d} | Train Loss: {avg_train_loss:.4f} | Valid Loss: {val_loss.item():.4f} | Valid AUC: {valid_auc:.4f} | LR: {optimizer.param_groups[0]['lr']:.1e}")
        
        # Print the learned branch weights each epoch!
        weights = model.branch_weights.data.cpu().numpy()
        print(f"  Branch Weights (GNN, FP, Desc): {weights[0]:.2f}, {weights[1]:.2f}, {weights[2]:.2f}")
        
        scheduler.step(valid_auc)
        
        if valid_auc > best_valid_auc:
            best_valid_auc = valid_auc
            epochs_no_improve = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  -> New best model saved to {MODEL_SAVE_PATH} (AUC: {best_valid_auc:.4f})")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"\n--- Early stopping at epoch {epoch+1} ---")
            break

    print("\n--- Test Run Training Complete ---")
    
    # ... (Final evaluation block is unchanged) ...
    print(f"Loading best model from {MODEL_SAVE_PATH} for final test evaluation...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    test_loss, test_mean_auc = eval_model(model, test_loader, print_scores=True)
    print(f"\n======================================")
    print(f" Final Test Loss: {test_loss.item():.4f}")
    print(f" Final Test Mean ROC-AUC: {test_mean_auc:.4f}")
    print(f"======================================")

if __name__ == "__main__":
    main()