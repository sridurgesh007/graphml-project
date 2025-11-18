# gine with branch weight
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score
import numpy as np
import os
import joblib

# Import our custom model (make sure it's the GINEConv one from model.py)
from gine_with_branchw import EarlyFusionModel

# --- Configuration ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PROCESSED_DATA_DIR = "processed_fusion_data"
N_TASKS = 12
MODEL_SAVE_PATH = "test_run_best_model_gine_branchw.pth" # Save file for this test run
N_EPOCHS = 30       # Number of epochs for this test run
EARLY_STOP_PATIENCE = 7 # Stop if valid AUC doesn't improve for 7 epochs

# These must match the order from build_fusion_dataset.py
TASK_NAMES = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
    "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
]

# --- 1. Set Default Hyperparameters (for this test run) ---
# These are NOT tuned. They are just reasonable defaults.
# You will find better ones by running tune.py
DEFAULT_PARAMS = {
    'lr': 1e-4,
    'weight_decay': 1e-5,
    'batch_size': 32,
    'gnn_dropout': 0.02,
    'classifier_dropout_1': 0.02,
    'classifier_dropout_2': 0.03,
    'graph_hidden_dim': 128,
    'graph_out_dim': 64,
    'fp_out_dim': 256
}

# --- Load Scalers (for model dimensions) ---
try:
    node_scaler = joblib.load(os.path.join(PROCESSED_DATA_DIR, "node_feature_scaler.joblib"))
    desc_imputer = joblib.load(os.path.join(PROCESSED_DATA_DIR, "desc_feature_imputer.joblib"))
    desc_scaler = joblib.load(os.path.join(PROCESSED_DATA_DIR, "desc_feature_scaler.joblib"))
except FileNotFoundError:
    print("Error: Scaler/imputer files not found. Please run 'build_fusion_dataset.py' first.")
    exit()

# --- Determine Feature Dimensions ---
try:
    _temp_data_list = torch.load(os.path.join(PROCESSED_DATA_DIR, "train_data.pt"), weights_only=False)
    if not _temp_data_list:
        print("Error: train_data.pt is empty.")
        exit()
    _temp_data = _temp_data_list[0]
    
    NODE_FEATURE_DIM = _temp_data.x.shape[1]
    EDGE_FEATURE_DIM = _temp_data.edge_attr.shape[1]
    FP_FEATURE_DIM = _temp_data.fp.shape[1]
    DESC_FEATURE_DIM = _temp_data.desc.shape[1]
    
    print(f"--- Feature Dimensions Detected ---")
    print(f"Node Features:   {NODE_FEATURE_DIM}")
    print(f"Edge Features:   {EDGE_FEATURE_DIM}")
    print(f"FP Features:     {FP_FEATURE_DIM}")
    print(f"Desc Features:   {DESC_FEATURE_DIM}")
    print(f"---------------------------------")
    
    # Load all data for the test run
    train_data_list = _temp_data_list
    valid_data_list = torch.load(os.path.join(PROCESSED_DATA_DIR, "valid_data.pt"), weights_only=False)
    test_data_list = torch.load(os.path.join(PROCESSED_DATA_DIR, "test_data.pt"), weights_only=False)
    
    del _temp_data
    del _temp_data_list

except FileNotFoundError:
    print("Error: Processed data files not found. Please run 'build_fusion_dataset.py' first.")
    exit()
except Exception as e:
    print(f"Error loading data for dimension check: {e}")
    exit()


# --- Loss Function (Weighted BCE) ---
def weighted_bce_loss(y_pred, y_true, weights):
    loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    raw_loss = loss_fn(y_pred, y_true)
    is_valid = ~torch.isnan(y_true)
    raw_loss = torch.where(is_valid, raw_loss, torch.zeros_like(raw_loss))
    weighted_loss = raw_loss * weights
    total_weight = weights.sum()
    final_loss = weighted_loss.sum() / (total_weight + 1e-8)
    return final_loss

# --- Evaluation Function (with per-task printing) ---
@torch.no_grad()
def eval_model(model, loader, print_scores=False):
    """
    Evaluates the model on the given data loader.
    If print_scores is True, it prints a detailed per-task AUC breakdown.
    Returns the mean ROC-AUC score.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_weights = [] 

    for data in loader:
        data = data.to(DEVICE)
        out = model(data)
        all_preds.append(out.cpu())
        all_labels.append(data.y.cpu())
        all_weights.append(data.w.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_weights = torch.cat(all_weights, dim=0)
    
    task_aucs = []
    valid_labels_mask = ~torch.isnan(all_labels) & (all_weights > 0)
    
    for i in range(N_TASKS):
        task_labels = all_labels[valid_labels_mask[:, i], i]
        task_preds = all_preds[valid_labels_mask[:, i], i]
        
        if len(task_labels) > 1 and len(torch.unique(task_labels)) > 1:
            try:
                task_auc = roc_auc_score(task_labels.numpy(), torch.sigmoid(task_preds).numpy())
                task_aucs.append(task_auc)
            except ValueError:
                task_aucs.append(np.nan)
        else:
            task_aucs.append(np.nan) 

    if print_scores:
        print("\n--- Per-Task ROC-AUC Scores ---")
        for name, auc in zip(TASK_NAMES, task_aucs):
            if not np.isnan(auc):
                print(f"  {name:<16}: {auc:.4f}")
            else:
                print(f"  {name:<16}: N/A (not enough samples)")
        print("---------------------------------")

    mean_auc = np.nanmean(task_aucs)
    return mean_auc

# --- Main Training Function ---
def main():
    
    print("--- Starting Test Run ---")
    print(f"Using device: {DEVICE}")
    print(f"Training for {N_EPOCHS} epochs with patience {EARLY_STOP_PATIENCE}.")

    # --- 2. Load Data ---
    print(f"Loading {len(train_data_list)} train, {len(valid_data_list)} valid, {len(test_data_list)} test samples.")
    
    train_loader = DataLoader(train_data_list, batch_size=DEFAULT_PARAMS['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_data_list, batch_size=DEFAULT_PARAMS['batch_size'], shuffle=False)
    test_loader = DataLoader(test_data_list, batch_size=DEFAULT_PARAMS['batch_size'], shuffle=False)

    # --- 3. Initialize Model and Optimizer ---
    model = EarlyFusionModel(
        node_feature_dim=NODE_FEATURE_DIM,
        edge_feature_dim=EDGE_FEATURE_DIM,
        fp_feature_dim=FP_FEATURE_DIM,
        desc_feature_dim=DESC_FEATURE_DIM,
        n_tasks=N_TASKS,
        graph_hidden_dim=DEFAULT_PARAMS['graph_hidden_dim'],
        graph_out_dim=DEFAULT_PARAMS['graph_out_dim'],
        gnn_dropout=DEFAULT_PARAMS['gnn_dropout'],
        fp_out_dim=DEFAULT_PARAMS['fp_out_dim'],
        classifier_dropout_1=DEFAULT_PARAMS['classifier_dropout_1'],
        classifier_dropout_2=DEFAULT_PARAMS['classifier_dropout_2']
    ).to(DEVICE)

    print(f"Model Architecture:{model}")
    
    optimizer = optim.Adam(
        model.parameters(), 
        lr=DEFAULT_PARAMS['lr'], 
        weight_decay=DEFAULT_PARAMS['weight_decay']
    )
    
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='max', # We are monitoring validation AUC
        factor=0.5,
        patience=3, # Reduce LR if no improvement for 3 epochs
        min_lr=1e-7
    )

    # --- 4. Training Loop (with validation and early stopping) ---
    best_valid_auc = 0.0
    epochs_no_improve = 0
    
    for epoch in range(N_EPOCHS):
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(DEVICE)
            optimizer.zero_grad()
            
            out = model(data)
            
            loss = weighted_bce_loss(out, data.y, data.w)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs

        avg_train_loss = total_loss / len(train_loader.dataset)
        
        # --- Validation ---
        # We don't print per-task scores every epoch, just the mean
        valid_auc = eval_model(model, valid_loader, print_scores=False) 
        
        print(f"Epoch {epoch+1:02d}/{N_EPOCHS:02d} | Train Loss: {avg_train_loss:.4f} | Valid AUC: {valid_auc:.4f} | LR: {optimizer.param_groups[0]['lr']:.1e}")
        
        scheduler.step(valid_auc)
        
        # --- Check for early stopping ---
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
    
    # --- 5. Final Evaluation on Test Set ---
    print(f"Loading best model from {MODEL_SAVE_PATH} for final test evaluation...")
    
    # Load the best model weights
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    
    # Run evaluation on the test set, with print_scores=True
    test_mean_auc = eval_model(model, test_loader, print_scores=True)
    
    print(f"\n======================================")
    print(f" Final Test Mean ROC-AUC: {test_mean_auc:.4f}")
    print(f"======================================")
    print("\nThis was a test run. For best results, run 'tune.py' to find optimal hyperparameters.")

# --- Main Execution ---
if __name__ == "__main__":
    main()
