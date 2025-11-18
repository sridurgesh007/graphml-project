# import os
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch_geometric.loader import DataLoader
# from sklearn.metrics import roc_auc_score
# from tqdm import tqdm

# # Import our custom model
# from temp import EarlyFusionModel

# # --- 1. Constants and Setup ---
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {DEVICE}")

# DATA_DIR = "processed_fusion_data"
# MODEL_SAVE_PATH = "best_fusion_model_with_gine.pth"

# # Hyperparameters
# BATCH_SIZE = 64
# LEARNING_RATE = 1e-4
# WEIGHT_DECAY = 1e-5
# EPOCHS = 100
# PATIENCE = 10  # For early stopping
# NUM_TASKS = 12

# # --- 2. Load Data and Create Loaders ---

# print("Loading processed data...")
# try:
#     train_data = torch.load(os.path.join(DATA_DIR, "train_data.pt"), weights_only=False)
#     valid_data = torch.load(os.path.join(DATA_DIR, "valid_data.pt"), weights_only=False)
#     test_data = torch.load(os.path.join(DATA_DIR, "test_data.pt"), weights_only=False)
# except FileNotFoundError:
#     print(f"Error: Processed data not found in '{DATA_DIR}'.")
#     print("Please run 'build_fusion_dataset.py' first.")
#     exit()

# if not train_data:
#     print("Error: Training data is empty.")
#     exit()

# # Create DataLoaders
# train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
# valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False)
# test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# print(f"Data loaded: {len(train_data)} train, {len(valid_data)} valid, {len(test_data)} test samples.")

# # --- 3. Initialize Model and Optimizer ---

# # Dynamically get feature dimensions from the first data object
# first_data = train_data[0]
# NODE_DIM = first_data.x.shape[1]
# EDGE_DIM = first_data.edge_attr.shape[1]
# FP_DIM = first_data.fp.shape[1]
# DESC_DIM = first_data.desc.shape[1]

# print(f"Feature Dims: Node={NODE_DIM}, Edge={EDGE_DIM}, FP={FP_DIM}, Desc={DESC_DIM}")

# model = EarlyFusionModel(
#     node_feature_dim=NODE_DIM,
#     edge_feature_dim=EDGE_DIM,
#     fp_feature_dim=FP_DIM,
#     desc_feature_dim=DESC_DIM,
#     n_tasks=NUM_TASKS
# ).to(DEVICE)

# optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# # Define the LOSS FUNCTION (using weights!)
# # We use 'none' reduction to get per-element losses, which we then multiply by our weights.
# loss_fn = nn.BCEWithLogitsLoss(reduction='none')

# # --- 4. Training and Evaluation Functions ---

# def train_epoch(model, loader, loss_fn, optimizer):
#     model.train()
#     total_loss = 0
#     for batch in tqdm(loader, desc="Training", leave=False):
#         batch = batch.to(DEVICE)
        
#         # Forward pass
#         logits = model(batch)
        
#         # Calculate weighted loss
#         y_true = batch.y
#         weights = batch.w
        
#         # raw_loss shape: [batch_size, num_tasks]
#         raw_loss = loss_fn(logits, y_true)
        
#         # Apply weights
#         weighted_loss = raw_loss * weights
        
#         # Calculate mean loss, ignoring 0-weight entries
#         # We sum all weighted losses and divide by the sum of all weights
#         # (This correctly handles missing labels)
#         final_loss = weighted_loss.sum() / weights.sum()

#         # Backpropagation
#         optimizer.zero_grad()
#         final_loss.backward()
#         optimizer.step()
        
#         total_loss += final_loss.item() * batch.num_graphs
    
#     return total_loss / len(loader.dataset)


# @torch.no_grad()
# def eval_model(model, loader):
#     model.eval()
#     all_preds = []
#     all_labels = []
#     all_weights = []
    
#     for batch in tqdm(loader, desc="Evaluating", leave=False):
#         batch = batch.to(DEVICE)
        
#         # Forward pass
#         logits = model(batch)
#         preds = torch.sigmoid(logits)
        
#         # Collect predictions, labels, and weights
#         all_preds.append(preds.cpu().numpy())
#         all_labels.append(batch.y.cpu().numpy())
#         all_weights.append(batch.w.cpu().numpy())
        
#     # Concatenate all results
#     all_preds = np.concatenate(all_preds, axis=0)
#     all_labels = np.concatenate(all_labels, axis=0)
#     all_weights = np.concatenate(all_weights, axis=0)
    
#     # Calculate per-task ROC-AUC
#     task_aucs = []
#     for i in range(NUM_TASKS):
#         task_labels = all_labels[:, i]
#         task_preds = all_preds[:, i]
#         task_weights = all_weights[:, i]
        
#         # Filter for valid labels (where weight > 0)
#         valid_indices = task_weights > 0
        
#         # Check if we have enough valid samples (at least one of each class)
#         if np.sum(valid_indices) > 0 and len(np.unique(task_labels[valid_indices])) == 2:
#             try:
#                 auc = roc_auc_score(task_labels[valid_indices], task_preds[valid_indices])
#                 task_aucs.append(auc)
#             except ValueError:
#                 task_aucs.append(np.nan) # Should not happen, but for safety
#         else:
#             task_aucs.append(np.nan) # Not enough data to calculate AUC

#     # Calculate mean AUC, ignoring tasks with NaN
#     mean_auc = np.nanmean(task_aucs)
#     return mean_auc

# # --- 5. Main Training Loop ---

# print("\n--- Starting Model Training ---")

# best_valid_auc = 0.0
# epochs_no_improve = 0

# for epoch in range(1, EPOCHS + 1):
#     train_loss = train_epoch(model, train_loader, loss_fn, optimizer)
#     valid_auc = eval_model(model, valid_loader)
    
#     print(f"Epoch {epoch:03d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Valid AUC: {valid_auc:.4f}")
    
#     # Check for improvement
#     if valid_auc > best_valid_auc:
#         best_valid_auc = valid_auc
#         epochs_no_improve = 0
#         torch.save(model.state_dict(), MODEL_SAVE_PATH)
#         print(f"🎉 New best model saved to {MODEL_SAVE_PATH} (AUC: {best_valid_auc:.4f})")
#     else:
#         epochs_no_improve += 1
        
#     # Check for early stopping
#     if epochs_no_improve >= PATIENCE:
#         print(f"Early stopping triggered after {PATIENCE} epochs with no improvement.")
#         break

# # --- 6. Final Evaluation ---

# print("\n--- Training Complete ---")

# # Load the best model
# print(f"Loading best model from {MODEL_SAVE_PATH} for final test.")
# model.load_state_dict(torch.load(MODEL_SAVE_PATH))

# test_auc = eval_model(model, test_loader)

# print("\n--- Final Results ---")
# print(f"Best Validation AUC: {best_valid_auc:.4f}")
# print(f"Final Test AUC:      {test_auc:.4f}")

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
from temp import EarlyFusionModel

# --- Configuration ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PROCESSED_DATA_DIR = "processed_fusion_data"
N_TASKS = 12
MODEL_SAVE_PATH = "test_run_best_model.pth" # Save file for this test run
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
    'lr': 0.00021725941083907706,
    'weight_decay': 0.0001627843815987054,
    'batch_size': 8,
    'gnn_dropout': 0.05901679328662685,
    'classifier_dropout_1': 0.49333487992422975,
    'classifier_dropout_2': 0.06278874854395061,
    'graph_hidden_dim': 64,
    'graph_out_dim': 32,
    'fp_out_dim': 512
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
