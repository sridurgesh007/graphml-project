# import os
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch_geometric.loader import DataLoader
# from sklearn.metrics import roc_auc_score
# from tqdm import tqdm
# import time

# from model import EarlyFusionModel
# from tune import train_epoch, eval_model, loss_fn # Reuse our functions

# # --- 1. Constants and Best Params ---
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {DEVICE}")

# DATA_DIR = "processed_fusion_data"
# FINAL_MODEL_PATH = "final_fusion_model.pth"
# NUM_TASKS = 12

# # === PASTE BEST PARAMS FROM OPTUNA HERE ===
# # (Example values, replace with your results)
# BEST_PARAMS = {
#     "lr": 0.00045,
#     "weight_decay": 3.16e-6,
#     "batch_size": 64,
#     "gat_heads": 8,
#     "gnn_dropout": 0.15,
#     "classifier_dropout_1": 0.45,
#     "classifier_dropout_2": 0.25,
#     "graph_hidden_dim": 128,
#     "graph_out_dim": 128,
#     "fp_out_dim": 256
# }
# # === END OF PARAMS ===

# # Use a fixed number of epochs, or train until loss plateaus
# # Since we don't have a validation set, 
# # we'll train for a "reasonable" number of epochs.
# # (Alternatively, find the best epoch from the tuning trial)
# FINAL_EPOCHS = 40 # Example: Best trial stopped around epoch 40

# # --- 2. Load and Combine Data ---
# print("Loading processed data...")
# train_data = torch.load(os.path.join(DATA_DIR, "train_data.pt"))
# valid_data = torch.load(os.path.join(DATA_DIR, "valid_data.pt"))
# test_data = torch.load(os.path.join(DATA_DIR, "test_data.pt"))

# # Combine train and validation sets for final training
# full_train_data = train_data + valid_data
# print(f"Combined train+valid data: {len(full_train_data)} samples")

# # Create final DataLoaders
# train_loader = DataLoader(full_train_data, batch_size=BEST_PARAMS["batch_size"], shuffle=True)
# test_loader = DataLoader(test_data, batch_size=BEST_PARAMS["batch_size"], shuffle=False)

# # --- 3. Initialize Final Model ---
# print("Initializing final model with best parameters...")

# # Get feature dimensions
# first_data = train_data[0]
# NODE_DIM = first_data.x.shape[1]
# FP_DIM = first_data.fp.shape[1]
# DESC_DIM = first_data.desc.shape[1]

# model = EarlyFusionModel(
#     node_feat_dim=NODE_DIM,
#     fp_feat_dim=FP_DIM,
#     desc_feat_dim=DESC_DIM,
#     num_tasks=NUM_TASKS,
#     graph_hidden_dim=BEST_PARAMS["graph_hidden_dim"],
#     graph_out_dim=BEST_PARAMS["graph_out_dim"],
#     fp_out_dim=BEST_PARAMS["fp_out_dim"],
#     gat_heads=BEST_PARAMS["gat_heads"],
#     gnn_dropout=BEST_PARAMS["gnn_dropout"],
#     classifier_dropout_1=BEST_PARAMS["classifier_dropout_1"],
#     classifier_dropout_2=BEST_PARAMS["classifier_dropout_2"]
# ).to(DEVICE)

# optimizer = optim.Adam(
#     model.parameters(), 
#     lr=BEST_PARAMS["lr"], 
#     weight_decay=BEST_PARAMS["weight_decay"]
# )

# # --- 4. Final Training ---
# print(f"\n--- Starting final training for {FINAL_EPOCHS} epochs ---")
# start_time = time.time()
# for epoch in range(1, FINAL_EPOCHS + 1):
#     train_loss = train_epoch(model, train_loader, loss_fn, optimizer)
#     print(f"Epoch {epoch:03d}/{FINAL_EPOCHS} | Train Loss: {train_loss:.4f}")

# end_time = time.time()
# print(f"--- Final Training Complete (Took {(end_time-start_time)/60:.2f} mins) ---")

# # Save the final model
# torch.save(model.state_dict(), FINAL_MODEL_PATH)
# print(f"Final model saved to {FINAL_MODEL_PATH}")

# # --- 5. Final Test Set Evaluation ---
# print("Evaluating final model on test set...")
# test_auc = eval_model(model, test_loader)

# print("\n--- FINAL TEST RESULTS ---")
# print(f"Final Test AUC: {test_auc:.4f}")

# final model training of GINEConv with optuna params
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

# Import our custom model (make sure it's the GINEConv one)
from temp import EarlyFusionModel

# --- Configuration ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PROCESSED_DATA_DIR = "processed_fusion_data"
N_TASKS = 12
MODEL_SAVE_PATH = "best_fusion_model.pth"

# --- 1. ADD TASK NAMES (for printing) ---
# These must match the order from build_fusion_dataset.py
TASK_NAMES = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
    "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
]

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
    # Load just one sample to check dims
    _temp_data_list = torch.load(os.path.join(PROCESSED_DATA_DIR, "train_data.pt"))
    if not _temp_data_list:
        print("Error: train_data.pt is empty.")
        exit()
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
    
    # Don't keep all data in memory yet
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
    
    # Mask out NaN labels
    is_valid = ~torch.isnan(y_true)
    raw_loss = torch.where(is_valid, raw_loss, torch.zeros_like(raw_loss))
    
    weighted_loss = raw_loss * weights
    
    # Normalize by the sum of weights
    total_weight = weights.sum()
    final_loss = weighted_loss.sum() / (total_weight + 1e-8)
    return final_loss

# --- 2. MODIFY EVAL_MODEL ---
@torch.no_grad()
def eval_model(model, loader, print_scores=False):
    """
    Evaluates the model on the given data loader.
    
    Args:
        model (nn.Module): The model to evaluate.
        loader (DataLoader): Data loader.
        print_scores (bool): If True, prints a detailed per-task AUC breakdown.
        
    Returns:
        float: The mean ROC-AUC score across all tasks.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_weights = [] 

    for data in loader:
        data = data.to(DEVICE)
        out = model(
            data.x, data.edge_index, data.edge_attr,
            data.fp_features, data.desc_features, data.batch
        )
        all_preds.append(out.cpu())
        all_labels.append(data.y.cpu())
        all_weights.append(data.w.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_weights = torch.cat(all_weights, dim=0)
    
    task_aucs = []
    
    # Get all valid (non-NaN, weight > 0) labels
    valid_labels_mask = ~torch.isnan(all_labels) & (all_weights > 0)
    
    for i in range(N_TASKS):
        # Get labels and preds for this task where data is valid
        task_labels = all_labels[valid_labels_mask[:, i], i]
        task_preds = all_preds[valid_labels_mask[:, i], i]
        
        # Ensure there are both positive and negative samples
        if len(task_labels) > 1 and len(torch.unique(task_labels)) > 1:
            try:
                task_auc = roc_auc_score(task_labels.numpy(), torch.sigmoid(task_preds).numpy())
                task_aucs.append(task_auc)
            except ValueError:
                task_aucs.append(np.nan) # Should not happen, but as a safeguard
        else:
            task_aucs.append(np.nan) # Not enough data to calculate AUC

    # --- 3. ADD PRINT BLOCK ---
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
    # --- 1. Get Best Hyperparameters ---
    # ** IMPORTANT! **
    # Paste your best params from Optuna here.
    # These are just placeholders from my GINEConv run.
    best_params = {
        'lr': 0.00015,
        'weight_decay': 1e-05,
        'batch_size': 128,
        'gnn_dropout': 0.1,
        'classifier_dropout_1': 0.1,
        'classifier_dropout_2': 0.1,
        'graph_hidden_dim': 256,
        'graph_out_dim': 128,
        'fp_out_dim': 512
    }
    
    print("--- Using Best Hyperparameters ---")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print("----------------------------------")

    # --- 2. Load Full Training Data ---
    print("Loading all training data (train + valid)...")
    try:
        # We combine train and valid for the final model
        train_data_list = torch.load(os.path.join(PROCESSED_DATA_DIR, "train_data.pt"))
        valid_data_list = torch.load(os.path.join(PROCESSED_DATA_DIR, "valid_data.pt"))
        full_train_data = train_data_list + valid_data_list
        
        test_data_list = torch.load(os.path.join(PROCESSED_DATA_DIR, "test_data.pt"))
    except FileNotFoundError:
        print("Error: Data files not found. Run build_fusion_dataset.py first.")
        return

    print(f"Total training samples: {len(full_train_data)}")
    print(f"Test samples: {len(test_data_list)}")

    # Create DataLoaders
    train_loader = DataLoader(full_train_data, batch_size=best_params['batch_size'], shuffle=True)
    test_loader = DataLoader(test_data_list, batch_size=best_params['batch_size'], shuffle=False)

    # --- 3. Initialize Model and Optimizer ---
    model = EarlyFusionModel(
        node_feature_dim=NODE_FEATURE_DIM,
        edge_feature_dim=EDGE_FEATURE_DIM,
        fp_feature_dim=FP_FEATURE_DIM,
        desc_feature_dim=DESC_FEATURE_DIM,
        n_tasks=N_TASKS,
        # Pass tunable params
        graph_hidden_dim=best_params['graph_hidden_dim'],
        graph_out_dim=best_params['graph_out_dim'],
        gnn_dropout=best_params['gnn_dropout'],
        fp_out_dim=best_params['fp_out_dim'],
        classifier_dropout_1=best_params['classifier_dropout_1'],
        classifier_dropout_2=best_params['classifier_dropout_2']
    ).to(DEVICE)
    
    optimizer = optim.Adam(
        model.parameters(), 
        lr=best_params['lr'], 
        weight_decay=best_params['weight_decay']
    )
    
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', # We train on loss for the final model
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=True
    )

    # --- 4. Final Training Loop ---
    # We train for a fixed number of epochs based on Optuna results
    # (e.g., if trials stopped around 30-40, we train for 40)
    n_epochs = 40 
    
    print(f"\n--- Starting Final Training for {n_epochs} Epochs ---")
    
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(DEVICE)
            optimizer.zero_grad()
            
            out = model(
                data.x, data.edge_index, data.edge_attr,
                data.fp_features, data.desc_features, data.batch
            )
            
            loss = weighted_bce_loss(out, data.y, data.w)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs

        avg_train_loss = total_loss / len(train_loader.dataset)
        
        # Step the scheduler on the training loss
        scheduler.step(avg_train_loss)
        
        print(f"Epoch {epoch+1:02d}/{n_epochs:02d} | Train Loss: {avg_train_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.1e}")

    print("--- Final Training Complete ---")
    
    # Save the final model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Final model saved to {MODEL_SAVE_PATH}")

    # --- 5. Final Evaluation on Test Set ---
    print("\n--- Evaluating on Test Set ---")
    
    # Load the just-saved model for a clean evaluation
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    
    # --- 4. CALL EVAL_MODEL WITH print_scores=True ---
    test_mean_auc = eval_model(model, test_loader, print_scores=True)
    
    print(f"\n======================================")
    print(f" Final Test Mean ROC-AUC: {test_mean_auc:.4f}")
    print(f"======================================")


# --- Main Execution ---
if __name__ == "__main__":
    main()