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
import optuna

# Import our simplified model
from gine_with_branchw import EarlyFusionModel

# --- Configuration ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PROCESSED_DATA_DIR = "processed_fusion_data_3d"
N_TASKS = 12
N_EPOCHS = 200       # Max epochs per trial
EARLY_STOP_PATIENCE = 15
N_TRIALS = 200      # Total number of Optuna trials to run

TASK_NAMES = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
    "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
]

# --- Load Data & Scalers (Done ONCE at the start) ---
print("Loading data and scalers...")
try:
    # We don't need the scalers, but we DO need the processed data
    train_data_list = torch.load(os.path.join(PROCESSED_DATA_DIR, "train_data.pt"), weights_only=False)
    valid_data_list = torch.load(os.path.join(PROCESSED_DATA_DIR, "valid_data.pt"), weights_only=False)
    
    if not train_data_list:
        print("Error: train_data.pt is empty. Run build_fusion_dataset_3d.py")
        exit()
        
    # --- DYNAMICALLY DETECT FEATURE DIMS ---
    _temp_data = train_data_list[0]
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
    
    del _temp_data

except Exception as e:
    print(f"Error loading data: {e}. Please run build_fusion_dataset_3D.py first.")
    exit()

# --- Calculate CLIPPED pos_weight (with BUG FIX) ---
def calculate_pos_weights(data_list):
    num_pos = torch.zeros(N_TASKS)
    num_neg = torch.zeros(N_TASKS)
    
    for data in data_list:
        labels = data.y.squeeze()
        weights = data.w.squeeze()
        is_valid = (weights > 0) & (~torch.isnan(labels))
        pos_mask = (labels == 1) & is_valid
        neg_mask = (labels == 0) & is_valid
        
        # Add element-wise (this was the bug fix)
        num_pos += pos_mask
        num_neg += neg_mask

    pos_weight = num_neg / (num_pos + 1e-6)
    # pos_weight = torch.clamp(pos_weight, min=1.0, max=15.0) # CLIPPED

    print("--- Class Imbalance (CLIPPED pos_weight) Calculated ---")
    for name, weight in zip(TASK_NAMES, pos_weight):
        print(f"  {name:<16}: {weight:.2f}")
    print("-----------------------------------------------")
    
    return pos_weight.to(DEVICE)

# Calculate weights *only* from the training set and send to device
pos_weight_tensor = calculate_pos_weights(train_data_list)

# --- Loss Function ---
def weighted_bce_loss(y_pred, y_true, weights, pos_weight):
    loss_fn = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)
    raw_loss = loss_fn(y_pred, y_true)
    is_valid = ~torch.isnan(y_true)
    raw_loss = torch.where(is_valid, raw_loss, torch.zeros_like(raw_loss))
    weighted_loss = raw_loss * weights
    total_weight = weights.sum()
    final_loss = weighted_loss.sum() / (total_weight + 1e-8)
    return final_loss

# --- Evaluation Function ---
@torch.no_grad()
def eval_model(model, loader):
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
    task_aucs = []
    valid_labels_mask = ~torch.isnan(all_labels) & (all_weights > 0)
    for i in range(N_TASKS):
        task_labels = all_labels[valid_labels_mask[:, i], i]
        task_preds = all_preds[valid_labels_mask[:, i], i]
        if len(task_labels) > 1 and len(torch.unique(task_labels)) > 1:
            try: task_aucs.append(roc_auc_score(task_labels.numpy(), torch.sigmoid(task_preds).numpy()))
            except ValueError: task_aucs.append(np.nan)
        else: task_aucs.append(np.nan)

    for i, auc in enumerate(task_aucs):
        print(f"  Task {i+1} ({TASK_NAMES[i]}): AUC = {auc:.4f}" if not np.isnan(auc) else f"  Task {i+1} ({TASK_NAMES[i]}): AUC = N/A")

    mean_auc = np.nanmean(task_aucs)
    return mean_auc

# --- 1. Define the Optuna Objective Function ---
def objective(trial):
    
    # --- 2. Define Hyperparameter Search Space ---
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128])
    
    graph_hidden_dim = trial.suggest_categorical("graph_hidden_dim", [64, 128, 256])
    graph_out_dim = trial.suggest_categorical("graph_out_dim", [32, 64, 128])
    fp_out_dim = trial.suggest_categorical("fp_out_dim", [64, 128, 256, 512])
    
    gnn_dropout = trial.suggest_float("gnn_dropout", 0.1, 0.5)
    classifier_dropout_1 = trial.suggest_float("classifier_dropout_1", 0.2, 0.7)
    classifier_dropout_2 = trial.suggest_float("classifier_dropout_2", 0.1, 0.5)

    # --- 3. Create DataLoaders, Model, Optimizer ---
    train_loader = DataLoader(train_data_list, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data_list, batch_size=batch_size, shuffle=False)

    model = EarlyFusionModel(
        node_feature_dim=NODE_FEATURE_DIM,
        edge_feature_dim=EDGE_FEATURE_DIM,
        fp_feature_dim=FP_FEATURE_DIM,
        desc_feature_dim=DESC_FEATURE_DIM,
        n_tasks=N_TASKS,
        graph_hidden_dim=graph_hidden_dim,
        graph_out_dim=graph_out_dim,
        gnn_dropout=gnn_dropout,
        fp_out_dim=fp_out_dim,
        classifier_dropout_1=classifier_dropout_1,
        classifier_dropout_2=classifier_dropout_2
    ).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-7)

    # --- 4. Training Loop ---
    best_valid_auc = 0.0
    epochs_no_improve = 0

    for epoch in range(N_EPOCHS):
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(DEVICE)
            optimizer.zero_grad()
            out = model(data)
            loss = weighted_bce_loss(out, data.y, data.w, pos_weight_tensor)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs

        avg_train_loss = total_loss / len(train_loader.dataset)
        valid_auc = eval_model(model, valid_loader)
        
        print(f"Trial {trial.number} Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.4f} | Valid AUC: {valid_auc:.4f}")

        scheduler.step(valid_auc)
        
        # --- 5. Early Stopping & Pruning ---
        if valid_auc > best_valid_auc:
            best_valid_auc = valid_auc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        trial.report(valid_auc, epoch)
        if trial.should_prune():
            print("--- Trial Pruned ---")
            raise optuna.exceptions.TrialPruned()

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"--- Early stopping at epoch {epoch+1} ---")
            break
            
    return best_valid_auc

# --- 6. Main Execution ---
if __name__ == "__main__":
    print(f"--- Starting Optuna Tuning ---")
    print("Running trials with 3d features incorporated")
    print(f"Device: {DEVICE}")
    print(f"Running {N_TRIALS} trials...")

    study = optuna.create_study(
        study_name="early_fusion_v3_3D_bond", 
        direction="maximize",
        storage="sqlite:///tox21_tuning.db",
        load_if_exists=True
    )
    
    # Start the optimization
    try:
        study.optimize(objective, n_trials=N_TRIALS)  # Run 200 trials
    except KeyboardInterrupt:
        print("Tuning interrupted by user.")

    print("\n--- Tuning Complete ---")
    pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
    completed_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

    print("Study statistics: ")
    print(f"  Number of finished trials: {len(study.trials)}")
    print(f"  Number of pruned trials:   {len(pruned_trials)}")
    print(f"  Number of complete trials: {len(completed_trials)}")

    print(f"Best trial number: {study.best_trial.number}")
    print(f"Best Valid AUC: {study.best_value:.4f}")
    print("Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")