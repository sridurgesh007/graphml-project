import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import optuna


# Import our custom *tunable* model
# from temp import EarlyFusionModel
# import gine_branchw model
from gine_with_branchw import EarlyFusionModel

# --- 1. Constants and Setup ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

DATA_DIR = "processed_fusion_data"
# DB_STORAGE_PATH = "sqlite:///tox21_tuning3.db"
# STUDY_NAME = "early_fusion_v3"
DB_STORAGE_PATH = "sqlite:///optuna_tuning_gine_bw_clsimb1.db" # <-- New DB file
STUDY_NAME = "early_fusion_v5_gine"      # <-- New study name
LABEL_COLUMNS = [
        'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 
        'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 
        'SR-HSE', 'SR-MMP', 'SR-p53'
    ]

# Hyperparameters
EPOCHS = 200      # Max epochs *per trial*
PATIENCE = 15    # Early stopping patience
N_TASKS = 12

# --- 2. Load Data (Load ONCE, outside the objective) ---
print("Loading processed data...")
try:
    train_data = torch.load(os.path.join(DATA_DIR, "train_data.pt"), weights_only=False)
    valid_data = torch.load(os.path.join(DATA_DIR, "valid_data.pt"), weights_only=False)
except FileNotFoundError:
    print(f"Error: Processed data not found in '{DATA_DIR}'.")
    print("Please run 'build_fusion_dataset.py' first.")
    exit()

# Get feature dimensions (ONCE)
first_data = train_data[0]
NODE_DIM = first_data.x.shape[1]
EDGE_DIM = first_data.edge_attr.shape[1]
FP_DIM = first_data.fp.shape[1]
DESC_DIM = first_data.desc.shape[1]
print(f"Data loaded. Feature Dims: Node={NODE_DIM}, Edge={EDGE_DIM}, FP={FP_DIM}, Desc={DESC_DIM}")

# --- Calculate pos_weight for Class Imbalance (NEW!) ---
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
    for name, weight in zip(LABEL_COLUMNS, pos_weight):
        print(f"  {name:<16}: {weight:.2f}") # Now these will all be different!
    print("-----------------------------------------------")
    
    return pos_weight.to(DEVICE)

# Calculate weights *only* from the training set
pos_weight = calculate_pos_weights(train_data)

# --- 3. Training/Evaluation Functions (copied from train.py) ---

# Loss function (defined globally)
loss_fn = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)

def train_epoch(model, loader, loss_fn, optimizer):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(DEVICE)
        logits = model(batch)
        y_true = batch.y
        weights = batch.w
        
        raw_loss = loss_fn(logits, y_true)
        weighted_loss = raw_loss * weights
        final_loss = weighted_loss.sum() / (weights.sum() + 1e-8)

        optimizer.zero_grad()
        final_loss.backward()
        optimizer.step()
        total_loss += final_loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)

@torch.no_grad()
def eval_model(model, loader):
    model.eval()
    all_preds, all_labels, all_weights = [], [], []
    for batch in loader:
        batch = batch.to(DEVICE)
        logits = model(batch)
        all_preds.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(batch.y.cpu().numpy())
        all_weights.append(batch.w.cpu().numpy())
        
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_weights = np.concatenate(all_weights, axis=0)
    
    task_aucs = []
    for i in range(N_TASKS):
        valid_indices = all_weights[:, i] > 0
        if np.sum(valid_indices) > 0 and len(np.unique(all_labels[valid_indices, i])) == 2:
            task_aucs.append(roc_auc_score(all_labels[valid_indices, i], all_preds[valid_indices, i]))
        else:
            task_aucs.append(np.nan)

    # print(f"--- Trial {trial.number} AUCs ---")
    # use task names
    for i, auc in enumerate(task_aucs):
        print(f"  Task {i+1} ({LABEL_COLUMNS[i]}): AUC = {auc:.4f}" if not np.isnan(auc) else f"  Task {i+1} ({LABEL_COLUMNS[i]}): AUC = N/A")
            
    return np.nanmean(task_aucs)

# --- 4. Optuna Objective Function ---

def objective(trial):
    """
    This function is called by Optuna for each trial.
    """
    # --- A. Suggest Hyperparameters ---
    print(f"\n--- Starting Trial {trial.number} ---")

    # gnn type
    # gnn = trial.suggest_categorical("gnn_type", ['gat', 'gin'])
    
    # Optimization params
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128])
    
    # Model architecture params
    # gat_heads = trial.suggest_categorical("gat_heads", [2, 4, 8, 16])
    gnn_dropout = trial.suggest_float("gnn_dropout", 0.0, 0.5)
    classifier_dropout_1 = trial.suggest_float("classifier_dropout_1", 0.0, 0.6)
    classifier_dropout_2 = trial.suggest_float("classifier_dropout_2", 0.0, 0.5)
    
    # Fixed model params
    graph_hidden_dim = trial.suggest_categorical("graph_hidden_dim", [64, 128, 256])
    graph_out_dim = trial.suggest_categorical("graph_out_dim", [32, 64, 128])
    fp_out_dim = trial.suggest_categorical("fp_out_dim", [128, 256, 512])

    # --- B. Setup Model, Loaders, Optimizer ---
    
    # DataLoaders must be created inside objective to use batch_size
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)

    # model = EarlyFusionModel(
    #     graph_in_dim=NODE_DIM,
    #     graph_edge_dim=EDGE_DIM,
    #     fp_in_dim=FP_DIM,
    #     desc_in_dim=DESC_DIM,
    #     num_tasks=NUM_TASKS,
    #     # gnn_type=gnn,
    #     graph_hidden_dim=graph_hidden_dim,
    #     graph_out_dim=graph_out_dim,
    #     fp_out_dim=fp_out_dim,
    #     gat_heads=gat_heads,
    #     gnn_dropout=gnn_dropout,
    #     classifier_dropout_1=classifier_dropout_1,
    #     classifier_dropout_2=classifier_dropout_2
    # ).to(DEVICE)
    model = EarlyFusionModel(
        node_feature_dim=NODE_DIM,
        edge_feature_dim=EDGE_DIM,
        fp_feature_dim=FP_DIM,
        desc_feature_dim=DESC_DIM,
        n_tasks=N_TASKS,
        graph_hidden_dim=graph_hidden_dim,
        graph_out_dim=graph_out_dim,
        gnn_dropout=gnn_dropout,
        fp_out_dim=fp_out_dim,
        classifier_dropout_1=classifier_dropout_1,
        classifier_dropout_2=classifier_dropout_2
    ).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='max',      # We monitor AUC, so we want to maximize it
        factor=0.5,      # Reduce LR by half
        patience=5,      # Wait 5 epochs for improvement
        min_lr=1e-7      # Don't go below this
    )

    # --- C. Run Training Loop ---
    
    best_valid_auc = 0.0
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer)
        valid_auc = eval_model(model, valid_loader)
        
        scheduler.step(valid_auc) 

        print(f"Trial {trial.number} Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Valid AUC: {valid_auc:.4f}")

        # Optuna Pruning: Stop unpromising trials early
        trial.report(valid_auc, epoch)
        if trial.should_prune():
            print("--- Trial Pruned ---")
            raise optuna.TrialPruned()

        # Early Stopping
        if valid_auc > best_valid_auc:
            best_valid_auc = valid_auc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= PATIENCE:
            print(f"--- Trial Early-Stopped ---")
            break
            
    return best_valid_auc # Return the best validation AUC for this trial

# --- 5. Main Study Execution ---

if __name__ == "__main__":
    print(f"Starting Optuna study: {STUDY_NAME}")
    print(f"Database will be saved to: {DB_STORAGE_PATH}")

    # Create a study object and specify direction to "maximize" AUC
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=DB_STORAGE_PATH,
        direction="maximize",
        load_if_exists=True  # Allows you to resume tuning
    )

    # redoing trials with branchw
    print("Running trials with branch weights")

    # Start the optimization
    try:
        study.optimize(objective, n_trials=200)  # Run 200 trials
    except KeyboardInterrupt:
        print("Tuning interrupted by user.")

    # --- 6. Print Results ---
    print("\n--- Tuning Complete ---")
    
    pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
    completed_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

    print("Study statistics: ")
    print(f"  Number of finished trials: {len(study.trials)}")
    print(f"  Number of pruned trials:   {len(pruned_trials)}")
    print(f"  Number of complete trials: {len(completed_trials)}")

    print("\n--- Best Trial ---")
    trial = study.best_trial
    print(f"  Value (AUC): {trial.value:.4f}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
