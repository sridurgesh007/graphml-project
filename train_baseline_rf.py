"""
Baseline Random Forest for Tox21
Uses tox21_features.csv and the same scaffold split (train/val/test)
from prep_tox21_unified_split.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# -----------------------------------
# 1. Load data
# -----------------------------------
print("📥 Loading Tox21 features...")
features_df = pd.read_csv("tox21_processed/tox21_features.csv")

train_smiles = set(pd.read_csv("tox21_processed/train_smiles.txt")['smiles'])
valid_smiles = set(pd.read_csv("tox21_processed/valid_smiles.txt")['smiles'])
test_smiles  = set(pd.read_csv("tox21_processed/test_smiles.txt")['smiles'])

def split_df(df, smiles_set):
    return df[df['smiles'].isin(smiles_set)].reset_index(drop=True)

train_df = split_df(features_df, train_smiles)
valid_df = split_df(features_df, valid_smiles)
test_df  = split_df(features_df, test_smiles)

print(f"✅ Split sizes: Train={len(train_df)}, Val={len(valid_df)}, Test={len(test_df)}")

# -----------------------------------
# 2. Extract features & targets
# -----------------------------------
feature_cols = [c for c in features_df.columns if c.startswith("morgan_") or c.startswith("maccs_") or not c.startswith("target_") and c not in ["smiles"]]

target_cols = [c for c in features_df.columns if c.startswith("target_")]

X_train = train_df[feature_cols].values
X_valid = valid_df[feature_cols].values
X_test  = test_df[feature_cols].values

# -----------------------------------
# 3. Train one model per toxicity target
# -----------------------------------
print("\n🧠 Training Random Forests per target...\n")
results = []

for target in tqdm(target_cols):
    y_train = train_df[target].values
    y_valid = valid_df[target].values
    y_test  = test_df[target].values

    # Skip target if too many missing values
    if np.isnan(y_train).all():
        continue

    # Remove rows with NaN labels
    valid_mask_train = ~np.isnan(y_train)
    valid_mask_val   = ~np.isnan(y_valid)
    valid_mask_test  = ~np.isnan(y_test)

    # Standardize continuous features (RDKit descriptors only)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[valid_mask_train])
    X_valid_scaled = scaler.transform(X_valid[valid_mask_val])
    X_test_scaled  = scaler.transform(X_test[valid_mask_test])

    # Random Forest
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X_train_scaled, y_train[valid_mask_train])

    # Evaluate
    y_pred_val = clf.predict_proba(X_valid_scaled)[:, 1]
    y_pred_test = clf.predict_proba(X_test_scaled)[:, 1]

    auc_val = roc_auc_score(y_valid[valid_mask_val], y_pred_val)
    auc_test = roc_auc_score(y_test[valid_mask_test], y_pred_test)

    results.append({
        "target": target,
        "val_auc": auc_val,
        "test_auc": auc_test
    })

# -----------------------------------
# 4. Summarize results
# -----------------------------------
results_df = pd.DataFrame(results)
mean_auc = results_df["test_auc"].mean()

print("\n📊 Baseline Random Forest Results:")
print(results_df.sort_values("test_auc", ascending=False))
print(f"\n✅ Mean Test ROC-AUC across targets: {mean_auc:.3f}")

results_df.to_csv("tox21_processed/baseline_rf_results.csv", index=False)
print("\n💾 Results saved to tox21_processed/baseline_rf_results.csv")