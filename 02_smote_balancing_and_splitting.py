"""
=============================================================================
SMOTE BALANCING AND DATASET PARTITIONING
=============================================================================
Study: Predicting Under-Five Mortality Using Tree-Based Ensemble Learning

Software environment: Python 3.12.3, pandas 3.0.2, numpy 2.4.4,
                       scikit-learn 1.8.0, imbalanced-learn 0.14.2

Order of operations (see manuscript Methods, "Dataset Partitioning"):
    Step 1: SMOTE applied to the full dataset to produce a balanced pool
    Step 2: Stratified subsample drawn from the balanced pool
    Step 3: Subsample split into Train (70%) / Validation (15%) / Test (15%)
=============================================================================
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

RANDOM_STATE = 2025
np.random.seed(RANDOM_STATE)

# -----------------------------------------------------------------------
# STEP 1: LOAD DATA AND SELECTED VARIABLES 
# -----------------------------------------------------------------------
df = pd.read_excel("Mortality_data.xlsx")
selected_vars = pd.read_csv("selected_variables.csv", header=None)[0].tolist()

df_enc = df.drop(columns=["COUNTRY"])
label_encoders = {}
for col in df_enc.columns:
    le = LabelEncoder()
    df_enc[col] = le.fit_transform(df_enc[col].astype(str))
    label_encoders[col] = le

y_full = df_enc["U5M"].values
if label_encoders["U5M"].classes_[1] != "Yes":
    y_full = 1 - y_full

X_selected = df_enc[selected_vars].values
print(f"Original class distribution: {dict(zip(*np.unique(y_full, return_counts=True)))}")

# -----------------------------------------------------------------------
# STEP 2: APPLY SMOTE TO THE FULL DATASET
# -----------------------------------------------------------------------
sm = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
X_balanced, y_balanced = sm.fit_resample(X_selected, y_full)
print(f"Balanced class distribution: {dict(zip(*np.unique(y_balanced, return_counts=True)))}")

# -----------------------------------------------------------------------
# STEP 3: DRAW STRATIFIED SUBSAMPLE FOR TRACTABLE HYPERPARAMETER TUNING
# -----------------------------------------------------------------------
N_PER_CLASS = 40000
idx0 = np.where(y_balanced == 0)[0]
idx1 = np.where(y_balanced == 1)[0]
np.random.shuffle(idx0)
np.random.shuffle(idx1)
subsample_idx = np.concatenate([idx0[:N_PER_CLASS], idx1[:N_PER_CLASS]])

X_sub = X_balanced[subsample_idx]
y_sub = y_balanced[subsample_idx]
print(f"Subsample for modelling: {X_sub.shape[0]} observations "
      f"({N_PER_CLASS} per class)")

# -----------------------------------------------------------------------
# STEP 4: STRATIFIED 70/15/15 TRAIN / VALIDATION / TEST SPLIT
# -----------------------------------------------------------------------
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X_sub, y_sub, test_size=0.15, stratify=y_sub, random_state=RANDOM_STATE
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.1765,   # 0.1765 * 0.85 ≈ 0.15
    stratify=y_trainval, random_state=RANDOM_STATE
)

print(f"\nFinal partition sizes:")
print(f"  Training:   {X_train.shape[0]} observations")
print(f"  Validation: {X_val.shape[0]} observations")
print(f"  Test:       {X_test.shape[0]} observations")

# Save partitions for downstream modelling scripts
np.save("X_train.npy", X_train); np.save("y_train.npy", y_train)
np.save("X_val.npy",   X_val);   np.save("y_val.npy",   y_val)
np.save("X_test.npy",  X_test);  np.save("y_test.npy",  y_test)
print("\nPartitions saved: X_train.npy, X_val.npy, X_test.npy (+ corresponding y_*.npy)")
