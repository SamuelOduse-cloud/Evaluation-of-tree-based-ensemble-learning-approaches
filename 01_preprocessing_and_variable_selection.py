"""
=============================================================================
DATA PREPROCESSING AND VARIABLE SELECTION
=============================================================================
Study: Predicting Under-Five Mortality Using Tree-Based Ensemble Learning
Data source: Demographic and Health Surveys (DHS) Program, pooled dataset
             across 27 Sub-Saharan African countries (N = 206,317)

Software environment (see environment.yml / requirements.txt for full spec):
    Python 3.12.3
    pandas 3.0.2, numpy 2.4.4, scikit-learn 1.8.0, scipy 1.17.1
    imbalanced-learn 0.14.2

Reproducibility: random_state = 2025 is fixed throughout for all stochastic
operations (subsampling, SMOTE, cross-validation folds, tree ensembles).
=============================================================================
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.over_sampling import SMOTE

RANDOM_STATE = 2025
np.random.seed(RANDOM_STATE)

# -----------------------------------------------------------------------
# STEP 1: LOAD DATA
# -----------------------------------------------------------------------
# Replace with the path to the DHS extract used in this study.
DATA_PATH = "Mortality_data.xlsx"
df = pd.read_excel(DATA_PATH)
print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# -----------------------------------------------------------------------
# STEP 2: ENCODE ALL VARIABLES AS CATEGORICAL FACTORS
# -----------------------------------------------------------------------
df_enc = df.drop(columns=["COUNTRY"])          # country excluded as predictor
feature_cols = [c for c in df_enc.columns if c != "U5M"]

label_encoders = {}
for col in df_enc.columns:
    le = LabelEncoder()
    df_enc[col] = le.fit_transform(df_enc[col].astype(str))
    label_encoders[col] = le

# Ensure U5M is coded with "Yes" (death) = 1
y_full = df_enc["U5M"].values
if label_encoders["U5M"].classes_[1] != "Yes":
    y_full = 1 - y_full
X_full = df_enc[feature_cols].values

print(f"U5M positive class (deaths) rate: {y_full.mean():.4f}")

# -----------------------------------------------------------------------
# STEP 3: VARIABLE SELECTION (THREE-METHOD CONSENSUS)
# -----------------------------------------------------------------------
# Applied to a SMOTE-balanced 15% stratified subsample (see manuscript
# Methods, "Rationale for Variable Selection on SMOTE-Balanced Data")
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.85, random_state=RANDOM_STATE)
idx, _ = next(sss.split(X_full, y_full))

sm = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
X_bal, y_bal = sm.fit_resample(X_full[idx], y_full[idx])
print(f"Balanced subsample for variable selection: {X_bal.shape[0]} observations")

# --- Method A: Random Forest Importance (Mean Decrease in Impurity) -----
rf_importance_model = RandomForestClassifier(
    n_estimators=200, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1
)
rf_importance_model.fit(X_bal, y_bal)
importances = pd.Series(
    rf_importance_model.feature_importances_, index=feature_cols
).sort_values(ascending=False)
rf_vars = importances[importances >= importances.mean()].index.tolist()

# --- Method B: Recursive Feature Elimination (RFE) -----------------------
rfe = RFE(
    estimator=RandomForestClassifier(
        n_estimators=80, max_depth=8, random_state=RANDOM_STATE, n_jobs=-1
    ),
    n_features_to_select=15,
    step=2,
)
rfe.fit(X_bal, y_bal)
rfe_vars = [feature_cols[i] for i, s in enumerate(rfe.support_) if s]

# --- Method C: Boruta Shadow-Feature Method (20 iterations) --------------
shadow_maxes = []
for _ in range(20):
    X_shadow = X_bal.copy()
    for c in range(X_shadow.shape[1]):
        np.random.shuffle(X_shadow[:, c])
    rf_shadow = RandomForestClassifier(
        n_estimators=50, max_depth=7, random_state=None, n_jobs=-1
    )
    rf_shadow.fit(X_shadow, y_bal)
    shadow_maxes.append(rf_shadow.feature_importances_.max())
shadow_threshold = np.percentile(shadow_maxes, 95)
boruta_vars = importances[importances > shadow_threshold].index.tolist()

# --- Consensus rule: retain variables selected by >= 2 of 3 methods ------
from collections import Counter
freq = Counter(rf_vars + rfe_vars + boruta_vars)
selected_vars = sorted(
    [v for v, c in freq.items() if c >= 2], key=lambda v: freq[v], reverse=True
)
print(f"\nConsensus-selected variables ({len(selected_vars)}): {selected_vars}")

# Save outputs for downstream scripts
pd.Series(selected_vars).to_csv("selected_variables.csv", index=False, header=False)
pd.Series(feature_cols).to_csv("all_feature_columns.csv", index=False, header=False)
importances.to_csv("rf_importance_scores.csv", header=["importance"])
pd.Series(rf_vars).to_csv("rf_selected_vars.csv", index=False, header=False)
pd.Series(rfe_vars).to_csv("rfe_selected_vars.csv", index=False, header=False)
pd.Series(boruta_vars).to_csv("boruta_selected_vars.csv", index=False, header=False)
print("\nOutputs saved: selected_variables.csv, all_feature_columns.csv, "
      "rf_importance_scores.csv, rf_selected_vars.csv, rfe_selected_vars.csv, "
      "boruta_selected_vars.csv")
