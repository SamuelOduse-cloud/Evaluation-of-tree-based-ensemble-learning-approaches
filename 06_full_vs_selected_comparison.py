"""
=============================================================================
SCRIPT 6 of 7: FULL PREDICTOR SET vs. CONSENSUS-SELECTED SET COMPARISON
=============================================================================
Study: Predicting Under-Five Mortality Using Tree-Based Ensemble Learning

Produces Table 2 in the manuscript: test-set performance of all five
algorithms when trained on (a) the 8-variable consensus-selected set
versus (b) the full 26-variable candidate predictor set.

This script re-runs the SMOTE + split + train + tune pipeline using the
full predictor set, then merges results with the consensus-selected
results already produced by Scripts 2-3.

Software environment: identical to Scripts 1-3 (see requirements.txt)
=============================================================================
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                               GradientBoostingClassifier)
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, recall_score, f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

RANDOM_STATE = 2025
np.random.seed(RANDOM_STATE)

# -----------------------------------------------------------------------
# LOAD DATA (full 26-predictor set this time, not just consensus set)
# -----------------------------------------------------------------------
df = pd.read_excel("Mortality_data.xlsx")
all_feature_cols = pd.read_csv("all_feature_columns.csv", header=None)[0].tolist()

df_enc = df.drop(columns=["COUNTRY"])
label_encoders = {}
for col in df_enc.columns:
    le = LabelEncoder()
    df_enc[col] = le.fit_transform(df_enc[col].astype(str))
    label_encoders[col] = le
y_full = df_enc["U5M"].values
if label_encoders["U5M"].classes_[1] != "Yes":
    y_full = 1 - y_full
X_full_predictors = df_enc[all_feature_cols].values

# -----------------------------------------------------------------------
# SMOTE + SUBSAMPLE + SPLIT (identical procedure to Script 2, full set)
# -----------------------------------------------------------------------
sm = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
X_bal, y_bal = sm.fit_resample(X_full_predictors, y_full)

N_PER_CLASS = 40000
idx0 = np.where(y_bal == 0)[0]; idx1 = np.where(y_bal == 1)[0]
np.random.shuffle(idx0); np.random.shuffle(idx1)
sub_idx = np.concatenate([idx0[:N_PER_CLASS], idx1[:N_PER_CLASS]])
X_sub, y_sub = X_bal[sub_idx], y_bal[sub_idx]

X_trainval, X_test_f, y_trainval, y_test_f = train_test_split(
    X_sub, y_sub, test_size=0.15, stratify=y_sub, random_state=RANDOM_STATE
)
X_train_f, _, y_train_f, _ = train_test_split(
    X_trainval, y_trainval, test_size=0.1765, stratify=y_trainval,
    random_state=RANDOM_STATE
)

# -----------------------------------------------------------------------
# TRAIN ALL FIVE ALGORITHMS ON THE FULL PREDICTOR SET (same grids as Script 3)
# -----------------------------------------------------------------------
PARAM_GRIDS = {
    "Random Forest": (
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced"),
        {"n_estimators": [200], "max_depth": [None, 10], "min_samples_leaf": [1, 5]}),
    "XGBoost": (
        XGBClassifier(random_state=RANDOM_STATE, n_jobs=-1, eval_metric="logloss", verbosity=0),
        {"n_estimators": [100, 200], "max_depth": [4, 6], "learning_rate": [0.05, 0.10]}),
    "GBM": (
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        {"n_estimators": [100, 200], "max_depth": [3, 5], "learning_rate": [0.05, 0.10]}),
    "LightGBM": (
        LGBMClassifier(random_state=RANDOM_STATE, class_weight="balanced", verbose=-1),
        {"n_estimators": [100, 200], "num_leaves": [31, 63], "learning_rate": [0.05, 0.10]}),
    "Extra Trees": (
        ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced"),
        {"n_estimators": [200], "max_depth": [None, 10], "min_samples_leaf": [1, 5]}),
}

full_results = []
for name, (estimator, grid) in PARAM_GRIDS.items():
    print(f"Training {name} on full 26-predictor set...")
    gs = GridSearchCV(estimator, grid,
                       cv=StratifiedKFold(3, shuffle=True, random_state=RANDOM_STATE),
                       scoring="roc_auc", n_jobs=-1)
    gs.fit(X_train_f, y_train_f)
    best_model = gs.best_estimator_
    best_model.fit(X_train_f, y_train_f)
    prob = best_model.predict_proba(X_test_f)[:, 1]
    pred = (prob >= 0.5).astype(int)
    full_results.append({
        "Algorithm": name,
        "AUC_full": round(roc_auc_score(y_test_f, prob), 4),
        "Sensitivity_full": round(recall_score(y_test_f, pred), 4),
        "F1_full": round(f1_score(y_test_f, pred), 4),
    })

full_df = pd.DataFrame(full_results)

# -----------------------------------------------------------------------
# MERGE WITH CONSENSUS-SELECTED (8-VARIABLE) RESULTS FROM SCRIPT 3
# -----------------------------------------------------------------------
selected_metrics = pd.read_csv("model_evaluation_metrics.csv")
selected_test = selected_metrics[selected_metrics["Set"] == "Test"][
    ["Algorithm", "AUC", "Sensitivity", "F1"]
].rename(columns={"AUC": "AUC_selected", "Sensitivity": "Sensitivity_selected",
                   "F1": "F1_selected"})

comparison_df = selected_test.merge(full_df, on="Algorithm")[
    ["Algorithm", "AUC_selected", "AUC_full",
     "Sensitivity_selected", "Sensitivity_full", "F1_selected", "F1_full"]
]
comparison_df.to_csv("table2_full_vs_selected_comparison.csv", index=False)

print("\n=== Table 2: Full (26 vars) vs Consensus-Selected (8 vars) ===")
print(comparison_df.to_string(index=False))
print("\nSaved: table2_full_vs_selected_comparison.csv")
