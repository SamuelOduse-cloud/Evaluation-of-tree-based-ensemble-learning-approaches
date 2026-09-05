"""
=============================================================================
MODEL TRAINING, HYPERPARAMETER TUNING, AND EVALUATION
=============================================================================
Study: Predicting Under-Five Mortality Using Tree-Based Ensemble Learning

Software environment:
    Python 3.12.3
    scikit-learn 1.8.0, xgboost 3.4.1, lightgbm 4.7.0, scipy 1.17.1

Algorithms: Random Forest, XGBoost, Gradient Boosting Machine (GBM),
            LightGBM, Extra Trees

Tuning strategy: GridSearchCV with 3-fold stratified cross-validation,
                 optimising ROC-AUC. See manuscript Table 3 for full grids.
=============================================================================
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                               GradientBoostingClassifier)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (roc_auc_score, accuracy_score, recall_score,
                              f1_score, confusion_matrix, brier_score_loss)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

RANDOM_STATE = 2025
np.random.seed(RANDOM_STATE)

# -----------------------------------------------------------------------
# LOAD PARTITIONED DATA 
# -----------------------------------------------------------------------
X_train = np.load("X_train.npy"); y_train = np.load("y_train.npy")
X_val   = np.load("X_val.npy");   y_val   = np.load("y_val.npy")
X_test  = np.load("X_test.npy");  y_test  = np.load("y_test.npy")

# -----------------------------------------------------------------------
# HYPERPARAMETER GRIDS 
# -----------------------------------------------------------------------
PARAM_GRIDS = {
    "Random Forest": (
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1,
                                class_weight="balanced"),
        {"n_estimators": [200], "max_depth": [None, 10], "min_samples_leaf": [1, 5]}
    ),
    "XGBoost": (
        XGBClassifier(random_state=RANDOM_STATE, n_jobs=-1,
                       eval_metric="logloss", verbosity=0),
        {"n_estimators": [100, 200], "max_depth": [4, 6], "learning_rate": [0.05, 0.10]}
    ),
    "GBM": (
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        {"n_estimators": [100, 200], "max_depth": [3, 5], "learning_rate": [0.05, 0.10]}
    ),
    "LightGBM": (
        LGBMClassifier(random_state=RANDOM_STATE, class_weight="balanced", verbose=-1),
        {"n_estimators": [100, 200], "num_leaves": [31, 63], "learning_rate": [0.05, 0.10]}
    ),
    "Extra Trees": (
        ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=-1,
                              class_weight="balanced"),
        {"n_estimators": [200], "max_depth": [None, 10], "min_samples_leaf": [1, 5]}
    ),
}

def compute_metrics(y_true, y_prob, set_name, algorithm):
    y_pred = (y_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    return {
        "Algorithm": algorithm, "Set": set_name,
        "AUC": round(roc_auc_score(y_true, y_prob), 4),
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Sensitivity": round(sensitivity, 4),
        "Specificity": round(specificity, 4),
        "F1": round(f1_score(y_true, y_pred), 4),
        "Balanced_Acc": round((sensitivity + specificity) / 2, 4),
        "Brier": round(brier_score_loss(y_true, y_prob), 4),
    }

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
all_results, cv_auc_by_algorithm, fitted_models = [], {}, {}

for name, (estimator, grid) in PARAM_GRIDS.items():
    print(f"\n=== Tuning {name} ===")
    gs = GridSearchCV(
        estimator, grid,
        cv=StratifiedKFold(3, shuffle=True, random_state=RANDOM_STATE),
        scoring="roc_auc", n_jobs=-1
    )
    gs.fit(X_train, y_train)
    best_model = gs.best_estimator_
    print(f"Best parameters: {gs.best_params_}")

    # 5-fold CV AUC (for the Friedman statistical test)
    fold_aucs = []
    for train_idx, val_idx in cv5.split(X_train, y_train):
        best_model.fit(X_train[train_idx], y_train[train_idx])
        p = best_model.predict_proba(X_train[val_idx])[:, 1]
        fold_aucs.append(roc_auc_score(y_train[val_idx], p))
    cv_auc_by_algorithm[name] = fold_aucs

    # Refit on full training set and evaluate on all three partitions
    best_model.fit(X_train, y_train)
    fitted_models[name] = best_model
    for split_name, X, y in [("Train", X_train, y_train),
                              ("Validation", X_val, y_val),
                              ("Test", X_test, y_test)]:
        prob = best_model.predict_proba(X)[:, 1]
        all_results.append(compute_metrics(y, prob, split_name, name))

    # Save test-set predicted probabilities (for ROC curves and SHAP)
    np.save(f"test_probs_{name.replace(' ', '_')}.npy",
            best_model.predict_proba(X_test)[:, 1])

results_df = pd.DataFrame(all_results)[
    ["Algorithm", "Set", "AUC", "Accuracy", "Sensitivity", "Specificity",
     "F1", "Balanced_Acc", "Brier"]
]
results_df.to_csv("model_evaluation_metrics.csv", index=False)
pd.DataFrame(cv_auc_by_algorithm).to_csv("cv_fold_auc_matrix.csv", index=False)

print("\n=== Final Test-Set Results ===")
print(results_df[results_df["Set"] == "Test"].to_string(index=False))
print("\nSaved: model_evaluation_metrics.csv, cv_fold_auc_matrix.csv, "
      "test_probs_<algorithm>.npy")
