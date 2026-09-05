"""
=============================================================================
STATISTICAL COMPARISON (FRIEDMAN + WILCOXON) AND SHAP
=============================================================================
Study: Predicting Under-Five Mortality Using Tree-Based Ensemble Learning

Software environment:
    Python 3.12.3
    scipy 1.17.1, shap 0.52.0, lightgbm 4.7.0

Statistical framework: Demsar (2006) — Friedman omnibus test followed by
                        post-hoc pairwise Wilcoxon signed-rank tests.
=============================================================================
"""
import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

from scipy.stats import friedmanchisquare, wilcoxon
from lightgbm import LGBMClassifier
import shap

RANDOM_STATE = 2025
np.random.seed(RANDOM_STATE)

ALGORITHMS = ["Random Forest", "XGBoost", "GBM", "LightGBM", "Extra Trees"]

# -----------------------------------------------------------------------
# PART A: FRIEDMAN TEST + POST-HOC WILCOXON 
# -----------------------------------------------------------------------
cv_auc = pd.read_csv("cv_fold_auc_matrix.csv")
cv_matrix = cv_auc[ALGORITHMS].values     # shape: (5 folds, 5 algorithms)

chi2_stat, p_value = friedmanchisquare(*[cv_matrix[:, i] for i in range(5)])
print(f"Friedman test: chi-squared = {chi2_stat:.4f}, df = 4, p = {p_value:.4f}")

n_alg = len(ALGORITHMS)
pval_matrix = np.ones((n_alg, n_alg))
for i in range(n_alg):
    for j in range(n_alg):
        if i != j:
            try:
                _, p = wilcoxon(cv_matrix[:, i], cv_matrix[:, j])
            except ValueError:
                p = 1.0
            pval_matrix[i, j] = p

pval_df = pd.DataFrame(pval_matrix, index=ALGORITHMS, columns=ALGORITHMS)
pval_df.to_csv("posthoc_wilcoxon_pvalues.csv")
pd.DataFrame({"chi_squared": [chi2_stat], "p_value": [p_value]}).to_csv(
    "friedman_test_result.csv", index=False
)
print("Saved: posthoc_wilcoxon_pvalues.csv, friedman_test_result.csv")

# -----------------------------------------------------------------------
# PART B: SHAP EXPLAINABILITY ANALYSIS
# -----------------------------------------------------------------------
X_train = np.load("X_train.npy"); y_train = np.load("y_train.npy")
X_test  = np.load("X_test.npy")
selected_vars = pd.read_csv("selected_variables.csv", header=None)[0].tolist()

# Refit LightGBM with its tuned hyperparameters
lgb_model = LGBMClassifier(
    n_estimators=200, num_leaves=31, learning_rate=0.1,
    random_state=RANDOM_STATE, class_weight="balanced", verbose=-1, n_jobs=-1
)
lgb_model.fit(X_train, y_train)

# Compute SHAP values on 1,000 randomly selected test-set observations
explainer = shap.TreeExplainer(lgb_model)
shap_values = explainer.shap_values(X_test[:1000])
sv = shap_values[1] if isinstance(shap_values, list) else shap_values

mean_abs_shap = pd.Series(
    np.abs(sv).mean(axis=0), index=selected_vars
).sort_values(ascending=False)
mean_abs_shap.to_csv("shap_mean_abs_importance.csv", header=["mean_abs_shap"])

print("\nSHAP mean |importance| ranking:")
print(mean_abs_shap.to_string())
print("\nSaved: shap_mean_abs_importance.csv")
