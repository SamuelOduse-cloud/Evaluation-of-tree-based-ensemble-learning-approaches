"""
=============================================================================
SCRIPT 5 of 6: ALL FIGURES FOR MANUSCRIPT
=============================================================================
Study: Predicting Under-Five Mortality Using Tree-Based Ensemble Learning

Generates every figure reported in the manuscript:
    Figure 1 - Variable selection (RF importance / RFE / Boruta panels)
    Figure 2 - Cramer's V correlation heatmap
    Figure 3 - SMOTE before/after class distribution bar chart
    Figure 4 - ROC curves (test set, all five algorithms)
    Figure 5 - Calibration plot + Brier score bar chart
    Figure 6 - Five-fold cross-validation AUC per fold
    Figure 7 - Test-set performance grouped bar chart
    Figure 8 - Post-hoc Wilcoxon pairwise p-value heatmap
    Figure 9 - SHAP beeswarm + mean |SHAP| bar chart

Software environment:
    Python 3.12.3, matplotlib 3.10.8, seaborn 0.13.2, scipy 1.17.1,
    scikit-learn 1.8.0, shap 0.52.0

Prerequisite: Scripts 1-4 must be run first; this script reads their
              saved CSV / NPY outputs and writes PNG figures.
=============================================================================
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings("ignore")

from scipy.stats import chi2_contingency
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import LabelEncoder

plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 150})

ALGORITHMS = ["Random Forest", "XGBoost", "GBM", "LightGBM", "Extra Trees"]
COLORS = {"Random Forest": "#1F78B4", "XGBoost": "#E31A1C", "GBM": "#33A02C",
          "LightGBM": "#FF7F00", "Extra Trees": "#6A3D9A"}

# =============================================================================
# FIGURE 1: VARIABLE SELECTION (RF IMPORTANCE / RFE / BORUTA)
# =============================================================================
# Requires: rf_importance_scores.csv (Script 1), plus rfe_vars.csv and
# boruta_vars.csv, which Script 1 should also export (see note below).
# If Script 1 was run exactly as provided, uncomment the corresponding
# export lines in Script 1 (rf_vars, rfe_vars, boruta_vars, selected_vars)
# before running this figure block.

importances = pd.read_csv("rf_importance_scores.csv", index_col=0)["importance"]
selected_vars = pd.read_csv("selected_variables.csv", header=None)[0].tolist()

fig, ax = plt.subplots(figsize=(10, 7))
colors_bar = ["#084594" if v in selected_vars else "#9ECAE1" for v in importances.index]
ax.barh(importances.index, importances.values, color=colors_bar, edgecolor="white")
ax.axvline(importances.mean(), color="red", linestyle="--", lw=1.8, label="Mean threshold")
ax.set_xlabel("Mean Decrease in Impurity")
ax.set_title("Figure 1. Random Forest Variable Importance\n(Blue = consensus-selected predictors)")
leg = [mpatches.Patch(facecolor="#084594", label="Selected"),
       mpatches.Patch(facecolor="#9ECAE1", label="Not selected")]
ax.legend(handles=leg, loc="lower right")
plt.tight_layout()
plt.savefig("Figure1_Variable_Importance.png", dpi=180, bbox_inches="tight")
plt.close()
print("Saved: Figure1_Variable_Importance.png")

# =============================================================================
# FIGURE 2: CRAMER'S V CORRELATION HEATMAP
# =============================================================================
df_orig = pd.read_excel("Mortality_data.xlsx")
hmap_vars = selected_vars + ["U5M"]
df_hmap = df_orig[hmap_vars].astype(str)

def cramers_v(x, y):
    ct = pd.crosstab(x, y)
    chi2 = chi2_contingency(ct, correction=False)[0]
    n = len(x)
    r, k = ct.shape
    phi2_corr = max(0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    k_corr = k - (k - 1) ** 2 / (n - 1)
    denom = min(r_corr - 1, k_corr - 1)
    return np.sqrt(phi2_corr / denom) if denom > 0 else 0.0

n_vars = len(hmap_vars)
cv_matrix = np.zeros((n_vars, n_vars))
for i in range(n_vars):
    for j in range(n_vars):
        cv_matrix[i, j] = 1.0 if i == j else cramers_v(df_hmap.iloc[:, i], df_hmap.iloc[:, j])
cv_df = pd.DataFrame(cv_matrix, index=hmap_vars, columns=hmap_vars)
cv_df.to_csv("cramers_v_matrix.csv")   # save underlying numeric table too

fig, ax = plt.subplots(figsize=(10, 9))
sns.heatmap(cv_df, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
            linewidths=0.5, linecolor="white", ax=ax,
            cbar_kws={"label": "Cramer's V"})
ax.set_title("Figure 2. Cramer's V Correlation Heatmap\nSelected Predictors + U5M Outcome")
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.savefig("Figure2_CramersV_Heatmap.png", dpi=180, bbox_inches="tight")
plt.close()
print("Saved: Figure2_CramersV_Heatmap.png, cramers_v_matrix.csv")

# =============================================================================
# FIGURE 3: SMOTE BEFORE / AFTER CLASS DISTRIBUTION
# =============================================================================
from imblearn.over_sampling import SMOTE

df_enc = df_orig.drop(columns=["COUNTRY"])
label_encoders = {}
for col in df_enc.columns:
    le = LabelEncoder()
    df_enc[col] = le.fit_transform(df_enc[col].astype(str))
    label_encoders[col] = le
y_full = df_enc["U5M"].values
if label_encoders["U5M"].classes_[1] != "Yes":
    y_full = 1 - y_full
X_selected = df_enc[selected_vars].values

before_counts = dict(zip(*np.unique(y_full, return_counts=True)))
sm = SMOTE(random_state=2025, k_neighbors=5)
_, y_bal = sm.fit_resample(X_selected, y_full)
after_counts = dict(zip(*np.unique(y_bal, return_counts=True)))

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
labels = ["Alive (No)", "Died (Yes)"]
bar_colors = ["#2171B5", "#CB181D"]
for ax, counts, title in zip(axes, [before_counts, after_counts],
                              ["Before SMOTE", "After SMOTE"]):
    vals = [counts.get(0, 0), counts.get(1, 0)]
    bars = ax.bar(labels, vals, color=bar_colors, width=0.55, edgecolor="white")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v:,}", ha="center", va="bottom", fontweight="bold")
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel("Frequency")
fig.suptitle("Figure 3. U5M Class Distribution Before and After SMOTE", fontweight="bold")
plt.tight_layout()
plt.savefig("Figure3_SMOTE_BarChart.png", dpi=180, bbox_inches="tight")
plt.close()
print("Saved: Figure3_SMOTE_BarChart.png")

# =============================================================================
# FIGURE 4: ROC CURVES (TEST SET)
# =============================================================================
y_test = np.load("y_test.npy")

fig, ax = plt.subplots(figsize=(8, 7))
for algo in ALGORITHMS:
    prob = np.load(f"test_probs_{algo.replace(' ', '_')}.npy")
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc_val = roc_auc_score(y_test, prob)
    ax.plot(fpr, tpr, lw=2.2, color=COLORS[algo], label=f"{algo} (AUC={auc_val:.3f})")
ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="Random (AUC=0.500)")
ax.set_xlabel("False Positive Rate (1 - Specificity)")
ax.set_ylabel("True Positive Rate (Sensitivity)")
ax.set_title("Figure 4. ROC Curves - Test Set\nFive Tree-Based Ensemble Algorithms")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("Figure4_ROC_Curves.png", dpi=180, bbox_inches="tight")
plt.close()
print("Saved: Figure4_ROC_Curves.png")

# =============================================================================
# FIGURE 5: CALIBRATION PLOT + BRIER SCORE BAR CHART
# =============================================================================
from sklearn.metrics import brier_score_loss

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
ax1, ax2 = axes
ax1.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calibration")
brier_scores = {}
for algo in ALGORITHMS:
    prob = np.load(f"test_probs_{algo.replace(' ', '_')}.npy")
    frac_pos, mean_pred = calibration_curve(y_test, prob, n_bins=10, strategy="uniform")
    ax1.plot(mean_pred, frac_pos, "o-", color=COLORS[algo], label=algo)
    brier_scores[algo] = round(brier_score_loss(y_test, prob), 4)
ax1.set_xlabel("Mean Predicted Probability")
ax1.set_ylabel("Fraction of Positives")
ax1.set_title("(A) Calibration Plot")
ax1.legend(fontsize=9)

bars = ax2.bar(list(brier_scores.keys()), list(brier_scores.values()),
               color=[COLORS[a] for a in brier_scores], edgecolor="white")
for bar, v in zip(bars, brier_scores.values()):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
              f"{v:.4f}", ha="center", va="bottom", fontweight="bold")
ax2.set_ylabel("Brier Score (lower = better)")
ax2.set_title("(B) Brier Scores - Test Set")
ax2.set_xticklabels(brier_scores.keys(), rotation=20, ha="right")
fig.suptitle("Figure 5. Model Calibration Assessment", fontweight="bold")
plt.tight_layout()
plt.savefig("Figure5_Calibration.png", dpi=180, bbox_inches="tight")
plt.close()
pd.Series(brier_scores, name="Brier").to_csv("brier_scores.csv", header=True)
print("Saved: Figure5_Calibration.png, brier_scores.csv")

# =============================================================================
# FIGURE 6: FIVE-FOLD CROSS-VALIDATION AUC PER FOLD
# =============================================================================
cv_auc = pd.read_csv("cv_fold_auc_matrix.csv")

fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(5)
offsets = np.linspace(-0.3, 0.3, 5)
width = 0.15
for i, algo in enumerate(ALGORITHMS):
    ax.bar(x + offsets[i], cv_auc[algo].values, width, label=algo,
           color=COLORS[algo], alpha=0.88, edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels([f"Fold {i+1}" for i in range(5)])
ax.set_ylabel("AUC (ROC)")
ax.set_title("Figure 6. Five-Fold Cross-Validation AUC - All Algorithms")
ax.legend(fontsize=9, loc="lower right")
plt.tight_layout()
plt.savefig("Figure6_CV_Fold_AUC.png", dpi=180, bbox_inches="tight")
plt.close()
print("Saved: Figure6_CV_Fold_AUC.png")

# =============================================================================
# FIGURE 7: TEST-SET PERFORMANCE GROUPED BAR CHART
# =============================================================================
metrics_df = pd.read_csv("model_evaluation_metrics.csv")
test_df = metrics_df[metrics_df["Set"] == "Test"]
metric_cols = ["AUC", "Accuracy", "Sensitivity", "Specificity", "F1"]

x = np.arange(len(metric_cols))
width = 0.15
fig, ax = plt.subplots(figsize=(13, 7))
for i, algo in enumerate(ALGORITHMS):
    row = test_df[test_df["Algorithm"] == algo][metric_cols].values.flatten()
    ax.bar(x + (i - 2) * width, row, width, label=algo, color=COLORS[algo],
           alpha=0.88, edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(metric_cols)
ax.set_ylabel("Metric Value")
ax.set_title("Figure 7. Test-Set Performance Comparison - Five Algorithms")
ax.legend()
plt.tight_layout()
plt.savefig("Figure7_TestSet_Performance.png", dpi=180, bbox_inches="tight")
plt.close()
print("Saved: Figure7_TestSet_Performance.png")

# =============================================================================
# FIGURE 8: POST-HOC WILCOXON PAIRWISE P-VALUE HEATMAP
# =============================================================================
pval_df = pd.read_csv("posthoc_wilcoxon_pvalues.csv", index_col=0)
friedman_res = pd.read_csv("friedman_test_result.csv")
chi2_val = friedman_res["chi_squared"].iloc[0]
p_val = friedman_res["p_value"].iloc[0]

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(pval_df, annot=True, fmt=".3f", cmap="Blues_r", vmin=0, vmax=0.2,
            linewidths=0.5, linecolor="white", ax=ax, cbar_kws={"label": "p-value"})
ax.set_title(f"Figure 8. Post-hoc Pairwise Comparison (Wilcoxon)\n"
             f"Friedman test: chi2={chi2_val:.3f}, p={p_val:.4f}")
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
plt.tight_layout()
plt.savefig("Figure8_Posthoc_Heatmap.png", dpi=180, bbox_inches="tight")
plt.close()
print("Saved: Figure8_Posthoc_Heatmap.png")

# =============================================================================
# FIGURE 9: SHAP BEESWARM + MEAN |SHAP| BAR CHART
# =============================================================================
import shap
from lightgbm import LGBMClassifier

X_train = np.load("X_train.npy"); y_train = np.load("y_train.npy")
X_test_full = np.load("X_test.npy")

lgb_model = LGBMClassifier(
    n_estimators=200, num_leaves=31, learning_rate=0.1,
    random_state=2025, class_weight="balanced", verbose=-1, n_jobs=-1
)
lgb_model.fit(X_train, y_train)

explainer = shap.TreeExplainer(lgb_model)
shap_values = explainer.shap_values(X_test_full[:1000])
sv = shap_values[1] if isinstance(shap_values, list) else shap_values

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
plt.sca(axes[0])
shap.summary_plot(sv, X_test_full[:1000], feature_names=selected_vars,
                   show=False, max_display=len(selected_vars), plot_type="dot")
axes[0].set_title("(A) SHAP Beeswarm Plot")
plt.sca(axes[1])
shap.summary_plot(sv, X_test_full[:1000], feature_names=selected_vars,
                   show=False, max_display=len(selected_vars), plot_type="bar")
axes[1].set_title("(B) Mean |SHAP| Feature Importance")
fig.suptitle("Figure 9. SHAP Explainability Analysis - LightGBM Model", fontweight="bold")
plt.tight_layout()
plt.savefig("Figure9_SHAP.png", dpi=180, bbox_inches="tight")
plt.close()
print("Saved: Figure9_SHAP.png")

print("\n=== All 9 figures generated successfully ===")
