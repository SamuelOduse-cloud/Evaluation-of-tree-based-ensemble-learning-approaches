"""
=============================================================================
SCRIPT 7 of 7: COMPILE ALL MANUSCRIPT TABLES
=============================================================================
Study: Predicting Under-Five Mortality Using Tree-Based Ensemble Learning

Compiles every table reported in the manuscript into individual CSV files
(and prints each to console for verification). This script assumes
Scripts 1, 2, 3, 4, and 6 have already been run.

Tables produced:
    Table 1 - Under-five mortality rate by country
    Table 2 - Full (26 vars) vs. consensus-selected (8 vars) comparison
    Table 3 - Model evaluation metrics (Train / Validation / Test)
    Table 4 - Brier scores (calibration)
    Table 5 - SHAP mean absolute importance
    Table 6 - Post-hoc Wilcoxon pairwise p-values
    Table 7 - Friedman test result summary
=============================================================================
"""
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------
# TABLE 1: UNDER-FIVE MORTALITY RATE BY COUNTRY
# -----------------------------------------------------------------------
df = pd.read_excel("Mortality_data.xlsx")

country_summary = (
    df.groupby("COUNTRY")
    .apply(lambda g: pd.Series({
        "N": len(g),
        "Deaths": (g["U5M"] == "Yes").sum(),
        "U5M_rate_per_1000": round((g["U5M"] == "Yes").sum() / len(g) * 1000, 1)
    }))
    .reset_index()
    .sort_values("U5M_rate_per_1000", ascending=False)
)

# Append pooled total row
pooled_row = pd.DataFrame([{
    "COUNTRY": "TOTAL / POOLED",
    "N": len(df),
    "Deaths": (df["U5M"] == "Yes").sum(),
    "U5M_rate_per_1000": round((df["U5M"] == "Yes").sum() / len(df) * 1000, 1)
}])
table1 = pd.concat([country_summary, pooled_row], ignore_index=True)
table1.to_csv("Table1_Country_U5M_Rates.csv", index=False)
print("=== Table 1: Under-Five Mortality Rate by Country ===")
print(table1.to_string(index=False))

# -----------------------------------------------------------------------
# TABLE 2: FULL vs. CONSENSUS-SELECTED PREDICTOR COMPARISON
# -----------------------------------------------------------------------
try:
    table2 = pd.read_csv("table2_full_vs_selected_comparison.csv")
    print("\n=== Table 2: Full vs. Consensus-Selected Predictor Comparison ===")
    print(table2.to_string(index=False))
except FileNotFoundError:
    print("\n[Table 2 not found - run Script 6 first]")

# -----------------------------------------------------------------------
# TABLE 3: MODEL EVALUATION METRICS (TRAIN / VALIDATION / TEST)
# -----------------------------------------------------------------------
try:
    table3 = pd.read_csv("model_evaluation_metrics.csv")
    table3.to_csv("Table3_Model_Evaluation_Metrics.csv", index=False)
    print("\n=== Table 3: Model Evaluation Metrics ===")
    print(table3.to_string(index=False))
except FileNotFoundError:
    print("\n[Table 3 not found - run Script 3 first]")

# -----------------------------------------------------------------------
# TABLE 4: BRIER SCORES (CALIBRATION)
# -----------------------------------------------------------------------
try:
    table4 = pd.read_csv("brier_scores.csv")
    table4.columns = ["Algorithm", "Brier_Score"]
    table4.to_csv("Table4_Brier_Scores.csv", index=False)
    print("\n=== Table 4: Brier Scores (Calibration) ===")
    print(table4.to_string(index=False))
except FileNotFoundError:
    print("\n[Table 4 not found - run Script 5 (figures) first, which computes Brier scores]")

# -----------------------------------------------------------------------
# TABLE 5: SHAP MEAN ABSOLUTE IMPORTANCE
# -----------------------------------------------------------------------
try:
    table5 = pd.read_csv("shap_mean_abs_importance.csv")
    table5.to_csv("Table5_SHAP_Importance.csv", index=False)
    print("\n=== Table 5: SHAP Mean Absolute Importance ===")
    print(table5.to_string(index=False))
except FileNotFoundError:
    print("\n[Table 5 not found - run Script 4 first]")

# -----------------------------------------------------------------------
# TABLE 6: POST-HOC WILCOXON PAIRWISE P-VALUES
# -----------------------------------------------------------------------
try:
    table6 = pd.read_csv("posthoc_wilcoxon_pvalues.csv", index_col=0)
    table6.to_csv("Table6_Posthoc_Wilcoxon_Pvalues.csv")
    print("\n=== Table 6: Post-hoc Pairwise Wilcoxon p-values ===")
    print(table6.to_string())
except FileNotFoundError:
    print("\n[Table 6 not found - run Script 4 first]")

# -----------------------------------------------------------------------
# TABLE 7: FRIEDMAN TEST RESULT SUMMARY
# -----------------------------------------------------------------------
try:
    table7 = pd.read_csv("friedman_test_result.csv")
    table7.to_csv("Table7_Friedman_Test_Result.csv", index=False)
    print("\n=== Table 7: Friedman Test Result ===")
    print(table7.to_string(index=False))
except FileNotFoundError:
    print("\n[Table 7 not found - run Script 4 first]")

print("\n=== All available tables compiled and saved as individual CSV files ===")
