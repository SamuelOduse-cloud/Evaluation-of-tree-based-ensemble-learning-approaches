# Reproducibility Package
## Predicting Under-Five Mortality Using Tree-Based Ensemble Learning

This repository contains the complete code required to reproduce all
analyses, figures, and tables reported in the manuscript.

### Data Access
The dataset is derived from the Demographic and Health Surveys (DHS)
Program (https://dhsprogram.com/data/), pooled across 27 Sub-Saharan
African countries. Access requires a data use agreement with the DHS
Program. Data were accessed on 19 Jan 2025.
Due to DHS data-sharing restrictions, the raw microdata cannot be
redistributed in this repository; researchers must request access
directly from https://dhsprogram.com/data/new-user-registration.aspx.

### Environment Setup
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Execution Order
Run scripts sequentially; each script writes intermediate outputs
(CSV / NPY files) consumed by later scripts.

1. `01_preprocessing_and_variable_selection.py`
   Loads raw data, encodes variables, runs the three-method consensus
   variable selection (RF importance, RFE, Boruta).

2. `02_smote_balancing_and_splitting.py`
   Applies SMOTE and partitions the data into train/validation/test
   (70/15/15) using the consensus-selected predictors.

3. `03_model_training_and_tuning.py`
   Trains and tunes all five algorithms (Random Forest, XGBoost, GBM,
   LightGBM, Extra Trees) via grid-search cross-validation; computes
   evaluation metrics for all three partitions.

4. `04_statistical_tests_and_shap.py`
   Runs the Friedman test and post-hoc Wilcoxon comparisons; computes
   SHAP values for the best-performing model (LightGBM).

5. `05_figures.py`
   Generates **all nine manuscript figures**: variable importance panel,
   Cramer's V heatmap, SMOTE bar chart, ROC curves, calibration plot +
   Brier scores, cross-validation fold AUC chart, test-set performance
   bar chart, post-hoc p-value heatmap, and SHAP beeswarm/bar charts.

6. `06_full_vs_selected_comparison.py`
   Re-runs the pipeline on the full 26-predictor set and compares
   performance against the 8-variable consensus model (Table 2).

7. `07_compile_all_tables.py`
   Compiles every manuscript table (Tables 1-7) into individual,
   clearly named CSV files for direct inclusion in the manuscript.

### Random Seed
`random_state = 2025` is fixed throughout for all stochastic operations.

### Outputs
Each script writes its outputs (CSV/PNG/NPY files) to the working
directory. Figures are saved with descriptive filenames matching their
manuscript figure numbers (e.g., `Figure4_ROC_Curves.png`). Tables are
saved with matching filenames (e.g., `Table3_Model_Evaluation_Metrics.csv`).

### Citation
If you use this code, please cite the associated manuscript:
[Author(s), Year, Journal, DOI — to be finalised upon publication]

### License
[Insert chosen license, e.g., MIT / CC-BY 4.0]
