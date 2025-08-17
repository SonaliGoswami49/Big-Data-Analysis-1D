# Disease Outbreak Prediction Using Symptoms, Weather, and Case Data

## Project Overview
I predict weekly disease outbreaks in Australia using:
- Symptom search trends (Google Trends)
- COVID-19 case data (OWID)
- Weather features (Meteostat)

I engineer lag, delta, and rolling features, then train and compare three models:
- Logistic Regression
- Random Forest
- XGBoost

Evaluation focuses on recall-oriented metrics (Fβ with β=1.5, PR-AUC) to prioritize detecting outbreaks.

 **Repository Structure**
 
.
├── raw/processed datasets

├── Big_Data_Analysis_Final_Assignment_Submission.ipynb # main notebook 

├── model_metrics_combined.csv # exported metrics table (created by notebook)

├── pr_validation.png # PR curve on validation (created by notebook)

├── pr_test.png # PR curve on test (created by notebook)

├── roc_validation.png # ROC curve on validation (created by notebook)

├── roc_test.png # ROC curve on test (created by notebook)

└── README.md

**Installation**

**Prerequisites**
- Python 3.9 or newer
- Jupyter Notebook or JupyterLab
- Google Colab

**Set up a virtual environment (recommended)**

Windows
.venv\Scripts\activate

macOS/Linux
source .venv/bin/activate


**Requirements**
 
numpy>=1.24
pandas>=2.0
matplotlib>=3.7
scikit-learn>=1.3
xgboost>=2.0
jupyter>=1.0

**Install dependencies**

pip install numpy pandas matplotlib scikit-learn xgboost


**How to Run**

Place datasets in data/ (optional, depending on your notebook’s data-loading code).

Launch Jupyter and open the notebook:

Run all cells in order:

Feature engineering

Outbreak label + stratified splits

Cross-validation and threshold policy

Hyperparameter tuning (Logistic, RF, XGB)

Probability calibration

Validation/Test evaluation with PR/ROC curves

Combined metrics table and plots

Outputs saved by the notebook:

model_metrics_combined.csv

pr_validation.png, pr_test.png

roc_validation.png, roc_test.png

**Reproducibility Notes**

I set random_state=42 in CV and models where applicable.

Exact results can vary with data filtering and library versions.

**Method Summary (for markers/reviewers)**

Outbreak definition: week is an outbreak if new_cases ≥ global 85th percentile.

Thresholding: choose threshold on validation by maximizing Fβ (β=1.5) with a recall floor, then evaluate on test.

Metrics: Precision, Recall, F1, ROC-AUC, PR-AUC; emphasis on Recall and PR-AUC for imbalanced detection.

Calibration: isotonic calibration to improve probability estimates (esp. tree-based models).

**Troubleshooting**

If xgboost is missing, install with pip install xgboost.

If plots/CSV don’t appear, confirm you ran the evaluation cells and that the working directory is the repo root.
