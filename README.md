# 🫀 Indian Liver Patient Predictor

End-to-end machine learning project to predict **liver disease (Yes/No)** in Indian patients using clinical biomarkers. Covers EDA, preprocessing, feature selection, model comparison (Logistic Regression, RF, XGBoost, SVM, KNN), SHAP explainability, and a Streamlit app.

Personal project by **Rayala Madhu Bhanu Varma**.

---

## Results

| Model | CV Accuracy | CV F1 | CV ROC-AUC |
|---|---|---|---|
| **Random Forest** ⭐ | **0.983** | **0.983** | **0.997** |
| SVM (RBF) | 0.985 | 0.985 | 0.995 |
| XGBoost | 0.983 | 0.983 | 0.994 |
| Logistic Regression | 0.976 | 0.975 | 0.994 |
| KNN (k=7) | 0.938 | 0.934 | 0.985 |

Test set: **98.3% accuracy · 0.999 AUC**

> Logistic Regression achieves 97.6% — confirming the original ILPD finding that simple linear models perform strongly on liver function test data.

---

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/indian-liver-patient-predictor
cd indian-liver-patient-predictor
pip install -r requirements.txt

# Run full ML pipeline (generates data + trains + saves all charts)
python src/pipeline.py

# Launch Streamlit app
streamlit run app.py
```

---

## Dataset

**583 synthetic patients** matching the original ILPD (UCI ML Repository, Ramana et al. 2012) distribution:
- ~71% liver disease (415 patients)
- ~29% no disease (168 patients)
- 28 missing values in Albumin/Globulin Ratio

### Features

| Feature | Unit | Clinical role |
|---|---|---|
| Age | years | Risk increases with age |
| Gender | M/F | Males at higher risk |
| Total Bilirubin | mg/dL | Jaundice marker — elevated in liver damage |
| Direct Bilirubin | mg/dL | Conjugated fraction — hepatocellular damage |
| Alkaline Phosphotase | IU/L | Cholestasis and biliary obstruction |
| SGPT / ALT | IU/L | Primary hepatocellular damage marker |
| SGOT / AST | IU/L | Liver + cardiac damage |
| Total Proteins | g/dL | Synthetic function of liver |
| Albumin | g/dL | Drops in chronic liver disease |
| Albumin/Globulin Ratio | — | Reversal = poor prognosis |

### Preprocessing steps
1. Median imputation for missing AG Ratio values
2. Gender label encoding (Male=1, Female=0)
3. Log-transform of 5 skewed enzyme columns (Total Bilirubin, Direct Bilirubin, ALP, ALT, AST)
4. Oversampling minority class (SMOTE-style resample) on training set
5. StandardScaler normalisation

---

## Pipeline Steps

```
src/pipeline.py
├── Step 1  Load / generate data
├── Step 2  EDA (8 charts)
│     class balance, gender split, age histogram, biomarker violins,
│     correlation heatmap, SGPT–SGOT trend, bilirubin scatter, albumin box
├── Step 3  Preprocessing (impute, encode, log-transform, balance, scale)
├── Step 4  Feature selection (mutual information)
├── Step 5  5-fold CV model comparison (5 models, 5 metrics, bar + radar)
├── Step 6  Best model evaluation (confusion matrix, ROC, PR, threshold analysis)
├── Step 7  SHAP (global importance, beeswarm, waterfall)
└── Step 8  Save artifacts + dashboard index
```

---

## Streamlit App

| Page | Content |
|---|---|
| **🩺 Patient Predictor** | Enter all biomarker values → prediction + confidence bars + clinical flags + SHAP waterfall |
| **📊 EDA Explorer** | 6 interactive tabs: class/gender, violin, correlation, SGPT–SGOT, bilirubin, albumin |
| **🏆 Model Performance** | CV comparison, radar, confusion matrix, ROC, PR, threshold sensitivity |
| **💡 SHAP Explainability** | Global importance, beeswarm, per-patient waterfall with slider |

<img width="1738" height="910" alt="image" src="https://github.com/user-attachments/assets/3aab15a2-74ed-4192-9c93-1436e6f46d04" />

---

## Project Structure

```
liver_predictor/
├── app.py                             # Streamlit app (4 pages)
├── src/
│   ├── data_generator.py              # Synthetic ILPD-style dataset
│   └── pipeline.py                    # 8-step ML pipeline
├── data/
│   └── indian_liver_patients.csv      # Generated on first run
├── models/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   ├── features.pkl
│   └── model_summary.json
├── output/
│   ├── index.html                     # Dashboard index
│   ├── eda_*.html                     # 8 EDA charts
│   ├── model_comparison.html
│   ├── model_radar.html
│   ├── confusion_matrix.html
│   ├── roc_curve.html
│   ├── pr_curve.html
│   ├── threshold_analysis.html
│   ├── feature_selection.html
│   └── shap_*.html                    # 3 SHAP charts
├── requirements.txt
└── README.md
```

---

## Tech Stack

| Layer | Tool |
|---|---|
| ML | scikit-learn, XGBoost |
| Explainability | SHAP |
| App | Streamlit |
| Charts | Plotly |
| Stats | statsmodels (OLS trendlines) |
| Data | pandas, numpy |

---

## Reference

Ramana, B.V., Babu, M.S.P., & Venkateswarlu, N.B. (2012). A Critical Comparative Study of Liver Patients with Healthy People using Classification Algorithms. *International Journal of Computer Science Issues*, 9(5).

UCI ML Repository: [Indian Liver Patient Dataset](https://archive.ics.uci.edu/ml/datasets/ILPD+(Indian+Liver+Patient+Dataset))

---

