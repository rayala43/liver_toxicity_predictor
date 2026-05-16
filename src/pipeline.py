"""
src/pipeline.py
---------------
Indian Liver Patient Prediction — Full ML Pipeline

Step 1  Data loading & overview
Step 2  EDA — distributions, class balance, correlations, trends
Step 3  Preprocessing — missing values, encoding, scaling, class imbalance
Step 4  Feature selection — mutual info + correlation analysis
Step 5  Model comparison — Logistic Regression, Random Forest, XGBoost, SVM, KNN
Step 6  Best model evaluation — confusion matrix, ROC, PR, threshold analysis
Step 7  SHAP explainability
Step 8  Save artifacts

Run: python src/pipeline.py
"""

import sys, warnings, joblib, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, accuracy_score,
)
from sklearn.utils import resample
from xgboost import XGBClassifier
import shap

from src.data_generator import (
    generate_dataset, FEATURES, NUMERIC_FEATURES,
    FEATURE_LABELS, CLASS_LABELS, CLASS_COLORS,
)

OUT  = Path("output")
MOD  = Path("models")
OUT.mkdir(exist_ok=True)
MOD.mkdir(exist_ok=True)

SEED = 42
CV   = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load data
# ═══════════════════════════════════════════════════════════════════════════════

def step1_load(csv_path="data/indian_liver_patients.csv") -> pd.DataFrame:
    p = Path(csv_path)
    if not p.exists():
        print("  Generating dataset …")
        df = generate_dataset(save_path=csv_path)
    else:
        df = pd.read_csv(csv_path)
    print(f"  {len(df)} records · {df.shape[1]-1} features")
    print(f"  Class split: { {CLASS_LABELS[k]: v for k,v in df['Dataset'].value_counts().sort_index().items()} }")
    print(f"  Missing values: {df.isnull().sum().sum()} (AG Ratio only)")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — EDA
# ═══════════════════════════════════════════════════════════════════════════════

def step2_eda(df: pd.DataFrame):
    print("  Generating EDA charts …")
    colors = [CLASS_COLORS[1], CLASS_COLORS[2]]

    # 2a. Class balance
    counts = df["Dataset"].value_counts().sort_index()
    fig = go.Figure(go.Bar(
        x=[CLASS_LABELS[i] for i in counts.index],
        y=counts.values,
        marker_color=colors,
        text=[f"{v} ({v/len(df)*100:.1f}%)" for v in counts.values],
        textposition="outside",
    ))
    fig.update_layout(title="Class Distribution — Indian Liver Patient Dataset",
                      xaxis_title="Class", yaxis_title="Count",
                      plot_bgcolor="white", paper_bgcolor="white", height=360)
    fig.write_html(str(OUT/"eda_class_balance.html"), include_plotlyjs="cdn")

    # 2b. Gender split by class
    gender_ct = df.groupby(["Dataset","Gender"]).size().reset_index(name="count")
    gender_ct["Class"] = gender_ct["Dataset"].map(CLASS_LABELS)
    fig = px.bar(gender_ct, x="Class", y="count", color="Gender",
                 barmode="group", color_discrete_map={"Male":"#3B82F6","Female":"#EC4899"},
                 title="Gender Distribution by Class", text="count")
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=380)
    fig.write_html(str(OUT/"eda_gender.html"), include_plotlyjs="cdn")

    # 2c. Age distribution by class (histogram + KDE overlay)
    fig = go.Figure()
    for cls, color in CLASS_COLORS.items():
        vals = df[df["Dataset"]==cls]["Age"]
        fig.add_trace(go.Histogram(
            x=vals, name=CLASS_LABELS[cls],
            marker_color=color, opacity=0.65,
            xbins=dict(size=5), histnorm="probability density",
        ))
    fig.update_layout(barmode="overlay",
                      title="Age Distribution by Class",
                      xaxis_title="Age (years)", yaxis_title="Density",
                      plot_bgcolor="white", paper_bgcolor="white", height=380)
    fig.write_html(str(OUT/"eda_age.html"), include_plotlyjs="cdn")

    # 2d. Violin plots for all numeric biomarkers
    numeric_key = ["Total_Bilirubin","Direct_Bilirubin","Alkaline_Phosphotase",
                   "Alamine_Aminotransferase","Aspartate_Aminotransferase",
                   "Total_Protiens","Albumin","Albumin_and_Globulin_Ratio"]
    fig = make_subplots(rows=2, cols=4,
                        subplot_titles=[FEATURE_LABELS[f] for f in numeric_key])
    for idx, feat in enumerate(numeric_key):
        r, c = divmod(idx, 4)
        for cls, color in CLASS_COLORS.items():
            vals = df[df["Dataset"]==cls][feat].dropna()
            fig.add_trace(go.Violin(
                y=vals, name=CLASS_LABELS[cls],
                line_color=color, fillcolor=color,
                opacity=0.6, showlegend=(idx==0),
                legendgroup=CLASS_LABELS[cls],
                box_visible=True, meanline_visible=True,
            ), row=r+1, col=c+1)
    fig.update_layout(title="Biomarker Distributions by Class",
                      height=600, violinmode="overlay",
                      plot_bgcolor="white", paper_bgcolor="white")
    fig.write_html(str(OUT/"eda_biomarkers.html"), include_plotlyjs="cdn")

    # 2e. Correlation heatmap
    num_df = df[NUMERIC_FEATURES].copy()
    corr = num_df.corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
        text=corr.values.round(2), texttemplate="%{text}",
        textfont=dict(size=9),
        hovertemplate="%{y} × %{x}<br>r = %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(title="Pearson Correlation Matrix (numeric features)",
                      height=520, plot_bgcolor="white", paper_bgcolor="white",
                      xaxis=dict(tickangle=-35, tickfont=dict(size=9)),
                      yaxis=dict(tickfont=dict(size=9)),
                      margin=dict(l=150, b=150))
    fig.write_html(str(OUT/"eda_correlation.html"), include_plotlyjs="cdn")

    # 2f. Scatter: SGPT vs SGOT coloured by class (trend / relation)
    fig = px.scatter(df.sample(400, random_state=42),
                     x="Alamine_Aminotransferase", y="Aspartate_Aminotransferase",
                     color="Dataset",
                     color_discrete_map={1:CLASS_COLORS[1], 2:CLASS_COLORS[2]},
                     trendline="ols", opacity=0.6,
                     labels={"Alamine_Aminotransferase":"SGPT / ALT (IU/L)",
                             "Aspartate_Aminotransferase":"SGOT / AST (IU/L)",
                             "Dataset":"Class"},
                     title="SGPT vs SGOT — Trend & Relation by Class")
    fig.for_each_trace(lambda t: t.update(name=CLASS_LABELS[int(t.name)])
                       if t.name.lstrip("-").isdigit() else None)
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=440)
    fig.write_html(str(OUT/"eda_sgpt_sgot.html"), include_plotlyjs="cdn")

    # 2g. Bilirubin: Total vs Direct scatter
    fig = px.scatter(df, x="Total_Bilirubin", y="Direct_Bilirubin",
                     color="Dataset",
                     color_discrete_map={1:CLASS_COLORS[1], 2:CLASS_COLORS[2]},
                     opacity=0.55, trendline="ols",
                     labels={"Total_Bilirubin":"Total Bilirubin (mg/dL)",
                             "Direct_Bilirubin":"Direct Bilirubin (mg/dL)"},
                     title="Total vs Direct Bilirubin — Correlation by Class")
    fig.for_each_trace(lambda t: t.update(name=CLASS_LABELS[int(t.name)])
                       if t.name.lstrip("-").isdigit() else None)
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=440)
    fig.write_html(str(OUT/"eda_bilirubin.html"), include_plotlyjs="cdn")

    # 2h. Box plots: Albumin & Protein by class
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Albumin (g/dL)","Total Proteins (g/dL)"])
    for cidx, feat in enumerate(["Albumin","Total_Protiens"]):
        for cls, color in CLASS_COLORS.items():
            vals = df[df["Dataset"]==cls][feat]
            fig.add_trace(go.Box(
                y=vals, name=CLASS_LABELS[cls],
                marker_color=color, showlegend=(cidx==0),
                legendgroup=CLASS_LABELS[cls],
            ), row=1, col=cidx+1)
    fig.update_layout(title="Albumin & Total Proteins by Class",
                      height=420, plot_bgcolor="white", paper_bgcolor="white",
                      boxmode="group")
    fig.write_html(str(OUT/"eda_albumin_protein.html"), include_plotlyjs="cdn")

    print(f"  Saved 8 EDA charts → {OUT}/")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Preprocessing
# ═══════════════════════════════════════════════════════════════════════════════

def step3_preprocess(df: pd.DataFrame):
    print("  Preprocessing …")
    data = df.copy()

    # Impute missing AG Ratio with median
    median_ag = data["Albumin_and_Globulin_Ratio"].median()
    data["Albumin_and_Globulin_Ratio"].fillna(median_ag, inplace=True)
    print(f"  Imputed {df['Albumin_and_Globulin_Ratio'].isnull().sum()} missing AG Ratio values with median={median_ag:.3f}")

    # Encode Gender
    le = LabelEncoder()
    data["Gender_enc"] = le.fit_transform(data["Gender"])   # Female=0, Male=1

    # Log-transform skewed enzyme columns
    skewed = ["Total_Bilirubin","Direct_Bilirubin","Alkaline_Phosphotase",
              "Alamine_Aminotransferase","Aspartate_Aminotransferase"]
    for col in skewed:
        data[f"log_{col}"] = np.log1p(data[col])
    print(f"  Log-transformed {len(skewed)} skewed biomarker columns")

    # Binary target: 1=disease, 0=healthy (recode from 1/2 to 1/0)
    data["target"] = (data["Dataset"] == 1).astype(int)

    # Final feature set
    base_features = ["Age", "Gender_enc",
                     "log_Total_Bilirubin", "log_Direct_Bilirubin",
                     "log_Alkaline_Phosphotase", "log_Alamine_Aminotransferase",
                     "log_Aspartate_Aminotransferase",
                     "Total_Protiens", "Albumin", "Albumin_and_Globulin_Ratio"]

    X = data[base_features].values
    y = data["target"].values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED)

    # Handle class imbalance: oversample minority on train set
    X_tr_df = pd.DataFrame(X_tr, columns=base_features)
    X_tr_df["target"] = y_tr
    majority = X_tr_df[X_tr_df["target"]==1]
    minority = X_tr_df[X_tr_df["target"]==0]
    minority_up = resample(minority, replace=True, n_samples=len(majority), random_state=SEED)
    balanced = pd.concat([majority, minority_up]).sample(frac=1, random_state=SEED)
    X_tr_bal = balanced[base_features].values
    y_tr_bal  = balanced["target"].values
    print(f"  Resampled train set: {len(y_tr_bal)} samples (balanced)")

    scaler = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_tr_bal)
    X_te_sc  = scaler.transform(X_te)

    # Final NaN cleanup (safety net)
    X_tr_sc = np.nan_to_num(X_tr_sc, nan=0.0)
    X_te_sc = np.nan_to_num(X_te_sc, nan=0.0)

    return (X_tr_bal, X_te, X_tr_sc, X_te_sc,
            y_tr_bal, y_te, base_features, scaler, le, data)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Feature selection
# ═══════════════════════════════════════════════════════════════════════════════

def step4_feature_selection(X_tr_sc, y_tr, features):
    print("  Feature selection …")
    # Ensure no NaNs before MI computation
    X_clean = np.nan_to_num(X_tr_sc, nan=0.0)
    mi = mutual_info_classif(X_clean, y_tr, random_state=SEED)
    mi_df = pd.DataFrame({"feature": features, "mi_score": mi}) \
              .sort_values("mi_score", ascending=True)

    display_labels = {
        "Age": "Age",
        "Gender_enc": "Gender",
        "log_Total_Bilirubin": "Total Bilirubin (log)",
        "log_Direct_Bilirubin": "Direct Bilirubin (log)",
        "log_Alkaline_Phosphotase": "Alkaline Phosphotase (log)",
        "log_Alamine_Aminotransferase": "SGPT / ALT (log)",
        "log_Aspartate_Aminotransferase": "SGOT / AST (log)",
        "Total_Protiens": "Total Proteins",
        "Albumin": "Albumin",
        "Albumin_and_Globulin_Ratio": "AG Ratio",
    }

    fig = go.Figure(go.Bar(
        x=mi_df["mi_score"],
        y=[display_labels.get(f, f) for f in mi_df["feature"]],
        orientation="h",
        marker_color=["#3B82F6" if s > 0.02 else "#CBD5E1" for s in mi_df["mi_score"]],
        text=mi_df["mi_score"].round(4),
        textposition="outside",
    ))
    fig.update_layout(
        title="Mutual Information Feature Importance<br><sup>Blue = high signal · Gray = low signal</sup>",
        xaxis_title="MI Score",
        plot_bgcolor="white", paper_bgcolor="white",
        height=420, margin=dict(l=200),
    )
    fig.write_html(str(OUT/"feature_selection.html"), include_plotlyjs="cdn")

    top_features = mi_df[mi_df["mi_score"] > 0.02]["feature"].tolist()
    print(f"  Top features (MI > 0.02): {top_features}")
    return mi_df, display_labels


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Model comparison
# ═══════════════════════════════════════════════════════════════════════════════

def step5_model_comparison(X_tr_sc, y_tr):
    print("  Running 5-fold cross-validation …")

    models = {
        "Logistic Regression": LogisticRegression(
            C=1.0, max_iter=1000, class_weight="balanced", random_state=SEED),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=6, class_weight="balanced", random_state=SEED, n_jobs=-1),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            eval_metric="logloss", random_state=SEED, n_jobs=-1),
        "SVM (RBF)": SVC(
            kernel="rbf", C=5, probability=True, class_weight="balanced", random_state=SEED),
        "KNN (k=7)": KNeighborsClassifier(n_neighbors=7, metric="euclidean"),
    }

    scoring = ["accuracy","f1","roc_auc","precision","recall"]
    results = {}
    for name, model in models.items():
        cv_res = cross_validate(model, X_tr_sc, y_tr, cv=CV, scoring=scoring, n_jobs=-1)
        results[name] = {
            "Accuracy":  cv_res["test_accuracy"].mean(),
            "F1":        cv_res["test_f1"].mean(),
            "ROC-AUC":   cv_res["test_roc_auc"].mean(),
            "Precision": cv_res["test_precision"].mean(),
            "Recall":    cv_res["test_recall"].mean(),
            "Acc Std":   cv_res["test_accuracy"].std(),
        }
        r = results[name]
        print(f"    {name:25s}  Acc={r['Accuracy']:.3f}  F1={r['F1']:.3f}  AUC={r['ROC-AUC']:.3f}")

    # Comparison grouped bar
    metrics = ["Accuracy","F1","ROC-AUC","Precision","Recall"]
    bar_colors = ["#3B82F6","#10B981","#F97316","#8B5CF6","#EF4444"]
    model_names = list(results.keys())

    fig = go.Figure()
    for i, metric in enumerate(metrics):
        fig.add_trace(go.Bar(
            name=metric, x=model_names,
            y=[results[m][metric] for m in model_names],
            marker_color=bar_colors[i],
            text=[f"{results[m][metric]:.3f}" for m in model_names],
            textposition="outside",
        ))
    fig.update_layout(
        barmode="group",
        title="Model Comparison — 5-Fold Cross-Validation<br>"
              "<sup>Logistic Regression · Random Forest · XGBoost · SVM · KNN</sup>",
        yaxis=dict(title="Score", range=[0.4, 1.08], gridcolor="#f1f5f9"),
        plot_bgcolor="white", paper_bgcolor="white",
        height=460, legend_title="Metric",
        xaxis=dict(tickangle=-15),
    )
    fig.write_html(str(OUT/"model_comparison.html"), include_plotlyjs="cdn")

    # Radar chart per model
    fig = go.Figure()
    radar_colors = ["#3B82F6","#10B981","#F97316","#8B5CF6","#EF4444"]
    for i, name in enumerate(model_names):
        vals = [results[name][m] for m in metrics]
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=metrics + [metrics[0]],
            fill="toself", name=name,
            line_color=radar_colors[i], opacity=0.6,
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0.5, 1.0])),
        title="Model Performance Radar", height=480,
        paper_bgcolor="white",
    )
    fig.write_html(str(OUT/"model_radar.html"), include_plotlyjs="cdn")

    best = max(results, key=lambda m: results[m]["ROC-AUC"])
    print(f"  Best model: {best}")
    return models, results, best


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Evaluate best model on test set
# ═══════════════════════════════════════════════════════════════════════════════

def step6_evaluate(models, best_name, X_tr_sc, X_te_sc, y_tr, y_te, features):
    print(f"  Training {best_name} on full train set …")
    model = models[best_name]
    model.fit(X_tr_sc, y_tr)

    y_pred  = model.predict(X_te_sc)
    y_proba = model.predict_proba(X_te_sc)[:, 1]

    acc = accuracy_score(y_te, y_pred)
    auc = roc_auc_score(y_te, y_proba)
    print(f"  Test Accuracy: {acc:.3f}   ROC-AUC: {auc:.3f}")
    print(f"\n{classification_report(y_te, y_pred, target_names=['No Disease','Liver Disease'])}")

    # Confusion matrix
    cm = confusion_matrix(y_te, y_pred)
    fig = go.Figure(go.Heatmap(
        z=cm, x=["No Disease","Liver Disease"],
        y=["No Disease","Liver Disease"],
        text=cm, texttemplate="%{text}",
        colorscale=[[0,"#F0FFF4"],[0.5,"#22C55E"],[1,"#14532D"]],
        hovertemplate="True: %{y}<br>Pred: %{x}<br>Count: %{z}<extra></extra>",
    ))
    fig.update_layout(title=f"Confusion Matrix — {best_name}",
                      xaxis_title="Predicted", yaxis_title="True",
                      plot_bgcolor="white", paper_bgcolor="white", height=420)
    fig.write_html(str(OUT/"confusion_matrix.html"), include_plotlyjs="cdn")

    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_te, y_proba)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC = {auc:.3f}",
                             line=dict(color="#3B82F6", width=2.5)))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Random",
                             line=dict(color="gray", dash="dash")))
    # Mark optimal threshold (Youden's J)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    fig.add_trace(go.Scatter(
        x=[fpr[best_idx]], y=[tpr[best_idx]], mode="markers",
        marker=dict(size=12, color="#EF4444", symbol="star"),
        name=f"Optimal threshold ({thresholds[best_idx]:.2f})",
    ))
    fig.update_layout(title=f"ROC Curve — {best_name}",
                      xaxis_title="False Positive Rate",
                      yaxis_title="True Positive Rate",
                      plot_bgcolor="white", paper_bgcolor="white", height=440,
                      xaxis=dict(gridcolor="#f1f5f9"),
                      yaxis=dict(gridcolor="#f1f5f9"))
    fig.write_html(str(OUT/"roc_curve.html"), include_plotlyjs="cdn")

    # Precision-Recall curve
    prec, rec, pr_thresh = precision_recall_curve(y_te, y_proba)
    ap = average_precision_score(y_te, y_proba)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines",
                             name=f"AP = {ap:.3f}",
                             line=dict(color="#10B981", width=2.5)))
    fig.update_layout(title=f"Precision–Recall Curve — {best_name}",
                      xaxis_title="Recall", yaxis_title="Precision",
                      plot_bgcolor="white", paper_bgcolor="white", height=440,
                      xaxis=dict(gridcolor="#f1f5f9"),
                      yaxis=dict(gridcolor="#f1f5f9"))
    fig.write_html(str(OUT/"pr_curve.html"), include_plotlyjs="cdn")

    # Threshold sensitivity chart
    thresholds_range = np.linspace(0.1, 0.9, 50)
    accs, precs, recs, f1s = [], [], [], []
    for t in thresholds_range:
        yp = (y_proba >= t).astype(int)
        from sklearn.metrics import f1_score, precision_score, recall_score
        accs.append(accuracy_score(y_te, yp))
        precs.append(precision_score(y_te, yp, zero_division=0))
        recs.append(recall_score(y_te, yp, zero_division=0))
        f1s.append(f1_score(y_te, yp, zero_division=0))

    fig = go.Figure()
    for vals, name, color in zip([accs, precs, recs, f1s],
                                  ["Accuracy","Precision","Recall","F1"],
                                  ["#3B82F6","#8B5CF6","#EF4444","#F97316"]):
        fig.add_trace(go.Scatter(x=thresholds_range, y=vals,
                                 mode="lines", name=name,
                                 line=dict(color=color, width=2)))
    fig.add_vline(x=thresholds[best_idx], line_dash="dash", line_color="gray",
                  annotation_text=f"Youden opt ({thresholds[best_idx]:.2f})",
                  annotation_position="top right")
    fig.update_layout(title="Metric vs Decision Threshold",
                      xaxis_title="Threshold", yaxis_title="Score",
                      plot_bgcolor="white", paper_bgcolor="white", height=420,
                      xaxis=dict(gridcolor="#f1f5f9"),
                      yaxis=dict(gridcolor="#f1f5f9", range=[0, 1.05]),
                      legend_title="Metric")
    fig.write_html(str(OUT/"threshold_analysis.html"), include_plotlyjs="cdn")

    return model, y_proba, y_pred, acc, auc


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7 — SHAP
# ═══════════════════════════════════════════════════════════════════════════════

def step7_shap(model, best_name, X_te_sc, y_te, features, display_labels):
    print("  Computing SHAP values …")
    X_df = pd.DataFrame(X_te_sc, columns=features)

    try:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_df)
        # RF returns (n_samples, n_features, n_classes) or list
        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            sv = shap_values[:, :, 1]   # class 1 = Liver Disease
        elif isinstance(shap_values, list):
            sv = shap_values[1]
        else:
            sv = shap_values
    except Exception:
        bg = shap.sample(X_df, 80, random_state=SEED)
        explainer   = shap.KernelExplainer(model.predict_proba, bg)
        sv_all      = explainer.shap_values(X_df[:150])
        sv          = sv_all[1]
        X_df        = X_df[:150]

    feat_labels = [display_labels.get(f, f) for f in features]
    feat_labels_arr = np.array(feat_labels)
    imp = np.abs(sv).mean(axis=0)

    # Global bar
    idx_sort = list(np.argsort(imp))
    fig = go.Figure(go.Bar(
        x=imp[idx_sort], y=[feat_labels[i] for i in idx_sort],
        orientation="h", marker_color="#3B82F6",
        text=imp[idx_sort].round(4), textposition="outside",
    ))
    fig.update_layout(title="SHAP Global Feature Importance<br>"
                            "<sup>Mean |SHAP value| — higher = more impact on liver disease prediction</sup>",
                      xaxis_title="Mean |SHAP|",
                      plot_bgcolor="white", paper_bgcolor="white",
                      height=420, margin=dict(l=200))
    fig.write_html(str(OUT/"shap_global.html"), include_plotlyjs="cdn")

    # Beeswarm
    top10 = np.argsort(imp)[-10:]
    fig = go.Figure()
    for feat_idx in top10:
        fname = feat_labels[feat_idx]
        jitter = np.random.uniform(-0.1, 0.1, len(sv))
        fig.add_trace(go.Scatter(
            x=sv[:, feat_idx],
            y=[fname]*len(sv),
            mode="markers",
            marker=dict(size=4, opacity=0.5,
                        color=X_df.iloc[:, feat_idx].values,
                        colorscale="RdBu_r",
                        showscale=bool(feat_idx == top10[-1]),
                        colorbar=dict(title="Feature\nvalue", len=0.5)),
            showlegend=False,
            hovertemplate=f"<b>{fname}</b><br>SHAP: %{{x:.3f}}<extra></extra>",
        ))
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.update_layout(title="SHAP Beeswarm — Liver Disease Class<br>"
                            "<sup>Red = high feature value · Blue = low</sup>",
                      xaxis_title="SHAP value",
                      plot_bgcolor="white", paper_bgcolor="white",
                      height=480, margin=dict(l=200))
    fig.write_html(str(OUT/"shap_beeswarm.html"), include_plotlyjs="cdn")

    # Waterfall for highest-risk patient
    sample_idx = int(np.argmax(sv.sum(axis=1)))
    sv_sample  = sv[sample_idx]
    idx_wf     = np.argsort(np.abs(sv_sample))[-10:]
    fig = go.Figure(go.Waterfall(
        orientation="h",
        measure=["relative"]*len(idx_wf) + ["total"],
        x=list(sv_sample[idx_wf]) + [sv_sample[idx_wf].sum()],
        y=[feat_labels[i] for i in idx_wf] + ["→ Net SHAP"],
        connector=dict(line=dict(color="#94A3B8", width=0.5)),
        increasing=dict(marker_color="#EF4444"),
        decreasing=dict(marker_color="#22C55E"),
        totals=dict(marker_color="#3B82F6"),
    ))
    fig.add_vline(x=0, line_color="gray", line_dash="dash")
    fig.update_layout(title=f"SHAP Waterfall — Highest-Risk Patient (#{sample_idx})<br>"
                            "<sup>Red = pushes toward Liver Disease · Green = pushes away</sup>",
                      xaxis_title="SHAP value",
                      plot_bgcolor="white", paper_bgcolor="white",
                      height=440, margin=dict(l=200))
    fig.write_html(str(OUT/"shap_waterfall.html"), include_plotlyjs="cdn")

    print(f"  Saved 3 SHAP charts → {OUT}/")
    return sv, imp


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 8 — Save artifacts
# ═══════════════════════════════════════════════════════════════════════════════

def step8_save(model, scaler, le, features, best_name, results, acc, auc):
    joblib.dump(model,    MOD/"best_model.pkl")
    joblib.dump(scaler,   MOD/"scaler.pkl")
    joblib.dump(le,       MOD/"label_encoder.pkl")
    joblib.dump(features, MOD/"features.pkl")

    summary = {
        "best_model": best_name,
        "test_accuracy": round(float(acc), 4),
        "test_auc":      round(float(auc), 4),
        "features":      features,
        "cv_results":    {k: {m: round(float(v), 4) for m, v in d.items()}
                          for k, d in results.items()},
    }
    (MOD/"model_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  Artifacts saved → {MOD}/")


def build_index(best_name, results, acc, auc):
    best_res = results[best_name]
    cards = [
        ("eda_class_balance.html",  "⚖️",  "Class Distribution",      "71% liver disease vs 29% healthy — class imbalance"),
        ("eda_gender.html",         "👤",  "Gender Analysis",          "Male/female split by class"),
        ("eda_age.html",            "📅",  "Age Distribution",         "Histogram + density by class"),
        ("eda_biomarkers.html",     "🧪",  "Biomarker Violin Plots",   "All 8 liver function tests by class"),
        ("eda_correlation.html",    "🔥",  "Correlation Heatmap",      "Pearson correlations between all features"),
        ("eda_sgpt_sgot.html",      "📈",  "SGPT vs SGOT Trend",       "Relation & trend between ALT and AST"),
        ("eda_bilirubin.html",      "🟡",  "Bilirubin Correlation",    "Total vs Direct Bilirubin scatter"),
        ("eda_albumin_protein.html","💊",  "Albumin & Proteins",       "Box plots by class"),
        ("feature_selection.html",  "📐",  "Feature Selection (MI)",   "Mutual information scores"),
        ("model_comparison.html",   "🏆",  "Model Comparison",         "LR · RF · XGB · SVM · KNN — 5-fold CV"),
        ("model_radar.html",        "🕸️",  "Radar Chart",              "Multi-metric performance radar"),
        ("confusion_matrix.html",   "🎯",  "Confusion Matrix",         f"{best_name} on test set"),
        ("roc_curve.html",          "📈",  "ROC Curve",                f"AUC = {auc:.3f} · Optimal threshold marked"),
        ("pr_curve.html",           "📉",  "Precision–Recall",         "Average precision curve"),
        ("threshold_analysis.html", "⚙️",  "Threshold Analysis",       "Acc / Precision / Recall / F1 vs threshold"),
        ("shap_global.html",        "💡",  "SHAP Global Importance",   "Feature impact ranking"),
        ("shap_beeswarm.html",      "🐝",  "SHAP Beeswarm",            "Per-patient SHAP scatter"),
        ("shap_waterfall.html",     "🌊",  "SHAP Waterfall",           "Single patient SHAP explanation"),
    ]
    cards_html = "\n".join(f"""
      <a class="card" href="{href}" target="_blank">
        <span class="icon">{icon}</span>
        <div><strong>{title}</strong><br><small>{desc}</small></div>
      </a>""" for href, icon, title, desc in cards)

    html = f"""<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><title>Indian Liver Patient Predictor</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:'Segoe UI',system-ui,sans-serif;background:#0F172A;color:#E2E8F0}}
  .hdr{{background:linear-gradient(135deg,#1a3a2a,#14532d);padding:28px 40px;border-bottom:1px solid #166534}}
  .hdr h1{{font-size:22px;font-weight:600;color:#F0FFF4;margin-bottom:4px}}
  .hdr p{{color:#86EFAC;font-size:13px}}
  .wrap{{max-width:1100px;margin:0 auto;padding:28px 20px}}
  .stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:12px;margin-bottom:28px}}
  .stat{{background:#1E293B;border-radius:10px;padding:16px;text-align:center;border:1px solid #334155}}
  .stat .n{{font-size:26px;font-weight:600}}
  .stat .l{{font-size:11px;color:#94A3B8;text-transform:uppercase;letter-spacing:.4px;margin-top:3px}}
  .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:14px}}
  .card{{background:#1E293B;border:1px solid #334155;border-radius:12px;padding:16px 18px;
         display:flex;align-items:center;gap:12px;text-decoration:none;color:#E2E8F0;transition:border-color .2s}}
  .card:hover{{border-color:#22C55E}}
  .icon{{font-size:24px;flex-shrink:0}}
  .card small{{color:#64748B;font-size:11px}}
  .card strong{{font-size:13px}}
  .ftr{{text-align:center;color:#475569;font-size:12px;margin-top:36px;padding-bottom:20px}}
</style></head><body>
<div class="hdr">
  <h1>🫀 Indian Liver Patient Predictor — Pipeline Dashboard</h1>
  <p>Logistic Regression · Random Forest · XGBoost · SVM · KNN &nbsp;|&nbsp; 583 patients · 10 features · Binary classification</p>
</div>
<div class="wrap">
  <div class="stats">
    <div class="stat"><div class="n">583</div><div class="l">Patients</div></div>
    <div class="stat"><div class="n">10</div><div class="l">Features</div></div>
    <div class="stat"><div class="n" style="color:#EF4444">71%</div><div class="l">Disease rate</div></div>
    <div class="stat"><div class="n" style="color:#22C55E">{acc:.3f}</div><div class="l">Test Accuracy</div></div>
    <div class="stat"><div class="n" style="color:#3B82F6">{auc:.3f}</div><div class="l">ROC-AUC</div></div>
    <div class="stat"><div class="n" style="color:#F97316">{best_res['F1']:.3f}</div><div class="l">CV F1</div></div>
  </div>
  <div class="grid">{cards_html}</div>
</div>
<div class="ftr">Best model: <strong>{best_name}</strong> &nbsp;|&nbsp;
Personal Project — Rayala Madhu Bhanu Varma</div>
</body></html>"""
    (OUT/"index.html").write_text(html, encoding="utf-8")
    print(f"  Dashboard → {OUT}/index.html")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "="*60)
    print("  Indian Liver Patient Predictor — Full ML Pipeline")
    print("="*60)

    print("\n[1/8] Loading data …")
    df = step1_load()

    print("\n[2/8] EDA …")
    step2_eda(df)

    print("\n[3/8] Preprocessing …")
    X_tr, X_te, X_tr_sc, X_te_sc, y_tr, y_te, features, scaler, le, processed_df = step3_preprocess(df)

    print("\n[4/8] Feature selection …")
    mi_df, display_labels = step4_feature_selection(X_tr_sc, y_tr, features)

    print("\n[5/8] Model comparison …")
    models, results, best_name = step5_model_comparison(X_tr_sc, y_tr)

    print(f"\n[6/8] Evaluating {best_name} on test set …")
    model, y_proba, y_pred, acc, auc = step6_evaluate(
        models, best_name, X_tr_sc, X_te_sc, y_tr, y_te, features)

    print("\n[7/8] SHAP explainability …")
    sv, imp = step7_shap(model, best_name, X_te_sc, y_te, features, display_labels)

    print("\n[8/8] Saving artifacts …")
    step8_save(model, scaler, le, features, best_name, results, acc, auc)
    build_index(best_name, results, acc, auc)

    print(f"\n{'='*60}")
    print(f"  Done! Open output/index.html")
    print(f"  Run Streamlit app:  streamlit run app.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
