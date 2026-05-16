"""
app.py — Indian Liver Patient Predictor · Streamlit App
=========================================================
4 pages:
  1. Patient Predictor  — enter biomarker values → prediction + SHAP
  2. EDA Explorer       — class balance, biomarkers, correlations, trends
  3. Model Performance  — comparison, confusion matrix, ROC, PR, threshold
  4. SHAP Explainability — global, beeswarm, waterfall

Run: streamlit run app.py
"""

import sys, json, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import shap
import streamlit as st

from src.data_generator import (
    generate_dataset, NUMERIC_FEATURES,
    FEATURE_LABELS, CLASS_LABELS, CLASS_COLORS,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Indian Liver Patient Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .block-container{padding-top:1.5rem;padding-bottom:2rem}
  div[data-testid="stMetric"]{background:white;border-radius:10px;
    padding:14px;box-shadow:0 1px 3px rgba(0,0,0,.07)}
</style>
""", unsafe_allow_html=True)

# ── Load artifacts ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model    = joblib.load("models/best_model.pkl")
    scaler   = joblib.load("models/scaler.pkl")
    le       = joblib.load("models/label_encoder.pkl")
    features = joblib.load("models/features.pkl")
    summary  = json.loads(Path("models/model_summary.json").read_text())
    return model, scaler, le, features, summary

@st.cache_data
def load_data():
    p = Path("data/indian_liver_patients.csv")
    if not p.exists():
        return generate_dataset(save_path=str(p))
    return pd.read_csv(p)

try:
    model, scaler, le, features, summary = load_artifacts()
    df_raw = load_data()
    ok = True
except Exception as e:
    st.error(f"Run `python src/pipeline.py` first.\n\n{e}")
    st.stop()

DISPLAY_LABELS = {
    "Age":                          "Age",
    "Gender_enc":                   "Gender",
    "log_Total_Bilirubin":          "Total Bilirubin (log)",
    "log_Direct_Bilirubin":         "Direct Bilirubin (log)",
    "log_Alkaline_Phosphotase":     "Alkaline Phosphotase (log)",
    "log_Alamine_Aminotransferase": "SGPT / ALT (log)",
    "log_Aspartate_Aminotransferase": "SGOT / AST (log)",
    "Total_Protiens":               "Total Proteins",
    "Albumin":                      "Albumin",
    "Albumin_and_Globulin_Ratio":   "AG Ratio",
}

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🫀 Liver Patient Predictor")
    st.markdown("*Indian Liver Patient Dataset*")
    st.markdown("---")
    page = st.radio("Navigate", [
        "🩺 Patient Predictor",
        "📊 EDA Explorer",
        "🏆 Model Performance",
        "💡 SHAP Explainability",
    ])
    st.markdown("---")
    best = summary.get("best_model", "Random Forest")
    cv   = summary.get("cv_results", {}).get(best, {})
    st.markdown(f"**Best model:** {best}")
    st.metric("Test Accuracy", f"{summary.get('test_accuracy',0):.3f}")
    st.metric("ROC-AUC",       f"{summary.get('test_auc',0):.3f}")
    st.metric("CV F1",         f"{cv.get('F1',0):.3f}")
    st.markdown("---")
    st.caption("Personal Project — Rayala Madhu Bhanu Varma")


# helper: build model-ready vector
def build_input(age, gender_val, tb, db, alkphos, sgpt, sgot, tp, alb, ag):
    vec = {
        "Age": age,
        "Gender_enc": gender_val,
        "log_Total_Bilirubin": np.log1p(tb),
        "log_Direct_Bilirubin": np.log1p(db),
        "log_Alkaline_Phosphotase": np.log1p(alkphos),
        "log_Alamine_Aminotransferase": np.log1p(sgpt),
        "log_Aspartate_Aminotransferase": np.log1p(sgot),
        "Total_Protiens": tp,
        "Albumin": alb,
        "Albumin_and_Globulin_Ratio": ag,
    }
    row = np.array([[vec[f] for f in features]])
    row = np.nan_to_num(row, nan=0.0)
    return scaler.transform(row)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Patient Predictor
# ══════════════════════════════════════════════════════════════════════════════
if page == "🩺 Patient Predictor":
    st.markdown("# 🩺 Patient Liver Disease Predictor")
    st.markdown("Enter patient biomarker values below to predict liver disease status with SHAP explanation.")
    st.markdown("---")

    col_l, col_r = st.columns([2, 1])

    with col_l:
        st.markdown("### Patient Details")
        c1, c2 = st.columns(2)
        age     = c1.slider("Age (years)", 4, 90, 45)
        gender  = c2.selectbox("Gender", ["Male","Female"])
        gender_val = 1 if gender == "Male" else 0

        st.markdown("### Liver Function Tests")
        c1, c2, c3 = st.columns(3)
        tb      = c1.number_input("Total Bilirubin (mg/dL)",    0.1, 75.0,  1.0, 0.1)
        db      = c2.number_input("Direct Bilirubin (mg/dL)",   0.0, 20.0,  0.3, 0.1)
        alkphos = c3.number_input("Alkaline Phosphotase (IU/L)",44,  2110,  187,  1)
        sgpt    = c1.number_input("SGPT / ALT (IU/L)",          7,   2000,   35,  1)
        sgot    = c2.number_input("SGOT / AST (IU/L)",          10,  4929,   35,  1)
        tp      = c3.number_input("Total Proteins (g/dL)",      2.7,  9.6,   7.0, 0.1)

        st.markdown("### Protein Markers")
        c1, c2 = st.columns(2)
        alb     = c1.number_input("Albumin (g/dL)",             0.9,  5.5,   3.8, 0.1)
        ag      = c2.number_input("Albumin/Globulin Ratio",     0.3,  2.8,   1.0, 0.01)

        predict_btn = st.button("🔮 Predict", type="primary", use_container_width=True)

    input_sc   = build_input(age, gender_val, tb, db, alkphos, sgpt, sgot, tp, alb, ag)
    proba      = model.predict_proba(input_sc)[0]
    pred_class = int(model.predict(input_sc)[0])
    label      = CLASS_LABELS[pred_class + 1] if pred_class + 1 in CLASS_LABELS else ("Liver Disease" if pred_class == 1 else "No Disease")
    color      = "#EF4444" if pred_class == 1 else "#22C55E"
    conf       = float(proba[pred_class]) * 100

    with col_r:
        st.markdown("### Prediction Result")
        st.markdown(f"""
        <div style="text-align:center;padding:24px 0 16px">
          <div style="display:inline-block;padding:10px 28px;border-radius:99px;
               font-size:22px;font-weight:700;background:{color}22;
               color:{color};border:2px solid {color}">
            {label}
          </div>
          <div style="margin-top:12px;font-size:14px;color:#64748b">
            Confidence: <strong>{conf:.1f}%</strong>
          </div>
        </div>""", unsafe_allow_html=True)

        # Probability bar
        for i, (lbl, clr) in enumerate(zip(["No Disease","Liver Disease"],["#22C55E","#EF4444"])):
            pct = float(proba[i]) * 100
            st.markdown(f"""
            <div style="margin:6px 0">
              <div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:3px">
                <span>{lbl}</span><span style="font-weight:600;color:{clr}">{pct:.1f}%</span>
              </div>
              <div style="background:#f1f5f9;border-radius:99px;height:8px">
                <div style="background:{clr};width:{pct}%;height:100%;border-radius:99px"></div>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**Clinical flags:**")
        flags = []
        if tb     > 1.2:  flags.append(("🔴", "Elevated Total Bilirubin"))
        if sgpt   > 56:   flags.append(("🔴", "Elevated SGPT/ALT"))
        if sgot   > 40:   flags.append(("🟡", "Elevated SGOT/AST"))
        if alkphos> 147:  flags.append(("🟡", "Elevated Alkaline Phosphotase"))
        if alb    < 3.5:  flags.append(("🟠", "Low Albumin (hypoalbuminemia)"))
        if ag     < 0.8:  flags.append(("🟠", "Low AG Ratio"))
        if tp     < 6.0:  flags.append(("🟡", "Low Total Proteins"))
        if not flags:     flags.append(("🟢", "All values within normal range"))
        for icon, msg in flags:
            st.markdown(f"{icon} {msg}")

    # SHAP waterfall
    st.markdown("---")
    st.markdown("### SHAP Explanation")
    try:
        explainer   = shap.TreeExplainer(model)
        X_df_in     = pd.DataFrame(input_sc, columns=features)
        sv_all      = explainer.shap_values(X_df_in)
        if isinstance(sv_all, np.ndarray) and sv_all.ndim == 3:
            sv_p = sv_all[0, :, pred_class]
        elif isinstance(sv_all, list):
            sv_p = sv_all[pred_class][0]
        else:
            sv_p = sv_all[0]

        idx_wf = np.argsort(np.abs(sv_p))[::-1][:8]
        fig = go.Figure(go.Waterfall(
            orientation="h",
            measure=["relative"]*len(idx_wf) + ["total"],
            x=list(sv_p[idx_wf]) + [sv_p[idx_wf].sum()],
            y=[DISPLAY_LABELS.get(features[i],features[i]) for i in idx_wf] + ["→ Net SHAP"],
            connector=dict(line=dict(color="#94A3B8",width=0.5)),
            increasing=dict(marker_color="#EF4444"),
            decreasing=dict(marker_color="#22C55E"),
            totals=dict(marker_color="#3B82F6"),
        ))
        fig.add_vline(x=0, line_color="gray", line_dash="dash")
        fig.update_layout(title=f"SHAP Waterfall — {label} prediction",
                          xaxis_title="SHAP value (red = toward Liver Disease)",
                          plot_bgcolor="white", paper_bgcolor="white",
                          height=400, margin=dict(l=200))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.info(f"SHAP not available: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EDA Explorer
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 EDA Explorer":
    st.markdown("# 📊 EDA Explorer")
    st.markdown(f"**{len(df_raw):,}** patients · **{df_raw.shape[1]-1}** raw features · ~28 missing values in AG Ratio")
    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    disease = len(df_raw[df_raw["Dataset"]==1])
    healthy = len(df_raw[df_raw["Dataset"]==2])
    c1.metric("Liver Disease", disease, f"{disease/len(df_raw)*100:.1f}%")
    c2.metric("No Disease",    healthy, f"{healthy/len(df_raw)*100:.1f}%")
    c3.metric("Missing (AG Ratio)", df_raw["Albumin_and_Globulin_Ratio"].isnull().sum())

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Class & Gender", "Biomarker Violin", "Correlations",
        "SGPT vs SGOT", "Bilirubin", "Albumin & Protein"
    ])
    colors = [CLASS_COLORS[1], CLASS_COLORS[2]]

    with tab1:
        col_a, col_b = st.columns(2)
        with col_a:
            counts = df_raw["Dataset"].value_counts().sort_index()
            fig = go.Figure(go.Bar(
                x=[CLASS_LABELS[i] for i in counts.index], y=counts.values,
                marker_color=colors,
                text=[f"{v} ({v/len(df_raw)*100:.1f}%)" for v in counts.values],
                textposition="outside",
            ))
            fig.update_layout(title="Class Distribution", xaxis_title="Class",
                              yaxis_title="Count", plot_bgcolor="white",
                              paper_bgcolor="white", height=360)
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            gender_ct = df_raw.groupby(["Dataset","Gender"]).size().reset_index(name="count")
            gender_ct["Class"] = gender_ct["Dataset"].map(CLASS_LABELS)
            fig = px.bar(gender_ct, x="Class", y="count", color="Gender", barmode="group",
                         color_discrete_map={"Male":"#3B82F6","Female":"#EC4899"},
                         title="Gender by Class", text="count")
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=360)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        feat = st.selectbox("Biomarker", [f for f in NUMERIC_FEATURES if f != "Age"],
                            format_func=lambda f: FEATURE_LABELS.get(f, f))
        fig = go.Figure()
        for cls, color in CLASS_COLORS.items():
            vals = df_raw[df_raw["Dataset"]==cls][feat].dropna()
            fig.add_trace(go.Violin(y=vals, name=CLASS_LABELS[cls],
                                    line_color=color, fillcolor=color,
                                    opacity=0.65, box_visible=True, meanline_visible=True))
        fig.update_layout(title=f"{FEATURE_LABELS.get(feat,feat)} by Class",
                          violinmode="overlay", plot_bgcolor="white",
                          paper_bgcolor="white", height=440)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_raw.groupby("Dataset")[feat].describe().round(2).rename(index=CLASS_LABELS),
                     use_container_width=True)

    with tab3:
        corr = df_raw[NUMERIC_FEATURES].corr()
        fig = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.index,
            colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
            text=corr.values.round(2), texttemplate="%{text}",
            textfont=dict(size=9),
        ))
        fig.update_layout(title="Pearson Correlation Matrix",
                          height=520, plot_bgcolor="white", paper_bgcolor="white",
                          xaxis=dict(tickangle=-35, tickfont=dict(size=9)),
                          yaxis=dict(tickfont=dict(size=9)),
                          margin=dict(l=150, b=150))
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        fig = px.scatter(df_raw.sample(400, random_state=42),
                         x="Alamine_Aminotransferase", y="Aspartate_Aminotransferase",
                         color="Dataset",
                         color_discrete_map={1:CLASS_COLORS[1],2:CLASS_COLORS[2]},
                         trendline="ols", opacity=0.6,
                         labels={"Alamine_Aminotransferase":"SGPT / ALT (IU/L)",
                                 "Aspartate_Aminotransferase":"SGOT / AST (IU/L)"},
                         title="SGPT vs SGOT — Trend & Relation by Class")
        fig.for_each_trace(lambda t: t.update(name=CLASS_LABELS[int(t.name)])
                           if t.name.lstrip("-").isdigit() else None)
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=440)
        st.plotly_chart(fig, use_container_width=True)

    with tab5:
        fig = px.scatter(df_raw, x="Total_Bilirubin", y="Direct_Bilirubin",
                         color="Dataset",
                         color_discrete_map={1:CLASS_COLORS[1],2:CLASS_COLORS[2]},
                         opacity=0.55, trendline="ols",
                         labels={"Total_Bilirubin":"Total Bilirubin (mg/dL)",
                                 "Direct_Bilirubin":"Direct Bilirubin (mg/dL)"},
                         title="Total vs Direct Bilirubin")
        fig.for_each_trace(lambda t: t.update(name=CLASS_LABELS[int(t.name)])
                           if t.name.lstrip("-").isdigit() else None)
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=440)
        st.plotly_chart(fig, use_container_width=True)

    with tab6:
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=["Albumin (g/dL)","Total Proteins (g/dL)"])
        for cidx, feat_name in enumerate(["Albumin","Total_Protiens"]):
            for cls, color in CLASS_COLORS.items():
                vals = df_raw[df_raw["Dataset"]==cls][feat_name]
                fig.add_trace(go.Box(y=vals, name=CLASS_LABELS[cls],
                                     marker_color=color, showlegend=(cidx==0),
                                     legendgroup=CLASS_LABELS[cls]), row=1, col=cidx+1)
        fig.update_layout(title="Albumin & Total Proteins by Class",
                          height=420, plot_bgcolor="white", paper_bgcolor="white",
                          boxmode="group")
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Model Performance
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏆 Model Performance":
    st.markdown("# 🏆 Model Performance")
    st.markdown("---")

    cv_res      = summary.get("cv_results", {})
    model_names = list(cv_res.keys())
    metrics     = ["Accuracy","F1","ROC-AUC","Precision","Recall"]
    bar_colors  = ["#3B82F6","#10B981","#F97316","#8B5CF6","#EF4444"]

    # CV comparison
    fig = go.Figure()
    for i, metric in enumerate(metrics):
        fig.add_trace(go.Bar(
            name=metric, x=model_names,
            y=[cv_res[m].get(metric, 0) for m in model_names],
            marker_color=bar_colors[i],
            text=[f"{cv_res[m].get(metric,0):.3f}" for m in model_names],
            textposition="outside",
        ))
    fig.update_layout(barmode="group",
                      title="5-Fold CV: Logistic Regression · RF · XGBoost · SVM · KNN",
                      yaxis=dict(range=[0.5,1.08], gridcolor="#f1f5f9"),
                      plot_bgcolor="white", paper_bgcolor="white",
                      height=440, legend_title="Metric",
                      xaxis=dict(tickangle=-15))
    st.plotly_chart(fig, use_container_width=True)

    # Radar
    fig = go.Figure()
    radar_colors = ["#3B82F6","#10B981","#F97316","#8B5CF6","#EF4444"]
    for i, name in enumerate(model_names):
        vals = [cv_res[name].get(m, 0) for m in metrics]
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=metrics + [metrics[0]],
            fill="toself", name=name,
            line_color=radar_colors[i], opacity=0.6,
        ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0.5, 1.0])),
                      title="Model Radar", height=460, paper_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

    # Live test-set evaluation
    st.markdown("### Test Set Evaluation")
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
    from sklearn.metrics import precision_recall_curve, average_precision_score

    data = df_raw.copy()
    data["Albumin_and_Globulin_Ratio"].fillna(
        data["Albumin_and_Globulin_Ratio"].median(), inplace=True)
    data["Gender_enc"] = le.transform(data["Gender"])
    for col in ["Total_Bilirubin","Direct_Bilirubin","Alkaline_Phosphotase",
                "Alamine_Aminotransferase","Aspartate_Aminotransferase"]:
        data[f"log_{col}"] = np.log1p(data[col])
    data["target"] = (data["Dataset"]==1).astype(int)
    X  = np.nan_to_num(data[features].values, nan=0.0)
    y  = data["target"].values
    _, X_te, _, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_te_sc  = np.nan_to_num(scaler.transform(X_te), nan=0.0)
    y_pred   = model.predict(X_te_sc)
    y_proba  = model.predict_proba(X_te_sc)[:,1]

    tab_cm, tab_roc, tab_pr, tab_thr = st.tabs([
        "Confusion Matrix","ROC Curve","Precision–Recall","Threshold Analysis"])

    with tab_cm:
        cm = confusion_matrix(y_te, y_pred)
        fig = go.Figure(go.Heatmap(
            z=cm, x=["No Disease","Liver Disease"],
            y=["No Disease","Liver Disease"],
            text=cm, texttemplate="%{text}",
            colorscale=[[0,"#F0FFF4"],[0.5,"#22C55E"],[1,"#14532D"]],
        ))
        fig.update_layout(title=f"Confusion Matrix — {best}",
                          xaxis_title="Predicted", yaxis_title="True",
                          plot_bgcolor="white", paper_bgcolor="white", height=400)
        st.plotly_chart(fig, use_container_width=True)
        c1,c2,c3,c4 = st.columns(4)
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        c1.metric("Accuracy",  f"{accuracy_score(y_te,y_pred):.3f}")
        c2.metric("F1",        f"{f1_score(y_te,y_pred):.3f}")
        c3.metric("Precision", f"{precision_score(y_te,y_pred):.3f}")
        c4.metric("Recall",    f"{recall_score(y_te,y_pred):.3f}")

    with tab_roc:
        fpr, tpr, thresh = roc_curve(y_te, y_proba)
        auc = roc_auc_score(y_te, y_proba)
        j   = int(np.argmax(tpr - fpr))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                 name=f"AUC = {auc:.3f}",
                                 line=dict(color="#3B82F6", width=2.5)))
        fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",name="Random",
                                 line=dict(color="gray",dash="dash")))
        fig.add_trace(go.Scatter(x=[fpr[j]],y=[tpr[j]],mode="markers",
                                 marker=dict(size=12,color="#EF4444",symbol="star"),
                                 name=f"Optimal ({thresh[j]:.2f})"))
        fig.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR",
                          plot_bgcolor="white", paper_bgcolor="white", height=440,
                          xaxis=dict(gridcolor="#f1f5f9"),yaxis=dict(gridcolor="#f1f5f9"))
        st.plotly_chart(fig, use_container_width=True)

    with tab_pr:
        prec, rec, _ = precision_recall_curve(y_te, y_proba)
        ap = average_precision_score(y_te, y_proba)
        fig = go.Figure(go.Scatter(x=rec, y=prec, mode="lines",
                                   name=f"AP={ap:.3f}",
                                   line=dict(color="#10B981",width=2.5)))
        fig.update_layout(title="Precision–Recall Curve",
                          xaxis_title="Recall", yaxis_title="Precision",
                          plot_bgcolor="white", paper_bgcolor="white", height=440,
                          xaxis=dict(gridcolor="#f1f5f9"),yaxis=dict(gridcolor="#f1f5f9"))
        st.plotly_chart(fig, use_container_width=True)

    with tab_thr:
        from sklearn.metrics import f1_score as _f1, precision_score as _prec, recall_score as _rec
        ts = np.linspace(0.1,0.9,60)
        accs,precs,recs,f1s=[],[],[],[]
        for t in ts:
            yp=(y_proba>=t).astype(int)
            accs.append(accuracy_score(y_te,yp))
            precs.append(_prec(y_te,yp,zero_division=0))
            recs.append(_rec(y_te,yp,zero_division=0))
            f1s.append(_f1(y_te,yp,zero_division=0))
        fig = go.Figure()
        for vals, name, color in zip([accs,precs,recs,f1s],
                                      ["Accuracy","Precision","Recall","F1"],
                                      ["#3B82F6","#8B5CF6","#EF4444","#F97316"]):
            fig.add_trace(go.Scatter(x=ts, y=vals, mode="lines",
                                     name=name, line=dict(color=color,width=2)))
        fig.add_vline(x=thresh[j], line_dash="dash", line_color="gray",
                      annotation_text=f"Optimal ({thresh[j]:.2f})")
        fig.update_layout(title="Metric vs Decision Threshold",
                          xaxis_title="Threshold", yaxis_title="Score",
                          plot_bgcolor="white", paper_bgcolor="white", height=420,
                          xaxis=dict(gridcolor="#f1f5f9"),
                          yaxis=dict(gridcolor="#f1f5f9",range=[0,1.05]))
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — SHAP Explainability
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💡 SHAP Explainability":
    st.markdown("# 💡 SHAP Explainability")
    st.markdown("Global and per-patient explanation of liver disease predictions.")
    st.markdown("---")

    @st.cache_data
    def compute_shap_test():
        from sklearn.model_selection import train_test_split
        data = df_raw.copy()
        data["Albumin_and_Globulin_Ratio"].fillna(data["Albumin_and_Globulin_Ratio"].median(), inplace=True)
        data["Gender_enc"] = le.transform(data["Gender"])
        for col in ["Total_Bilirubin","Direct_Bilirubin","Alkaline_Phosphotase",
                    "Alamine_Aminotransferase","Aspartate_Aminotransferase"]:
            data[f"log_{col}"] = np.log1p(data[col])
        data["target"] = (data["Dataset"]==1).astype(int)
        X = np.nan_to_num(data[features].values, nan=0.0)
        y = data["target"].values
        _, X_te, _, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        X_te_sc = np.nan_to_num(scaler.transform(X_te), nan=0.0)
        X_df    = pd.DataFrame(X_te_sc, columns=features)
        exp     = shap.TreeExplainer(model)
        sv_all  = exp.shap_values(X_df)
        if isinstance(sv_all, np.ndarray) and sv_all.ndim == 3:
            sv = sv_all[:, :, 1]
        elif isinstance(sv_all, list):
            sv = sv_all[1]
        else:
            sv = sv_all
        return sv, X_df, y_te

    with st.spinner("Computing SHAP values …"):
        sv, X_df, y_te = compute_shap_test()

    feat_labels = [DISPLAY_LABELS.get(f, f) for f in features]
    imp = np.abs(sv).mean(axis=0)

    tab_g, tab_b, tab_w = st.tabs(["Global Importance","Beeswarm","Waterfall"])

    with tab_g:
        idx_s = list(np.argsort(imp))
        fig = go.Figure(go.Bar(
            x=imp[idx_s], y=[feat_labels[i] for i in idx_s],
            orientation="h", marker_color="#3B82F6",
            text=[f"{imp[i]:.4f}" for i in idx_s], textposition="outside",
        ))
        fig.update_layout(title="Global SHAP Feature Importance — Liver Disease class",
                          xaxis_title="Mean |SHAP|",
                          plot_bgcolor="white", paper_bgcolor="white",
                          height=420, margin=dict(l=200))
        st.plotly_chart(fig, use_container_width=True)

    with tab_b:
        top10 = list(np.argsort(imp)[-10:])
        fig = go.Figure()
        for feat_idx in top10:
            fname = feat_labels[feat_idx]
            fig.add_trace(go.Scatter(
                x=sv[:, feat_idx],
                y=[fname]*len(sv),
                mode="markers",
                marker=dict(size=4, opacity=0.5,
                            color=X_df.iloc[:,feat_idx].values,
                            colorscale="RdBu_r",
                            showscale=bool(feat_idx == top10[-1]),
                            colorbar=dict(title="Feature\nvalue", len=0.5)),
                showlegend=False,
            ))
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        fig.update_layout(title="SHAP Beeswarm — Liver Disease class<br>"
                                "<sup>Red = high feature value · Blue = low</sup>",
                          xaxis_title="SHAP value",
                          plot_bgcolor="white", paper_bgcolor="white",
                          height=480, margin=dict(l=200))
        st.plotly_chart(fig, use_container_width=True)

    with tab_w:
        sample_idx = st.slider("Patient index", 0, len(X_df)-1,
                                int(np.argmax(sv.sum(axis=1))))
        sv_s   = sv[sample_idx]
        idx_wf = list(np.argsort(np.abs(sv_s))[-10:])
        fig = go.Figure(go.Waterfall(
            orientation="h",
            measure=["relative"]*len(idx_wf) + ["total"],
            x=list(sv_s[idx_wf]) + [sv_s[idx_wf].sum()],
            y=[feat_labels[i] for i in idx_wf] + ["→ Net SHAP"],
            connector=dict(line=dict(color="#94A3B8",width=0.5)),
            increasing=dict(marker_color="#EF4444"),
            decreasing=dict(marker_color="#22C55E"),
            totals=dict(marker_color="#3B82F6"),
        ))
        fig.add_vline(x=0, line_color="gray", line_dash="dash")
        pred_lbl = "Liver Disease" if model.predict(X_df.iloc[[sample_idx]])[0]==1 else "No Disease"
        fig.update_layout(title=f"SHAP Waterfall — Patient #{sample_idx} · Predicted: {pred_lbl}",
                          xaxis_title="SHAP value",
                          plot_bgcolor="white", paper_bgcolor="white",
                          height=440, margin=dict(l=200))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Patient biomarker values (scaled):**")
        row = X_df.iloc[sample_idx]
        disp = {feat_labels[i]: round(float(row[features[i]]),3) for i in range(len(features))}
        st.dataframe(pd.DataFrame([disp]).T.rename(columns={0:"Value (scaled)"}),
                     use_container_width=True, height=240)
