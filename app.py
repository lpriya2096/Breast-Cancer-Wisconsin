import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, RocCurveDisplay
)

# Configure page with a professional layout
st.set_page_config(page_title="Breast Cancer Diagnostic Center", page_icon="🩺", layout="wide")

# Custom CSS styling for premium look
st.markdown("""
<style>
.block-container {max-width: 1200px; padding-top: 2rem; padding-bottom: 2rem;}
.metric-card {
    background-color: #ffffff;
    border: 1px solid #e6ebf1;
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    text-align: center;
    margin-bottom: 15px;
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #1e3d59;
}
.metric-label {
    font-size: 0.9rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    margin-top: 5px;
}
</style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILES = {
    "Logistic Regression": "model/Logistic Regression.joblib",
    "Decision Tree": "model/Decision Tree.joblib",
    "K-Nearest Neighbors (KNN)": "model/KNN.joblib",
    "Gaussian Naive Bayes": "model/Naive Bayes.joblib",
    "Support Vector Machine (SVM)": "model/SVM.joblib",
}

@st.cache_resource(show_spinner=False, max_entries=5)
def load_model(model_path: str):
    return joblib.load(model_path)

# Header & Title Block
st.title("🩺 Breast Cancer Diagnostic Evaluation")
st.markdown("""
Welcome to the Diagnostic evaluation dashboard. This interactive platform evaluates 
and compares machine learning models trained on the Breast Cancer Wisconsin dataset.
""")
st.markdown("---")

# Sidebar controls
st.sidebar.image("https://img.icons8.com/external-flatart-icons-outline-flatarticons/128/external-cancer-medical-health-flatart-icons-outline-flatarticons.png", width=80)
st.sidebar.title("Evaluation Settings")
selected_model = st.sidebar.selectbox("Active Classifier", list(MODEL_FILES))
st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("Upload Test CSV (containing target column)", type=["csv"])

if uploaded_file is None:
    # Default view explaining the setup
    st.warning("👈 Please upload a test dataset (`test_data.csv`) using the sidebar to begin analysis.")
    
    # Display baseline model performance comparison from previous tests
    if Path("model_comparison.csv").is_file():
        st.subheader("📊 Pre-computed Model Comparison (Baseline Results)")
        try:
            comparison_df = pd.read_csv("model_comparison.csv", index_index=True if "Model" in pd.read_csv("model_comparison.csv").columns else False)
            st.dataframe(comparison_df.style.format(precision=4).highlight_max(axis=0, color="#e2f0d9"))
        except Exception:
            pass
    st.stop()

try:
    test_df = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"Cannot read uploaded CSV file: {exc}")
    st.stop()

# Dynamic target detection
target_col = None
for col in ["target", "Revenue", "revenue"]:
    if col in test_df.columns:
        target_col = col
        break

if not target_col:
    st.error("⚠️ CSV must contain a target column named 'target'.")
    st.stop()

y = pd.to_numeric(test_df[target_col], errors="coerce").fillna(0).astype(int)
X = test_df.drop(columns=[target_col])

model_path = BASE_DIR / MODEL_FILES[selected_model]
if not model_path.is_file():
    st.error(f"Model file not found: {MODEL_FILES[selected_model]}")
    st.stop()

try:
    with st.spinner("Processing diagnostic predictions..."):
        model = load_model(str(model_path))
        
        # Handle pipeline or standard model features alignment
        if hasattr(model, "feature_names_in_"):
            expected = list(model.feature_names_in_)
        elif hasattr(model, "named_steps") and "preprocessing" in model.named_steps and hasattr(model.named_steps["preprocessing"], "feature_names_in_"):
            expected = list(model.named_steps["preprocessing"].feature_names_in_)
        else:
            expected = None

        if expected is not None:
            missing = [name for name in expected if name not in X.columns]
            if missing:
                st.error("Missing input columns expected by model: " + ", ".join(missing))
                st.stop()
            X = X[expected]

        predictions = model.predict(X)
        
        # Try getting probabilities
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X)[:, 1]
        elif hasattr(model, "decision_function"):
            probabilities = model.decision_function(X)
        else:
            probabilities = predictions
except Exception as exc:
    st.error(f"Model testing failed: {exc}")
    st.stop()

# Compute standard metrics
acc = accuracy_score(y, predictions)
auc = roc_auc_score(y, probabilities)
prec = precision_score(y, predictions, zero_division=0)
rec = recall_score(y, predictions, zero_division=0)
f1 = f1_score(y, predictions, zero_division=0)
mcc = matthews_corrcoef(y, predictions)

# Organize interface into tabs
tab_eval, tab_comp = st.tabs(["🎯 Evaluation & Visualizations", "📊 Model Comparison"])

with tab_eval:
    st.subheader(f"Results for {selected_model}")
    
    # Display beautiful metric card grids
    m_cols = st.columns(6)
    metrics_dict = {
        "Accuracy": acc,
        "ROC-AUC": auc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "MCC": mcc
    }
    
    for col, (lbl, val) in zip(m_cols, metrics_dict.items()):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{val:.4f}</div>
                <div class="metric-label">{lbl}</div>
            </div>
            """, unsafe_allow_html=True)
            
    st.markdown("---")
    
    # Interactive visuals side-by-side
    v_col1, v_col2 = st.columns(2, gap="large")
    
    with v_col1:
        st.subheader("📈 Receiver Operating Curve (ROC)")
        fig, ax = plt.subplots(figsize=(6, 5))
        RocCurveDisplay.from_predictions(y, probabilities, name=selected_model, ax=ax, color="#1e3d59", linewidth=2.5)
        ax.plot([0, 1], [0, 1], "r--", alpha=0.7, label="Random Classifier")
        ax.set_title("ROC Curve (Model Discrimination Power)", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="lower right")
        st.pyplot(fig)
        
    with v_col2:
        st.subheader("📊 Diagnostic Confusion Matrix")
        cm = confusion_matrix(y, predictions)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0][0], 0, 0, 0)
        
        fig, ax = plt.subplots(figsize=(5.5, 4.8))
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Malignant (0)", "Benign (1)"]).plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title("Confusion Matrix", fontsize=12, fontweight="bold")
        st.pyplot(fig)
        st.info(f"🔍 **Diagnosis Verdict:** True Negatives: **{tn}** | False Positives: **{fp}** | False Negatives: **{fn}** | True Positives: **{tp}**")

with tab_comp:
    st.subheader("📊 Classifier Benchmark Overview")
    st.write("Below is the benchmark metrics dataframe for all five pre-trained classifiers.")
    if Path("model_comparison.csv").is_file():
        comparison_df = pd.read_csv("model_comparison.csv")
        st.dataframe(comparison_df.style.format(precision=4).highlight_max(axis=0, color="#e2f0d9"))
    else:
        st.info("Run the notebook training step to save the comparison CSV data.")
