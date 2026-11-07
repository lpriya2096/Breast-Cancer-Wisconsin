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

st.set_page_config(page_title="Breast Cancer Model Evaluation Dashboard", page_icon="🩺", layout="wide")

# Custom styling for a clean, modern look
st.markdown("""
<style>
.block-container {max-width: 1200px; padding-top: 2rem; padding-bottom: 2rem;}
.metric-container {background-color: #f8f9fa; border-radius: 8px; padding: 15px; border-left: 5px solid #1e3d59; margin-bottom: 10px;}
[data-testid="stMetricValue"] {font-size: 1.8rem !important; font-weight: bold;}
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

st.title("🩺 Breast Cancer Model Evaluation & Comparison")
st.caption("Compare five pre-trained classification models on labeled test data with metrics and interactive ROC visualizations.")

with st.form("evaluation_form"):
    left, right = st.columns([1, 1.3], gap="medium")
    with left:
        selected_model = st.selectbox("Select Classification Model", list(MODEL_FILES))
    with right:
        uploaded_file = st.file_uploader("Upload Test CSV (containing target column)", type=["csv"])
    run_evaluation = st.form_submit_button("Run Evaluation", type="primary")

if not run_evaluation:
    st.info("💡 Choose a model, upload **test_data.csv**, and click **Run Evaluation**.")
    st.stop()

if uploaded_file is None:
    st.error("⚠️ Please upload a test CSV dataset before running the evaluation.")
    st.stop()

try:
    test_df = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"Cannot read CSV: {exc}")
    st.stop()

# Detect target column (handle case-insensitive variations like 'target' or 'Revenue')
target_col = None
for col in ["target", "Revenue", "revenue"]:
    if col in test_df.columns:
        target_col = col
        break

if not target_col:
    st.error("CSV must contain a binary target column (e.g., 'target').")
    st.stop()

raw_target = test_df[target_col].astype(str).str.strip().str.lower()
y = raw_target.map({"true": 1, "false": 0, "1": 1, "0": 0, "benign": 1, "malignant": 0})
if y.isna().any():
    # Fallback to standard numeric conversion
    y = pd.to_numeric(test_df[target_col], errors="coerce")

if y.isna().any() or y.nunique() != 2:
    st.error("Target column must contain exactly 2 unique binary classes (0/1 or True/False/benign/malignant) without missing values.")
    st.stop()

y = y.astype(int)
X = test_df.drop(columns=[target_col])

model_path = BASE_DIR / MODEL_FILES[selected_model]
if not model_path.is_file():
    st.error(f"Model file missing: {MODEL_FILES[selected_model]}")
    st.stop()

try:
    with st.spinner("Evaluating model predictions..."):
        model = load_model(str(model_path))
        
        # Standardize features matching model expectations if feature names are present
        if hasattr(model, "feature_names_in_"):
            expected = list(model.feature_names_in_)
        elif hasattr(model, "named_steps") and "preprocessing" in model.named_steps and hasattr(model.named_steps["preprocessing"], "feature_names_in_"):
            expected = list(model.named_steps["preprocessing"].feature_names_in_)
        else:
            expected = None

        if expected is not None:
            missing = [name for name in expected if name not in X.columns]
            if missing:
                st.error("Missing columns from the model expectation: " + ", ".join(missing))
                st.stop()
            X = X[expected]

        predictions = model.predict(X)
        class_labels = list(model.classes_)
        
        # Obtain prediction probabilities or decision values for ROC calculation
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X)[:, 1]
        elif hasattr(model, "decision_function"):
            probabilities = model.decision_function(X)
        else:
            probabilities = predictions
except Exception as exc:
    st.error(f"Model loading or prediction failed: {exc}")
    st.stop()

# Calculate metrics
acc = accuracy_score(y, predictions)
auc = roc_auc_score(y, probabilities)
prec = precision_score(y, predictions, zero_division=0)
rec = recall_score(y, predictions, zero_division=0)
f1 = f1_score(y, predictions, zero_division=0)
mcc = matthews_corrcoef(y, predictions)

scores = [
    ("Accuracy", acc),
    ("AUC ROC", auc),
    ("Precision", prec),
    ("Recall", rec),
    ("F1 Score", f1),
    ("MCC", mcc)
]

st.subheader("🎯 Model Performance Metrics")
cols = st.columns(6)
for col, (label, score) in zip(cols, scores):
    with col:
        st.metric(label=label, value=f"{score:.4f}")

st.markdown("---")

left_col, right_col = st.columns(2)

with left_col:
    st.subheader("📈 ROC Curve")
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y, probabilities, name=selected_model, ax=ax, color="#1e3d59")
    ax.plot([0, 1], [0, 1], "r--", alpha=0.7, label="Random Guess")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig)

with right_col:
    st.subheader("📊 Confusion Matrix")
    cm = confusion_matrix(y, predictions)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0][0], 0, 0, 0)
    
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Malignant (0)", "Benign (1)"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix Visualization", fontsize=12, fontweight="bold")
    st.pyplot(fig)
    st.info(f"**Summary:** True Negatives (TN): `{tn}` | False Positives (FP): `{fp}` | False Negatives (FN): `{fn}` | True Positives (TP): `{tp}`")
