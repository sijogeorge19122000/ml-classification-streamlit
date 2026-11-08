import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report, roc_curve
)

# Page configuration
st.set_page_config(
    page_title="ML Classification Suite | BITS Pilani Assignment 2",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1A365D;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4A5568;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #F7FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🔬 Machine Learning Classification Evaluator</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">M.Tech (AIML) – Machine Learning Assignment 2 | BITS Pilani</div>', unsafe_allow_html=True)

# Model file mappings
MODEL_FILES = {
    'Logistic Regression': 'model/logistic_regression.joblib',
    'Decision Tree': 'model/decision_tree.joblib',
    'KNN': 'model/knn.joblib',
    'Naive Bayes': 'model/naive_bayes.joblib',
    'Support Vector Machine': 'model/svm.joblib'
}

MODEL_DESCRIPTIONS = {
    'Logistic Regression': 'Linear probabilistic classifier with L2 regularization and feature standardization.',
    'Decision Tree': 'Non-linear tree-based classifier with max depth = 5 to prevent overfitting.',
    'KNN': 'Instance-based non-parametric classifier using k = 5 nearest neighbors with Euclidean distance.',
    'Naive Bayes': 'Gaussian Naive Bayes classifier assuming class-conditional feature independence.',
    'Support Vector Machine': 'Support Vector Classifier with Radial Basis Function (RBF) kernel and probability calibration.'
}

# --- Sidebar Controls ---
st.sidebar.header("⚙️ App Configuration")

# 1. Dataset Upload Option (Required Feature a)
st.sidebar.subheader("1. Upload Test Dataset (CSV)")
uploaded_file = st.sidebar.file_uploader(
    "Upload Test CSV (with input features & target)",
    type=["csv"],
    help="Upload a test CSV containing the 30 continuous features and the 'target' column."
)

# 2. Model Selection Dropdown (Required Feature b)
st.sidebar.subheader("2. Model Selection")
selected_model_name = st.sidebar.selectbox(
    "Choose a Classifier:",
    options=list(MODEL_FILES.keys()),
    index=0
)

# 3. View All Models Comparison Toggle
compare_all = st.sidebar.checkbox("📊 Compare All 5 Models", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Student Details:**  
- **Course:** M.Tech AIML  
- **Subject:** Machine Learning (Assignment 2)  
- **Institution:** BITS Pilani (WILP)
""")

# --- Data Loading Logic ---
@st.cache_data
def load_default_data():
    if os.path.exists('test_data.csv'):
        return pd.read_csv('test_data.csv')
    elif os.path.exists('ml_assignment_2/test_data.csv'):
        return pd.read_csv('ml_assignment_2/test_data.csv')
    else:
        return None

if uploaded_file is not None:
    try:
        data_df = pd.read_csv(uploaded_file)
        st.sidebar.success(f"Uploaded: {uploaded_file.name} ({data_df.shape[0]} rows, {data_df.shape[1]} cols)")
    except Exception as e:
        st.error(f"Error reading uploaded CSV: {e}")
        st.stop()
else:
    data_df = load_default_data()
    if data_df is not None:
        st.info("ℹ️ Using default bundled test dataset (`test_data.csv` - 114 samples). You can also upload your own test CSV in the sidebar.")
    else:
        st.warning("⚠️ No test data found. Please upload a test dataset CSV via the sidebar.")
        st.stop()

# Validate target column
target_col = None
for col_candidate in ['target', 'diagnosis', 'Class', 'class', 'Target']:
    if col_candidate in data_df.columns:
        target_col = col_candidate
        break

if target_col is None:
    st.error("❌ The dataset must contain a target column named 'target' (or 'diagnosis'). Please check your CSV.")
    st.stop()

X_test = data_df.drop(columns=[target_col])
y_test = data_df[target_col]

# Convert categorical target if needed
if y_test.dtype == object:
    y_test = y_test.map({'M': 0, 'B': 1, 'Malignant': 0, 'Benign': 1, '0': 0, '1': 1})

# --- Model Loading & Evaluation Function ---
@st.cache_resource
def load_model(model_name):
    path = MODEL_FILES[model_name]
    if not os.path.exists(path):
        # Try alternate path inside ml_assignment_2
        alt_path = os.path.join('ml_assignment_2', path)
        if os.path.exists(alt_path):
            path = alt_path
        else:
            return None
    return joblib.load(path)

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    try:
        y_prob = model.predict_proba(X)[:, 1]
    except Exception:
        y_prob = y_pred # fallback if decision_function
    
    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_prob)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y, y_pred)
    
    return {
        'Accuracy': acc,
        'AUC': auc,
        'Precision': prec,
        'Recall': rec,
        'F1': f1,
        'MCC': mcc,
        'y_pred': y_pred,
        'y_prob': y_prob
    }

# --- Main Evaluation View ---
with st.expander("📄 View Test Dataset Preview", expanded=False):
    st.write(f"**Shape:** {data_df.shape[0]} rows × {data_df.shape[1]} columns")
    st.dataframe(data_df.head(10), use_container_width=True)
    st.caption("Class distribution: 0 = Malignant (positive finding), 1 = Benign (negative finding)")

# Load selected model
model = load_model(selected_model_name)
if model is None:
    st.error(f"❌ Could not find model artifact for {selected_model_name}. Please verify that `model/` contains trained `.joblib` files.")
    st.stop()

# Evaluate
eval_res = evaluate_model(model, X_test, y_test)

# Display Selected Model Details
st.markdown(f"### 🤖 Active Model: **{selected_model_name}**")
st.write(f"*{MODEL_DESCRIPTIONS[selected_model_name]}*")

# 3. Display of Evaluation Metrics (Required Feature c)
st.markdown("#### 📈 Key Performance Indicators (KPIs)")
col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Accuracy", f"{eval_res['Accuracy']:.4f}")
col2.metric("AUC Score", f"{eval_res['AUC']:.4f}")
col3.metric("Precision", f"{eval_res['Precision']:.4f}")
col4.metric("Recall", f"{eval_res['Recall']:.4f}")
col5.metric("F1 Score", f"{eval_res['F1']:.4f}")
col6.metric("MCC Score", f"{eval_res['MCC']:.4f}")

st.markdown("---")

# 4. Confusion Matrix and Classification Report (Required Feature d)
tab1, tab2, tab3 = st.tabs(["🔲 Confusion Matrix", "📋 Classification Report", "📉 ROC Curve"])

with tab1:
    col_cm1, col_cm2 = st.columns([1.2, 1])
    cm = confusion_matrix(y_test, eval_res['y_pred'])
    
    with col_cm1:
        fig_cm, ax_cm = plt.subplots(figsize=(4.5, 3.5), dpi=150)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Malignant (0)', 'Benign (1)'],
                    yticklabels=['Malignant (0)', 'Benign (1)'], ax=ax_cm)
        ax_cm.set_xlabel('Predicted Class', fontweight='bold', fontsize=9)
        ax_cm.set_ylabel('Actual Class', fontweight='bold', fontsize=9)
        ax_cm.set_title(f'Confusion Matrix: {selected_model_name}', fontweight='bold', fontsize=10, pad=10)
        plt.tight_layout()
        st.pyplot(fig_cm)
        plt.close()
        
    with col_cm2:
        st.markdown("##### Detailed Breakdown:")
        tn, fp, fn, tp = cm.ravel()
        st.write(f"- **True Negatives (TN):** {tn} (Correctly classified Malignant)")
        st.write(f"- **False Positives (FP):** {fp} (Malignant misclassified as Benign)")
        st.write(f"- **False Negatives (FN):** {fn} (Benign misclassified as Malignant)")
        st.write(f"- **True Positives (TP):** {tp} (Correctly classified Benign)")
        st.info("💡 High recall on malignant cases is critical in medical diagnostics to prevent untreated malignancy.")

with tab2:
    st.markdown("##### Classification Report (Precision, Recall, F1 per class):")
    report_dict = classification_report(y_test, eval_res['y_pred'], target_names=['Malignant (0)', 'Benign (1)'], output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    st.dataframe(report_df.style.format(precision=4), use_container_width=True)

with tab3:
    fpr, tpr, _ = roc_curve(y_test, eval_res['y_prob'])
    fig_roc, ax_roc = plt.subplots(figsize=(5.5, 3.5), dpi=150)
    ax_roc.plot(fpr, tpr, color='#1A365D', lw=2.2, label=f'{selected_model_name} (AUC = {eval_res["AUC"]:.4f})')
    ax_roc.plot([0, 1], [0, 1], color='#CBD5E0', linestyle='--', lw=1.5, label='Random Guess')
    ax_roc.set_xlabel('False Positive Rate', fontweight='bold', fontsize=9)
    ax_roc.set_ylabel('True Positive Rate', fontweight='bold', fontsize=9)
    ax_roc.set_title('Receiver Operating Characteristic (ROC) Curve', fontweight='bold', fontsize=10)
    ax_roc.legend(loc='lower right', fontsize=8.5)
    ax_roc.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig_roc)
    plt.close()

# --- All Models Comparison Section ---
if compare_all:
    st.markdown("---")
    st.markdown("### 🏆 Comprehensive Comparison: All 5 Classifiers")
    
    all_results = []
    for m_name in MODEL_FILES.keys():
        m_obj = load_model(m_name)
        if m_obj is not None:
            m_eval = evaluate_model(m_obj, X_test, y_test)
            all_results.append({
                'ML Model Name': m_name,
                'Accuracy': round(m_eval['Accuracy'], 4),
                'AUC': round(m_eval['AUC'], 4),
                'Precision': round(m_eval['Precision'], 4),
                'Recall': round(m_eval['Recall'], 4),
                'F1': round(m_eval['F1'], 4),
                'MCC': round(m_eval['MCC'], 4)
            })
            
    comp_df = pd.DataFrame(all_results)
    st.dataframe(
        comp_df.style.highlight_max(axis=0, subset=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC'], color='#C6F6D5'),
        use_container_width=True
    )
    
    # Comparison Bar Chart
    fig_bar, ax_bar = plt.subplots(figsize=(9, 4), dpi=150)
    metrics_to_plot = ['Accuracy', 'AUC', 'F1', 'MCC']
    comp_plot_df = comp_df.set_index('ML Model Name')[metrics_to_plot]
    comp_plot_df.plot(kind='bar', ax=ax_bar, colormap='viridis', width=0.75)
    ax_bar.set_title('Side-by-Side Model Performance Comparison', fontweight='bold', fontsize=11, pad=12)
    ax_bar.set_ylabel('Score', fontweight='bold', fontsize=9.5)
    ax_bar.set_ylim(0.75, 1.02)
    ax_bar.set_xticklabels(ax_bar.get_xticklabels(), rotation=15, ha='right', fontsize=8.5)
    ax_bar.legend(loc='lower right', fontsize=8.5)
    ax_bar.grid(True, linestyle='--', alpha=0.5, axis='y')
    plt.tight_layout()
    st.pyplot(fig_bar)
    plt.close()

st.markdown("---")
st.caption("Developed for BITS Pilani WILP M.Tech AIML – Machine Learning Assignment 2.")
