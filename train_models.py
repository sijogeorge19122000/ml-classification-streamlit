import os
import joblib
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
)

# 1. Load Dataset
data = load_breast_cancer(as_frame=True)
df = data.frame
print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns ({len(data.feature_names)} features + 1 target)")

X = df[data.feature_names]
y = df['target'] # 0 = Malignant, 1 = Benign

# 2. Train-Test Split (80% Train, 20% Test, Stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

# Save test dataset with target column for Streamlit evaluation
test_df = X_test.copy()
test_df['target'] = y_test
os.makedirs('ml_assignment_2', exist_ok=True)
os.makedirs('ml_assignment_2/model', exist_ok=True)
test_csv_path = 'ml_assignment_2/test_data.csv'
test_df.to_csv(test_csv_path, index=False)
print(f"Saved test dataset to: {test_csv_path}")

# 3. Define the 5 Models using Pipelines
models = {
    'Logistic Regression': (
        Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(max_iter=1000, random_state=42))
        ]),
        'logistic_regression.joblib'
    ),
    'Decision Tree': (
        Pipeline([
            ('classifier', DecisionTreeClassifier(max_depth=5, random_state=42))
        ]),
        'decision_tree.joblib'
    ),
    'KNN': (
        Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', KNeighborsClassifier(n_neighbors=5))
        ]),
        'knn.joblib'
    ),
    'Naive Bayes': (
        Pipeline([
            ('classifier', GaussianNB())
        ]),
        'naive_bayes.joblib'
    ),
    'Support Vector Machine': (
        Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC(kernel='rbf', probability=True, random_state=42))
        ]),
        'svm.joblib'
    )
}

# 4. Train, Evaluate, and Save Models
results = []
for name, (pipeline, filename) in models.items():
    pipeline.fit(X_train, y_train)
    
    # Save trained pipeline
    save_path = os.path.join('ml_assignment_2/model', filename)
    joblib.dump(pipeline, save_path)
    
    # Predict on test set
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    # Calculate all 6 required metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    results.append({
        'ML Model Name': name,
        'Accuracy': round(acc, 4),
        'AUC': round(auc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1': round(f1, 4),
        'MCC': round(mcc, 4)
    })
    print(f"Trained and saved {name} -> {save_path}")

# Also save standalone scaler just in case
scaler = StandardScaler()
scaler.fit(X_train)
joblib.dump(scaler, 'ml_assignment_2/model/scaler.joblib')
joblib.dump(list(data.feature_names), 'ml_assignment_2/model/feature_names.joblib')

results_df = pd.DataFrame(results)
print("\n" + "="*70)
print("FINAL MODEL COMPARISON TABLE (TEST DATA)")
print("="*70)
print(results_df.to_string(index=False))
print("="*70)

# Save comparison table to CSV as reference
results_df.to_csv('ml_assignment_2/model_comparison_metrics.csv', index=False)
