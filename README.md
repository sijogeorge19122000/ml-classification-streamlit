# End-to-End Machine Learning Classification & Streamlit Deployment
## BITS Pilani – M.Tech (AIML) | Machine Learning Assignment 2

---

### **Mandatory Submission Details**
- **Student Name:** Sijo George
- **BITS Student ID:** 2026AK05604
- **Program:** M.Tech (AIML) – 1st Semester
- **Course:** Machine Learning
- **GitHub Repository Link:** [https://github.com/sijogeorge/ml-classification-streamlit](https://github.com/sijogeorge/ml-classification-streamlit) *(Update with your repository URL)*
- **Live Streamlit App Link:** [https://ml-classification-evaluator.streamlit.app](https://ml-classification-evaluator.streamlit.app) *(Update with your deployed app URL)*

---

## **a. Problem Statement**
Breast cancer diagnosis requires rapid and highly reliable discrimination between **malignant** (cancerous) and **benign** (non-cancerous) tissue masses. Fine needle aspiration (FNA) biopsy samples produce digitized microscopic images of cell nuclei. 

The objective of this assignment is to develop, evaluate, and deploy an interactive end-to-end machine learning system that utilizes 30 continuous geometric and morphological characteristics of cell nuclei to classify tumors accurately. Crucially, in clinical oncology, minimizing false negatives (misclassifying malignant tumors as benign) while maintaining high overall precision and Matthew's Correlation Coefficient (MCC) is vital for patient survival and effective treatment planning.

---

## **b. Dataset Description [1 Mark]**
- **Dataset Name:** Breast Cancer Wisconsin (Diagnostic) Dataset
- **Repository Source:** UCI Machine Learning Repository (Dataset ID: 17) & Kaggle
- **Total Instances (Rows):** 569 samples *(Satisfies requirement of Minimum 500 instances)*
- **Total Predictive Features:** 30 continuous real-valued numeric features *(Satisfies requirement of Minimum 12 features)*
- **Target Variable:** `target` (Binary Classification: `0 = Malignant`, `1 = Benign`)
- **Class Distribution:**
  - Benign ($1$): 357 instances (62.74%)
  - Malignant ($0$): 212 instances (37.26%)
- **Feature Computation:** Ten fundamental characteristics of cell nuclei are captured:
  1. *Radius* (mean of distances from center to points on the perimeter)
  2. *Texture* (standard deviation of gray-scale values)
  3. *Perimeter*
  4. *Area*
  5. *Smoothness* (local variation in radius lengths)
  6. *Compactness* ($\text{perimeter}^2 / \text{area} - 1.0$)
  7. *Concavity* (severity of concave portions of the contour)
  8. *Concave Points* (number of concave portions of the contour)
  9. *Symmetry*
  10. *Fractal Dimension* ("coastline approximation" - 1.0)
  
  For each image, the **Mean**, **Standard Error (SE)**, and **"Worst"** (mean of the three largest values) are computed, producing $10 \times 3 = 30$ predictive features.
- **Data Splitting & Preprocessing:** 
  - Stratified 80% train ($455$ instances) and 20% test ($114$ instances) split.
  - Features standardized using `StandardScaler` within unified Scikit-Learn pipelines.
  - Zero missing values.

---

## **c. GitHub Repository Link [1 Mark]**
- **Repository URL:** `https://github.com/sijogeorge/ml-classification-streamlit`
- **Repository Contents:**
  - `app.py`: Interactive Streamlit multi-model evaluation dashboard.
  - `requirements.txt`: Clean, version-pinned deployment dependencies.
  - `README.md`: Complete documentation, metrics comparison, and observations.
  - `test_data.csv`: Unseen test partition ($114$ samples with 30 features + `target`).
  - `model/`: Serialized pre-trained pipelines (`.joblib` files) for all 5 classifiers.
  - `train_models.py`: Complete reproducible training and evaluation script.

---

## **d. Models Used & Comparison Table [5 Marks]**
Five distinct classification algorithms were trained on the identical training set ($n=455$) and evaluated on the identical unseen test set ($n=114$).

### **Evaluation Metrics Comparison Table**
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.9825** | **0.9954** | **0.9861** | **0.9861** | **0.9861** | **0.9623** |
| **Decision Tree** | 0.9211 | 0.9163 | 0.9565 | 0.9167 | 0.9362 | 0.8341 |
| **KNN** | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| **Naive Bayes** | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| **Support Vector Machine** | **0.9825** | 0.9950 | **0.9861** | **0.9861** | **0.9861** | **0.9623** |

*Note: In medical evaluation, Recall reflects the true positive rate for benign cases while maintaining only 1 false positive on malignant cases (41 out of 42 malignant cases correctly identified).*

---

## **e. Observations on Model Performance [3 Marks]**

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Achieves top-tier performance across all evaluation dimensions (**Accuracy: 98.25%**, **AUC: 0.9954**, **MCC: 0.9623**). Because the 30 standardized features exhibit strong linear separability in high-dimensional space, the L2-regularized sigmoid decision surface separates classes cleanly with only 2 total misclassifications out of 114 test samples. |
| **Decision Tree** | Lowest performing model in the benchmark (**Accuracy: 92.11%**, **AUC: 0.9163**, **MCC: 0.8341**). Because standard decision trees execute axis-parallel orthogonal splits, they struggle to model smooth diagonal correlations between geometric features (e.g., radius, perimeter, and area), leading to step-function approximations and lower generalization on unseen test data. |
| **KNN** | Strong non-parametric performance (**Accuracy: 95.61%**, **AUC: 0.9788**, **MCC: 0.9054**). By computing standardized Euclidean distances across $k=5$ neighbors, KNN captures local geometric clustering effectively. However, inference latency scales with sample size, and it is sensitive to feature noise in higher dimensions. |
| **Naive Bayes** | Solid probability ranking with high discriminative capability (**AUC: 0.9878**, **Accuracy: 93.86%**, **MCC: 0.8676**). Although Gaussian Naive Bayes makes the strong assumption of class-conditional feature independence (which is violated due to high collinearity between radius and perimeter), it remains robust and requires minimal computational overhead. |
| **Support Vector Machine** | Tied for the highest accuracy (**98.25%**) and MCC (**0.9623**) with an outstanding AUC (**0.9950**). The Radial Basis Function (RBF) kernel projects the 30 continuous features into an infinite-dimensional Hilbert space, maximizing the geometric margin between malignant and benign support vectors. |
| **Overall, Winner for your dataset?** | **Logistic Regression is the overall winner.** While SVM achieves identical accuracy and F1 score, Logistic Regression slightly edges out SVM in **AUC (0.9954 vs. 0.9950)**, provides faster inference, requires fewer hyperparameters to tune, and outputs directly interpretable, calibrated class probabilities that are highly valued in clinical diagnostic workflows. |

---

## **f. Streamlit Web Application Features**
The interactive web application includes:
1. **Dataset Upload Option:** Allows users to upload custom test CSV files containing input features and target labels. If no file is uploaded, the app automatically falls back to the bundled `test_data.csv`.
2. **Model Selection Dropdown:** Seamlessly switch between all 5 trained classifiers without retraining.
3. **Display of Evaluation Metrics:** Real-time calculation and display of Accuracy, AUC, Precision, Recall, F1, and MCC score cards.
4. **Interactive Visualizations:**
   - Color-coded Confusion Matrix with TP, TN, FP, FN breakdown.
   - Comprehensive Scikit-Learn Classification Report.
   - Receiver Operating Characteristic (ROC) curve with AUC indicator.
   - Side-by-side comparative bar chart across all 5 models.

---

## **g. How to Run Locally & Deploy**

### **1. Run Locally**
```bash
# Clone repository
git clone https://github.com/sijogeorge/ml-classification-streamlit.git
cd ml-classification-streamlit

# Install dependencies
pip install -r requirements.txt

# Run Streamlit App
streamlit run app.py
```

### **2. Deploy on Streamlit Community Cloud (FREE)**
1. Push this folder to a public GitHub repository.
2. Visit [https://streamlit.io/cloud](https://streamlit.io/cloud) and sign in with GitHub.
3. Click **"New App"**.
4. Select your repository, branch (`main`), and file path (`app.py`).
5. Click **"Deploy"** — the live app URL will be available immediately!
