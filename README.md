#  Traffic Accident Prediction using Machine Learning
---

## 🚧 Problem Statement  
Traffic accidents seriously threaten public safety, often resulting in injuries, fatalities, and economic losses. While various factors contribute to accidents, developing a data-driven system capable of identifying high-risk scenarios is vital to enable proactive interventions. Our objective is to build a predictive model that classifies whether an accident is likely to occur (`Accident` or `No Accident`) based on historical accident data and road/environment-related features.

---

## 💡 Proposed Solution Framework  

To address the problem, we developed a machine learning pipeline with the following core components:

1. **Data Exploration & Visualization**
2. **Data Preprocessing & Feature Engineering**
3. **Handling Class Imbalance (SMOTE-ENN)**
4. **Model Building using Random Forest**
5. **Hyperparameter Tuning with GridSearchCV**
6. **Final Evaluation using ROC-AUC, Classification Report, Confusion Matrix**

---

## 🔍 Check the code

### 1️⃣ Data Exploration & Visualization

We started by understanding the dataset structure and distributions:

- **Correlation Matrix:**  
  A heatmap was generated for numeric features to identify strongly correlated variables. This helped guide feature selection and highlight multicollinearity.

- **Target Distribution:**  
  Visualized using `countplot` to assess class imbalance in accident severity and binary labels.

- **Categorical Insight (Chi-Square Test):**  
  Used to examine relationships (e.g, between lighting condition and accident) via p-values.

- **Outlier Detection:**  
  Implemented using boxplots for each numeric feature. This helped identify variables with extreme values that could affect model performance.

- **Pairplots:**  
  Visualized relationships between selected features like `Speed_Limit`, `Driver_Age`, and the target class, revealing clusters and patterns.

---

### 2️⃣ Data Preprocessing & Feature Engineering

- **Encoding & Scaling:**  
  Categorical features were encoded, and numerical features were scaled to ensure consistent ranges.

- **Feature Selection:**  
  Choose statistically and visually relevant features based on correlation analysis, domain knowledge, and pairplots.

- **Label Binarization:**  
  Transformed the multi-class severity column into a binary target (`Accident`, `No Accident`) for simplified classification.

---

### 3️⃣ Handling Class Imbalance

Due to the natural imbalance between accident and non-accident cases, we used **SMOTE-ENN**:
- **SMOTE** synthesizes new minority class samples.
- **ENN (Edited Nearest Neighbors)** removes ambiguous or noisy samples.
- Result: A more balanced and cleaner dataset for robust learning.

---

### 4️⃣ Model Building – Random Forest

A `RandomForestClassifier` was selected due to its:
- Robustness to outliers and noisy data
- Built-in feature importance
- Scalability and interpretability

Model was trained using balanced class weights to further mitigate imbalance effects.

---

### 5️⃣ Hyperparameter Tuning – GridSearchCV

Used an extensive grid search over:
- `n_estimators`, `max_depth`, `criterion`
- `max_features`, `min_samples_split`, `min_samples_leaf`

✅ Optimization goal: **maximize ROC-AUC**  
🧠 Cross-validation: StratifiedKFold to preserve class ratio across folds  
⏱ Training time and iteration count were logged to evaluate complexity

---

### 6️⃣ Evaluation & Interpretation

- **Best ROC-AUC Score** was reported from training.
- **Final Test Evaluation** included:
  - Accuracy
  - Precision, Recall, F1-score for each class
  - **Confusion Matrix** annotated with percentages
  - **ROC Curve** to visualize discrimination ability across thresholds


---
