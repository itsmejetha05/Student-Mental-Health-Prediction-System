import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# ==========================================
# 1. DATASET AND PREPROCESSING
# ==========================================
# Load the 1,000-row dataset
df = pd.read_csv("Student Mental Health.csv")

# --- MANDATORY INITIAL DATA INSPECTION (This shows the actual data) ---
print("\n" + "="*60)
print("SECTION 3.1: INITIAL DATA INSPECTION (RAW DATA)")
print("="*60)
print("1. ACTUAL DATA PREVIEW (First 5 Rows):")
print(df.head()) # <--- THIS IS THE COMMAND THAT SHOWS THE ACTUAL DATA

print("\n2. DATASET STRUCTURE (Metadata):")
print(df.info()) 

print("\n3. MISSING VALUES ASSESSMENT:")
print(df.isnull().sum())
print("="*60 + "\n")

# Data Cleaning Logic
def clean_cgpa(val):
    try:
        return float(val)
    except:
        return np.nan

# Transformations
df['CGPA_Cleaned'] = df['CGPA'].apply(clean_cgpa)
df['Year_Cleaned'] = df['YearOfStudy'].str.extract('(\d+)').astype(float).fillna(1)

# Encoding
le = LabelEncoder()
df['Gender_Encoded'] = le.fit_transform(df['Gender'].astype(str))

# Feature Selection (7 features)
feature_cols = ['Age', 'Year_Cleaned', 'CGPA_Cleaned', 'Gender_Encoded', 
                'Anxiety', 'PanicAttack', 'SpecialistTreatment']

X = df[feature_cols].fillna(df[feature_cols].mean())
y = df['Depression']

print("\n--- Dataset Structure after Encoding (Head) ---")
print(X.head())

# ==========================================
# 2. DATA PARTITIONING (70/15/15)
# ==========================================
# Split into Train (70%) and Temp (30%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
# Split Temp into Validation (15%) and Test (15%)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

print("\n--- Data Split Sizes ---")
print(f"Training set: {len(X_train)} rows")
print(f"Validation set: {len(X_val)} rows")
print(f"Test set: {len(X_test)} rows")

# ==========================================
# 3. NORMALIZATION EVIDENCE
# ==========================================
scaler_viz = StandardScaler()
X_scaled_viz = pd.DataFrame(scaler_viz.fit_transform(X), columns=X.columns)

print("\n--- Data after Normalization (Head) ---")
print(X_scaled_viz[['Age', 'CGPA_Cleaned']].head())

print("\n--- Descriptive Stats for Normalized Columns ---")
print(X_scaled_viz[['Age', 'CGPA_Cleaned']].describe())

# ==========================================
# 4. CLASS IMBALANCE HANDLING (SMOTE)
# ==========================================
# Distribution BEFORE SMOTE
plt.figure(figsize=(6, 5))
sns.countplot(x=y_train, palette='viridis')
plt.title('Class Distribution Before SMOTE')
plt.xlabel('Depression Status (0: No, 1: Yes)')
plt.ylabel('Number of Samples')
plt.tight_layout()
plt.savefig('class_distribution_before_smote.png')

# Apply SMOTE for training
smote_visual = SMOTE(random_state=42)
X_train_res, y_train_res = smote_visual.fit_resample(X_train, y_train)

# Distribution AFTER SMOTE
plt.figure(figsize=(6, 5))
sns.countplot(x=y_train_res, palette='magma')
plt.title('Class Distribution After SMOTE')
plt.xlabel('Depression Status (0: No, 1: Yes)')
plt.ylabel('Number of Samples')
plt.tight_layout()
plt.savefig('class_distribution_after_smote.png')

# ==========================================
# 5. MODEL TRAINING & TUNING
# ==========================================

# Scenario 1: Baseline Logistic Regression (no SMOTE)
lr_base = LogisticRegression(max_iter=1000, random_state=42)
lr_base.fit(X_train, y_train)
y_pred_lr_base = lr_base.predict(X_val)
y_prob_lr_base = lr_base.predict_proba(X_val)[:, 1]

# Scenario 2: Tuned SMOTE Pipeline (Standardization -> SMOTE -> Model)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('model', LogisticRegression(max_iter=1000, random_state=42))
])

param_grid = {'model__C': [0.01, 0.1, 1, 10]}
grid_lr = GridSearchCV(pipeline, param_grid, scoring='f1', cv=5)
grid_lr.fit(X_train, y_train)

# Save best model pipeline for app.py
with open('mental_health_model.pkl', 'wb') as f:
    pickle.dump(grid_lr.best_estimator_, f)

# Output Hyperparameter Results for Figure 13
print("\n" + "="*50)
print("--- HYPERPARAMETER TUNING RESULTS  ---")
print("="*50)
results_df = pd.DataFrame(grid_lr.cv_results_)
tuning_results = results_df[['param_model__C', 'mean_test_score', 'std_test_score']].rename(columns={
    'param_model__C': 'C Value',
    'mean_test_score': 'Mean F1-Score (5-Fold CV)',
    'std_test_score': 'Std Dev F1-Score'
}).sort_values(by='Mean F1-Score (5-Fold CV)', ascending=False)
print(tuning_results.to_string(index=False))
print(f"\nBest Parameters: {grid_lr.best_params_}")
print("="*50 + "\n")

# ==========================================
# 6. EVALUATION VISUALS
# ==========================================

def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    group_names = ['TN', 'FP', 'FN', 'TP']
    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts, group_names)]
    labels = np.asarray(labels).reshape(2, 2)
    
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', cbar=False,
                xticklabels=['Predicted No (0)', 'Predicted Yes (1)'],
                yticklabels=['Actual No (0)', 'Actual Yes (1)'],
                annot_kws={"size": 14})
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(filename)

# Confusion Matrices
plot_confusion_matrix(y_val, y_pred_lr_base, "Baseline Confusion Matrix", "confusion_baseline.png")
plot_confusion_matrix(y_val, grid_lr.predict(X_val), "SMOTE-Enhanced Confusion Matrix", "confusion_smote.png")

# ROC Curve Comparison
fpr1, tpr1, _ = roc_curve(y_val, y_prob_lr_base)
fpr2, tpr2, _ = roc_curve(y_val, grid_lr.predict_proba(X_val)[:, 1])
plt.figure(figsize=(8, 6))
plt.plot(fpr1, tpr1, label=f'Baseline (AUC = {auc(fpr1, tpr1):.3f})')
plt.plot(fpr2, tpr2, label=f'SMOTE (AUC = {auc(fpr2, tpr2):.3f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve Performance Comparison')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.tight_layout()
plt.savefig('roc_comparison.png')

# Feature Weights
best_model = grid_lr.best_estimator_.named_steps['model']
coef_df = pd.DataFrame({'Feature': feature_cols, 'Weight': best_model.coef_[0]}).sort_values(by='Weight', ascending=False)

plt.figure(figsize=(14, 8)) 
sns.barplot(x='Weight', y='Feature', data=coef_df, palette='coolwarm')
plt.title('Logistic Regression Feature Weights (1000 Rows)')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.subplots_adjust(left=0.3) 
plt.savefig('feature_importance.png')

# Final Test Set Evaluation
print("\n--- Final Evaluation on Test Set (SMOTE Model) ---")
print(classification_report(y_test, grid_lr.predict(X_test)))