import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# ==========================================
# 3.1 DATASET AND PREPROCESSING
# ==========================================
# Sourcing data from Kaggle Student Mental Health Dataset
df = pd.read_csv("Student Mental health.csv")

print("--- Figure 3: Initial Dataset Structure & Missing Values ---")
print(df.info()) 
print("\nMissing Values Count:\n", df.isnull().sum())

# --- Data Cleaning Logic ---
def clean_cgpa(val):
    val = str(val).strip()
    if '-' in val:
        try:
            low, high = val.split('-')
            return (float(low) + float(high)) / 2
        except: return np.nan
    try: return float(val)
    except: return np.nan

df['CGPA_Cleaned'] = df['What is your CGPA?'].apply(clean_cgpa)
df['Year_Cleaned'] = df['Your current year of Study'].str.extract('(\d+)').astype(float)

# --- Encoding Categorical Features ---
le = LabelEncoder()
categorical_cols = ['Choose your gender', 'Marital status', 'Do you have Anxiety?', 
                  'Do you have Panic attack?', 'Did you seek any specialist for a treatment?']

for col in categorical_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# Encoding target variable: Depression
df['Depression'] = le.fit_transform(df['Do you have Depression?'].astype(str))

# --- Feature Selection (8 key columns) ---
feature_cols = ['Age', 'Year_Cleaned', 'CGPA_Cleaned', 'Choose your gender', 
                'Do you have Anxiety?', 'Do you have Panic attack?', 
                'Marital status', 'Did you seek any specialist for a treatment?']

# Handle any missing values in Age/CGPA with mean imputation for consistency
X = df[feature_cols].fillna(df[feature_cols].mean())
y = df['Depression']

print("\n--- Figure 4: Dataset Structure after Encoding (Head) ---")
print(X.head())

# ==========================================
# 3.3 DATA PARTITIONING (70/15/15)
# ==========================================
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

print("\n--- 3.3 Data Split Sizes ---")
print(f"Training set: {len(X_train)} rows")
print(f"Validation set: {len(X_val)} rows")
print(f"Test set: {len(X_test)} rows")

# ==========================================
# 3.1.5 NORMALIZATION EVIDENCE
# ==========================================
scaler_viz = StandardScaler()
X_scaled_viz = pd.DataFrame(scaler_viz.fit_transform(X), columns=X.columns)

print("\n--- Figure 5: Data after Normalization (Head) ---")
print(X_scaled_viz[['Age', 'CGPA_Cleaned']].head())

print("\n--- 3.1.5 Descriptive Stats for Relevant Columns ---")
print(X_scaled_viz[['Age', 'CGPA_Cleaned']].describe())

# ==========================================
# 3.4 HANDLING CLASS IMBALANCE (SMOTE)
# ==========================================
# Generating Visual Evidence for SMOTE
plt.figure(figsize=(6, 5))
sns.countplot(x=y_train, palette='viridis')
plt.title('Class Distribution Before SMOTE')
plt.savefig('class_distribution_before_smote.png')

# Apply SMOTE for visualization purposes
smote_viz = SMOTE(random_state=42)
X_res_viz, y_res_viz = smote_viz.fit_resample(X_train, y_train)

plt.figure(figsize=(6, 5))
sns.countplot(x=y_res_viz, palette='magma')
plt.title('Class Distribution After SMOTE')
plt.savefig('class_distribution_after_smote.png')

# ==========================================
# 3.5 MODEL IMPLEMENTATION (PIPELINE)
# ==========================================
# Automated Pipeline for Scaling -> Balancing -> Classification

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('model', LogisticRegression(max_iter=1000, random_state=42))
])

# Inverse Regularization Strength Tuning
param_grid = {'model__C': [0.01, 0.1, 1, 10]}
grid_lr = GridSearchCV(pipeline, param_grid, scoring='f1', cv=5)
grid_lr.fit(X_train, y_train)

# ==========================================
# 4.1 FINAL EVALUATION & FEATURE IMPORTANCE
# ==========================================
best_model = grid_lr.best_estimator_.named_steps['model']
coef_df = pd.DataFrame({'Feature': feature_cols, 'Weight': best_model.coef_[0]}).sort_values(by='Weight')

# Save Feature Importance Plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Weight', y='Feature', data=coef_df)
plt.title('Logistic Regression Feature Importance Weights')
plt.savefig('lr_coefficients.png')

# Output final performance metrics for Chapter 4
y_pred = grid_lr.predict(X_test)
print("\n--- Section 4.1: Final Evaluation on Test Set ---")
print(classification_report(y_test, y_pred))

# Save the Best Pipeline for the Streamlit App
with open('mental_health_model.pkl', 'wb') as f:
    pickle.dump(grid_lr.best_estimator_, f)

print("\n✅ All processes complete. Screenshots and Model ready.")