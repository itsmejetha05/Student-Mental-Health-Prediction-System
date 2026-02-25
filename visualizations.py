import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# --- 1. DATA PREPARATION ---
df = pd.read_csv('Student Mental health.csv')

def clean_data(data):
    data['CGPA_Cleaned'] = data['What is your CGPA?'].apply(lambda x: float(str(x).split('-')[0]) if '-' in str(x) else float(x))
    data['Year_Cleaned'] = data['Your current year of Study'].str.extract('(\d+)').astype(float).fillna(1)
    le = LabelEncoder()
    cols = ['Choose your gender', 'Marital status', 'Do you have Anxiety?', 'Do you have Panic attack?', 'Did you seek any specialist for a treatment?']
    for col in cols:
        data[col] = le.fit_transform(data[col].astype(str))
    data['Depression'] = le.fit_transform(data['Do you have Depression?'].astype(str))
    return data

df = clean_data(df)
feature_cols = ['Age', 'Year_Cleaned', 'CGPA_Cleaned', 'Choose your gender', 'Do you have Anxiety?', 'Do you have Panic attack?', 'Marital status', 'Did you seek any specialist for a treatment?']
X = df[feature_cols].fillna(df[feature_cols].mean())
y = df['Depression']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# --- 2. GENERATE FIGURES FOR REPORT ---

# Figure 9 & 10: Class Distribution (Before & After SMOTE)
plt.figure(figsize=(6,5))
sns.countplot(x=y_train, palette='viridis')
plt.title('Figure 9: Class Distribution Before SMOTE')
plt.savefig('figure_9_before.png')

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

plt.figure(figsize=(6,5))
sns.countplot(x=y_res, palette='magma')
plt.title('Figure 10: Class Distribution After SMOTE')
plt.savefig('figure_10_after.png')

# Figure 14: Baseline Confusion Matrix (Matches Jashmina style)
lr_base = LogisticRegression(max_iter=1000)
lr_base.fit(X_train, y_train)
y_pred_base = lr_base.predict(X_test)

def plot_cm(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    group_names = ['TN','FP','FN','TP']
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(cm.flatten(), group_names)]
    labels = np.asarray(labels).reshape(2,2)
    plt.figure(figsize=(7,6))
    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', xticklabels=['No','Yes'], yticklabels=['No','Yes'])
    plt.title(title)
    plt.savefig(filename)

plot_cm(y_test, y_pred_base, 'Figure 14: Logistic Regression Baseline', 'figure_14_baseline.png')

# Figure 15: SMOTE Confusion Matrix
lr_smote = LogisticRegression(max_iter=1000)
lr_smote.fit(X_res, y_res)
y_pred_smote = lr_smote.predict(X_test)
plot_cm(y_test, y_pred_smote, 'Figure 15: Logistic Regression with SMOTE', 'figure_15_smote.png')

print("✅ All figures (9, 10, 14, 15) saved for Chapter 4.")