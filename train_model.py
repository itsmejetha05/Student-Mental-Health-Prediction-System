import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer  
from imblearn.over_sampling import SMOTE   
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

# --- 1. LOAD DATA ---
print("Step 1: Loading Data...")
try:
    df = pd.read_csv("Student Mental health.csv")
except FileNotFoundError:
    print("ERROR: File not found. Please make sure 'Student Mental health.csv' is in this folder.")
    exit()

# --- 2. DATA CLEANING ---
# Clean CGPA
def clean_cgpa(val):
    val = str(val).strip()
    if '-' in val:
        try:
            low, high = val.split('-')
            return (float(low) + float(high)) / 2
        except:
            return np.nan
    try:
        return float(val)
    except:
        return np.nan

df['CGPA_Cleaned'] = df['What is your CGPA?'].apply(clean_cgpa)

# Clean Year of Study (Turns "Year 1" or "year 2" into just 1 or 2)
def clean_year(val):
    val = str(val).lower().replace('year', '').strip()
    try:
        return int(val)
    except:
        return 1 # Default to year 1 if missing

df['Year_Cleaned'] = df['Your current year of Study'].apply(clean_year)

# Encode Text Columns -> Numbers
le = LabelEncoder()
cols_to_encode = [
    'Choose your gender', 
    'Marital status', 
    'Do you have Anxiety?', 
    'Do you have Panic attack?',
    'Did you seek any specialist for a treatment?' # ADDED THIS!
]

for col in cols_to_encode:
    df[col] = le.fit_transform(df[col].astype(str))

# Define Target (Depression: Yes/No)
target_col = 'Do you have Depression?'
df[target_col] = le.fit_transform(df[target_col].astype(str))

# --- 3. DEFINE FEATURES (X) AND TARGET (y) ---
# Now we have 8 features!
feature_columns = [
    'Age', 
    'Year_Cleaned',            # Added!
    'CGPA_Cleaned', 
    'Choose your gender', 
    'Do you have Anxiety?', 
    'Do you have Panic attack?',
    'Marital status',
    'Did you seek any specialist for a treatment?' # Added!
]

X = df[feature_columns]
y = df[target_col]

# --- 4. HANDLE MISSING VALUES ---
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# --- 5. BALANCE DATA WITH SMOTE ---
print("\nStep 2: Balancing Data with SMOTE...")
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

# --- 6. SPLIT & SCALE ---
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 7. TRAIN MODELS ---
print("\nStep 3: Training & Comparing Models...")

log_reg = LogisticRegression()
log_reg.fit(X_train_scaled, y_train)
acc_log = accuracy_score(y_test, log_reg.predict(X_test_scaled))

print(f"Logistic Regression Accuracy: {acc_log*100:.2f}%")

# Generate Confusion Matrix Image 
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, log_reg.predict(X_test_scaled)), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix: Logistic Regression')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('confusion_matrix.png')

# --- 8. SAVE SYSTEM ---
data_to_save = {"model": log_reg, "scaler": scaler}
with open('mental_health_model.pkl', 'wb') as f:
    pickle.dump(data_to_save, f)

print("\nSUCCESS: System saved as 'mental_health_model.pkl'")