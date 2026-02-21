import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# --- 1. LOAD & CLEAN DATA (Matching your 8-feature model) ---
df = pd.read_csv('Student Mental health.csv')

def clean_cgpa(val):
    val = str(val).strip()
    if '-' in val:
        try:
            low, high = val.split('-')
            return (float(low) + float(high)) / 2
        except:
            return 0.0
    try:
        return float(val)
    except:
        return 0.0

def clean_year(val):
    val = str(val).lower().replace('year', '').strip()
    try:
        return int(val)
    except:
        return 1 

df['CGPA_Cleaned'] = df['What is your CGPA?'].apply(clean_cgpa)
df['Year_Cleaned'] = df['Your current year of Study'].apply(clean_year)

# Encode text to numbers for the heatmap
le = LabelEncoder()
cols_to_encode = [
    'Choose your gender', 'Marital status', 'Do you have Anxiety?', 
    'Do you have Panic attack?', 'Did you seek any specialist for a treatment?',
    'Do you have Depression?'
]

df_encoded = df.copy()
for col in cols_to_encode:
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

# --- 2. GENERATE THE IMAGES ---

# Graph 1: Pie Chart (Proving the need for SMOTE)
plt.figure(figsize=(6,6))
df['Do you have Depression?'].value_counts().plot(
    kind='pie', autopct='%1.1f%%', colors=['#66b3ff','#ff9999'], startangle=90
)
plt.title('Depression Distribution (Before SMOTE)')
plt.ylabel('')
plt.savefig('graph_1_imbalance.png')
print("✅ Saved 'graph_1_imbalance.png'")

# Graph 2: CGPA vs Depression (Boxplot)
plt.figure(figsize=(8,5))
sns.boxplot(x='Do you have Depression?', y='CGPA_Cleaned', data=df, palette='Set2')
plt.title('Impact of CGPA on Depression Risk')
plt.savefig('graph_2_cgpa.png')
print("✅ Saved 'graph_2_cgpa.png'")

# Graph 3: Specialist vs Depression (Bar Chart for the new feature)
plt.figure(figsize=(8,5))
sns.countplot(x='Did you seek any specialist for a treatment?', hue='Do you have Depression?', data=df, palette='Pastel1')
plt.title('Specialist Treatment vs. Depression')
plt.savefig('graph_3_specialist.png')
print("✅ Saved 'graph_3_specialist.png'")

# Graph 4: The Massive 8-Feature Correlation Heatmap
plt.figure(figsize=(10,8))
feature_cols = [
    'Age', 'Year_Cleaned', 'CGPA_Cleaned', 'Choose your gender', 
    'Do you have Anxiety?', 'Do you have Panic attack?', 'Marital status', 
    'Did you seek any specialist for a treatment?', 'Do you have Depression?'
]
sns.heatmap(df_encoded[feature_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('8-Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('graph_4_heatmap.png')
print("✅ Saved 'graph_4_heatmap.png'")