# %% [markdown]
# # Submission 1 : Menyelesaikan Permasalahan Human Resources

# %% [markdown]
# - Nama: Labib Ammar Fadhali
# - Email: labibfadhali12@gmail.com
# - Id Dicoding: labibaf

# %% [markdown]
# ## Persiapan

# %% [markdown]
# ### Menyiapkan library yang dibutuhkan

# %%
import pandas as pd
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report, confusion_matrix
import joblib

import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# ### Menyiapkan data yang akan diguankan

# %% [markdown]
# ## Data Understanding

# %%
df=pd.read_csv('./dataset/employee_data.csv')
df.head()

# %%
df.shape

# %%
df.info()

# %%
df.isnull().sum()

# %%
df.describe(include='all')

# %% [markdown]
# ## Data Preparation / Preprocessing

# %% [markdown]
# Handling Missing Value

# %%
df['Attrition'].value_counts()

# %%
df['Attrition']=df['Attrition'].fillna(df['Attrition'].mode()[0])

# %%
df['Attrition'].isna().sum()

# %% [markdown]
# Duplicate Check

# %%
df.duplicated().sum()

# %%
df.nunique()

# %% [markdown]
# ### Feature Engineering

# %%
df.head()

# %%
attrition_map={1:'Yes',0:'No'}
df['Attrition']=df['Attrition'].map(attrition_map)

education_map={1:'Below College',2:'College',3:'Bachelor',4:'Master',5:'Doctor'}
df['Education']=df['Education'].map(education_map)

performance_map={1:'Low',2:'Good',3:'Excelllent',4:'Outstanding'}
df['PerformanceRating']=df['PerformanceRating'].map(performance_map)
df['WorkLifeBalance']=df['WorkLifeBalance'].map(performance_map)

sactification_map={1:'Low',2:'Medium',3:'High',4:'Very High'}
df['EnvironmentSatisfaction']=df['EnvironmentSatisfaction'].map(sactification_map)
df['JobSatisfaction']=df['JobSatisfaction'].map(sactification_map)
df['RelationshipSatisfaction']=df['RelationshipSatisfaction'].map(sactification_map)
df['JobInvolvement']=df['JobInvolvement'].map(sactification_map)

# %%
df['AgeAtStartEmployment'] = df['Age'] - df['TotalWorkingYears']

# %%
drop_cols=['EmployeeCount','StandardHours','Over18']
df=df.drop(columns=drop_cols,axis=1)

# %%
df.head()

# %%
df.to_csv('./dataset/employee_data_cleaned.csv',index=False)

# %% [markdown]
# ## EDA

# %%
sns.countplot(df['Attrition'])
plt.title('Attrition Distribution')
plt.show()

# %%
plt.pie(df['Attrition'].value_counts(),labels=df['Attrition'].value_counts().index,autopct='%1.1f%%',startangle=90,colors=['#6ff2fc','#66b3ff'])
plt.show()

# %%
def categorical_plot(features,df,segment_feature=None):
  """
   Parameter:
    - df : DataFrame
        DataFrame yang berisi data.
    - features : list of strings
        Daftar nama kolom yang akan diplot.
    - segment_feature : string, opsional
        Nama kolom untuk segmentasi data (misalnya untuk plot yang dihasilkan menggunakan hue).
  """
  num_plot=len(features)
  fig,ax=plt.subplots(num_plot,figsize=(8,5*num_plot))
  for i, feature in enumerate(features):
    if segment_feature:
      sns.countplot(data=df,x=segment_feature,hue=feature,ax=ax[i])
    else:
      sns.countplot(data=df,x=feature,ax=ax[i])
  plt.tight_layout()
  plt.show()

# %%
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
categorical_columns.remove('Attrition')

# %%
categorical_plot(
   features=categorical_columns,
   df=df,
   segment_feature='Attrition'
)

# %%
df_num = df.select_dtypes(include=['int64', 'float64'])
df_num['Attrition'] = df['Attrition']
df_num['Attrition'] = df_num['Attrition'].map({'Yes': 1, 'No': 0})

# %%
correlation = df_num.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation, annot=False, cmap='coolwarm')
plt.title('Korelasi antara Fitur Numerik dan Attrition')
plt.show()

# %% [markdown]
# ## Modeling

# %%
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# %%
X = df.drop(columns=['EmployeeId', 'Attrition'])
y = df['Attrition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# %%
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Neural Network': MLPClassifier(random_state=42)
}

# %%
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Akurasi model {name}: {accuracy}")

    # Classification Report
    print(f"Classification Report model {name}:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# %% [markdown]
# ## Evaluation

# %%
oversampler = RandomOverSampler(random_state=42)
X_resampled, y_resampled = oversampler.fit_resample(X, y)

# %%
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# %%
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Neural Network': MLPClassifier(random_state=42)
}

# %%
best_model = None
best_accuracy = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Akurasi model {name}: {accuracy}")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        
    # Classification Report
    print(f"Classification Report model {name}:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

print(f"Model terbaik: {best_model}")

# %%
joblib.dump(best_model, './model/best_model.pkl')
print("Model terbaik telah disimpan sebagai 'best_model.pkl'.")


