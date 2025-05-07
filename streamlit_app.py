import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import joblib

# Set Streamlit page
st.set_page_config(page_title="Fraud Detection", layout="wide")

st.title("🧠 Fraud Detection with Comparative Machine Learning Models")
st.markdown("Upload a CSV file, see EDA, and compare multiple ML models.")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Raw Dataset")
    st.write(df.head())

    # --- Data Preprocessing ---
    df.drop_duplicates(inplace=True)
    df.drop(['Transaction_ID'], axis=1, inplace=True, errors='ignore')

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('Unknown')
        else:
            df[col] = df[col].fillna(df[col].mean())

  # Label encoding for EDA
    le = LabelEncoder()
    df_eda = df.copy()
    cat_cols = ['Transaction_Type', 'Device_Used', 'Location', 'Payment_Method']
    for col in cat_cols:
        if col in df_eda.columns:
            df_eda[col] = le.fit_transform(df_eda[col])

    # --- EDA Section ---
    st.subheader("📊 EDA - Correlation Matrix")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df_eda.corr(), annot=True, ax=ax)
    st.pyplot(fig)

    # st.subheader("📊 Fraudulent Distribution")
    # fig, ax = plt.subplots()
    # sns.countplot(data=df, x='Fraudulent', palette={0: 'blue', 1: 'red'}, ax=ax)
    # st.pyplot(fig)

    # --- Model Training ---
    st.subheader("⚙️ Model Comparison")

    # One-hot encoding for training
    df_train = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    X = df_train.drop(['Fraudulent'], axis=1)
    y = df_train['Fraudulent']

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, stratify=y_res, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
       "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(eval_metric='logloss', use_label_encoder=False),
        "CatBoost": CatBoostClassifier(verbose=0),
        "LightGBM": LGBMClassifier(),
        "Logistic Regression": LogisticRegression(max_iter=500)
    }

    results = []
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        roc_auc = roc_auc_score(y_test, y_pred)

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "F1 Score": f1,
            "ROC AUC": roc_auc
        })

        st.markdown(f"#### Confusion Matrix - {name}")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax)
        st.pyplot(fig)

    st.subheader("📈 Results Summary")
    results_df = pd.DataFrame(results).sort_values(by="F1 Score", ascending=False)
    st.dataframe(results_df.reset_index(drop=True))


