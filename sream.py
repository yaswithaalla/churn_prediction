import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_auc_score, roc_curve, classification_report)

# -------------------------------
# Streamlit App
# -------------------------------
st.title("📊 Customer Churn Predictor")

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv(r"D:\aicte mini project\WA_Fn-UseC_-Telco-Customer-Churn.csv")  # default dataset

# Drop irrelevant ID columns
if "customerID" in df.columns:
    df = df.drop(columns=["customerID"])

# Sidebar controls
st.sidebar.header("⚙ Settings")
target_col = st.sidebar.selectbox("Target column", options=df.columns, index=len(df.columns) - 1)
test_size = st.sidebar.slider("Test Size (%)", 10, 50, 20) / 100
model_choice = st.sidebar.selectbox("Model", ["Logistic Regression", "Random Forest"])

# Split features and target
y = df[target_col]
if y.dtype == "object":
    y = pd.factorize(y)[0]  # Convert categorical target into numeric codes

X = df.drop(columns=[target_col])

# Train-test split
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
except ValueError:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

# Preprocessing
numeric_selector = selector(dtype_include=['int64', 'float64'])
categorical_selector = selector(dtype_include=['object', 'category'])
preprocess = ColumnTransformer([
    ("num", StandardScaler(), numeric_selector),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_selector)
])

# Model selection
if model_choice == "Logistic Regression":
    model = LogisticRegression(max_iter=2000, class_weight="balanced")
else:
    model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)

# Build pipeline
clf = Pipeline(steps=[("prep", preprocess), ("model", model)])
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None

# -------------------------------
# Metrics
# -------------------------------
st.subheader("📈 Model Performance Metrics")
st.write("*Accuracy:*", round(accuracy_score(y_test, y_pred), 3))
st.write("*Precision:*", round(precision_score(y_test, y_pred, average="weighted"), 3))
st.write("*Recall:*", round(recall_score(y_test, y_pred, average="weighted"), 3))
st.write("*F1 Score:*", round(f1_score(y_test, y_pred, average="weighted"), 3))
st.text("Classification Report:\n" + classification_report(y_test, y_pred))

# -------------------------------
# Confusion Matrix
# -------------------------------
st.subheader("🔹 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# -------------------------------
# ROC Curve
# -------------------------------
if y_proba is not None and len(np.unique(y)) == 2:
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    st.subheader("🔹 ROC Curve")
    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax2.plot([0, 1], [0, 1], '--')
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.legend()
    st.pyplot(fig2)

# -------------------------------
# Customer-Level Predictions (Test Data)
# -------------------------------
st.subheader("📌 Sample Predictions")

results_df = X_test.copy()
results_df["Actual"] = y_test
results_df["Predicted"] = y_pred

if len(np.unique(y)) == 2:
    results_df["Predicted_Label"] = results_df["Predicted"].map({0: "Not Churn", 1: "Churn"})
    results_df["Actual_Label"] = results_df["Actual"].map({0: "Not Churn", 1: "Churn"})

st.dataframe(results_df.head(20))  # show first 20 rows

# -------------------------------
# Predict on New Data (CSV Upload)
# -------------------------------
st.subheader("🆕 Predict Churn for New Customers (Batch CSV)")

new_file = st.file_uploader("Upload new customer data (CSV)", type=["csv"], key="newdata")

if new_file is not None:
    new_data = pd.read_csv(new_file)

    if "customerID" in new_data.columns:
        new_data = new_data.drop(columns=["customerID"])

    predictions = clf.predict(new_data)
    probabilities = clf.predict_proba(new_data)[:, 1] if hasattr(clf, "predict_proba") else None

    result_df = new_data.copy()
    result_df["Predicted"] = predictions

    if len(np.unique(y)) == 2:
        result_df["Predicted_Label"] = result_df["Predicted"].map({0: "Not Churn", 1: "Churn"})
        if probabilities is not None:
            result_df["Churn_Probability"] = np.round(probabilities, 3)

    st.write("### 🔍 Prediction Results")
    st.dataframe(result_df)

# -------------------------------
# Manual Input for Single Customer
# -------------------------------
st.subheader("✍ Manual Entry: Predict Churn for One Customer")

with st.form("manual_entry"):
    input_data = {}
    for col in X.columns:
        if str(X[col].dtype) in ["object", "category"]:
            input_data[col] = st.selectbox(f"{col}", options=df[col].dropna().unique())
        else:
            input_data[col] = st.number_input(f"{col}", value=float(df[col].dropna().median()))
    submit_btn = st.form_submit_button("Predict Churn")

if submit_btn:
    single_df = pd.DataFrame([input_data])
    pred_single = clf.predict(single_df)[0]
    proba_single = clf.predict_proba(single_df)[0, 1] if hasattr(clf, "predict_proba") else None

    label = "Churn" if pred_single == 1 else "Not Churn"
    st.write(f"### 🧾 Prediction: *{label}*")
    if proba_single is not None:
        st.write(f"### 🔮 Churn Probability: *{proba_single:.2f}*")