import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, roc_curve, auc)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# App configuration
st.set_page_config(page_title="ML Explorer", layout="wide")
st.title("🤖 Interactive Machine Learning Explorer")

# Sidebar
with st.sidebar:
    st.header("📊 Data Configuration")
    data_source = st.radio("Choose data source:", ["Sample Dataset", "Upload Your Own"])

    dataset_name = None
    uploaded_file = None
    if data_source == "Sample Dataset":
        dataset_name = st.selectbox("Select sample dataset:", ["Iris", "Breast Cancer", "Wine Quality"])
    else:
        uploaded_file = st.file_uploader("Upload CSV file:", type=["csv"])

    st.header("⚙️ Model Settings")
    model_type = st.selectbox("Choose model:", ["Logistic Regression", "Decision Tree"])

    # Model parameters
    if model_type == "Logistic Regression":
        C = st.slider("Inverse Regularization (C)", 0.01, 10.0, 1.0)
        max_iter = st.slider("Maximum Iterations", 50, 1000, 100)
    else:
        max_depth = st.slider("Max Depth", 1, 20, 5)
        criterion = st.selectbox("Split Criterion", ["gini", "entropy"])

    st.header("🧪 Experiment Setup")
    test_size = st.slider("Test Set Size (%)", 10, 50, 20) / 100
    random_state = st.number_input("Random Seed", 0, 1000, 42)

# Load sample data
def load_sample_data(name):
    data_funcs = {
        "Iris": datasets.load_iris,
        "Breast Cancer": datasets.load_breast_cancer,
        "Wine Quality": datasets.load_wine
    }
    data = data_funcs[name]()
    return pd.DataFrame(data.data, columns=data.feature_names), pd.Series(data.target), data.feature_names, data.target_names

# Load data
X, y, feature_names, class_names = None, None, None, None
if data_source == "Upload Your Own" and uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.header("🔍 Data Overview")
    st.write("First 5 rows of uploaded data:")
    st.dataframe(df.head())
    
    target_col = st.selectbox("Select target column:", df.columns)
    if target_col:
        X, y = df.drop(columns=[target_col]), df[target_col]
        if y.dtype == 'object':
            y = LabelEncoder().fit_transform(y)
        feature_names = X.columns.tolist()
        class_names = [str(c) for c in np.unique(y)]
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Feature Summary**")
            st.write(X.describe())
        with col2:
            st.write("**Class Distribution**")
            st.bar_chart(pd.Series(y).value_counts())

elif data_source == "Sample Dataset":
    X, y, feature_names, class_names = load_sample_data(dataset_name)
    st.header("🔍 Data Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Feature Summary**")
        st.write(X.describe())
    with col2:
        st.write("**Class Distribution**")
        st.bar_chart(y.value_counts())

# Model Training
if st.sidebar.button("🚀 Train Model"):
    try:
        if X is None or y is None:
            st.warning("⚠️ Please complete data configuration first")
            st.stop()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        if model_type == "Logistic Regression":
            model = LogisticRegression(C=C, max_iter=max_iter, random_state=random_state)
        else:
            model = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, random_state=random_state)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

        # Metrics
        st.header("📈 Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")
        col2.metric("Precision", f"{precision_score(y_test, y_pred, average='weighted'):.2%}")
        col3.metric("Recall", f"{recall_score(y_test, y_pred, average='weighted'):.2%}")
        col4.metric("F1 Score", f"{f1_score(y_test, y_pred, average='weighted'):.2%}")

        # Confusion Matrix
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        # ROC Curve
        if y_proba is not None and len(np.unique(y)) == 2:
            st.subheader("ROC Curve")
            fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f'ROC curve (area = {auc(fpr, tpr):.2f})', color='darkorange')
            ax.plot([0, 1], [0, 1], linestyle='--', color='navy')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.legend(loc="lower right")
            st.pyplot(fig)

        # Feature Importance
        if model_type == "Decision Tree":
            st.subheader("Feature Importance")
            importances = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values("Importance", ascending=False)

            fig, ax = plt.subplots()
            sns.barplot(x='Importance', y='Feature', data=importances, ax=ax)
            ax.set_title("Feature Importance Scores")
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Error in model training: {str(e)}")
