import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Page configuration
st.set_page_config(page_title="ML Explorer", layout="wide")
st.title("🤖 Interactive Machine Learning Explorer")

# Sidebar controls
with st.sidebar:
    st.header("📊 Data Configuration")
    data_source = st.radio("Choose data source:", ["Sample Dataset", "Upload Your Own"])
    
    if data_source == "Sample Dataset":
        dataset_name = st.selectbox("Select sample dataset:", 
                                  ["Iris", "Breast Cancer", "Wine Quality"])
    else:
        uploaded_file = st.file_uploader("Upload CSV file:", type=["csv"])
    
    st.header("⚙️ Model Settings")
    model_type = st.selectbox("Choose model:", 
                            ["Logistic Regression", "Decision Tree"])
    
    # Model-specific parameters
    if model_type == "Logistic Regression":
        C = st.slider("Inverse Regularization (C)", 0.01, 10.0, 1.0)
        max_iter = st.slider("Maximum Iterations", 50, 1000, 100)
    elif model_type == "Decision Tree":
        max_depth = st.slider("Max Depth", 1, 20, 5)
        criterion = st.selectbox("Split Criterion", ["gini", "entropy"])
    
    st.header("🧪 Experiment Setup")
    test_size = st.slider("Test Set Size (%)", 10, 50, 20) / 100
    random_state = st.number_input("Random Seed", 0, 1000, 42)

# Load sample data function
def load_sample_data():
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    else:  # Wine Quality
        data = datasets.load_wine()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    return X, y, data.feature_names, data.target_names

# uploaded files
if data_source == "Upload Your Own" and uploaded_file is not None:
    st.session_state.df = pd.read_csv(uploaded_file)
    st.header("🔍 Data Overview")
    st.write("First 5 rows of uploaded data:")
    st.dataframe(st.session_state.df.head())
    
    cols = st.session_state.df.columns.tolist()
    target_col = st.selectbox("Select target column:", cols)
    
    if target_col:
        st.session_state.X = st.session_state.df.drop(columns=[target_col])
        st.session_state.y = st.session_state.df[target_col]
        
        # Encode categorical target
        if st.session_state.y.dtype == 'object':
            le = LabelEncoder()
            st.session_state.y = le.fit_transform(st.session_state.y)
        
        # Show data summary
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Feature Summary**")
            st.write(st.session_state.X.describe())
        with col2:
            st.write("**Class Distribution**")
            st.bar_chart(pd.Series(st.session_state.y).value_counts())

# sample dataset
if data_source == "Sample Dataset":
    X_sample, y_sample, feature_names, class_names = load_sample_data()
    st.header("🔍 Data Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Feature Summary**")
        st.write(X_sample.describe())
    with col2:
        st.write("**Class Distribution**")
        st.bar_chart(y_sample.value_counts())

# Model training and evaluation
if st.sidebar.button("🚀 Train Model"):
    try:
        # Get data based on source
        if data_source == "Sample Dataset":
            X, y = X_sample, y_sample
        else:
            if st.session_state.X is None or st.session_state.y is None:
                st.warning("⚠️ Please complete data configuration first")
                st.stop()
            X = st.session_state.X
            y = st.session_state.y
            feature_names = X.columns.tolist()
            class_names = [str(c) for c in np.unique(y)]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)
        
        # Initialize model
        if model_type == "Logistic Regression":
            model = LogisticRegression(C=C, max_iter=max_iter, random_state=random_state)
        elif model_type == "Decision Tree":
            model = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion,
                                          random_state=random_state)
        
        # Train model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Display metrics
        st.header("📈 Performance Metrics")
        metric_cols = st.columns(4)
        metric_cols[0].metric("Accuracy", f"{accuracy:.2%}")
        metric_cols[1].metric("Precision", f"{precision:.2%}")
        metric_cols[2].metric("Recall", f"{recall:.2%}")
        metric_cols[3].metric("F1 Score", f"{f1:.2%}")
        
        # Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names,
                    yticklabels=class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        
        # ROC Curve 
        if y_proba is not None and len(np.unique(y)) == 2:
            st.subheader("ROC Curve")
            fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (area = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.legend(loc="lower right")
            st.pyplot(fig)
        
        # Feature Importance 
        if model_type == "Decision Tree":
            st.subheader("Feature Importance")
            importances = model.feature_importances_
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            fig, ax = plt.subplots()
            sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
            ax.set_title("Feature Importance Scores")
            st.pyplot(fig)
            
    except Exception as e:
        st.error(f"Error in model training: {str(e)}")
