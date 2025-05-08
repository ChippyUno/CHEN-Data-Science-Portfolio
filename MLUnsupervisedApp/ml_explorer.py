import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# App configuration
st.set_page_config(page_title="ML Explorer", layout="wide")
st.title("🔍 Unsupervised Learning Explorer")

# Sidebar controls
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
    model_type = st.selectbox("Choose analysis type:", ["K-Means Clustering", "PCA"])
    
    # Model parameters
    if model_type == "K-Means Clustering":
        max_clusters = st.slider("Max clusters for elbow method:", 2, 15, 10)
        selected_clusters = st.slider("Number of clusters:", 1, 10, 3)
    else:
        n_components = st.slider("Number of components:", 2, 10, 2)
    
    st.header("🧪 Experiment Setup")
    scale_data = st.checkbox("Standardize data", value=True)
    random_state = st.number_input("Random Seed", 0, 1000, 42)

# Load sample data
def load_sample_data(name):
    data_funcs = {
        "Iris": datasets.load_iris,
        "Breast Cancer": datasets.load_breast_cancer,
        "Wine Quality": datasets.load_wine
    }
    data = data_funcs[name]()
    return pd.DataFrame(data.data, columns=data.feature_names), data.feature_names

# Load and display data
X, feature_names = None, None
if data_source == "Upload Your Own" and uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.header("🔍 Data Overview")
    st.write("First 5 rows of uploaded data:")
    st.dataframe(df.head())
    
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    selected_features = st.multiselect("Select features:", numeric_cols, default=numeric_cols)
    X = df[selected_features]
    feature_names = selected_features

elif data_source == "Sample Dataset":
    X, feature_names = load_sample_data(dataset_name)
    st.header("🔍 Data Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Feature Summary**")
        st.write(X.describe())
    with col2:
        st.write("**Data Distribution**")
        st.line_chart(X.sample(n=100, random_state=42))

# Analysis execution
if st.sidebar.button("🚀 Run Analysis"):
    try:
        if X is None or len(X) == 0:
            st.warning("⚠️ Please complete data configuration first")
            st.stop()

        # Preprocess data
        if scale_data:
            X_scaled = StandardScaler().fit_transform(X)
        else:
            X_scaled = X.values

        if model_type == "K-Means Clustering":
            st.header("📈 Clustering Results")
            
            # Elbow method
            st.subheader("Elbow Method Analysis")
            wcss = []
            for i in range(1, max_clusters+1):
                kmeans = KMeans(n_clusters=i, init='k-means++', random_state=random_state)
                kmeans.fit(X_scaled)
                wcss.append(kmeans.inertia_)
            
            fig, ax = plt.subplots()
            sns.lineplot(x=range(1, max_clusters+1), y=wcss, marker='o', ax=ax)
            ax.set_xlabel("Number of Clusters")
            ax.set_ylabel("Within-Cluster Sum of Squares (WCSS)")
            st.pyplot(fig)

            # Clustering results
            st.subheader("Cluster Visualization")
            kmeans = KMeans(n_clusters=selected_clusters, random_state=random_state)
            labels = kmeans.fit_predict(X_scaled)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Silhouette Score", 
                         f"{silhouette_score(X_scaled, labels):.3f}")
            
            # PCA projection
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            
            fig, ax = plt.subplots()
            sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, 
                           palette="viridis", s=50, ax=ax)
            ax.set_xlabel("Principal Component 1")
            ax.set_ylabel("Principal Component 2")
            ax.set_title("Cluster Projection")
            st.pyplot(fig)

        else:  # PCA Analysis
            st.header("📈 PCA Results")
            pca = PCA(n_components=n_components)
            X_pca = pca.fit_transform(X_scaled)
            
            # Explained variance
            st.subheader("Explained Variance")
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots()
                sns.barplot(x=np.arange(1, n_components+1), 
                            y=pca.explained_variance_ratio_, 
                            palette="Blues", ax=ax)
                ax.set_xlabel("Principal Components")
                ax.set_ylabel("Variance Ratio")
                st.pyplot(fig)
            
            with col2:
                fig, ax = plt.subplots()
                sns.lineplot(x=np.arange(1, n_components+1), 
                            y=np.cumsum(pca.explained_variance_ratio_), 
                            marker='o', ax=ax)
                ax.set_xlabel("Number of Components")
                ax.set_ylabel("Cumulative Explained Variance")
                st.pyplot(fig)

            # Component visualization
            if n_components >= 2:
                st.subheader("Component Projection")
                fig, ax = plt.subplots()
                sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], 
                               palette="viridis", s=50, ax=ax)
                ax.set_xlabel("Principal Component 1")
                ax.set_ylabel("Principal Component 2")
                st.pyplot(fig)

            # Component loadings
            st.subheader("Component Loadings")
            loadings = pd.DataFrame(
                pca.components_.T,
                columns=[f"PC{i+1}" for i in range(n_components)],
                index=feature_names
            )
            st.dataframe(loadings.style.background_gradient(cmap="coolwarm", axis=0))

    except Exception as e:
        st.error(f"Error in analysis: {str(e)}")