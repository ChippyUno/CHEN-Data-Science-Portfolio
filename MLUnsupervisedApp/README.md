# MLUnsupervised App

## Goal
The goal of the Unsupervised Learning Explorer is to provide users with an interactive environment to explore and understand unsupervised machine leanring techniques. This tool is ideal for data sicence learners, analysts, and anyone curious about clustering. The app is designed to:
- Simplify experimentation with unsupervised ML models
- Provide intuitive visualizations to interpret results
- Allow flexible input through sample datasets or user-uploaded files
- Promote hands-on learning by offering real-time feedback based on user-selected hyperparameters


## Features
# Data Input
- Choose from built-in datasets: Iris, Wine, or Breast Cancer
- Upload your own CSV file for custom exploration
- Select specifc numeric features from uploaded data for anlaysis

# Configuration Controls
- Model selection: Choose between K-Means Clustering and PCA
- Hyperparameter Tuning:
  - For K-Means:
    - Select the maximum number of clusters for elbow plot
    - Choose the number of clusters for final modleing
  - For PCA: Select the number of principal components to retain
- Data Preprocessing: Option to standardize data
- Reproducinility: Set a random seed

# Visual Outputs & Metrics
- Data Overview: Summary stats and sample line chart
- Elbow Plot: Helps determine optimal k for K-Means via WCSS curve
- Silhouette Score: Measures clsutering quality
- PCA Projection:
  - For K-Means: View clusters on PCA-reduced axes
  - For PCA: Explore component projections
- Explained Variance Charts:
  - Bar chart for variance by component
  - Line plot for cumulative variance
- Component Loadings Table: Shows how each original feature contributes to the components

# Error Handling
Alerts users if data input in incomplete or analysis settings are invalid 

   
    - 
