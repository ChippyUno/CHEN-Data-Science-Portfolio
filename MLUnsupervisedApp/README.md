# MLUnsupervised App

## Goal
The goal of the Unsupervised Learning Explorer is to provide users with an interactive environment to explore and understand unsupervised machine leanring techniques. This tool is ideal for data sicence learners, analysts, and anyone curious about clustering. The app is designed to:
- Simplify experimentation with unsupervised ML models
- Provide intuitive visualizations to interpret results
- Allow flexible input through sample datasets or user-uploaded files
- Promote hands-on learning by offering real-time feedback based on user-selected hyperparameters


## Features
### Data Input
- Choose from built-in datasets: Iris, Wine, or Breast Cancer
- Upload your own CSV file for custom exploration
- Select specifc numeric features from uploaded data for anlaysis

### Configuration Controls
- Model selection: Choose between K-Means Clustering and PCA
- Hyperparameter Tuning:
  - For K-Means:
    - Select the maximum number of clusters for elbow plot
    - Choose the number of clusters for final modleing
  - For PCA: Select the number of principal components to retain
- Data Preprocessing: Option to standardize data
- Reproducinility: Set a random seed

### Visual Outputs & Metrics
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

### Error Handling
Alerts users if data input in incomplete or analysis settings are invalid 


## Instructions
First, clone the repository git clone: https://github.com/ChippyUno/CHEN-Data-Science-Portfolio/blob/main/MLUnsupervisedApp/ml_explorer.py 

cd chen-data-science-portfolio/MLStreamlitAPP

Then install the required libraries pip install -r requirements.txt

The libraries and versions are matplotlib 3.10.1 numpy 2.2.4 pandas 2.2.3 scikit-learn 1.6.1 seaborn 0.13.2 streamlit 1.37.1


## Resources
Matplotlib: https://matplotlib.org/stable/users/explain/quick_start.html 

Streamlit: https://docs.streamlit.io/develop/tutorials

Seaborn: https://seaborn.pydata.org/tutorial.html

Scikit-learn: https://scikit-learn.org/stable/user_guide.html


## Visuals
<img width="1440" alt="Screenshot 2025-05-08 at 17 13 05" src="https://github.com/user-attachments/assets/9c40f10e-6664-48b8-8d59-55b9af7b41d7" />



  <img width="1413" alt="Screenshot 2025-05-08 at 20 22 03" src="https://github.com/user-attachments/assets/3a25045e-1351-4e6a-85e3-ef633cc6632f" />

