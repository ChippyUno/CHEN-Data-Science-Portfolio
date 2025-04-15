# ML 
welcome to the **ML** - a user freindly Streamlit app that lets you epxlore and experiment with basic machine learning models on sample datasets or your own uploaded CSV files.

## Goal
The goal of this project is to provide an interactive, educational envrionment where users can:
- Use built in datasets or upload their own
- Explore the dataset and visualize basic statistics and and class distributions 
- Choose between Logistic Regression, Decision Tree, and K-Nearest Neighbors classifiers
- Tune hyperparameters using sliders and dropdowns
- Train a model and evlauate its performance usign metrics like accuracy, preciison, recall, F1 score, confusion matrix, and ROC curve
- Visualize feature importances

## Instructions
First, clone the repository
git clone https://github.com/ChippyUno/CHEN-Data-Science-Portfolio.git

cd chen-data-science-portfolio/MLStreamlitAPP

Then install the required libraries
pip install -r requirements.txt

The libraries and versions are 
matplotlib	3.10.1
numpy	2.2.4
pandas	2.2.3
scikit-learn	1.6.1
seaborn	0.13.2
streamlit	1.37.1

## Features
Logistic Regression is a linear model used for classfication problems. Outputs probabilities using the logistic function. Best for problems with linear decision boundaries.

Tuning Hyperparameters
- C: inverse of regularization strength. Smaller values mean stronger regularization
- max_iter: maximum number of iterations for onvergence
- random_state: ensures reproducible results

Decision Tree Classifier is a non-linear model that splits the data into branches based on feature values. It is easy to interpret and visualize, it can overfit if not properly tuned. 

Tuning Hyperparameters
- max_depth: Limits how deep the tree can grow. Prevents overfitting
- criterion: measrues the qulaity of a split
- random_state: ensures consistency in training results 


## Resources
Matplotlib: https://matplotlib.org/stable/users/explain/quick_start.html 

Streamlit: https://docs.streamlit.io/develop/tutorials

Seaborn: https://seaborn.pydata.org/tutorial.html

Scikit-learn: https://scikit-learn.org/stable/user_guide.html


## Image for representation 
<img width="1138" alt="Screenshot 2025-04-14 at 22 51 37" src="https://github.com/user-attachments/assets/cd3a34eb-cb88-496a-b133-48853c2eb0aa" />


<img width="1147" alt="Screenshot 2025-04-14 at 21 41 55" src="https://github.com/user-attachments/assets/70277ae7-fb51-4130-8def-21b7fa8f3a6e" />

