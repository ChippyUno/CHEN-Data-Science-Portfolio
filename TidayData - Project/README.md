# Project Overview
This project focuses on data clenaing, transformation, and visualization using the 2008 Olympics medalist dataset. The goal is to apply tidy data principles to restructure the dataset for better analysis and visualization.


## Tidy Data Principles
1. Each variable is stored in its own column
2. Each observation has its own row
3. Each type of observational unit forms its own table
By following these principles, the dataset is transformed into a format suitable for efficient analysis and visualization.


## Instructions
### Prerequisites
Install pandas, matplotlib, and seaborn

### Running the Notebook
1. Open Jupyter Notebook or any compatible Python environment
2. Load the provided Jupyter Notebook file
3. Ensure the dataset (olympics_08_medalists.csv) is placed in the appropriate directory
4. Run all cells sequentially to clean the data, transform it into a tidy format, and generate visualizations


## Dataset Description 
### Source
The dataset consists of medalists from the 2008 Olympics, including information on athlete names, gender, and the sports they competed in

### Pre-Processing Steps
* Stripped unnecessary whitespace from column names and values.
* Handled missing values by removing rows with null entries.
* Melted the dataset to transform it into a long format, ensuring each variable has its own column.
* Split composite column names into separate categorical variables.
* Generated summary statistics and visualizations for exploratory data analysis.


### Images for Reference
<img width="864" alt="Tidydata Visualization" src="https://github.com/user-attachments/assets/57508dc2-427a-41bf-b7e5-bc48cfdf179d" />


<img width="855" alt="Screenshot 2025-03-17 at 21 01 36" src="https://github.com/user-attachments/assets/56c0e80f-1a35-4578-8ce1-e19c8a314a6d" />


## Link for Further Reading
Tidy Data Paper: https://vita.had.co.nz/papers/tidy-data.pdf 

Pandas Cheat Sheet: https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf 
