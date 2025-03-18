### Import necessary libraries and loading the dataset using relative path
import pandas as pd
import matplotlib.pyplot as plt
relative_path = "Data/olympics_08_medalists.csv"  
df = pd.read_csv(relative_path)
### Identifying missing value and drop rows where all values are missing
df.dropna(how='all', inplace=True)
print(df.isnull().sum())

### Melting data to transform it into a tidy format. Splitting Cateogry column to extract Gender and Sport. Then drop rows where Gender or Sport is NA. Drop the original cateogry column for the newest cleaned dataset.


df_melted = df.melt(id_vars=['medalist_name'], var_name='Category', value_name='Medal')
df_melted[['Gender', 'Sport']] = df_melted['Category'].str.split('_', n=1, expand=True)
df_melted.drop(columns=['Category'], inplace=True)
df_melted.dropna(subset=['Gender', 'Sport'], inplace=True)
print(df_melted.head())

### Creating Pivot Tables for medal count
medal_count = df_melted.pivot_table(index='Sport', columns='Gender', values='Medal', aggfunc='count', fill_value=0)
print("Pivot Table (medal_count):")
print(medal_count)
print("Sum of medal_count by gender:")
print(medal_count.sum())

### Visualizations: One compares the medals earned in different sports by gender. Another shows the cumulative medals earned by sport

medal_count.plot(kind='bar', figsize=(10, 6), color=['blue', 'pink'])
plt.title("Total Medal Counts by Gender")
plt.xlabel("Gender")
plt.ylabel("Total Medals")
plt.show()

df_melted = df_melted[df_melted['Medal'].notna()]

medal_count_sport = df_melted.groupby('Sport')['Medal'].count().sort_values(ascending=False).head(10)
medal_count_sport.plot(kind='bar', figsize=(10, 6), color='green')
plt.title("Top 10 Sports by Medal Count")
plt.xlabel("Sport")
plt.ylabel("Total Medals")
plt.show()
