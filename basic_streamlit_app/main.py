import streamlit as st
import pandas as pd
st.title("Welcome to my first Streamlit app")
st.write("The Streamlit app helps to trasnform Python scripts into interactive web apps")

file_path = "/Users/jameschen/Downloads/penguins.csv"
df=pd.read_csv(file_path)
st.dataframe(df.head())

species = st.selectbox("Select a species", df["species"].unique())
filtered_df = df[df["species"] == species]
st.write(f"Plants in {species}:")

st.dataframe(filtered_df)


