import streamlit as st
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def main():
    st.title("Exploratory Data Analysis (EDA)")
    eda_file = st.file_uploader("Upload a dataset of penguins", type=["csv", "xlsx"])

    if eda_file is not None:
            
            if os.path.splitext(eda_file.name)[1] == ".xlsx":
                df = pd.read_excel(eda_file, engine='openpyxl')
            else:
                df = pd.read_csv(eda_file)

            st.dataframe(df)

            try:
                st.subheader("Pairplot")
                fig = sns.pairplot(df, hue='species')
                st.pyplot(fig)
            except Exception as e:
                print(e)
                st.write("Error: Missing column species or file has different columns")

            try:
                st.subheader("Species Count by Island")
                fig1, ax1 = plt.subplots()
                sns.countplot(x='island', hue='species', data=df, ax=ax1)
                ax1.set(xlabel='Island', ylabel='Count')
                st.pyplot(fig1)
            except Exception as e:
                print(e)
                st.write("Error: Missing column species or file has different columns")
            try:
                st.subheader("Correlation Matrix Heatmap")
                plt.figure(figsize=(8, 6))
                sns.heatmap(df.corr(), annot=True)
                st.pyplot()
            except Exception as e:
                print(e)
                st.write("Error: Unable to generate correlation heatmap.")


if __name__ == "__main__":
    main()