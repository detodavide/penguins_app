import streamlit as st
import pandas as pd
import numpy as np
import joblib
from io import BytesIO
import base64
import os
import matplotlib.pyplot as plt
import seaborn as sns

def download_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False)
    writer.save()
    excel_data = output.getvalue()
    b64 = base64.b64encode(excel_data).decode()
    href = f'<a href="data:application/vnd.ms-excel;base64,{b64}" download="predicted_profit.xlsx">Download Excel</a>'
    return href

@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://w.wallhaven.cc/full/m9/wallhaven-m9omem.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )



def main():
    st.set_option('deprecation.showPyplotGlobalUse', False)

    add_bg_from_url()

    st.subheader("The app is used to do predictive analysis or exploration on data on the famous penguin dataset")

    options = ['EDA', 'INFERENCE', 'MANUAL_INFERENCE']
    selected_option = st.selectbox('Select an option: ', options)
    

    #EDA
    if selected_option == "EDA":
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
           
 

    if selected_option == "INFERENCE":
        model = joblib.load("penguins_logreg.pkl")
        #inference
        st.title("Penguins Classification") 

        st.subheader("Inference uploading a dataset")
        file = st.file_uploader("Upload a dataset of penguins", type=["csv", "xlsx"])

        if file is not None:
            
            if os.path.splitext(file.name)[1] == ".xlsx":
                df = pd.read_excel(file, engine='openpyxl')
            else:
                df = pd.read_csv(file)

            st.dataframe(df)

            st.write('Dataframe Description')
            dfdesc = df.describe(include='all').T.fillna("")
            st.write(dfdesc)

            X = pd.get_dummies(df)
            df_pred = model.predict(X)
            df['species'] = df_pred
            st.write('Updated Dataframe')
            st.dataframe(df)


            # Download buttons
            csv = convert_df(df)
            filename = "filename"

            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name=f'{filename}.csv',
                mime='text/csv',
            )

            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                # Write each dataframe to a different worksheet.
                df.to_excel(writer, sheet_name='Sheet1', index=False)
                # Close the Pandas Excel writer and output the Excel file to the buffer
                writer.save()

                download2 = st.download_button(
                    label="Download data as Excel",
                    data=buffer,
                    file_name=f'{filename}.xlsx',
                    mime='application/vnd.ms-excel'
                )
        

    if selected_option == "MANUAL_INFERENCE":
        st.subheader("in Progress...")
        # input1 = st.number_input("input 1", value=0.00)
        # input2 = st.number_input("input 2", value=0.00)
        # input3 = st.number_input("input 3", value=0.00)
        # input4 = st.number_input("input 4", value=0.00)
        # final_input = np.array([input1, input2, input3, input4])
        # final_input = final_input.reshape(-1,4)
        # pred = model.predict(final_input)
        # pred_str = pred[0]
        # st.write("Prediction: ", pred_str)

if __name__ == '__main__':
    
    main()