import streamlit as st
import pandas as pd
import numpy as np
import joblib
from io import BytesIO
import base64
import os

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

add_bg_from_url()


def main():

    
    model = joblib.load("iris_model.pkl")
    #inference
    st.title("Title") 

    st.subheader("Inference uploading a dataset")
    file = st.file_uploader("Upload a dataset of iris", type=["csv", "xlsx"])

    if file is not None:
        
        if os.path.splitext(file.name)[1] == ".xlsx":
            df = pd.read_excel(file, engine='openpyxl')
        else:
            df = pd.read_csv(file)



        df = df.round()
        st.dataframe(df)

        st.write('Dataframe Description')
        dfdesc = df.describe(include='all').T.fillna("")
        st.write(dfdesc)

        df_pred = model.predict(df)
        df['class'] = df_pred
        st.write('Updated Dataframe')
        st.dataframe(df)

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
        

    if file is None:
        st.subheader("Inference with manual inputs")
        input1 = st.number_input("input 1", value=0.00)
        input2 = st.number_input("input 2", value=0.00)
        input3 = st.number_input("input 3", value=0.00)
        input4 = st.number_input("input 4", value=0.00)
        final_input = np.array([input1, input2, input3, input4])
        final_input = final_input.reshape(-1,4)
        pred = model.predict(final_input)
        pred_str = pred[0]
        st.write("Prediction: ", pred_str)

if __name__ == '__main__':
    
    main()