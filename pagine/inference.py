import streamlit as st
import os
import pandas as pd
import joblib
from io import BytesIO

from utils.cache_convert import convert_df

def main():
    absolute_path = os.path.dirname(__file__)
    relative_path = "penguins_pipe.pkl"
    full_path = os.path.join(absolute_path, relative_path)

    model_pipe = joblib.load(full_path)
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

        
        df_pred = model_pipe.predict(df)
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

if __name__=="__main__":
    main()