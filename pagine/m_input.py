import streamlit as st
import pandas as pd
from utils.load_model import load_model

def main():

    model = load_model()

    st.subheader("in Progress...")
    islands = ['Torgersen', 'Biscoe Island', 'Dream']
    island = st.selectbox('Select island: ', islands)
    input2 = st.number_input("bill_length_mm", value=0.00)
    input3 = st.number_input("bill_depth_mm", value=0.00)
    input4 = st.number_input("flipper_length_mm", value=0.00)
    input5 = st.number_input("body_mass_g", value=0.00)
    sex_options = ['Male', 'Female']
    sex = st.selectbox('Select sex: ', sex_options)

    data = {
        "Island": [island],
        "bill_length_mm": [input2],
        "bill_depth_mm": [input3],
        "flipper_length_mm": [input4],
        "body_mass_g": [input5],
        "sex": [sex]
    }

    df = pd.DataFrame(data)
    X = pd.get_dummies(df).to_numpy()
    st.dataframe(X)
    y = model.predict(X)

    # Display the prediction
    st.write("Prediction: ", y)


if __name__ == "__main__":
    main()