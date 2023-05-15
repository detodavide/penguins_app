import streamlit as st
import pandas as pd

from pagine.eda import main as eda1
from pagine.inference import main as inf1
from pagine.m_input import main as input1
from utils.bg_image import add_bg_from_url

def main():
    st.set_option('deprecation.showPyplotGlobalUse', False)

    add_bg_from_url()

    st.subheader("The app is used to do predictive analysis or exploration on data on the famous penguin dataset")

    options = ['EDA', 'INFERENCE', 'MANUAL_INFERENCE']
    selected_option = st.selectbox('Select an option: ', options)
    

    #EDA
    if selected_option == "EDA":
        eda1()
 

    if selected_option == "INFERENCE":
        inf1()
        

    if selected_option == "MANUAL_INFERENCE":
        input1()

if __name__ == '__main__':
    
    main()