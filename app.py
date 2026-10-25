import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
# Stripping any accidental spaces in the filename
@st.cache_resource
def load_model():
    return joblib.load('model.sav')

model = load_model()

st.title("Αdvertising Sales Prediction App")
st.write("This app predicts **Sales** based on TV, Radio, and Newspaper advertising budgets.")

# Input fields for features
TV = st.number_input("TV Advertising Budget ($)", min_value=0.0, value=150.0, step=1.0)
Radio = st.number_input("Radio Advertising Budget ($)", min_value=0.0, value=25.0, step=1.0)
Newspaper = st.number_input("Newspaper Advertising Budget ($)", min_value=0.0, value=30.0, step=1.0)

# Predict button
if st.button("Predict Sales"):
    # Create a DataFrame with the exact feature names the model was trained on
    input_data = pd.DataFrame([[TV, Radio, Newspaper]], columns=['TV', 'Radio', 'Newspaper'])
    prediction = model.predict(input_data)[0]
    
    st.success(f"Predicted Sales: **{prediction:.2f} M**")
