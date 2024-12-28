import streamlit as st
import joblib

import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load your trained model (Replace 'your_model.joblib' with the actual model file)
model = joblib.load('FishWeightPrediction.joblib')

# Label Encoder for 'Species'
species_encoder = LabelEncoder()
species_encoder.fit(['Bream', 'Roach', 'Whitefish', 'Parkki', 'Perch', 'Pike', 'Smelt'])

# Standard Scaler (Assuming you have fitted a scaler before)
scaler = StandardScaler()

# Streamlit Interface
st.title("Fish Weight Prediction")

# Input ranges based on species
species = st.selectbox('Species', species_encoder.classes_)

if species == 'Bream':
    Length1 = st.slider('Length1 (cm)', 20.0, 40.0)
    Length2 = st.slider('Length2 (cm)', 22.0, 45.0)
    Length3 = st.slider('Length3 (cm)', 25.0, 50.0)
    Height = st.slider('Height (cm)', 5.0, 12.0)
    Width = st.slider('Width (cm)', 3.0, 6.0)
elif species == 'Roach':
    Length1 = st.slider('Length1 (cm)', 12.0, 30.0)
    Length2 = st.slider('Length2 (cm)', 14.0, 32.0)
    Length3 = st.slider('Length3 (cm)', 15.0, 34.0)
    Height = st.slider('Height (cm)', 2.0, 8.0)
    Width = st.slider('Width (cm)', 1.5, 5.0)
elif species == 'Whitefish':
    Length1 = st.slider('Length1 (cm)', 15.0, 45.0)
    Length2 = st.slider('Length2 (cm)', 16.0, 47.0)
    Length3 = st.slider('Length3 (cm)', 17.0, 50.0)
    Height = st.slider('Height (cm)', 4.0, 10.0)
    Width = st.slider('Width (cm)', 2.0, 5.0)
elif species == 'Parkki':
    Length1 = st.slider('Length1 (cm)', 10.0, 25.0)
    Length2 = st.slider('Length2 (cm)', 11.0, 28.0)
    Length3 = st.slider('Length3 (cm)', 12.0, 30.0)
    Height = st.slider('Height (cm)', 1.0, 5.0)
    Width = st.slider('Width (cm)', 1.0, 4.0)
elif species == 'Perch':
    Length1 = st.slider('Length1 (cm)', 15.0, 40.0)
    Length2 = st.slider('Length2 (cm)', 17.0, 42.0)
    Length3 = st.slider('Length3 (cm)', 18.0, 45.0)
    Height = st.slider('Height (cm)', 4.0, 10.0)
    Width = st.slider('Width (cm)', 2.0, 5.0)
elif species == 'Pike':
    Length1 = st.slider('Length1 (cm)', 20.0, 50.0)
    Length2 = st.slider('Length2 (cm)', 22.0, 52.0)
    Length3 = st.slider('Length3 (cm)', 25.0, 55.0)
    Height = st.slider('Height (cm)', 5.0, 12.0)
    Width = st.slider('Width (cm)', 3.0, 6.0)
elif species == 'Smelt':
    Length1 = st.slider('Length1 (cm)', 5.0, 15.0)
    Length2 = st.slider('Length2 (cm)', 6.0, 17.0)
    Length3 = st.slider('Length3 (cm)', 7.0, 20.0)
    Height = st.slider('Height (cm)', 1.0, 4.0)
    Width = st.slider('Width (cm)', 0.5, 2.0)

Species_encoded = species_encoder.transform([species])[0]

if st.button("Predict"):
    # Prepare the input data
    input_data = [[Length1, Length2, Length3, Height, Width, Species_encoded]]


    # Make prediction
    prediction = model.predict(input_data)

    # Display the prediction
    st.write(f"Predicted Fish Weight: {prediction[0]:.2f} grams")