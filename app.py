import streamlit as st
import pickle
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Wine Quality Prediction",
    page_icon="🍷",
    layout="centered"
)

# Load trained model

from pathlib import Path
import pickle
import streamlit as st
import numpy as np

BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_model_and_scaler():
    with open(BASE_DIR / "wine_model.pkl", "rb") as file:
        model = pickle.load(file)

    with open(BASE_DIR / "scaler.pkl", "rb") as file:
        scaler = pickle.load(file)

    return model, scaler

model, scaler = load_model_and_scaler()

# Title
st.title("🍷 Wine Quality Prediction App")
st.write("Enter the chemical properties of wine to predict its quality")

# Input fields
fixed_acidity = st.number_input("Fixed Acidity", 0.0, 20.0, 7.4)
volatile_acidity = st.number_input("Volatile Acidity", 0.0, 2.0, 0.7)
citric_acid = st.number_input("Citric Acid", 0.0, 1.0, 0.0)
residual_sugar = st.number_input("Residual Sugar", 0.0, 20.0, 1.9)
chlorides = st.number_input("Chlorides", 0.0, 1.0, 0.076)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", 0.0, 100.0, 11.0)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", 0.0, 300.0, 34.0)
density = st.number_input("Density", 0.9900, 1.0050, 0.9978, format="%.4f")
pH = st.number_input("pH", 2.5, 4.5, 3.51)
sulphates = st.number_input("Sulphates", 0.0, 2.0, 0.56)
alcohol = st.number_input("Alcohol (%)", 8.0, 15.0, 9.4)

# Predict button
if st.button("🔍 Predict Wine Quality"):
    input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid,
                             residual_sugar, chlorides, free_sulfur_dioxide,
                             total_sulfur_dioxide, density, pH,
                             sulphates, alcohol]])

    prediction = model.predict(input_data)[0]

    st.success(f"🍾 Predicted Wine Quality: **{prediction} / 10**")

    if prediction >= 7:
        st.balloons()
        st.write("✅ Excellent quality wine!")
    elif prediction >= 5:
        st.write("🙂 Average quality wine")
    else:
        st.write("⚠️ Poor quality wine")
