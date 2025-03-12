import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import os

# Load dataset
file_path = 'Crop_recommendation.csv'
try:
    crop = pd.read_csv(file_path)
except FileNotFoundError:
    st.error("Error: Crop_recommendation.csv not found! Please upload the correct file.")
    st.stop()

crop_dict = {
    'rice': 1, 'maize': 2, 'chickpea': 3, 'kidneybeans': 4, 'pigeonpeas': 5,
    'mothbeans': 6, 'mungbean': 7, 'blackgram': 8, 'lentil': 9, 'pomegranate': 10,
    'banana': 11, 'mango': 12, 'grapes': 13, 'watermelon': 14, 'muskmelon': 15,
    'apple': 16, 'orange': 17, 'papaya': 18, 'coconut': 19, 'cotton': 20,
    'jute': 21, 'coffee': 22
}

crop['crop_num'] = crop['label'].map(crop_dict)
crop.drop('label', axis=1, inplace=True)
X = crop.drop('crop_num', axis=1)
y = crop['crop_num']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize and Standardize
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = GaussianNB()
model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))

# Crop Dictionary Reverse Mapping
crop_reverse_dict = {v: k for k, v in crop_dict.items()}

def recommend_crop(N, P, K, temperature, humidity, ph, rainfall):
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(features)[0]
    return crop_reverse_dict.get(prediction, "No suitable crop found")

# Streamlit UI
st.set_page_config(page_title="Crop Recommendation System", layout="wide", initial_sidebar_state="collapsed")

# Theme Selection
st.sidebar.title("Theme Selection")
theme = st.sidebar.radio("Select Theme:", ["Light", "Dark"], horizontal=True)

if theme == "Dark":
    st.markdown(
        """
        <style>
            body, .stApp {
                background-color: black;
                color: white;
            }
        </style>
        """, unsafe_allow_html=True)
else:
    st.markdown(
        """
        <style>
            body, .stApp {
                background-color: white;
                color: black;
            }
        </style>
        """, unsafe_allow_html=True)

st.title("🌾 Crop Recommendation System")
st.subheader("Find the best crop to grow based on soil and climate conditions")

# User Inputs - 4 options in a row
col1, col2, col3, col4 = st.columns(4)
with col1:
    N = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=50, step=1)
    P = st.number_input("Phosphorus (P)", min_value=0, max_value=200, value=50, step=1)
with col2:
    K = st.number_input("Potassium (K)", min_value=0, max_value=200, value=50, step=1)
    temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0, step=0.1)
with col3:
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    ph = st.number_input("pH Level", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
with col4:
    rainfall = st.number_input("Rainfall (mm)", min_value=0, max_value=300, value=100, step=1)

if st.button("🌱 Recommend Crop", key="recommend_button"):
    recommended_crop = recommend_crop(N, P, K, temperature, humidity, ph, rainfall)
    st.success(f"🌾 Recommended Crop: {recommended_crop}")
    
    # Display crop image
    image_path = f"images/{recommended_crop.lower()}.jpg"  # Ensure images are named correctly and stored in an 'images' folder
    if os.path.exists(image_path):
        st.image(image_path, caption=f"{recommended_crop}", use_container_width=True)
    else:
        st.warning("Image not found for the recommended crop.")

st.markdown(f"### 📊 Model Accuracy: **{accuracy:.2f}**")
