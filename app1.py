import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# ------------------- PAGE CONFIG -------------------
st.set_page_config(page_title="🩺 Breast Cancer Survival Predictor", layout="centered")

# ------------------- CUSTOM CSS -------------------
st.markdown("""
    <style>
    body { background-color: #f8fafc; }
    .main {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }
    h1 {
        color: #7c3aed;
        text-align: center;
    }
    .stButton>button {
        background-color: #7c3aed;
        color: white;
        border-radius: 8px;
        height: 3em;
        width: 100%;
        font-weight: 600;
        border: none;
    }
    .stButton>button:hover {
        background-color: #5b21b6;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------- LOAD MODEL -------------------
MODEL_PATH = r"C:\Users\arumu\Downloads\CAPSTONE PROJECT BREAST CANCER RISK ANALYSIS\breast_cancer.sav"
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

# ------------------- LABEL ENCODERS -------------------
categorical_columns = {
    'Type of Breast Surgery': ['Mastectomy', 'Lumpectomy'],
    'Cellularity': ['High', 'Moderate', 'Low'],
    'Chemotherapy': [0, 1],  # Already numeric
    'ER Status': ['Positive', 'Negative'],
    'PR Status': ['Positive', 'Negative'],
    'HER2 Status': ['Positive', 'Negative', 'Indeterminate'],
    'Pam50 + Claudin-low subtype': ['LumA', 'LumB', 'Basal', 'Her2', 'Normal'],
    'Relapse Free Status': ['Yes', 'No'],
    'Tumor Stage': ['I', 'II', 'III', 'IV']  # Added this line
}

encoders = {}
for col, classes in categorical_columns.items():
    le = LabelEncoder()
    le.classes_ = np.array(classes)
    encoders[col] = le

# ------------------- PAGE TITLE -------------------
st.title("🩺 Breast Cancer Patient Survival Prediction App")
st.markdown("### Predict whether a patient is **Alive or Dead** based on their medical details.")
st.divider()

# ------------------- USER INPUT -------------------
st.subheader("🔹 Enter Patient Details")
data = {}
col1, col2 = st.columns(2)

with col1:
    data['Age at Diagnosis'] = st.number_input("Age at Diagnosis", 20, 120, 45)
    data['Type of Breast Surgery'] = st.selectbox("Type of Breast Surgery", ["Mastectomy", "Lumpectomy"])
    data['Cellularity'] = st.selectbox("Cellularity", ["High", "Moderate", "Low"])
    data['Chemotherapy'] = st.selectbox("Chemotherapy", [0, 1])
    data['ER Status'] = st.selectbox("ER Status", ["Positive", "Negative"])
    data['PR Status'] = st.selectbox("PR Status", ["Positive", "Negative"])
    data['HER2 Status'] = st.selectbox("HER2 Status", ["Positive", "Negative", "Indeterminate"])
    data['Pam50 + Claudin-low subtype'] = st.selectbox("Pam50 + Claudin-low subtype", ["LumA", "LumB", "Basal", "Her2", "Normal"])

with col2:
    data['Neoplasm Histologic Grade'] = st.selectbox("Neoplasm Histologic Grade", [1, 2, 3])
    data['Tumor Size'] = st.number_input("Tumor Size (mm)", 0.0, 200.0, 30.0)
    data['Tumor Stage'] = st.selectbox("Tumor Stage", ["I", "II", "III", "IV"])  # Will now be encoded
    data['Lymph nodes examined positive'] = st.number_input("Lymph Nodes Positive", 0, 50, 3)
    data['Mutation Count'] = st.number_input("Mutation Count", 0, 5000, 100)
    data['Nottingham prognostic index'] = st.number_input("Nottingham Prognostic Index", 1.0, 10.0, 4.5)
    data['Relapse Free Status'] = st.selectbox("Relapse Free Status", ["Yes", "No"])

# ------------------- ENCODE INPUT -------------------
for col in categorical_columns:
    if col in data:
        data[col] = encoders[col].transform([data[col]])[0]

# ------------------- PREDICTION -------------------
if st.button("🔮 Predict Survival Status"):
    df = pd.DataFrame([data])
    try:
        prediction = model.predict(df)
        pred_label = "🟢 Alive" if prediction[0] == 0 else "🔴 Dead"
        st.success(f"**Prediction:** {pred_label}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info("Ensure your model supports categorical or encoded input as used in training.")

# ------------------- HORIZONTAL LINE -------------------
st.markdown("<hr>", unsafe_allow_html=True)
