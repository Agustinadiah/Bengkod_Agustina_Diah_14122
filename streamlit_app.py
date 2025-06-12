import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ================= Setup & Load Assets =================
st.set_page_config(page_title="Prediksi Obesitas", page_icon="🧁", layout="wide")

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# ================= CSS Custom =================
st.markdown("""
    <style>
    body {
        background-color: #fff8f0;
    }
    .main {
        background-color: #fff8f5;
    }
    h1, h2, h3 {
        color: #ff6f91;
    }
    .title-style {
        background: linear-gradient(to right, #ffe0f0, #dfe7fd);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        color: #5f4b8b;
        font-size: 30px;
        font-weight: bold;
    }
    .sidebar .sidebar-content {
        background-color: #fef6fb;
    }
    </style>
""", unsafe_allow_html=True)

# ================= Sidebar =================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/854/854894.png", width=100)
    st.title("🩷 Obesity Check App")
    st.markdown("""
    Aplikasi ini membantu Anda memprediksi tingkat obesitas berdasarkan kebiasaan hidup Anda.

    **💡 Instruksi**:
    - Isi data dengan lengkap 🎯
    - Klik tombol prediksi 🔍
    - Dapatkan hasil dan tipsnya! 🌈
    """)

# ================= Title & Description =================
st.markdown('<div class="title-style">🏃‍♀️ Prediksi Tingkat Obesitas Berdasarkan Gaya Hidup 🍰</div>', unsafe_allow_html=True)
st.markdown("💬 *Masukkan informasi pribadi dan gaya hidup Anda untuk memprediksi kategori obesitas dengan penuh warna!*")

# ================= Input Form =================
with st.form("form_prediksi"):
    st.header("📋 Masukkan Informasi Anda")
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("🎂 Usia", 10, 100, 25)
        gender = st.radio("🚻 Jenis Kelamin", ["Male", "Female"])
        weight = st.number_input("⚖️ Berat Badan (kg)", 20, 200, 70)
        favc = st.selectbox("🍟 Sering Makan Makanan Tinggi Kalori?", ["yes", "no"])
        fcvc = st.slider("🥦 Konsumsi Sayur (1–3)", 1.0, 3.0, 2.0)
        scc = st.selectbox("🧮 Pantau Kalori Harian?", ["yes", "no"])

    with col2:
        calc = st.selectbox("🍷 Konsumsi Alkohol", ["no", "Sometimes", "Frequently", "Always"])
        ch2o = st.slider("💧 Konsumsi Air (liter/hari)", 0.0, 3.0, 2.0)
        fhwo = st.selectbox("👨‍👩‍👧 Riwayat Keluarga Overweight?", ["yes", "no"])
        faf = st.slider("🏋️‍♀️ Aktivitas Fisik Mingguan (jam)", 0.0, 3.0, 1.0)
        caec = st.selectbox("🧁 Sering Ngemil?", ["no", "Sometimes", "Frequently", "Always"])

    submitted = st.form_submit_button("🌟 Prediksi Sekarang")

# ================= Prediction Logic =================
if submitted:
    input_dict = {
        "Age": age,
        "Gender": 1 if gender == "Male" else 0,
        "Weight": weight,
        "CALC": {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}[calc],
        "FAVC": 1 if favc == "yes" else 0,
        "FCVC": fcvc,
        "SCC": 1 if scc == "yes" else 0,
        "CH2O": ch2o,
        "family_history_with_overweight": 1 if fhwo == "yes" else 0,
        "FAF": faf,
        "CAEC": {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}[caec]
    }

    user_input = pd.DataFrame([input_dict])
    user_input = user_input[[ 
        'Age', 'Gender', 'Weight', 'CALC', 'FAVC', 'FCVC', 'SCC', 
        'CH2O', 'family_history_with_overweight', 'FAF', 'CAEC'
    ]]

    X_scaled = scaler.transform(user_input)
    prediction = model.predict(X_scaled)
    result = label_encoder.inverse_transform(prediction)[0]

    # ================= Result Output =================
    st.markdown("----")
    st.subheader("🧸 Hasil Prediksi Anda:")
    st.success(f"🎯 Tingkat obesitas Anda diprediksi sebagai: **{result.replace('_', ' ')}**")

    # ========== Personalized Feedback ==========
    st.markdown("💡 **Saran Gaya Hidup Sehat:**")
    if "Obesity" in result:
        st.warning("🚨 Anda termasuk dalam kategori obesitas. Yuk mulai aktivitas fisik rutin dan perhatikan makananmu! 💪")
    elif "Overweight" in result:
        st.info("📌 Anda dalam kategori kelebihan berat badan. Ayo jaga pola makan dan tambah gerak ya! 🧘‍♀️")
    elif "Normal_Weight" in result:
        st.success("🍀 Berat badan Anda normal! Pertahankan gaya hidup sehat dan tetap aktif ✨")
    else:
        st.info("🌸 Kategori lain terdeteksi. Untuk hasil akurat, silakan konsultasikan ke ahli gizi.")

    with st.expander("📁 Lihat Data Masukan"):
        st.dataframe(user_input)
