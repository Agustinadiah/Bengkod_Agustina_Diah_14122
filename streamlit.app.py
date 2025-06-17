import streamlit as st
import joblib
import numpy as np

# Load model dan scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Judul aplikasi
st.title("Prediksi Tingkat Obesitas")
st.write("Masukkan data pribadi dan gaya hidup Anda untuk memprediksi tingkat obesitas.")

with st.form("form_obesitas"):
    st.markdown("### Data Pribadi dan Gaya Hidup")

    # === Urutan input sesuai dataset: Age,Gender,Height,Weight,CALC,FAVC,FCVC,NCP,SCC,SMOKE,CH2O,family_history_with_overweight,FAF,TUE,CAEC,MTRANS ===

    age = st.slider("Usia", 10, 100, 25)
    gender = st.selectbox("Jenis Kelamin", ["Female", "Male"])

    height = st.number_input("Tinggi Badan (meter)", min_value=1.00, max_value=2.50, value=1.65)
    weight = st.number_input("Berat Badan (kg)", min_value=20.0, max_value=200.0, value=65.0)

    calc = st.selectbox("Konsumsi Alkohol", ["no", "Sometimes", "Frequently", "Always"])
    favc = st.selectbox("Sering konsumsi makanan tinggi kalori?", ["yes", "no"])
    fcvc = st.slider("Frekuensi makan sayur (0.0 - 3.0)", 0.0, 3.0, 2.0)
    ncp = st.slider("Jumlah makan utama per hari", 1, 5, 3)
    scc = st.selectbox("Apakah Anda memantau konsumsi kalori?", ["yes", "no"])
    smoke = st.selectbox("Apakah Anda merokok?", ["yes", "no"])
    ch2o = st.slider("Konsumsi air per hari (liter)", 0.0, 3.0, 2.0)
    family_history = st.selectbox("Riwayat keluarga obesitas", ["yes", "no"])
    faf = st.slider("Aktivitas fisik mingguan (0.0 - 3.0)", 0.0, 3.0, 1.0)
    tue = st.slider("Penggunaan perangkat teknologi (jam/hari)", 0.0, 4.0, 2.0)
    caec = st.selectbox("Ngemil / fast food", ["no", "Sometimes", "Frequently", "Always"])
    mtrans = st.selectbox("Transportasi utama", ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"])

    submit = st.form_submit_button("Prediksi")

if submit:
    # === Encoding manual fitur kategorikal ===
    gender = 1 if gender == "Male" else 0
    family_history = 1 if family_history == "yes" else 0
    favc = 1 if favc == "yes" else 0
    scc = 1 if scc == "yes" else 0
    smoke = 1 if smoke == "yes" else 0

    calc_map = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
    caec_map = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
    mtrans_map = {
        "Public_Transportation": 0,
        "Walking": 1,
        "Automobile": 2,
        "Motorbike": 3,
        "Bike": 4
    }

    calc = calc_map[calc]
    caec = caec_map[caec]
    mtrans = mtrans_map[mtrans]

    # Susun urutan sesuai dataset
    data = np.array([[age, gender, height, weight,
                      calc, favc, fcvc, ncp,
                      scc, smoke, ch2o, family_history,
                      faf, tue, caec, mtrans]])

    # Skala fitur
    scaled = scaler.transform(data)

    # Prediksi
    pred = model.predict(scaled)[0]

    label = {
        0: "Insufficient Weight",
        1: "Normal Weight",
        2: "Overweight Level I",
        3: "Overweight Level II",
        4: "Obesity Type I",
        5: "Obesity Type II"
    }

    st.markdown("---")
    st.success(f"Hasil Prediksi: **{label[pred]}**")
