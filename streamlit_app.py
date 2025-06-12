import streamlit as st
import pandas as pd
import joblib
import streamlit_lottie as st_lottie
import requests

# ========== Halaman dan Gaya ==========
st.set_page_config(page_title="Obesity Predictor 🌈", page_icon="🧁", layout="wide")

# Custom CSS untuk nuansa pastel lucu
st.markdown("""
    <style>
    body, .main {
        background-color: #FFF5F9;
        font-family: 'Comic Sans MS', cursive;
    }
    h1, h2, h3 {
        color: #F07AA2;
        text-shadow: 1px 1px 2px #ffd6e8;
    }
    .big-title {
        background: linear-gradient(to right, #ffe0f0, #dff6ff);
        padding: 20px;
        border-radius: 20px;
        text-align: center;
        font-size: 35px;
        font-weight: bold;
        color: #6a4c93;
        margin-bottom: 30px;
    }
    .css-1v3fvcr {
        background-color: #fff0f5 !important;
    }
    .stButton > button {
        background-color: #ffd6e8;
        color: #6a4c93;
        font-size: 18px;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 20px;
    }
    .stButton > button:hover {
        background-color: #fbc4de;
        color: #4a2b75;
    }
    </style>
""", unsafe_allow_html=True)

# ========== Load Model ==========
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# ========== Load Lottie Animation ==========
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Gunakan URL dari lottiefiles yang stabil
lottie_health = load_lottie_url("https://assets6.lottiefiles.com/packages/lf20_m9dytwez.json")

# ========== Sidebar ==========
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/854/854894.png", width=80)
    st.title("💖 Obesity Predictor App")
    st.markdown("Selamat datang! 🌸 Aplikasi ini akan membantumu memprediksi tingkat obesitas berdasarkan gaya hidupmu.")
    st.markdown("✨ **Isi data**, klik prediksi, dan temukan saran sehatmu!")

# ========== Header ==========
st.markdown('<div class="big-title">🧁 Prediksi Obesitas dengan Gaya Lucu dan Ceria 🍬</div>', unsafe_allow_html=True)
if lottie_health:
    st_lottie.st_lottie(lottie_health, height=200, key="health")
else:
    st.warning("⚠️ Gagal memuat animasi, lanjutkan tanpa animasi.")

# ========== Form Input ==========
with st.form("form_prediksi"):
    st.header("🌷 Masukkan Informasimu Yuk!")
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("🎂 Usia", 10, 100, 25)
        gender = st.radio("🛋️ Jenis Kelamin", ["Male", "Female"])
        weight = st.number_input("⚖️ Berat Badan (kg)", 20, 200, 60)
        favc = st.selectbox("🍟 Suka Makanan Kalori Tinggi?", ["yes", "no"])
        fcvc = st.slider("🥦 Konsumsi Sayur (1–3)", 1.0, 3.0, 2.0)
        scc = st.selectbox("🧮 Pantau Kalori Harian?", ["yes", "no"])

    with col2:
        calc = st.selectbox("🍷 Konsumsi Alkohol", ["no", "Sometimes", "Frequently", "Always"])
        ch2o = st.slider("💧 Konsumsi Air (liter/hari)", 0.0, 3.0, 2.0)
        fhwo = st.selectbox("👨‍👩‍👧 Riwayat Keluarga Obesitas?", ["yes", "no"])
        faf = st.slider("🏃‍♀️ Aktivitas Mingguan (jam)", 0.0, 3.0, 1.0)
        caec = st.selectbox("🧁 Sering Ngemil?", ["no", "Sometimes", "Frequently", "Always"])

    submitted = st.form_submit_button("🌟 Prediksi Sekarang!")

# ========== Proses Prediksi ==========
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
    X_scaled = scaler.transform(user_input)
    prediction = model.predict(X_scaled)
    result = label_encoder.inverse_transform(prediction)[0]

    # 🎉 Animasi balon saat selesai
    st.balloons()

    st.subheader("🎀 Hasil Prediksi Kamu:")
    st.success(f"✨ Tingkat obesitas kamu diprediksi sebagai: **{result.replace('_', ' ')}**")

    st.markdown("📬 **Saran Lucu Sehat Untukmu:**")
    if "Obesity" in result:
        st.error("🍔 Kamu termasuk kategori **Obesitas**. Yuk kurangi junk food dan mulai aktivitas ringan seperti jalan kaki!")
    elif "Overweight" in result:
        st.warning("🍩 Kamu sedikit kelebihan berat badan. Kurangi ngemil tengah malam yaa 🌙")
    elif "Normal_Weight" in result:
        st.success("🍎 Berat badan kamu ideal! Pertahankan dengan rutin olahraga dan makan sehat 💪")
    else:
        st.info("🤔 Hasil unik! Mungkin kamu spesial banget. Konsultasikan lebih lanjut ke ahli ya 💖")

    with st.expander("📋 Lihat Data Masukanmu"):
        st.dataframe(user_input)
