#Aplikasi ini dibangun dengan Streamlit dan model ML yang sudah dilatih. User hanya mengisi data kebiasaan hidupnya, lalu sistem akan memproses input tersebut dan memprediksi tingkat obesitas. Aplikasi ini memberikan prediksi yang dibalut dengan tampilan pastel lucu agar user-friendly dan nyaman digunakan.
import streamlit as st #untuk membangun antarmuka aplikasi.
import pandas as pd #untuk manipulasi data input pengguna dalam bentuk tabel.
import numpy as np #digunakan untuk perhitungan numerik (meskipun tidak dipakai banyak di sini).
import joblib #untuk memuat model Machine Learning dan scaler yang sudah dilatih sebelumnya.

# ================= Konfigurasi Halaman dan Tema =================
st.set_page_config(page_title="Prediksi Obesitas", page_icon="🧁", layout="wide")
#Mengatur tampilan halaman Streamlit:
#page_title = judul tab browser.
#page_icon = ikon kecil di tab browser (favicon).
#layout="wide" = membuat tampilan lebih lebar.

# ================= Load Model & Tools ================
model = joblib.load("model.pkl") #Memuat file model yang sudah disimpan dari proses pelatihan sebelumnya.
scaler = joblib.load("scaler.pkl") #normalisasi data yang sudah disimpan dari proses pelatihan sebelumnya.
label_encoder = joblib.load("label_encoder.pkl") #konversi label yang sudah disimpan dari proses pelatihan sebelumnya.

# ================= Tambahan CSS Custom untuk Estetika =================
#Bagian ini menggunakan HTML + CSS untuk mempercantik tampilan:
#Mengatur warna latar belakang pastel.
#Membuat judul lebih lucu dan feminin.
#Mempercantik sidebar dan teks warna-warni.
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

# ================= Sidebar Aplikasi =================
#Sidebar berisi:
#Logo ikon lucu.
#Judul sidebar.
#Instruksi penggunaan aplikasi: mengisi data dan menekan tombol prediksi.
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

# ================= Judul dan Deskripsi Halaman Utama =================
#Menampilkan judul utama aplikasi di tengah halaman dan memberi sedikit deskripsi untuk membimbing pengguna.
st.markdown('<div class="title-style">🏃‍♀️ Prediksi Tingkat Obesitas Berdasarkan Gaya Hidup 🍰</div>', unsafe_allow_html=True)
st.markdown("💬 *Masukkan informasi pribadi dan gaya hidup Anda untuk memprediksi kategori obesitas dengan penuh warna!*")

# ================= Form Input Pengguna =================
#Bagian ini menampilkan formulir tempat pengguna mengisi:
#Usia, jenis kelamin, berat badan
#Kebiasaan makan, minum, aktivitas fisik
#Riwayat keluarga, dan kebiasaan ngemil
with st.form("form_prediksi"):
    st.header("📋 Masukkan Informasi Anda")
    col1, col2 = st.columns(2) #Input ini dibagi menjadi dua kolom (col1 dan col2) agar tampilan lebih rapi.

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

    submitted = st.form_submit_button("🌟 Prediksi Sekarang") #Jika tombol ditekan, semua input akan dikumpulkan dan diproses untuk predik

# ================= Konversi Data Input ke Bentuk Siap Model =================
if submitted:
    input_dict = {       #input_dict = input dari pengguna dalam bentuk dictionary.
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

    user_input = pd.DataFrame([input_dict])    #pd.DataFrame = diubah jadi tabel agar bisa dibaca model.
    user_input = user_input[[ 
        'Age', 'Gender', 'Weight', 'CALC', 'FAVC', 'FCVC', 'SCC', 
        'CH2O', 'family_history_with_overweight', 'FAF', 'CAEC'
    ]]
    # ================ Prediksi dengan Model Machine Learning ===================
    X_scaled = scaler.transform(user_input)    #scaler.transform = menyesuaikan skala data (normalisasi) agar cocok dengan model.
    prediction = model.predict(X_scaled)    #model.predict = menghasilkan label prediksi.
    result = label_encoder.inverse_transform(prediction)[0] #label_encoder.inverse_transform = mengubah label angka jadi kategori seperti Normal_Weight, Overweight, Obesity, dll.

    # ================= Tampilkan Hasil Prediksi =================
    #Menampilkan hasil prediksi dengan warna hijau dan emoji lucu.
    st.markdown("----")
    st.subheader("🧸 Hasil Prediksi Anda:")
    st.success(f"🎯 Tingkat obesitas Anda diprediksi sebagai: **{result.replace('_', ' ')}**")

    # ========== Berikan Saran Sesuai Kategori ==========
    #Memberi saran yang personal dan edukatif berdasarkan kategori hasil prediksi.
    st.markdown("💡 **Saran Gaya Hidup Sehat:**")
    if "Obesity" in result:
        st.warning("🚨 Anda termasuk dalam kategori obesitas. Yuk mulai aktivitas fisik rutin dan perhatikan makananmu! 💪")
    elif "Overweight" in result:
        st.info("📌 Anda dalam kategori kelebihan berat badan. Ayo jaga pola makan dan tambah gerak ya! 🧘‍♀️")
    elif "Normal_Weight" in result:
        st.success("🍀 Berat badan Anda normal! Pertahankan gaya hidup sehat dan tetap aktif ✨")
    else:
        st.info("🌸 Kategori lain terdeteksi. Untuk hasil akurat, silakan konsultasikan ke ahli gizi.")

    # =================== Tampilkan Data Input Kembali ==============
    with st.expander("📁 Lihat Data Masukan"): #Menampilkan kembali data yang diinput agar pengguna bisa mengecek.Menggunakan expander agar tampilannya tetap rapi.
        st.dataframe(user_input)
