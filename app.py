import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import gdown
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Set visualization defaults
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)

# Load dataset from Google Drive
@st.cache_data
def load_data():
    file_id = "1kjRfgWkgo8BP_BGlkAK3FBJ-OsbTK13x"
    download_url = f"https://drive.google.com/uc?id={file_id}"
    output_path = "dataset_mahasiswa_DO_4000_mahasiswa.csv"
    gdown.download(download_url, output_path, quiet=False)
    df = pd.read_csv(output_path)
    return df

# Function to load model from .pkl file
def load_model_from_file(file_path):
    try:
        model = joblib.load(file_path)
        return model
    except Exception as e:
        st.error(f"Error loading model from {file_path}: {e}")
        return None

# Preprocess data for prediction
def preprocess_data_for_prediction(input_data):
    kolom_numerik = ['IPK_Semester_1', 'IPK_Semester_2', 'IPK_Semester_3',
                     'IPK_Semester_4', 'IPK_Semester_5', 'IPK_Semester_6',
                     'Kehadiran_Per_Mata_Kuliah', 'Riwayat_Pengambilan_Ulang',
                     'Aktivitas_Sistem_Pembelajaran_Daring', 'Beban_Kerja_JamPerMinggu']
    kolom_kategorikal = ['Status_Pekerjaan', 'Status_Ekonomi']
    
    input_num = np.array(input_data[:10]).reshape(1, -1)
    input_cat = np.array(input_data[10:]).reshape(1, -1)
    
    scaler = MinMaxScaler()
    scaler.fit(df_cleaned[kolom_numerik])
    input_num_scaled = scaler.transform(input_num)
    
    input_final = np.hstack((input_num_scaled, input_cat))
    return input_final

# Function to predict dropout
def predict_dropout(model, input_data):
    try:
        prediction = model.predict(input_data)
        probabilities = model.predict_proba(input_data)
        return prediction, probabilities
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None

# Load dataset
df_original = load_data()
if df_original is None:
    st.stop()

# Clean data for analysis
df_cleaned = df_original.copy()
df_cleaned = df_cleaned.dropna().drop_duplicates()
df_cleaned = df_cleaned.drop(columns=['NIM'])

# Prepare df_head for preview
df_head = df_original.head(5)

# Title
st.title("🎓 Dashboard Prediksi Risiko Dropout Mahasiswa")

# Sidebar untuk informasi tambahan
st.sidebar.title("📋 Informasi Tambahan")

st.sidebar.subheader("Tentang Dataset")
st.sidebar.write("Dataset ini berisi data 4000 mahasiswa dengan fitur seperti IPK, kehadiran, dan status ekonomi untuk analisis risiko dropout.")

st.sidebar.subheader("ℹ Tentang Kelompok")
st.sidebar.write("Anggota:")
st.sidebar.write("- Reksi Hendra Pratama (G1A022032)")
st.sidebar.write("- Baim Mudrik Aziz (G1A022071)")
st.sidebar.markdown("<p style='text-align: center; color: #A9A9A9;'>© 2025 Dropout Analytics</p>", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📈 Data Mahasiswa", "📊 Hasil Analisis", "🔮 Prediksi Dropout", "📝 Kesimpulan & Pengembangan"])

# Tab 1: Data Mahasiswa
with tab1:
    st.subheader("📊 Dataset Exploration")

    # Dataset Awal
    st.write("### Dataset Awal")
    st.write("Dataset asli yang diunduh dari Google Drive dengan 4000 baris dan 14 kolom:")
    st.dataframe(df_original)

    # Statistik Deskriptif
    st.write("Statistik Deskriptif Dataset Awal (hanya kolom numerik):")
    st.write(df_cleaned.describe())

    # Dataset Preview
    st.write("### 5 Baris Pertama Dataset")
    st.dataframe(df_head)

    # Distribusi Risiko Dropout
    st.write("### Persentase Risiko Dropout Mahasiswa")
    dropout_risk_counts = df_cleaned['Status_Risiko_DO'].value_counts(normalize=True)
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    dropout_risk_counts.plot(kind='bar', ax=ax1)
    ax1.set_title('Persentase Risiko Dropout Mahasiswa')
    ax1.set_xlabel('Status Risiko DO (0 = Aman, 1 = Risiko Tinggi)')
    ax1.set_ylabel('Persentase')
    ax1.set_xticklabels(['Aman', 'Risiko Tinggi'], rotation=0)
    st.pyplot(fig1)

    # Frekuensi dan Persentase Status Pekerjaan
    st.write("### Analisis Status Pekerjaan")
    status_pekerjaan_counts = df_cleaned['Status_Pekerjaan'].value_counts()
    status_pekerjaan_percent = df_cleaned['Status_Pekerjaan'].value_counts(normalize=True) * 100
    st.write(pd.DataFrame({
        'Jumlah': status_pekerjaan_counts,
        'Persentase (%)': status_pekerjaan_percent
    }))

    fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(16, 6))
    sns.countplot(x='Status_Pekerjaan', data=df_cleaned, ax=ax2)
    ax2.set_title('Frekuensi Status Pekerjaan')
    ax2.set_xlabel('Status Pekerjaan')
    ax2.set_ylabel('Jumlah')
    ax2.set_xticklabels(['Tidak Bekerja', 'Bekerja'], rotation=45)

    df_cleaned['Status_Pekerjaan'].value_counts().plot(kind='pie', autopct='%1.2f%%', startangle=90, ax=ax3)
    ax3.set_title('Distribusi Persentase Status Pekerjaan')
    ax3.set_ylabel('')
    st.pyplot(fig2)

    # Frekuensi dan Persentase Status Ekonomi
    st.write("### Analisis Status Ekonomi")
    status_ekonomi_counts = df_cleaned['Status_Ekonomi'].value_counts()
    status_ekonomi_percent = df_cleaned['Status_Ekonomi'].value_counts(normalize=True) * 100
    st.write(pd.DataFrame({
        'Jumlah': status_ekonomi_counts,
        'Persentase (%)': status_ekonomi_percent
    }))

    fig3, (ax4, ax5) = plt.subplots(1, 2, figsize=(16, 6))
    sns.countplot(x='Status_Ekonomi', data=df_cleaned, ax=ax4)
    ax4.set_title('Frekuensi Status Ekonomi')
    ax4.set_xlabel('Status Ekonomi')
    ax4.set_ylabel('Jumlah')
    ax4.set_xticklabels(['Rendah', 'Menengah', 'Tinggi'], rotation=45)

    df_cleaned['Status_Ekonomi'].value_counts().plot(kind='pie', autopct='%1.2f%%', startangle=90, ax=ax5)
    ax5.set_title('Distribusi Persentase Status Ekonomi')
    ax5.set_ylabel('')
    st.pyplot(fig3)

# Tab 2: Hasil Analisis
with tab2:
    st.subheader("📊 Hasil Analisis")

    # Evaluasi Model Random Forest
    st.write("### Evaluasi Model Random Forest")
    st.code("""
    precision    recall  f1-score  support
0       0.98      0.99      0.98      440
1       0.99      0.97      0.98      360

accuracy                           0.98      800
macro avg       0.98      0.98      0.98      800
weighted avg       0.98      0.98      0.98      800

Akurasi: 0.9825
Presisi: 0.9859550561797753
Recall: 0.975
F1-Score: 0.9804469273743017
    """, language="text")

    # Confusion Matrix Random Forest
    st.write("### Confusion Matrix - Random Forest")
    conf_matrix_rf = [[435, 5], [9, 351]]
    fig_rf, ax_rf = plt.subplots()
    sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Blues', ax=ax_rf,
                xticklabels=['Aman', 'Risiko Tinggi'],
                yticklabels=['Aman', 'Risiko Tinggi'])
    ax_rf.set_title('Confusion Matrix - Random Forest')
    ax_rf.set_xlabel('Predicted Label')
    ax_rf.set_ylabel('True Label')
    st.pyplot(fig_rf)

    # Evaluasi Model Logistic Regression
    st.write("### Evaluasi Model Logistic Regression")
    st.code("""
    precision    recall  f1-score  support
0       0.90      0.91      0.90      440
1       0.89      0.88      0.88      360

accuracy                           0.89      800
macro avg       0.89      0.89      0.89      800
weighted avg       0.89      0.89      0.89      800

Akurasi: 0.89375
Presisi: 0.8873239436619719
Recall: 0.875
F1-Score: 0.8811188811188811
    """, language="text")

    # Confusion Matrix Logistic Regression
    st.write("### Confusion Matrix - Logistic Regression")
    conf_matrix_lr = [[400, 40], [45, 315]]
    fig_lr, ax_lr = plt.subplots()
    sns.heatmap(conf_matrix_lr, annot=True, fmt='d', cmap='Blues', ax=ax_lr,
                xticklabels=['Aman', 'Risiko Tinggi'],
                yticklabels=['Aman', 'Risiko Tinggi'])
    ax_lr.set_title('Confusion Matrix - Logistic Regression')
    ax_lr.set_xlabel('Predicted Label')
    ax_lr.set_ylabel('True Label')
    st.pyplot(fig_lr)

# Tab 3: Prediksi Dropout
with tab3:
    st.subheader("🔮 Prediksi Risiko Dropout dengan Random Forest")
    st.write("Masukkan data mahasiswa untuk prediksi risiko dropout menggunakan model Random Forest.")

    model_path = "model_prediksi_DO.pkl"
    selected_model = load_model_from_file(model_path)

    if selected_model is None:
        st.warning("Model Random Forest belum tersedia. Melatih model baru...")
        X = df_cleaned.drop(columns=['Status_Risiko_DO'])
        y = df_cleaned['Status_Risiko_DO']
        X = pd.get_dummies(X, columns=['Status_Pekerjaan', 'Status_Ekonomi'], drop_first=True)
        if X.shape[1] > 12:
            X = X.iloc[:, :12]
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        from sklearn.ensemble import RandomForestClassifier
        selected_model = RandomForestClassifier(n_estimators=100, random_state=42)
        selected_model.fit(X_scaled, y)
        joblib.dump(selected_model, model_path)
        st.success("Model Random Forest berhasil dilatih dan disimpan.")

    col1, col2 = st.columns(2)
    with col1:
        ipk1 = st.number_input("IPK Semester 1", min_value=0.0, max_value=4.0, step=0.1, value=3.0)
        ipk2 = st.number_input("IPK Semester 2", min_value=0.0, max_value=4.0, step=0.1, value=3.0)
        ipk3 = st.number_input("IPK Semester 3", min_value=0.0, max_value=4.0, step=0.1, value=3.0)
        ipk4 = st.number_input("IPK Semester 4", min_value=0.0, max_value=4.0, step=0.1, value=3.0)
        ipk5 = st.number_input("IPK Semester 5", min_value=0.0, max_value=4.0, step=0.1, value=3.0)
        kehadiran = st.number_input("Kehadiran Per Mata Kuliah (%)", min_value=0.0, max_value=100.0, step=1.0, value=80.0)
    with col2:
        ipk6 = st.number_input("IPK Semester 6", min_value=0.0, max_value=4.0, step=0.1, value=3.0)
        riwayat = st.number_input("Riwayat Pengambilan Ulang", min_value=0, step=1, value=0)
        aktivitas = st.number_input("Aktivitas Sistem Pembelajaran Daring (%)", min_value=0.0, max_value=100.0, step=1.0, value=80.0)
        pekerjaan = st.selectbox("Status Pekerjaan", options=[0, 1], format_func=lambda x: 'Tidak Bekerja' if x == 0 else 'Bekerja')
        ekonomi = st.selectbox("Status Ekonomi", options=[0, 1, 2], format_func=lambda x: 'Rendah' if x == 0 else 'Menengah' if x == 1 else 'Tinggi')
        beban = st.number_input("Beban Kerja Jam Per Minggu", min_value=0.0, max_value=168.0, step=1.0, value=0.0, disabled=(pekerjaan == 0))

    input_data = [ipk1, ipk2, ipk3, ipk4, ipk5, ipk6, kehadiran, riwayat, aktivitas, beban, pekerjaan, ekonomi]

    if st.button("Prediksi"):
        processed_input = preprocess_data_for_prediction(input_data)
        prediction, probabilities = predict_dropout(selected_model, processed_input)
        if prediction is not None:
            label_mapping = {0: "Aman", 1: "Risiko Tinggi"}
            result = label_mapping.get(prediction[0], "Tidak Diketahui")
            if result == "Aman":
                st.success(f"**Hasil Prediksi: {result}**")
                st.write("**Penjelasan:** Mahasiswa dinyatakan aman dari risiko dropout. Pihak universitas dapat memantau performa akademik secara berkala untuk menjaga status ini.")
            else:
                st.error(f"**Hasil Prediksi: {result}**")
                st.write("**Penjelasan dan Langkah untuk Pihak Universitas:** Mahasiswa memiliki risiko tinggi untuk dropout. Disarankan untuk: \n- Menjadwalkan konseling akademik dengan dosen pembimbing. \n- Menawarkan program pendampingan untuk meningkatkan kehadiran dan aktivitas daring. \n- Mengevaluasi beban kerja dan dukungan finansial jika perlu.")
            if probabilities is not None:
                st.write(f"Probabilitas: Aman: {probabilities[0][0]:.2f}, Risiko Tinggi: {probabilities[0][1]:.2f}")

# Tab 4: Kesimpulan & Pengembangan
with tab4:
    st.subheader("📝 Kesimpulan & Pengembangan")

    st.write("### Kesimpulan")
    st.write("""
    - Model Random Forest menunjukkan performa yang sangat baik dengan akurasi 98.25% dan F1-Score 0.98, menjadikannya lebih unggul dibandingkan Logistic Regression yang memiliki akurasi 89.38%.
    - Analisis distribusi menunjukkan bahwa sebagian besar mahasiswa memiliki status aman dari risiko dropout, dengan variasi pada status pekerjaan dan ekonomi yang memengaruhi risiko.
    - Dashboard ini efektif untuk memprediksi risiko dropout berdasarkan data akademik dan demografis.
    """)

    st.write("### Potensi Pengembangan")
    st.write("""
    - **Integrasi dengan Sistem Akademik:** Menghubungkan dashboard dengan sistem informasi akademik universitas untuk input data real-time dan notifikasi otomatis.
    - **Visualisasi untuk Pihak Rektorat:** Menambahkan dashboard interaktif dengan filter (misalnya, per fakultas atau semester) untuk analisis strategis oleh pihak rektorat.
    - **Prediksi Lanjutan:** Mengembangkan model untuk memprediksi faktor spesifik yang berkontribusi pada dropout, seperti tekanan psikologis atau dukungan sosial.
    """)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #A9A9A9;'>© 2025 Dropout Analytics</p>", unsafe_allow_html=True)