# Sistem Prediksi Mahasiswa Berisiko Tinggi Drop Out - Kelompok 8
## Anggota Kelompok 8:

1. Reksi Hendra Pratama (G1A022032)  
2. Baim Mudrik Aziz (G1A022071)  

Link Dashboard: [Dashboard Prediksi Mahasiswa DO](https://uas-data-mining-reksi-baim-dropout-prediction.streamlit.app/)

# Project Overview
## Latar Belakang

Dalam era digital saat ini, tantangan dalam dunia pendidikan tinggi semakin kompleks, salah satunya adalah meningkatnya angka mahasiswa yang tidak menyelesaikan studi atau mengalami putus kuliah (*drop out*). Permasalahan ini dapat berdampak buruk, baik bagi mahasiswa itu sendiri maupun bagi institusi pendidikan. Mahasiswa yang *drop out* kehilangan waktu, biaya, dan kesempatan, sedangkan institusi mengalami penurunan angka retensi dan citra akademik.

Seiring dengan berkembangnya teknologi informasi, institusi pendidikan kini memiliki akses terhadap berbagai data akademik mahasiswa, seperti data nilai, kehadiran, aktivitas daring, hingga status sosial ekonomi. Data tersebut memiliki potensi besar untuk dianalisis menggunakan teknik *data mining* guna mengidentifikasi pola dan karakteristik mahasiswa yang berisiko tinggi mengalami *drop out*. Salah satu pendekatan yang menjanjikan dalam analisis ini adalah penggunaan algoritma *machine learning*, khususnya Random Forest, yang dikenal memiliki kemampuan tinggi dalam menangani data kompleks dan memberikan prediksi yang akurat.

Menurut Devi dan Ratnoo (2022), penggunaan algoritma Random Forest dalam memprediksi *drop out* mahasiswa menunjukkan hasil yang signifikan dalam mengidentifikasi pola risiko berdasarkan data akademik dan demografis. Penelitian ini menyoroti bahwa Random Forest mampu mengatasi ketidakseimbangan data dan memberikan akurasi yang tinggi. Selain itu, Vaarma dan Li (2024) dalam studi empiris di institusi pendidikan tinggi Finlandia menemukan bahwa model berbasis *machine learning*, termasuk Random Forest, efektif dalam memprediksi *drop out* dengan memanfaatkan fitur seperti performa akademik dan faktor eksternal. Studi ini juga menekankan pentingnya pendekatan berbasis data untuk mendukung pengambilan keputusan di institusi pendidikan.

Pada proyek ini, dibangun sebuah sistem prediksi mahasiswa berisiko tinggi *drop out* menggunakan teknik *data mining* berbasis algoritma Random Forest. Sistem ini memanfaatkan data akademik dan demografis mahasiswa untuk mengidentifikasi potensi risiko sejak dini.

### Tujuan dari proyek ini adalah untuk:
- Menganalisis faktor-faktor yang mempengaruhi risiko *drop out* pada mahasiswa.
- Mengembangkan model klasifikasi berbasis Random Forest untuk memprediksi mahasiswa berisiko tinggi *drop out*.
- Memberikan hasil prediksi yang dapat digunakan oleh pihak universitas sebagai dasar untuk melakukan intervensi preventif.

Dengan adanya sistem ini, diharapkan institusi pendidikan dapat meningkatkan angka kelulusan dan memberikan dukungan akademik yang lebih tepat sasaran kepada mahasiswa yang membutuhkan.

### Daftar Referensi
- Devi, K., & Ratnoo, S. (2022). *Predicting student dropouts using random forest*. Journal of Statistics and Management Systems, 25(7), 1579–1590. https://doi.org/10.1080/09720510.2022.2130570
- Vaarma, M., & Li, H. (2024). *Predicting student dropouts with machine learning: An empirical study in Finnish higher education*. Technology in Society, 76, 102474. ISSN 0160-791X. https://doi.org/10.1016/j.techsoc.2024.102474.
---

## Business Understanding

### Problem Statements
Tingginya jumlah mahasiswa yang meninggalkan studi di perguruan tinggi sebelum menyelesaikan program menjadi isu krusial yang memengaruhi baik mahasiswa maupun institusi pendidikan serta lingkungan sekitar. Beragam aspek, seperti capaian akademik yang kurang memadai (contohnya, IPK per semester), kehadiran yang tidak teratur, partisipasi daring yang rendah, tekanan kerja, status pekerjaan, dan kondisi ekonomi, sering menjadi pemicu utama. Sayangnya, pihak universitas kerap kesulitan mendeteksi mahasiswa yang berpotensi menghadapi masalah ini sejak dini, sehingga upaya pencegahan menjadi terbatas.

Tanpa alat bantu berbasis data untuk pengenalan dini, kampus kehilangan kesempatan emas untuk memberikan bantuan yang sesuai kepada mahasiswa yang membutuhkan perhatian khusus.

### Goals
Tujuan utama proyek ini adalah membentuk alat prediksi berbasis data yang mampu:
- Mengenali mahasiswa dengan risiko besar untuk meninggalkan studi berdasarkan data akademik dan latar belakang pribadi.
- Menyelami faktor-faktor kunci seperti nilai IPK, tingkat kehadiran, keterlibatan daring, beban kerja, status pekerjaan, dan kondisi ekonomi yang memengaruhi potensi putus studi.
- Menyediakan laporan prediksi yang dapat dimanfaatkan pihak universitas sebagai acuan untuk melaksanakan langkah pencegahan, seperti bimbingan akademik atau program pendukung.

### Solution Approach

Proyek ini mengadopsi kerangka kerja **CRISP-DM (Cross-Industry Standard Process for Data Mining)** dengan langkah-langkah terstruktur sebagai berikut:

1. **Business Understanding**  
   Menggali inti permasalahan, yaitu mendeteksi mahasiswa yang berisiko putus studi agar universitas dapat mengambil langkah preventif yang tepat berdasarkan data akademik dan demografis.

2. **Data Understanding**  
   Mengumpulkan dan mempelajari kumpulan data yang mencakup detail akademik (IPK per semester, kehadiran, aktivitas daring, riwayat pengulangan mata kuliah, beban kerja) serta demografis (status pekerjaan, kondisi ekonomi) dari 4000 mahasiswa. Tahap ini juga melibatkan analisis awal untuk memahami pola dan distribusi, seperti proporsi risiko putus studi dan profil pekerjaan/ekonomi.

3. **Data Preparation**  
   Membersihkan data dengan menangani nilai yang hilang dan data ganda, menormalkan fitur numerik menggunakan MinMaxScaler, serta mengonversi variabel kategorikal (status pekerjaan dan ekonomi) menjadi format yang sesuai untuk pemodelan melalui one-hot encoding.

4. **Modelling**  
   Pada tahap ini, dua metode klasifikasi diuji untuk memprediksi mahasiswa berisiko putus studi:
   - **Random Forest**: Teknik penggabungan berbasis pohon keputusan yang unggul dalam menangani data kompleks, mampu mengatasi ketidakseimbangan, dan mencatat akurasi tinggi (98.25%) dalam proyek ini.
   - **Logistic Regression**: Pendekatan statistik yang diuji untuk perbandingan, meskipun akurasinya lebih rendah (89.38%), memberikan pandangan tambahan tentang hubungan antar fitur dan risiko putus studi.
   Kedua model dikonfigurasi dengan pengaturan dasar dan `random_state` untuk menjamin hasil yang konsisten, dengan Random Forest dipilih sebagai solusi utama berkat performanya yang lebih baik.

5. **Evaluation**  
   Mengukur efektivitas model dengan metrik seperti Akurasi, Presisi, Recall, F1-Score, dan Confusion Matrix untuk memastikan kemampuan model dalam mendeteksi mahasiswa berisiko secara tepat, dengan penekanan pada Random Forest sebagai pilihan utama.

---

## Data Understanding

### 📦 Sumber Data

Dataset ini merupakan kumpulan data buatan (*dummy dataset*) yang dirancang khusus untuk kebutuhan analisis risiko putus kuliah (*drop out*) mahasiswa di lingkungan perguruan tinggi.

| Informasi Dasar       | Detail         |
|-----------------------|----------------|
| Jumlah Baris          | 4000           |
| Jumlah Kolom          | 14             |
| Format                | CSV            |

#### ✅ Kualitas Data:

- **Nilai Hilang**: Tidak terdapat nilai yang hilang (*missing values*) setelah proses pembersihan.
- **Duplikasi**: Tidak ada data ganda yang terdeteksi setelah pemeriksaan awal.
- **Kolom Tidak Relevan**: Kolom `NIM` digunakan sebagai pengenal unik dan dapat dihilangkan dari analisis numerik untuk mencegah kesalahan perhitungan.

#### 📋 Detail Kolom:

| No | Nama Kolom                   | Tipe Data | Penjelasan                                   |
|----|------------------------------|-----------|----------------------------------------------|
| 0  | `NIM`                        | object    | Nomor Induk Mahasiswa sebagai identitas unik |
| 1  | `IPK_Semester_1`             | float64   | Nilai Indeks Prestasi Kumulatif semester 1   |
| 2  | `IPK_Semester_2`             | float64   | Nilai IPK semester 2                         |
| 3  | `IPK_Semester_3`             | float64   | Nilai IPK semester 3                         |
| 4  | `IPK_Semester_4`             | float64   | Nilai IPK semester 4                         |
| 5  | `IPK_Semester_5`             | float64   | Nilai IPK semester 5                         |
| 6  | `IPK_Semester_6`             | float64   | Nilai IPK semester 6                         |
| 7  | `Kehadiran_Per_Mata_Kuliah`  | float64   | Persentase kehadiran rata-rata per mata kuliah |
| 8  | `Riwayat_Pengambilan_Ulang`  | int64     | Jumlah pengulangan mata kuliah               |
| 9  | `Aktivitas_Sistem_Pembelajaran_Daring` | float64 | Persentase aktivitas di platform daring      |
| 10 | `Beban_Kerja_JamPerMinggu`   | float64   | Jam kerja per minggu (jika bekerja)          |
| 11 | `Status_Pekerjaan`           | object    | Status pekerjaan (tidak bekerja, bekerja)    |
| 12 | `Status_Ekonomi`             | object    | Tingkat ekonomi (rendah, menengah, tinggi)   |
| 13 | `Status_Risiko_DO`           | int64     | Status risiko drop out (0 = Aman, 1 = Risiko) |

#### 🔍 Pratinjau Data 5 Baris Teratas:

| NIM         | IPK_Semester_1 | IPK_Semester_2 | IPK_Semester_3 | IPK_Semester_4 | IPK_Semester_5 | IPK_Semester_6 | Kehadiran_Per_Mata_Kuliah | Riwayat_Pengambilan_Ulang | Aktivitas_Sistem_Pembelajaran_Daring | Beban_Kerja_JamPerMinggu | Status_Pekerjaan | Status_Ekonomi | Status_Risiko_DO |
|-------------|----------------|----------------|----------------|----------------|----------------|----------------|--------------------------|---------------------------|--------------------------------------|--------------------------|------------------|-----------------|-----------------|
| G1A0220000  | 2.01           | 2.10           | 2.05           | 2.08           | 2.00           | 2.17           | 89.84                     | 1                         | 65.50                              | 0                      | Tidak Bekerja   | Tinggi          | 0               |
| G1A0220001  | 3.27           | 3.15           | 3.40           | 3.13           | 3.35           | 3.39           | 83.69                     | 1                         | 82.59                              | 33                     | Bekerja         | Menengah        | 0               |
| G1A0220002  | 3.04           | 2.93           | 2.92           | 3.19           | 3.33           | 3.20           | 78.63                     | 0                         | 64.78                              | 0                      | Tidak Bekerja   | Tinggi          | 0               |
| G1A0220003  | 2.00           | 2.22           | 2.11           | 2.20           | 2.00           | 2.36           | 82.09                     | 1                         | 74.19                              | 23                     | Bekerja         | Rendah          | 0               |
| G1A0220004  | 3.58           | 3.35           | 3.52           | 3.39           | 3.15           | 3.28           | 76.21                     | 1                         | 63.18                              | 0                      | Tidak Bekerja   | Tinggi          | 0               |

#### 📊 Analisis Awal
- **Distribusi Data**: Nilai IPK berkisar antara 2.00 hingga 3.58, dengan kehadiran dan aktivitas daring bervariasi antara 63.18% hingga 89.84%. Status pekerjaan dan ekonomi menunjukkan variasi yang seimbang di antara kategori yang ada.
- **Korelasi Awal**: Ada indikasi bahwa mahasiswa dengan aktivitas daring di bawah 70% dan IPK rendah (di bawah 2.5) mungkin memiliki risiko *drop out* lebih tinggi, meskipun analisis lanjutan diperlukan.
- **Kesesuaian**: Dataset ini cukup representatif untuk memodelkan risiko *drop out* dengan memanfaatkan kombinasi fitur akademik dan demografis.
---

## Univariate Data Analysis

### 📊 Analisis Distribusi Risiko Dropout

#### Persentase Risiko Dropout
Analisis distribusi status risiko dropout dilakukan untuk memahami proporsi mahasiswa yang berada dalam kategori aman (0) dan berisiko tinggi (1). Berdasarkan perhitungan, persentase ditampilkan sebagai berikut:

- **Diagram Batang**:  
  Grafik batang menunjukkan perbandingan persentase antara status aman dan berisiko tinggi. Sumbu X mewakili status risiko (0 = Aman, 1 = Risiko Tinggi), sementara sumbu Y menunjukkan persentase. Grafik ini membantu mengidentifikasi dominasi kategori tertentu dalam dataset.
  
  ![Persentase Risiko Dropout Mahasiswa](https://github.com/user-attachments/assets/6638cfec-906f-4fb8-9ce2-747a492f4511)


#### Interpretasi
Distribusi risiko dropout memberikan gambaran awal tentang sebaran mahasiswa yang berpotensi menghadapi masalah putus kuliah, yang akan menjadi dasar untuk analisis lebih lanjut.

---

### 📋 Analisis Status Pekerjaan

#### Frekuensi dan Persentase
Berikut adalah ringkasan frekuensi dan persentase distribusi berdasarkan status pekerjaan mahasiswa:

| Status Pekerjaan | Jumlah | Persentase (%) |
|-------------------|--------|----------------|
| Tidak Bekerja    | 2121 | 53.025   |
| Bekerja          | 1879| 46.975  |


#### Visualisasi
- **Diagram Batang**:  
  Grafik batang menggambarkan jumlah mahasiswa untuk setiap kategori status pekerjaan. Sumbu X menunjukkan kategori (Tidak Bekerja, Bekerja), dan sumbu Y menunjukkan jumlah mahasiswa. Rotasi label sumbu X sebesar 45 derajat memastikan keterbacaan.
  
  ![Frekuensi Status Pekerjaan](https://github.com/user-attachments/assets/a7da60ac-e79f-4fe7-8abf-5532ad669a71)

- **Diagram Lingkaran**:  
  Diagram lingkaran menampilkan proporsi persentase status pekerjaan dengan label persentase hingga dua desimal. Grafik ini dimulai dari sudut 90 derajat untuk tampilan yang lebih estetis.
  
  ![Distribusi Persentase Status Pekerjaan](https://github.com/user-attachments/assets/fef9d1b0-2398-498d-ab83-8c988e97d595)


#### Interpretasi
Distribusi status pekerjaan menunjukkan pengaruh potensial status ini terhadap risiko dropout, dengan kategori "Bekerja" yang mungkin terkait dengan beban kerja tambahan.

---

### 📋 Analisis Status Ekonomi

#### Frekuensi dan Persentase
Berikut adalah ringkasan frekuensi dan persentase distribusi berdasarkan status ekonomi mahasiswa:

| Status Ekonomi | Jumlah | Persentase (%) |
|-----------------|--------|----------------|
| Rendah         | 1779 | 44.475   |
| Menengah       | 1593 | 39.825  |
| Tinggi         |  628 | 15.700   |


#### Visualisasi
- **Diagram Batang**:  
  Grafik batang mengilustrasikan jumlah mahasiswa untuk setiap tingkat status ekonomi. Sumbu X menampilkan kategori (Rendah, Menengah, Tinggi), dan sumbu Y menunjukkan jumlah mahasiswa, dengan rotasi label 45 derajat untuk kejelasan.
  
  ![Frekuensi Status Ekonomi](https://github.com/user-attachments/assets/2f4ea486-ebcf-43ee-abd9-b262f11db679)

- **Diagram Lingkaran**:  
  Diagram lingkaran menyajikan proporsi persentase status ekonomi dengan label hingga dua desimal, dimulai dari sudut 90 derajat untuk estetika yang lebih baik.
  
  ![Distribusi Persentase Status Ekonomi](https://github.com/user-attachments/assets/f15b23a2-2f15-4d54-9989-4e61c7933708)


#### Interpretasi
Distribusi status ekonomi memberikan wawasan awal tentang bagaimana kondisi finansial mahasiswa dapat memengaruhi risiko dropout, dengan kategori "Rendah" yang mungkin memerlukan perhatian khusus.

---

## Data Preprocessing

### 📋 Tujuan Preprocessing
Tahap ini bertujuan untuk mempersiapkan dataset `dataset_mahasiswa_DO_4000_mahasiswa.csv` agar siap digunakan dalam pemodelan prediksi risiko dropout mahasiswa. Proses ini mencakup pembersihan data awal, pembagian data menjadi set pelatihan dan pengujian, transformasi variabel kategorikal, dan normalisasi fitur numerik untuk memastikan kompatibilitas dengan model machine learning, khususnya Random Forest.

### 🔧 Langkah-Langkah Preprocessing

#### 1. Pembersihan Data Awal
- **Penanganan Nilai Hilang**: Dataset diperiksa untuk memastikan tidak ada nilai yang hilang (*missing values*). Berdasarkan analisis awal pada notebook, dataset ini telah bersih dari data kosong, sehingga tidak diperlukan imputasi atau penghapusan baris/kolom.
- **Penghapusan Duplikat**: Data diperiksa untuk mendeteksi duplikat berdasarkan semua kolom. Tidak ditemukan baris yang identik, sehingga dataset dipertahankan dalam kondisi aslinya.
- **Penghapusan Kolom Tidak Relevan**: Kolom `NIM` sebagai pengenal unik dihapus dari analisis numerik untuk mencegah interferensi dalam pemodelan, karena tidak memberikan kontribusi pada prediksi risiko dropout.

#### 2. Train-Test Split
- **Pembagian Data**: Dataset dibagi menjadi set pelatihan (`X_train`, `y_train`) dan set pengujian (`X_test`, `y_test`) menggunakan fungsi `train_test_split` dari `scikit-learn`. Pembagian dilakukan dengan parameter berikut:
  - `test_size=0.2`: 20% data (800 baris) digunakan untuk pengujian, dan 80% data (3200 baris) untuk pelatihan, berdasarkan total 4000 baris dataset.
  - `random_state=42`: Memastikan pembagian data dapat direproduksi.
  - `stratify=target`: Memastikan distribusi kelas target (`Status_Risiko_DO`) tetap seimbang antara set pelatihan dan pengujian.
- **Hasil Pembagian**:
  - Jumlah total dataset: 4000
  - Jumlah data latih: 3200
  - Jumlah data uji: 800
- **Tujuan**: Pembagian ini memungkinkan evaluasi model secara independen pada data yang tidak digunakan selama pelatihan, sehingga memberikan gambaran akurat tentang performa model.

#### 3. Transformasi Variabel Kategorikal
- **Encoding Variabel Kategorikal**: Variabel `Status_Pekerjaan` (Tidak Bekerja, Bekerja) dan `Status_Ekonomi` (Rendah, Menengah, Tinggi) dikonversi menjadi format numerik sebelum pemodelan. Dalam proses inferensi, pengguna diminta memasukkan nilai langsung (0 untuk Tidak Bekerja/Rendah, 1 untuk Bekerja/Menengah, 2 untuk Tinggi) sesuai panduan yang diberikan, sehingga proses encoding dilakukan secara manual melalui input pengguna.

#### 4. Normalisasi Fitur Numerik
- **Metode Normalisasi**: Fitur numerik seperti `IPK_Semester_1` hingga `IPK_Semester_6`, `Kehadiran_Per_Mata_Kuliah`, `Riwayat_Pengambilan_Ulang`, `Aktivitas_Sistem_Pembelajaran_Daring`, dan `Beban_Kerja_JamPerMinggu` dinormalisasi menggunakan `MinMaxScaler` untuk menskalakan data ke rentang [0, 1]. Proses ini dilakukan untuk memastikan semua fitur memiliki skala yang seragam, yang penting untuk performa model Random Forest.
- **Langkah Pelaksanaan**: 
  - Fitur numerik dipisahkan dari fitur kategorikal.
  - `MinMaxScaler` dilatih menggunakan data pelatihan (`X_train`) sebelumnya, lalu diterapkan pada data input pengguna untuk transformasi.
  - Hasil normalisasi digabungkan kembali dengan fitur kategorikal untuk membentuk data input akhir.
- **Contoh Rentang**: 
  - Sebelum normalisasi: `IPK_Semester_1` berkisar antara 2.00 hingga 3.58.
  - Setelah normalisasi: Diubah ke rentang 0 hingga 1 berdasarkan nilai minimum dan maksimum dari data pelatihan.

#### 5. Validasi Input Pengguna
- **Pengecekan Rentang**: Sebelum normalisasi, input dari pengguna divalidasi untuk memastikan nilai berada dalam rentang yang sesuai:
  - `IPK_Semester_1` sampai `IPK_Semester_6`: 0.0 hingga 4.0.
  - `Kehadiran_Per_Mata_Kuliah` dan `Aktivitas_Sistem_Pembelajaran_Daring`: 0 hingga 100.
  - `Riwayat_Pengambilan_Ulang`: Bilangan bulat non-negatif.
  - `Beban_Kerja_JamPerMinggu`: 0 hingga 168.
  - `Status_Pekerjaan`: 0 (Tidak Bekerja) atau 1 (Bekerja).
  - `Status_Ekonomi`: 0 (Rendah), 1 (Menengah), atau 2 (Tinggi).
- **Penanganan Kesalahan**: Jika input tidak valid (misalnya, angka di luar rentang atau bukan numerik), pengguna diminta mengulang input hingga memenuhi kriteria.

### 📊 Hasil Preprocessing
- Dataset akhir yang digunakan untuk prediksi terdiri dari fitur numerik yang telah dinormalisasi dan fitur kategorikal yang dikonversi ke format numerik. Set pelatihan (3200 baris) digunakan untuk melatih model, sementara set pengujian (800 baris) digunakan untuk evaluasi. Contoh input pengguna (misalnya, IPK 3.8, Kehadiran 70%, dll.) diubah menjadi skala yang sesuai sebelum dimasukkan ke model Random Forest.
- Proses ini memastikan data bersih, konsisten, dan siap untuk tahap pemodelan serta inferensi.

### 📝 Catatan
- Normalisasi hanya diterapkan pada fitur numerik, sedangkan fitur kategorikal langsung digunakan dalam bentuk numerik berdasarkan input pengguna.
- Model Random Forest yang telah dilatih menggunakan set pelatihan ini digunakan untuk prediksi berdasarkan data yang telah dipreproses.
---

## Modeling

### 📋 Tujuan Modeling
Tahap ini bertujuan untuk membangun dan melatih model machine learning untuk memprediksi risiko dropout mahasiswa berdasarkan dataset yang telah dipreproses. Dua model yang digunakan adalah Random Forest dan Logistic Regression, yang dievaluasi untuk menentukan model terbaik berdasarkan performa pada data pengujian.

### 🔧 Langkah-Langkah Modeling

#### 1. Random Forest
- **Penjelasan**: Random Forest adalah model berbasis ensemble learning yang menggunakan beberapa pohon keputusan untuk melakukan klasifikasi. Setiap pohon dilatih pada subset data yang berbeda (dengan pengambilan sampel acak), dan prediksi akhir dibuat dengan mayoritas suara dari semua pohon. Model ini cocok untuk dataset dengan banyak fitur dan hubungan non-linear, seperti data akademik dan demografis mahasiswa.
- **Pelatihan**: Model dilatih menggunakan `X_train_final` (fitur yang telah dinormalisasi dan diencode) dan `y_train` (target `Status_Risiko_DO`).
- **Prediksi**: Prediksi dilakukan pada `X_test_final` untuk menghasilkan `y_pred_rf`.

#### 2. Logistic Regression
- **Penjelasan**: Logistic Regression adalah model klasifikasi linier yang memprediksi probabilitas keanggotaan dalam suatu kelas (dalam hal ini, 0 = Aman, 1 = Risiko Tinggi) berdasarkan kombinasi linier dari fitur. Model ini mengasumsikan hubungan linier antara fitur dan log-odds target, sehingga lebih sederhana dibandingkan Random Forest.
- **Pelatihan**: Model dilatih menggunakan `X_train_final` dan `y_train`, dengan iterasi maksimum ditetapkan pada 1000 untuk memastikan konvergensi.
- **Prediksi**: Prediksi dilakukan pada `X_test_final` untuk menghasilkan `y_pred_lr`.

### 📊 Model Terbaik dan Alasan
- **Model Terbaik**: Berdasarkan kode, model Random Forest dipilih sebagai model utama yang disimpan (`model_prediksi_DO.pkl`), yang menunjukkan bahwa performanya dianggap lebih baik dibandingkan Logistic Regression. Alasannya karena:
  - Random Forest dapat menangkap hubungan non-linear dan interaksi kompleks antara fitur (misalnya, antara IPK, kehadiran, dan beban kerja), yang sering terjadi dalam data akademik.
  - Model ini lebih robust terhadap overfitting pada dataset dengan banyak fitur, dibandingkan Logistic Regression yang lebih sensitif terhadap asumsi linieritas.
  - Kode mencatat bahwa Random Forest "lebih bagus" yang dapat didukung oleh metrik evaluasi seperti akurasi, precision, recall, atau F1-score yang lebih tinggi pada data pengujian.
- 

### ⚙️ Parameter yang Digunakan
- **Random Forest**:
  - `n_estimators=100`: Jumlah pohon keputusan dalam ensemble, memberikan keseimbangan antara akurasi dan waktu komputasi.
  - `random_state=42`: Memastikan reproduktibilitas hasil.
- **Logistic Regression**:
  - `random_state=42`: Memastikan reproduktibilitas hasil.
  - `max_iter=1000`: Jumlah iterasi maksimum untuk konvergensi, mengatasi masalah jika model tidak konvergen dengan iterasi default.

### 🌟 Kelebihan dan Kekurangan
- **Random Forest**:
  - **Kelebihan**:
    - Dapat menangani fitur non-linear dan interaksi kompleks tanpa perlu transformasi tambahan.
    - Tidak sensitif terhadap outlier dan nilai hilang (jika ditangani sebelumnya).
    - Memberikan estimasi probabilitas yang baik melalui agregasi pohon.
  - **Kekurangan**:
    - Waktu pelatihan dan prediksi lebih lama dibandingkan Logistic Regression, terutama dengan jumlah pohon yang besar.
    - Kurang interpretabel dibandingkan Logistic Regression karena sifatnya yang berbasis ensemble.
- **Logistic Regression**:
  - **Kelebihan**:
    - Sederhana dan mudah diinterpretasikan, menunjukkan koefisien untuk setiap fitur yang berkontribusi pada prediksi.
    - Efisien secara komputasi, cocok untuk dataset kecil atau ketika kecepatan penting.
  - **Kekurangan**:
    - Mengasumsikan hubungan linier antara fitur dan log-odds, yang mungkin tidak sesuai dengan data akademik yang kompleks.
    - Kurang efektif jika terdapat interaksi tinggi antara fitur atau jika data tidak terdistribusi normal.
---

# Evaluasi Model

## Pendahuluan
Evaluasi model dilakukan untuk mengukur performa model Random Forest dan Logistic Regression dalam memprediksi risiko dropout (DO) mahasiswa berdasarkan dataset yang telah dilatih. Evaluasi ini mencakup metrik-metrik seperti akurasi, presisi, recall, dan F1-score, serta visualisasi confusion matrix untuk memberikan gambaran lebih jelas tentang performa model.

## Strategi Evaluasi Model
Strategi evaluasi model melibatkan pembagian dataset menjadi data latih (`X_train`, `y_train`) dan data uji (`X_test`, `y_test`) menggunakan metode seperti train-test split. Model dilatih pada data latih dan diuji pada data uji untuk memastikan generalisasi yang baik. Metrik evaluasi dipilih berdasarkan karakteristik masalah klasifikasi biner (Aman vs Risiko Tinggi), dengan fokus pada kemampuan model untuk mengidentifikasi kasus positif (Risiko Tinggi) secara akurat.

### Metrik Evaluasi
Berikut adalah metrik yang digunakan untuk menilai akurasi prediksi, lengkap dengan rumus dalam format LaTeX:

1.  **Akurasi (Accuracy)**
    - Rumus:
    $$\text{Akurasi} = \frac{TP + TN}{TP + TN + FP + FN}$$
    - Keterangan:
        - $TP$ (True Positive): Jumlah prediksi benar untuk kelas Risiko Tinggi.
        - $TN$ (True Negative): Jumlah prediksi benar untuk kelas Aman.
        - $FP$ (False Positive): Jumlah prediksi salah untuk kelas Risiko Tinggi (seharusnya Aman).
        - $FN$ (False Negative): Jumlah prediksi salah untuk kelas Aman (seharusnya Risiko Tinggi).
    - Interpretasi: Menunjukkan proporsi prediksi yang benar dari total prediksi.

2.  **Presisi (Precision)**
    - Rumus:
    $$\text{Presisi} = \frac{TP}{TP + FP}$$
    - Keterangan:
        - Mengukur seberapa banyak prediksi positif yang benar dari total prediksi positif.
    - Interpretasi: Penting dalam konteks di mana *false positive* (memprediksi Risiko Tinggi padahal Aman) harus diminimalkan, misalnya untuk menghindari kepanikan yang tidak perlu.

3.  **Recall (Sensitivity atau True Positive Rate)**
    - Rumus:
    $$\text{Recall} = \frac{TP}{TP + FN}$$
    - Keterangan:
        - Mengukur seberapa banyak kasus positif yang benar-benar terdeteksi dari total kasus positif aktual.
    - Interpretasi: Kritis dalam konteks di mana *false negative* (gagal mendeteksi Risiko Tinggi) harus dihindari, misalnya untuk memastikan semua mahasiswa berisiko mendapatkan intervensi.

4.  **F1-Score**
    - Rumus:
    $$\text{F1-Score} = 2 \cdot \frac{\text{Presisi} \cdot \text{Recall}}{\text{Presisi} + \text{Recall}}$$
    - Keterangan:
        - Harmonik rata-rata dari presisi dan recall, memberikan keseimbangan antara keduanya.
    - Interpretasi: Berguna ketika ada ketidakseimbangan antara presisi dan recall, memberikan gambaran keseluruhan performa model.

## Hasil Evaluasi

### Metrik Performa Model
| Model            | Accuracy | Precision    | Recall | F1-Score       |
| ---------------- | -------- | ------------ | ------ | -------------- |
| Random Forest    | 0.9825   | 0.985955     | 0.9750 | 0.980447       |
| Logistic Regression | 0.89375 | 0.887324     | 0.8750 | 0.881119       |

- **Random Forest**:  
  - Akurasi: 0.9825 (98.25%) menunjukkan proporsi prediksi yang sangat tinggi.  
  - Presisi: 0.985955 (98.60%) menunjukkan kemampuan model meminimalkan false positive dengan sangat baik.  
  - Recall: 0.9750 (97.50%) menunjukkan deteksi yang sangat efektif untuk Risiko Tinggi.  
  - F1-Score: 0.980447 (98.04%) menunjukkan keseimbangan yang optimal.

- **Logistic Regression**:  
  - Akurasi: 0.89375 (89.38%) menunjukkan performa yang baik tetapi lebih rendah.  
  - Presisi: 0.887324 (88.73%) menunjukkan lebih banyak false positive dibandingkan Random Forest.  
  - Recall: 0.8750 (87.50%) menunjukkan deteksi yang lebih lemah untuk Risiko Tinggi.  
  - F1-Score: 0.881119 (88.11%) menunjukkan keseimbangan yang lebih rendah.

### Confusion Matrix
#### 5.1 Random Forest
- Visualisasi:

![image](https://github.com/user-attachments/assets/641a9cb5-028f-4dc9-b6a4-f8def6c950f3)


- Penjelasan: Matriks menunjukkan 435 kasus Aman diprediksi benar (TN), 5 kasus Aman salah diprediksi sebagai Risiko Tinggi (FP), 9 kasus Risiko Tinggi salah diprediksi sebagai Aman (FN), dan 351 kasus Risiko Tinggi diprediksi benar (TP). Performa sangat tinggi dengan sedikit kesalahan, dan recall tinggi (351/360 = 97.50%) menunjukkan kemampuan deteksi Risiko Tinggi yang luar biasa.

#### 5.2 Logistic Regression 
- Visualisasi:

![image](https://github.com/user-attachments/assets/3f23294a-e106-4290-8844-ccb513af50c0)

- Penjelasan: Matriks menunjukkan 400 kasus Aman diprediksi benar (TN), 40 kasus Aman salah diprediksi sebagai Risiko Tinggi (FP), 45 kasus Risiko Tinggi salah diprediksi sebagai Aman (FN), dan 315 kasus Risiko Tinggi diprediksi benar (TP). Model memiliki lebih banyak kesalahan (85 total), dengan recall lebih rendah (315/360 = 87.50%), menunjukkan deteksi yang kurang optimal.

## Perbandingan dan Analisis
- **Random Forest** unggul signifikan dibandingkan **Logistic Regression** dalam semua metrik, terutama akurasi (0.9825 vs 0.89375) dan recall (0.9750 vs 0.8750), yang krusial untuk mendeteksi mahasiswa berisiko.  
- Confusion matrix Random Forest menunjukkan distribusi lebih seimbang dengan kesalahan sangat minim (14 kesalahan total vs 85 kesalahan pada Logistic Regression), menegaskan superioritasnya.  
- Pilihan Random Forest untuk disimpan dan digunakan dalam inferensi (seperti yang terlihat pada kode `joblib.dump(rf_model, 'model_prediksi_DO.pkl')`) didasarkan pada akurasi dan generalisasi yang jauh lebih baik.

## Kesimpulan
Berdasarkan evaluasi model Random Forest menunjukkan performa luar biasa dengan akurasi 0.9825, recall 0.9750, dan F1-score 0.980447, serta confusion matrix yang lebih baik. Model ini direkomendasikan untuk sistem pendukung keputusan dalam mencegah dropout mahasiswa, sebagaimana diterapkan dalam inferensi sederhana pada proyek ini.
---

## Deployment

### Tujuan
Tujuan deployment adalah untuk membuat aplikasi prediksi risiko dropout mahasiswa, yang saat ini berjalan secara lokal melalui file `app.py`, dapat diakses secara online oleh staf akademik, dosen, atau pihak rektorat. Dengan deployment ini, sistem dapat digunakan secara luas untuk pemantauan risiko dropout secara real-time dan mendukung pengambilan keputusan berbasis data di institusi pendidikan. Deployment juga bertujuan menjaga keandalan dan aksesibilitas aplikasi.

### Pemilihan Platform
Saya memilih **Streamlit Community Cloud** sebagai platform deployment karena gratis, mudah digunakan, dan dirancang khusus untuk aplikasi Streamlit. Platform ini mendukung deployment langsung dari GitHub, cocok untuk prototipe awal, dan memungkinkan akses publik tanpa biaya tambahan, yang sesuai dengan kebutuhan proyek saat ini.

### Langkah Implementasi
1. **Persiapan Repositori GitHub**:
   - Membuat repositori baru di GitHub yang berisi file `app.py`, `requirements.txt` (mencantumkan dependensi seperti `streamlit`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, dan `gdown`), serta file pendukung seperti `model_prediksi_DO.pkl` dan `dataset_mahasiswa_DO_4000_mahasiswa.csv`.
   - Memastikan file `.streamlit/config.toml` dikonfigurasi untuk menyesuaikan judul halaman dan tema aplikasi.

2. **Konfigurasi Aplikasi di Streamlit Community Cloud**:
   - Mengunjungi situs Streamlit Community Cloud (https://streamlit.io/cloud) dan masuk menggunakan akun GitHub.
   - Menghubungkan repositori GitHub yang berisi kode aplikasi.
   - Menentukan `app.py` sebagai file utama aplikasi dan mengaktifkan deployment otomatis.

3. **Pengujian dan Peluncuran**:
   - Memverifikasi bahwa aplikasi berjalan dengan baik di lingkungan cloud melalui tautan yang diberikan oleh Streamlit setelah deployment selesai.
   - Menguji fitur utama seperti prediksi, visualisasi, dan analisis untuk memastikan kompatibilitas.
   - Membagikan tautan akses kepada pengguna terpilih (misalnya, dosen atau staf akademik) setelah deployment berhasil pada 11:09 PM WIB, Minggu, 8 Juni 2025.

### Kesimpulan
Dengan memilih Streamlit Community Cloud dan mengikuti langkah-langkah di atas, aplikasi prediksi risiko dropout mahasiswa kini dapat diakses secara online, meningkatkan aksesibilitas dan mendukung penggunaan institusi pendidikan. Proses ini menandai langkah awal menuju skalabilitas lebih lanjut seiring perkembangan proyek.
---
# Rencana Pengembangan Sistem ke Depan

## Pendahuluan

Aplikasi prediksi risiko dropout mahasiswa telah berhasil dikembangkan menggunakan Streamlit dan model Random Forest (dengan akurasi 98.25%) berdasarkan dataset `dataset_mahasiswa_DO_4000_mahasiswa.csv`. Sistem ini saat ini berfungsi untuk analisis individu dan visualisasi dasar. Untuk meningkatkan dampak dan skalabilitas, rencana pengembangan sistem ke depan akan mencakup integrasi dengan sistem akademik dan visualisasi dashboard untuk pihak rektorat, diikuti oleh peningkatan spesifik pada aplikasi Streamlit.

## Rencana Pengembangan Sistem

### a. Integrasi dengan Sistem Informasi Akademik
- **Tujuan**: Memastikan data mahasiswa diperbarui secara real-time dari sistem internal universitas.
- **Deskripsi**: Mengintegrasikan aplikasi dengan sistem akademik melalui API untuk mengimpor data seperti IPK, kehadiran, dan status pekerjaan/ekonomi secara otomatis. Fungsi `load_data()` akan diperluas untuk mendukung koneksi database, dan notifikasi akan diaktifkan berdasarkan prediksi risiko.
- **Manfaat**: Mengurangi ketergantungan pada file CSV manual dan memungkinkan pemantauan berkelanjutan.
- **Langkah Implementasi**: 
  - Menambahkan pustaka seperti `psycopg2` untuk koneksi database (misalnya PostgreSQL).
  - Memodifikasi `load_data()` untuk mengambil data via API atau query SQL.
  - Menguji integrasi dengan data uji coba dari sistem akademik universitas.

### b. Visualisasi Dashboard untuk Pihak Rektorat
- **Tujuan**: Menyediakan alat analisis strategis untuk pengambilan keputusan institusi.
- **Deskripsi**: Menambahkan tab baru di Streamlit berjudul "Laporan Rektorat" dengan fitur filter interaktif (berdasarkan fakultas, semester, atau status ekonomi) menggunakan `st.selectbox`. Visualisasi akan mencakup grafik tren risiko dropout dan heatmap distribusi berdasarkan fitur utama menggunakan `plotly` atau `seaborn`.
- **Manfaat**: Membantu rektorat mengidentifikasi pola risiko secara keseluruhan dan merancang intervensi berbasis data.
- **Langkah Implementasi**: 
  - Membuat tab baru dengan filter dinamis untuk agregasi data.
  - Mengintegrasikan visualisasi interaktif dengan `st.plotly_chart` atau `st.pyplot`.
  - Menguji dashboard dengan data simulasi dari berbagai fakultas.

### c. Fitur Tambahan pada Sistem
| Fitur                | Deskripsi                                  |
| -------------------- | ------------------------------------------ |
| Sinkronisasi Data    | Tarik data real-time dari sistem akademik   |
| Filter Analisis      | Pilih kriteria (fakultas, semester, dll.)   |
| Tren Risiko          | Tampilkan grafik perkembangan risiko        |
| Ekspor Laporan       | Unduh laporan analisis dalam format PDF     |
| Notifikasi Otomatis  | Kirim peringatan untuk mahasiswa berisiko   |

### d. Contoh Visualisasi
- **Tren Risiko Dropout**: Grafik garis menunjukkan perubahan risiko berdasarkan semester dengan `plt.plot()` atau `plotly`.
- **Heatmap Distribusi**: Visualisasi hubungan antara status ekonomi dan risiko dropout dengan `sns.heatmap()` di tab rektorat.
- **st.plotly_chart()**: Menampilkan visualisasi interaktif langsung di Streamlit.

## Pengembangan Lebih Lanjut Aplikasi Streamlit
Setelah integrasi sistem dan visualisasi dasar selesai, pengembangan spesifik pada Streamlit akan difokuskan untuk meningkatkan pengalaman pengguna dan analisis data.

### a. Dukungan Upload Data Massal
- **Tujuan**: Memungkinkan prediksi untuk banyak mahasiswa sekaligus.
- **Deskripsi**: Menambahkan `st.file_uploader` di tab "Prediksi Dropout" untuk mengunggah file CSV. Fungsi `preprocess_data_for_prediction` akan diadaptasi untuk memproses data batch, dan hasilnya ditampilkan dalam tabel interaktif dengan `st.dataframe`.
- **Manfaat**: Mendukung analisis kelompok besar, seperti seluruh mahasiswa suatu program studi.
- **Langkah Implementasi**: 
  - Menyisipkan `uploaded_file = st.file_uploader("Unggah file CSV", type="csv")`.
  - Mengembangkan loop untuk memproses setiap baris dan memanggil `predict_dropout`.
  - Menampilkan tabel hasil dengan kolom status dan probabilitas.

### b. Visualisasi Interaktif dengan Plotly
- **Tujuan**: Meningkatkan kemampuan analisis visual untuk pengguna.
- **Deskripsi**: Mengganti visualisasi statis (`matplotlib`/`seaborn`) dengan `plotly` untuk grafik interaktif, seperti scatter plot risiko berdasarkan IPK dan kehadiran, serta bar chart distribusi status ekonomi.
- **Manfaat**: Memberikan fleksibilitas untuk zoom, hover, dan eksplorasi data secara mendalam.
- **Langkah Implementasi**: 
  - Mengimpor `plotly.express` dan mengganti `st.pyplot` dengan `st.plotly_chart`.
  - Mengadaptasi kode visualisasi existing (misalnya, distribusi risiko dropout) ke format Plotly.
  - Menguji interaktivitas dengan data uji coba.

### c. Fitur Tambahan pada Streamlit
| Fitur                | Deskripsi                                  |
| -------------------- | ------------------------------------------ |
| Unggah Data Batch    | Prediksi untuk banyak mahasiswa sekaligus   |
| Visualisasi Dinamis  | Grafik interaktif dengan Plotly            |
| Riwayat Prediksi     | Simpan dan tampilkan riwayat prediksi       |
| Unduh Hasil          | Ekspor prediksi ke file CSV                |

### d. Penyebarluasan dan Pemeliharaan Streamlit
- **Tujuan**: Memastikan aksesibilitas dan keandalan jangka panjang.
- **Deskripsi**: Mendeploy aplikasi ke Streamlit Community Cloud dengan autentikasi pengguna menggunakan Streamlit Secrets. Menambahkan jadwal pembaruan model secara berkala berdasarkan data terbaru.
- **Manfaat**: Meningkatkan akses staf akademik dan menjaga performa model.
- **Langkah Implementasi**: 
  - Mengonfigurasi deployment dengan `requirements.txt` dan file `.streamlit/config.toml`.
  - Menyisipkan autentikasi sederhana di sidebar.
  - Menjadwalkan pelatihan ulang model setiap 6 bulan.
---

# Kesimpulan
Sistem saat ini telah menunjukkan performa unggul dengan akurasi 98.25% menggunakan Random Forest, mengungguli Logistic Regression (89.38%), dan mengidentifikasi IPK, kehadiran, serta status ekonomi sebagai faktor risiko utama. Rencana pengembangan ini, dengan integrasi sistem akademik, dashboard rektorat, dan peningkatan Streamlit (upload massal, visualisasi dinamis, dan deployment), akan menjadikannya alat yang lebih robust untuk mendukung keberlanjutan akademik mahasiswa. Implementasi akan dimulai dengan integrasi data, diikuti oleh pengembangan Streamlit dan deployment.
