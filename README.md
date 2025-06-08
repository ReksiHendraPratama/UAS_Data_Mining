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
| Tidak Bekerja    | [Jumlah] | [Persentase]   |
| Bekerja          | [Jumlah] | [Persentase]   |

*Catatan: Nilai pasti akan tergantung pada data lengkap dataset. Berdasarkan sampel awal, distribusi ini mencerminkan variasi status pekerjaan.*

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
| Rendah         | [Jumlah] | [Persentase]   |
| Menengah       | [Jumlah] | [Persentase]   |
| Tinggi         | [Jumlah] | [Persentase]   |

*Catatan: Nilai pasti akan bergantung pada data lengkap dataset. Sampel awal menunjukkan variasi yang seimbang di antara ketiga kategori.*

#### Visualisasi
- **Diagram Batang**:  
  Grafik batang mengilustrasikan jumlah mahasiswa untuk setiap tingkat status ekonomi. Sumbu X menampilkan kategori (Rendah, Menengah, Tinggi), dan sumbu Y menunjukkan jumlah mahasiswa, dengan rotasi label 45 derajat untuk kejelasan.
  ![Frekuensi Status Ekonomi](https://github.com/user-attachments/assets/2f4ea486-ebcf-43ee-abd9-b262f11db679)

- **Diagram Lingkaran**:  
  Diagram lingkaran menyajikan proporsi persentase status ekonomi dengan label hingga dua desimal, dimulai dari sudut 90 derajat untuk estetika yang lebih baik.
  ![Distribusi Persentase Status Ekonomi](https://github.com/user-attachments/assets/f15b23a2-2f15-4d54-9989-4e61c7933708)


#### Interpretasi
Distribusi status ekonomi memberikan wawasan awal tentang bagaimana kondisi finansial mahasiswa dapat memengaruhi risiko dropout, dengan kategori "Rendah" yang mungkin memerlukan perhatian khusus.
