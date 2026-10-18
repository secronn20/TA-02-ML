# 🌾 Prediksi Hasil Panen Padi Berdasarkan Curah Hujan

**Proyek Akhir Modul 1 — Implementasi Regresi Linear Sederhana dengan Aplikasi Web Flask**

Studi kasus: memprediksi **produktivitas padi (hg/ha)** dari **curah hujan tahunan rata-rata (mm/tahun)**
menggunakan Regresi Linear Sederhana (Scikit-Learn), dikemas dalam aplikasi web interaktif (Flask + Bootstrap 5).

---

## 1. Ringkasan Studi Kasus

| Aspek | Keterangan |
|---|---|
| Variabel independen (X) | Curah hujan tahunan rata-rata (`average_rain_fall_mm_per_year`, mm) |
| Variabel dependen (y) | Produktivitas padi (`hg/ha_yield`, hg/ha) |
| Dataset | `yield_df.csv` — gabungan data **FAO + World Bank** via Kaggle (101 negara, 10 komoditas, 1990–2013) |
| Subset final | Padi (*Rice, paddy*) di **8 negara Asia Monsun**: Bangladesh, India, Indonesia, Malaysia, Nepal, Pakistan, Sri Lanka, Thailand |
| Persamaan model | **y = 25.562,87 + 3,73 × X** |
| Performa | R² uji = **0,313** · RMSE uji = **6.638 hg/ha (±0,66 ton/ha)** · r = **+0,480** (p ≈ 5,5×10⁻¹²) |

### Justifikasi pemilihan dataset & subset
1. **Komoditas padi** — pangan pokok utama Asia; air adalah input paling menentukan sehingga hubungan
   hujan → hasil panen logis secara agronomis (sebab-akibat, bukan sekadar korelasi).
2. **Kawasan Asia Monsun** — pada data global, korelasi justru negatif lemah (r = −0,13) karena
   terkonfound oleh irigasi/teknologi (contoh: Australia hujan 534 mm namun hasil sangat tinggi berkat irigasi).
   Pada kawasan monsun yang iklimnya homogen dan padi mayoritas *rain-fed*, hubungannya menjadi positif dan
   signifikan (r = +0,48; korelasi naik menjadi +0,58 pada agregat per negara).
3. **Indonesia termasuk dalam subset** — menjadikan studi kasus relevan dengan konteks lokal
   (Indonesia adalah negara berproduktivitas padi tertinggi di subset: 4,58 ton/ha).

## 2. Struktur Proyek

```
prediksi-panen-flask/
├── app.py                  # Aplikasi web Flask (form prediksi + analitik)
├── eda.py                  # Eksplorasi data + 5 grafik PNG (tema gelap)
├── train_model.py          # Pra-pemrosesan, pelatihan, evaluasi, Joblib
├── requirements.txt        # Daftar dependensi Python
├── README.md               # Dokumen ini
├── data/
│   ├── yield_df.csv        # Dataset mentah (FAO + World Bank via Kaggle)
│   └── padi_monsun_bersih.csv  # Subset bersih hasil eda.py
├── model/                  # (dibuat train_model.py)
│   ├── model_regresi_padi.joblib   # Model final (Joblib)
│   └── meta_model.json             # Metrik, parameter, rentang input
├── outputs/                # (dibuat eda.py)
│   ├── eda_ringkasan.json  # Ringkasan EDA untuk halaman analitik
│   └── screenshot/         # Tangkapan layar halaman web
├── static/
│   ├── charts/             # 5 grafik EDA (dipakai web)
│   ├── css/style.css       # Tema gelap slate & lime
│   └── vendor/             # Bootstrap 5.3.3 lokal (jalan offline)
└── templates/
    ├── base.html           # Kerangka halaman (navbar + footer)
    ├── index.html          # Formulir prediksi
    ├── hasil.html          # Halaman hasil prediksi
    └── analitik.html       # Dashboard analitik model
```

## 3. Cara Menjalankan

```bash
# 1) (disarankan) buat virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2) instal dependensi
pip install -r requirements.txt

# 3) jalankan EDA -> menghasilkan grafik PNG + outputs/eda_ringkasan.json
python eda.py

# 4) latih model -> menghasilkan model/*.joblib + meta_model.json
python train_model.py

# 5) jalankan aplikasi web
python app.py
# buka http://127.0.0.1:5000  (atau set PORT=... sesuai kebutuhan)
```

> Aplikasi sudah menyertakan model terlatih, grafik, dan Bootstrap lokal —
> langkah 3–4 boleh dilewati untuk sekadar demo; jalankan ulang bila ingin mereproduksi.

## 4. Pemetaan ke Instruksi Tugas

| Langkah Tugas | Lokasi Implementasi |
|---|---|
| 1. Studi kasus & dataset + justifikasi | README §1, justifikasi subset juga tampil di halaman `/analitik` |
| 2. EDA & pra-pemrosesan (scatterplot validasi linearitas, missing values, outlier) | `eda.py` (5 grafik), `train_model.py` Tahap 1 (deduplikasi 15.112 baris, cek missing, outlier IQR) |
| 3. Model Scikit-Learn + split + R² & RMSE + interpretasi intersep/koefisien | `train_model.py` Tahap 2–5; interpretasi juga tampil di `/analitik` |
| 4. Penyimpanan model dengan Joblib | `train_model.py` Tahap 6 → `model/model_regresi_padi.joblib` (+ verifikasi muat-ulang) |
| 5. Aplikasi web Flask + Bootstrap, form sesuai X, backend memuat model & menampilkan prediksi | `app.py` (route `/`, `/prediksi`, `/analitik`), `templates/`, `static/` |

## 5. Interpretasi Model (Konteks Studi Kasus)

- **Intersep (b₀) = 25.562,87 hg/ha** → estimasi hasil panen saat curah hujan 0 mm; secara praktis
  dibaca sebagai *baseline* produktivitas yang ditopang sumber air non-hujan (irigasi, air tanah).
  Karena 0 mm jauh di luar rentang data (494–2.875 mm), intersep **tidak boleh diekstrapolasi**.
- **Koefisien (b₁) = 3,73 hg/ha per mm** → setiap kenaikan curah hujan **+100 mm/tahun** diasosiasikan
  dengan kenaikan produktivitas padi **+373 hg/ha (≈ +0,37 ton/ha)**. Arah positif konsisten dengan
  logika agronomis padi sawah di kawasan monsun.
- **R² uji 0,313** → ±31% variasi produktivitas dijelaskan curah hujan saja; sisanya oleh irigasi,
  pemupukan, varietas, hama, dan praktik budidaya — wajar untuk model sederhana 1 variabel.

## 6. Keputusan Pra-pemrosesan (Bahan Presentasi)

1. **Duplikat**: 15.112 dari 28.242 baris dibuang (artefak penggabungan dua sumber data suhu);
   tersisa 13.130 baris unik per (negara, komoditas, tahun).
2. **Missing values**: 0 — data lengkap.
3. **Outlier (IQR)**: 6 titik (hasil panen tinggi Indonesia pada tahun-tahun tertentu) diverifikasi
   sebagai data riil FAO, bukan kesalahan pencatatan → **dipertahankan** agar model mewakili rentang nyata.
4. **Split**: 147 baris latih : 37 baris uji (80:20), `random_state=42` (reproducible).

## 7. Fitur Aplikasi Web

- **Formulir prediksi** sesuai variabel X: *"Curah Hujan Tahunan Rata-rata (mm/tahun)"* dengan validasi
  (angka, rentang 0–5.000) + chip klik-cepat nilai riil per negara.
- **Halaman hasil**: prediksi hg/ha + konversi ton/ha, substitusi persamaan, peringatan ekstrapolasi
  bila input di luar rentang pelatihan, perbandingan dengan rata-rata kawasan.
- **Halaman analitik**: kartu metrik (R², RMSE, r, n), alur pra-pemrosesan, 5 grafik EDA, interpretasi
  intersep & koefisien, batasan model, tabel statistik per negara.
- Tema **dark mode** (slate & lime), responsif (Bootstrap 5 grid), Bootstrap di-*vendor* lokal → tetap
  berjalan tanpa internet.
# TA-02-ML
