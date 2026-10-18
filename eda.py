import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

# ---------------------------------------------------------------
# 0. KONFIGURASI PATH & GAYA GRAFIK
# ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parent          # folder proyek
DATA_RAW = ROOT / "data" / "yield_df.csv"
OUT_CHART = ROOT / "static" / "charts"          # grafik utk web Flask
OUT_JSON = ROOT / "outputs"
OUT_CHART.mkdir(parents=True, exist_ok=True)
OUT_JSON.mkdir(parents=True, exist_ok=True)

# 8 negara Asia Monsun (Asia Selatan + Asia Tenggara) - subset studi kasus
NEGARA_MONSUN = [
    "Bangladesh", "India", "Indonesia", "Malaysia",
    "Nepal", "Pakistan", "Sri Lanka", "Thailand",
]
KOMODITAS = "Rice, paddy"   # padi

# Palet tema gelap (konsisten dengan UI aplikasi Flask)
BG, PANEL, FG, GRID = "#0f172a", "#1e293b", "#e2e8f0", "#334155"
ACCENT, ACCENT2 = "#84cc16", "#38bdf8"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
    "axes.edgecolor": GRID, "axes.labelcolor": FG,
    "xtick.color": FG, "ytick.color": FG, "text.color": FG,
    "grid.color": GRID, "grid.linewidth": 0.6,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
})

# ---------------------------------------------------------------
# 1. MUAT & BERSIHKAN DATA
# ---------------------------------------------------------------
print("=" * 64)
print("TAHAP 1 - PEMUATAN & PEMBERSIHAN AWAL")
print("=" * 64)
df = pd.read_csv(DATA_RAW)
print(f"Data mentah            : {len(df):,} baris x {df.shape[1]} kolom")

df = df.drop(columns=["Unnamed: 0"])          # buang artefak indeks lama

# Deduplikasi: 1 baris seharusnya unik per (negara, komoditas, tahun).
# Duplikat muncul akibat penggabungan dua sumber data suhu.
n0 = len(df)
df = df.drop_duplicates(subset=["Area", "Item", "Year"], keep="first")
print(f"Deduplikasi            : {n0:,} -> {len(df):,} baris "
      f"({n0 - len(df):,} duplikat dibuang)")

# Penanganan missing values (cek defensif - dataset ini ternyata lengkap)
n_missing = int(df[["hg/ha_yield", "average_rain_fall_mm_per_year"]].isna().sum().sum())
df = df.dropna(subset=["hg/ha_yield", "average_rain_fall_mm_per_year"])
print(f"Missing values X/y     : {n_missing} (dihapus jika ada)")

# ---------------------------------------------------------------
# 2. SARING SUBSET STUDI KASUS
# ---------------------------------------------------------------
print()
print("=" * 64)
print("TAHAP 2 - PENYARINGAN SUBSET STUDI KASUS")
print("=" * 64)
sub = df[(df["Item"] == KOMODITAS) & (df["Area"].isin(NEGARA_MONSUN))].copy()
sub = sub.rename(columns={
    "Area": "negara", "Year": "tahun",
    "average_rain_fall_mm_per_year": "curah_hujan_mm",
    "hg/ha_yield": "hasil_panen_hg_ha",
})
sub = sub[["negara", "tahun", "curah_hujan_mm", "hasil_panen_hg_ha"]]
print(f"Subset final           : {len(sub)} baris, {sub['negara'].nunique()} negara, "
      f"tahun {sub['tahun'].min()}-{sub['tahun'].max()}")
print(f"Rentang curah hujan    : {sub['curah_hujan_mm'].min():.0f} - "
      f"{sub['curah_hujan_mm'].max():.0f} mm/tahun")
print(f"Rentang hasil panen    : {sub['hasil_panen_hg_ha'].min():,.0f} - "
      f"{sub['hasil_panen_hg_ha'].max():,.0f} hg/ha")

# Simpan subset bersih untuk keperluan aplikasi & dokumentasi
sub.to_csv(ROOT / "data" / "padi_monsun_bersih.csv", index=False)

# ---------------------------------------------------------------
# 3. STATISTIK DESKRIPTIF & KORELASI
# ---------------------------------------------------------------
print()
print("=" * 64)
print("TAHAP 3 - STATISTIK DESKRIPTIF & KORELASI")
print("=" * 64)
print("\nStatistik deskriptif:")
print(sub[["curah_hujan_mm", "hasil_panen_hg_ha"]].describe().round(1).to_string())

r_p, p_p = pearsonr(sub["curah_hujan_mm"], sub["hasil_panen_hg_ha"])
r_s, p_s = spearmanr(sub["curah_hujan_mm"], sub["hasil_panen_hg_ha"])
print(f"\nKorelasi Pearson  r = {r_p:+.3f}  (p-value = {p_p:.2e})")
print(f"Korelasi Spearman r = {r_s:+.3f}  (p-value = {p_s:.2e})")
print("-> Hubungan linear positif signifikan" if p_p < 0.05 else "-> TIDAK signifikan!")

# statistik per negara (untuk tabel di halaman analitik web)
per_negara = (sub.groupby("negara")
              .agg(curah_hujan_mm=("curah_hujan_mm", "first"),
                   hasil_rata_hg_ha=("hasil_panen_hg_ha", "mean"),
                   hasil_min=("hasil_panen_hg_ha", "min"),
                   hasil_max=("hasil_panen_hg_ha", "max"),
                   jumlah_data=("hasil_panen_hg_ha", "count"))
              .sort_values("curah_hujan_mm").round(0))
print("\nRata-rata per negara (urut curah hujan):")
print(per_negara.to_string())

# ---------------------------------------------------------------
# 4. GRAFIK EDA
# ---------------------------------------------------------------
print()
print("=" * 64)
print("TAHAP 4 - PEMBUATAN GRAFIK EDA (PNG tema gelap)")
print("=" * 64)

# --- 4a. Scatterplot validasi linearitas + garis tren OLS ----------
fig, ax = plt.subplots(figsize=(8.6, 5.6), dpi=150, constrained_layout=True)
warna = dict(zip(NEGARA_MONSUN, plt.cm.tab10.colors))
for neg, g in sub.groupby("negara"):
    ax.scatter(g["curah_hujan_mm"], g["hasil_panen_hg_ha"],
               s=28, alpha=0.75, color=warna[neg], label=neg, edgecolors="none")
b1, b0 = np.polyfit(sub["curah_hujan_mm"], sub["hasil_panen_hg_ha"], 1)
xs = np.linspace(sub["curah_hujan_mm"].min(), sub["curah_hujan_mm"].max(), 100)
ax.plot(xs, b0 + b1 * xs, color=ACCENT, lw=2.6, label="Garis tren (OLS)")
ax.set_title("Validasi Linearitas: Curah Hujan vs Hasil Panen Padi")
ax.set_xlabel("Curah hujan tahunan rata-rata (mm)")
ax.set_ylabel("Produktivitas padi (hg/ha)")
ax.grid(True, alpha=0.5)
ax.legend(fontsize=7.5, ncol=3, loc="upper left", framealpha=0.25)
ax.annotate(f"r Pearson = {r_p:+.3f}", xy=(0.98, 0.04), xycoords="axes fraction",
            ha="right", fontsize=10, color=ACCENT, fontweight="bold")
fig.savefig(OUT_CHART / "scatter_linearitas.png")
plt.close(fig)
print("[1/5] scatter_linearitas.png")

# --- 4b. Histogram distribusi tiap variabel ------------------------
fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), dpi=150, constrained_layout=True)
axes[0].hist(sub["curah_hujan_mm"], bins=18, color=ACCENT2, alpha=0.85,
             edgecolor=BG)
axes[0].set_title("Distribusi Curah Hujan (X)")
axes[0].set_xlabel("mm/tahun"); axes[0].set_ylabel("Frekuensi")
axes[0].grid(True, alpha=0.4)
axes[1].hist(sub["hasil_panen_hg_ha"], bins=18, color=ACCENT, alpha=0.85,
             edgecolor=BG)
axes[1].set_title("Distribusi Hasil Panen (y)")
axes[1].set_xlabel("hg/ha"); axes[1].set_ylabel("Frekuensi")
axes[1].grid(True, alpha=0.4)
fig.savefig(OUT_CHART / "distribusi.png")
plt.close(fig)
print("[2/5] distribusi.png")

# --- 4c. Boxplot deteksi outlier -----------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), dpi=150, constrained_layout=True)
axes[0].boxplot(sub["curah_hujan_mm"], vert=False, patch_artist=True,
                boxprops=dict(facecolor=ACCENT2, alpha=0.7),
                medianprops=dict(color=FG))
axes[0].set_title("Boxplot Curah Hujan (X)")
axes[0].set_xlabel("mm/tahun"); axes[0].set_yticks([])
axes[1].boxplot(sub["hasil_panen_hg_ha"], vert=False, patch_artist=True,
                boxprops=dict(facecolor=ACCENT, alpha=0.7),
                medianprops=dict(color=FG))
axes[1].set_title("Boxplot Hasil Panen (y)")
axes[1].set_xlabel("hg/ha"); axes[1].set_yticks([])
for a in axes:
    a.grid(True, axis="x", alpha=0.4)
fig.savefig(OUT_CHART / "boxplot_outlier.png")
plt.close(fig)
print("[3/5] boxplot_outlier.png")

# --- 4d. Heatmap korelasi variabel numerik -------------------------
sub_full = df[(df["Item"] == KOMODITAS) & (df["Area"].isin(NEGARA_MONSUN))]
korel = sub_full[["average_rain_fall_mm_per_year", "hg/ha_yield",
                  "pesticides_tonnes", "avg_temp"]].corr()
label = ["Curah hujan", "Hasil panen", "Pestisida", "Suhu rata-rata"]
fig, ax = plt.subplots(figsize=(6.4, 5.2), dpi=150, constrained_layout=True)
im = ax.imshow(korel.values, cmap="RdYlGn", vmin=-1, vmax=1)
ax.set_xticks(range(4), label, rotation=20, ha="right")
ax.set_yticks(range(4), label)
for i in range(4):
    for j in range(4):
        ax.text(j, i, f"{korel.values[i, j]:+.2f}", ha="center", va="center",
                color="#0f172a", fontsize=10, fontweight="bold")
ax.set_title("Heatmap Korelasi Variabel Numerik (subset padi)")
fig.colorbar(im, ax=ax, shrink=0.82, label="Koefisien korelasi")
fig.savefig(OUT_CHART / "heatmap_korelasi.png")
plt.close(fig)
print("[4/5] heatmap_korelasi.png")

# --- 4e. Tren produktivitas tahunan per negara ---------------------
fig, ax = plt.subplots(figsize=(8.6, 5.0), dpi=150, constrained_layout=True)
for neg in NEGARA_MONSUN:
    g = sub[sub["negara"] == neg].groupby("tahun")["hasil_panen_hg_ha"].mean()
    ax.plot(g.index, g.values, marker="o", ms=3, lw=1.4, color=warna[neg], label=neg)
ax.set_title("Tren Produktivitas Padi per Negara (1990-2013)")
ax.set_xlabel("Tahun"); ax.set_ylabel("hg/ha")
ax.grid(True, alpha=0.5)
ax.legend(fontsize=8, ncol=4, loc="upper left", framealpha=0.25)
fig.savefig(OUT_CHART / "tren_tahunan.png")
plt.close(fig)
print("[5/5] tren_tahunan.png")

# ---------------------------------------------------------------
# 5. SIMPAN RINGKASAN EDA (dipakai halaman analitik Flask)
# ---------------------------------------------------------------
ringkasan = {
    "baris_mentah": int(n0 + 15112),  # sebelum dedup (dihitung ulang di bawah)
    "baris_setelah_dedup": int(len(pd.read_csv(DATA_RAW).drop(
        columns=["Unnamed: 0"]).drop_duplicates(subset=["Area", "Item", "Year"]))),
    "baris_subset": int(len(sub)),
    "jumlah_negara": int(sub["negara"].nunique()),
    "tahun_min": int(sub["tahun"].min()),
    "tahun_max": int(sub["tahun"].max()),
    "missing_values_xy": n_missing,
    "pearson_r": round(float(r_p), 4),
    "pearson_p": float(p_p),
    "spearman_r": round(float(r_s), 4),
    "hujan_min": float(sub["curah_hujan_mm"].min()),
    "hujan_max": float(sub["curah_hujan_mm"].max()),
    "hujan_rata": round(float(sub["curah_hujan_mm"].mean()), 1),
    "hasil_min": float(sub["hasil_panen_hg_ha"].min()),
    "hasil_max": float(sub["hasil_panen_hg_ha"].max()),
    "hasil_rata": round(float(sub["hasil_panen_hg_ha"].mean()), 1),
    "per_negara": per_negara.reset_index().to_dict(orient="records"),
    "daftar_negara": NEGARA_MONSUN,
}
with open(OUT_JSON / "eda_ringkasan.json", "w", encoding="utf-8") as f:
    json.dump(ringkasan, f, ensure_ascii=False, indent=2)
print("\nRingkasan EDA -> outputs/eda_ringkasan.json")
print("EDA SELESAI ✔")
