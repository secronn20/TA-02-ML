import json
import os
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, render_template, request

# ---------------------------------------------------------------
# KONFIGURASI & KONSTANTA
# ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
PATH_MODEL = ROOT / "model" / "model_regresi_padi.joblib"
PATH_META = ROOT / "model" / "meta_model.json"
PATH_EDA = ROOT / "outputs" / "eda_ringkasan.json"

HG_PER_TON = 10_000          # 1 ton = 10.000 hektogram
BATAS_INPUT_MIN, BATAS_INPUT_MAKS = 0, 5000   # batas masuk yang wajar

app = Flask(__name__)

_model = None
_meta = None
_eda = None


def fmt_id(angka, desimal=2):
    """Format angka gaya Indonesia: ribuan '.', desimal ','."""
    teks = f"{angka:,.{desimal}f}"           # contoh: 25,562.87
    return teks.replace(",", "@").replace(".", ",").replace("@", ".")


def muat_model():
    """Muat model Joblib + metadata JSON (dipanggil sekali di startup)."""
    global _model, _meta, _eda
    _model = joblib.load(PATH_MODEL)                     # <- Joblib di sini
    with open(PATH_META, encoding="utf-8") as f:
        _meta = json.load(f)
    if PATH_EDA.exists():
        with open(PATH_EDA, encoding="utf-8") as f:
            _eda = json.load(f)
    print(f"[startup] Model dimuat: {PATH_MODEL.name} | "
          f"persamaan y = {_meta['parameter_model']['intersep_hg_ha']:,.2f} + "
          f"{_meta['parameter_model']['koefisien_hg_ha_per_mm']:,.2f}x")


muat_model()


def konteks_bersama():
    """Variabel yang dipakai bersama oleh semua halaman."""
    p = _meta["parameter_model"]
    return {
        "persamaan": (f"y  =  {fmt_id(p['intersep_hg_ha'])}  +  "
                      f"{fmt_id(p['koefisien_hg_ha_per_mm'])} x X"),
        "intersep": fmt_id(p["intersep_hg_ha"]),
        "koefisien": fmt_id(p["koefisien_hg_ha_per_mm"]),
        "koef_100mm": fmt_id(p["koefisien_hg_ha_per_100mm"], 0),
        "koef_100mm_ton": fmt_id(p["koefisien_ton_ha_per_100mm"]),
        "r2_uji": f"{_meta['metrik']['r2_uji']:.3f}".replace(".", ","),
        "rmse_ton": fmt_id(_meta["metrik"]["rmse_uji_ton_ha"]),
        "korelasi_r": f"{_meta['metrik']['r_korelasi']:.3f}".replace(".", ","),
        "rentang_min": fmt_id(_meta["rentang_input_valid"]["hujan_min"], 0),
        "rentang_maks": fmt_id(_meta["rentang_input_valid"]["hujan_maks"], 0),
        "contoh_negara": _meta["contoh_prediksi"],
    }


# ---------------------------------------------------------------
# RUTE 1 : HALAMAN UTAMA (FORMULIR PREDIKSI)
# ---------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html", menu_aktif="prediksi",
                           **konteks_bersama())


# ---------------------------------------------------------------
# RUTE 2 : PROSES PREDIKSI (POST)
# ---------------------------------------------------------------
@app.route("/prediksi", methods=["POST"])
def prediksi():
    # --- validasi: harus angka ---
    input_mentah = request.form.get("curah_hujan", "").strip().replace(",", ".")
    try:
        hujan = float(input_mentah)
    except ValueError:
        return render_template("index.html", menu_aktif="prediksi",
                               pesan_error="Input tidak valid. Masukkan angka "
                                           "curah hujan, contoh: 2702",
                               **konteks_bersama()), 400

    # --- validasi: rentang masuk yang wajar ---
    if not (BATAS_INPUT_MIN <= hujan <= BATAS_INPUT_MAKS):
        return render_template("index.html", menu_aktif="prediksi",
                               pesan_error="Curah hujan wajib antara 0 sampai "
                                           "5.000 mm/tahun.",
                               **konteks_bersama()), 400

    # --- prediksi memakai model hasil Joblib.load ---
    hasil_hg_ha = float(_model.predict(np.array([[hujan]]))[0])
    hasil_ton = hasil_hg_ha / HG_PER_TON

    # --- konteks tambahan untuk ditampilkan ---
    p = _meta["parameter_model"]
    rent_lo = _meta["rentang_input_valid"]["hujan_min"]
    rent_hi = _meta["rentang_input_valid"]["hujan_maks"]
    rata_ton = _eda["hasil_rata"] / HG_PER_TON if _eda else None
    selisih = hasil_ton - rata_ton
    arah = "lebih tinggi" if selisih >= 0 else "lebih rendah"
    perbandingan = f"{fmt_id(abs(selisih))} ton/ha {arah}"

    return render_template(
        "hasil.html",
        menu_aktif="prediksi",
        hujan_input=fmt_id(hujan, 0),
        prediksi_hg_ha=fmt_id(hasil_hg_ha, 0),
        prediksi_ton_ha=fmt_id(hasil_ton),
        kecil_ke_besar=perbandingan,
        rata_dataset_ton=fmt_id(rata_ton),
        di_luar_rentang=not (rent_lo <= hujan <= rent_hi),
        intersep=fmt_id(p["intersep_hg_ha"]),
        koefisien=fmt_id(p["koefisien_hg_ha_per_mm"]),
    )


# ---------------------------------------------------------------
# RUTE 3 : DASHBOARD ANALITIK MODEL
# ---------------------------------------------------------------
@app.route("/analitik")
def analitik():
    per_negara = []
    if _eda:
        for b in _eda["per_negara"]:
            rata = float(b["hasil_rata_hg_ha"])
            per_negara.append({
                "negara": b["negara"],
                "curah_hujan_mm": fmt_id(b["curah_hujan_mm"], 0),
                "hasil_rata_hg_ha": fmt_id(rata, 0),
                "hasil_rata_ton": fmt_id(rata / HG_PER_TON),
                "hasil_min": fmt_id(b["hasil_min"], 0),
                "hasil_max": fmt_id(b["hasil_max"], 0),
                "jumlah_data": b["jumlah_data"],
            })

    p = _meta["parameter_model"]
    ctx = konteks_bersama()
    ctx.update({
        "meta": _meta,
        "per_negara": per_negara,
        "p_value": "5,5 &times; 10<sup>-12</sup>",
        "r2_uji_fmt": f"{_meta['metrik']['r2_uji']:.3f}".replace(".", ","),
        "r2_latih_fmt": f"{_meta['metrik']['r2_latih']:.3f}".replace(".", ","),
        "rmse_uji_hg_fmt": fmt_id(_meta["metrik"]["rmse_uji_hg_ha"], 0),
        "rmse_uji_ton": fmt_id(_meta["metrik"]["rmse_uji_ton_ha"]),
        "persen_variasi": round(_meta["metrik"]["r2_uji"] * 100),
    })
    return render_template("analitik.html", menu_aktif="analitik", **ctx)


# ---------------------------------------------------------------
# MENJALANKAN SERVER
# ---------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # debug=True hanya untuk pengembangan; matikan saat produksi
    app.run(host="0.0.0.0", port=port, debug=True)
