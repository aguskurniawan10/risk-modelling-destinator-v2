"""
features.py -- Feature engineering dari tabel per jam hasil data_loader.py.
Mengikuti pola anti-leakage PLTU Jeranjang: seluruh fitur HANYA memakai data
hingga waktu t (rolling/lag ke BELAKANG), tidak pernah mengintip masa depan.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    # --- Coal flow: level & dinamika (sinyal utama risiko CFM) ---
    "coal_flow_tph", "coal_flow_roll_mean_3h", "coal_flow_roll_std_3h",
    "coal_flow_roll_max_3h", "coal_flow_roll_mean_12h", "coal_flow_lag1", "coal_flow_lag2",
    "coal_flow_delta_1h",
    # --- Kualitas & rasio blending (harian, di-ffill) ---
    "cv_blend_kcal_kg", "rasio_mrc", "rasio_lrc",
    "cv_blend_roll_mean_7d",
    # --- Stok HOP ---
    "hop_hari_inventory", "hop_depletion_rate_1d", "hop_below_combined_safety",
    # --- Beban (PADAT, dari sheet redispatch) & forecast H+1 ---
    # [KLARIFIKASI USER] rantai risiko: ROH meleset dari redispatch -> blending
    # MRC tak selaras kebutuhan riil -> boros MRC saat beban rendah -> stok
    # menipis -> saat beban tinggi MRC kurang -> coal flow naik -> CFM.
    "beban_netto_mw", "beban_forecast_168h_mw", "boros_mrc_indicator",
    "deviasi_roh_redispatch_pct", "roh_data_available",
    # --- Kalender ---
    "hour", "day_of_week", "is_weekend", "is_holiday", "month",
]


def engineer(df: pd.DataFrame, mrc_min_days: float, lrc_min_days: float) -> pd.DataFrame:
    d = df.sort_values("time").reset_index(drop=True).copy()

    # Coal flow rolling (anti-leakage: rolling(...).shift(1) -> HANYA pakai data s.d t-1,
    # nilai di kolom coal_flow_tph sendiri (t) tetap dipakai sbg fitur "kondisi saat ini")
    d["coal_flow_roll_mean_3h"] = d["coal_flow_tph"].rolling(3, min_periods=1).mean()
    d["coal_flow_roll_std_3h"] = d["coal_flow_tph"].rolling(3, min_periods=1).std()
    d["coal_flow_roll_max_3h"] = d["coal_flow_tph"].rolling(3, min_periods=1).max()
    d["coal_flow_roll_mean_12h"] = d["coal_flow_tph"].rolling(12, min_periods=1).mean()
    d["coal_flow_lag1"] = d["coal_flow_tph"].shift(1)
    d["coal_flow_lag2"] = d["coal_flow_tph"].shift(2)
    d["coal_flow_delta_1h"] = d["coal_flow_tph"] - d["coal_flow_lag1"]

    # Blending: rolling 7-hari (168 jam) rata-rata CV -- indikasi tren kualitas coal yard
    d["cv_blend_roll_mean_7d"] = d["cv_blend_kcal_kg"].rolling(168, min_periods=24).mean()

    # Stok HOP: laju penurunan per hari (24 jam) -- pakai diff pada resolusi harian
    d["hop_depletion_rate_1d"] = d["hop_hari_inventory"] - d["hop_hari_inventory"].shift(24)

    # Ambang stok aman GABUNGAN, tertimbang rasio blending saat ini:
    # [STATED BY USER] MRC min 5 hari, LRC min 10 hari -- ditimbang rasio
    # blending riil hari itu (bukan asumsi 50:50).
    combined_safety = d["rasio_mrc"] * mrc_min_days + d["rasio_lrc"] * lrc_min_days
    d["hop_below_combined_safety"] = (d["hop_hari_inventory"] < combined_safety).astype(float)

    # --- BOROS MRC [KLARIFIKASI USER] ---------------------------------------
    # Indikator "MRC dipakai saat beban rendah" -- rasio_mrc dinilai BOROS
    # bila beban_netto_mw sedang di tertil TERENDAH (bawah P33 historis)
    # TAPI rasio_mrc tetap tinggi -- coal premium terpakai padahal energi
    # yg dibutuhkan sedang rendah, mempercepat penipisan stok MRC tanpa
    # kebutuhan mendesak.
    p33_beban = d["beban_netto_mw"].quantile(0.33)
    is_low_load = d["beban_netto_mw"] < p33_beban
    d["boros_mrc_indicator"] = np.where(is_low_load, d["rasio_mrc"], 0.0)

    # --- Penanganan NaN utk kesiapan model (LogReg tak toleran NaN) --------
    # deviasi_roh_redispatch_pct: sumber ROH jarang (lihat data_loader.py)
    # -> isi 0 (netral, BUKAN statistik data spy tak ada leakage), keberadaan
    # nilai asli tetap terekam via kolom roh_data_available (flag terpisah).
    d["deviasi_roh_redispatch_pct"] = d["deviasi_roh_redispatch_pct"].fillna(0.0)
    # lag/rolling di awal seri: bfill (nilai valid terdekat), lalu fallback 0
    for c in ["coal_flow_lag1", "coal_flow_lag2", "coal_flow_delta_1h",
              "cv_blend_roll_mean_7d", "hop_depletion_rate_1d"]:
        d[c] = d[c].bfill().fillna(0.0)
    # beban_forecast_168h_mw: NaN di 24 jam pertama data (lag blm cukup panjang)
    # -> bfill dari prediksi valid terdekat (BUKAN nilai masa depan -- cuma
    # mengisi celah start-of-series yg tak terhindarkan).
    if "beban_forecast_168h_mw" in d.columns:
        d["beban_forecast_168h_mw"] = d["beban_forecast_168h_mw"].bfill().fillna(d["beban_netto_mw"])

    # Jaring pengaman terakhir: celah coal_flow_tph yg LEBIH PANJANG dari batas
    # interpolasi di data_loader.py (>3 jam berturut-turut) masih bisa
    # merembet ke kolom turunan (rolling/lag) -- di sini diisi dgn
    # forward/backward-fill sbg upaya terakhir spy model bisa dilatih,
    # dihitung & dilaporkan eksplisit (bukan didiamkan).
    remaining_na = d[FEATURE_COLUMNS].isna().sum()
    remaining_na = remaining_na[remaining_na > 0]
    if len(remaining_na) > 0:
        print(f"  [features] NaN tersisa setelah penanganan utama (celah panjang coal_flow_tph, "
              f"di-ffill/bfill sbg jaring pengaman): {dict(remaining_na)}")
        d[FEATURE_COLUMNS] = d[FEATURE_COLUMNS].ffill().bfill()

    return d
