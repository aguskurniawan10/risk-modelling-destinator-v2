"""
data_loader.py -- Harmonisasi 5 sumber data mentah DESTINATOR jadi SATU
tabel per jam (Unit 1, Jan 2025 - Agu 2026). Mengikuti pola PLTU Jeranjang:
setiap fungsi loader mendokumentasikan cara pembersihan & keputusan
imputasi, bukan diam-diam.
"""
from __future__ import annotations
import warnings

import numpy as np
import pandas as pd
import openpyxl

from . import config as C

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

# ==============================================================================
# Kalender libur nasional & cuti bersama 2025-2026 (SKB 3 Menteri) -- di-hardcode
# dari dokumen resmi yg diberikan user (Nomor 1017/2/2 Tahun 2024 utk 2025;
# Nomor 1497/2/5 Tahun 2025 utk 2026), BUKAN estimasi.
# ==============================================================================
LIBUR_NASIONAL_2025 = [
    "2025-01-01", "2025-01-27", "2025-01-29", "2025-03-29", "2025-03-31", "2025-04-01",
    "2025-04-18", "2025-04-20", "2025-05-01", "2025-05-12", "2025-05-29", "2025-06-01",
    "2025-06-06", "2025-06-27", "2025-08-17", "2025-09-05", "2025-12-25",
]
CUTI_BERSAMA_2025 = [
    "2025-01-28", "2025-03-28", "2025-04-02", "2025-04-03", "2025-04-04", "2025-04-07",
    "2025-05-13", "2025-05-30", "2025-06-09", "2025-12-26",
]
LIBUR_NASIONAL_2026 = [
    "2026-01-01", "2026-01-16", "2026-02-17", "2026-03-19", "2026-03-21", "2026-03-22",
    "2026-04-03", "2026-04-05", "2026-05-01", "2026-05-14", "2026-05-27", "2026-05-31",
    "2026-06-01", "2026-06-16", "2026-08-17", "2026-08-25", "2026-12-25",
]
CUTI_BERSAMA_2026 = [
    "2026-02-16", "2026-03-18", "2026-03-20", "2026-03-23", "2026-03-24",
    "2026-05-15", "2026-05-28", "2026-12-24",
]
ALL_HOLIDAYS = set(pd.to_datetime(
    LIBUR_NASIONAL_2025 + CUTI_BERSAMA_2025 + LIBUR_NASIONAL_2026 + CUTI_BERSAMA_2026
).date)


# ==============================================================================
# 1. TOTAL COAL FLOW (per jam, Unit 1) -- sinyal utama risiko CFM
# ==============================================================================
def load_coal_flow() -> pd.DataFrame:
    """Bersihkan kode error DCS (I/O Timeout, No Data, Arc Off-line, Configure)
    jadi NaN eksplisit -- BUKAN 0 (0 T/h valid = unit shutdown, beda makna dari
    sensor gagal baca)."""
    path = C.DATA_DIR / "data_total_coal_flow_destinator.xlsx"
    df = pd.read_excel(path, sheet_name="Sheet1")
    df.columns = ["time", "coal_flow_tph"]
    n_before = len(df)
    is_error_code = df["coal_flow_tph"].isin(C.DQ.dcs_error_codes)
    df["coal_flow_is_dcs_error"] = is_error_code
    df.loc[is_error_code, "coal_flow_tph"] = np.nan
    df["coal_flow_tph"] = pd.to_numeric(df["coal_flow_tph"], errors="coerce")
    df["time"] = pd.to_datetime(df["time"])
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    n_dcs_err = int(is_error_code.sum())
    # Interpolasi waktu utk celah PENDEK (<=3 jam berurutan) akibat gangguan DCS
    # sesaat -- celah lebih panjang dibiarkan NaN (bukan dipaksa diisi), spy
    # tak menyamarkan gangguan sensor berkepanjangan sbg data valid.
    df["coal_flow_tph"] = df["coal_flow_tph"].interpolate(method="linear", limit=3, limit_area="inside")
    n_remaining_na = int(df["coal_flow_tph"].isna().sum())
    print(f"  [coal_flow] {n_before} baris, {n_dcs_err} error DCS -> NaN "
          f"({n_dcs_err/n_before*100:.1f}%), rentang {df['time'].min()} - {df['time'].max()}")
    print(f"  [coal_flow] setelah interpolasi celah pendek (<=3 jam): {n_remaining_na} NaN tersisa "
          f"({n_remaining_na/n_before*100:.2f}%)")
    return df


# ==============================================================================
# 2. ROH vs REALISASI BEBAN NETTO (per 30 menit -> agregat per jam, Unit 1)
# ==============================================================================
# ==============================================================================
# 1b. SFC AKTUAL PER JAM -- data riil langsung (BUKAN diturunkan dari
#     coal_flow/beban spt versi sebelumnya). HANYA tersedia penuh Jan-Agu
#     2025 (100% jam terisi) -- KOSONG TOTAL sejak Sep 2025 [ditemukan saat
#     parsing, bukan asumsi]. Jauh lebih padat drpd proksi ROH (~12% jam)
#     utk periode yg tersedia, jadi dipakai sbg SUMBER UTAMA model
#     SFC~CV_blend, dgn proksi ROH+beban sbg pelengkap kontrol beban.
# ==============================================================================
def load_sfc_actual() -> pd.DataFrame:
    path = C.DATA_DIR / "DATA_SFC_DESTINATOR.xlsx"
    df = pd.read_excel(path, sheet_name="Sheet1")
    df.columns = ["time", "sfc_actual"]
    n_before = len(df)
    is_error = df["sfc_actual"].isin(["#VALUE!"])
    df.loc[is_error, "sfc_actual"] = np.nan
    df["sfc_actual"] = pd.to_numeric(df["sfc_actual"], errors="coerce")
    out_of_range = df["sfc_actual"].notna() & ((df["sfc_actual"] < 0.2) | (df["sfc_actual"] > 2.0))
    n_out = int(out_of_range.sum())
    df.loc[out_of_range, "sfc_actual"] = np.nan
    df["time"] = pd.to_datetime(df["time"])
    n_missing = int(df["sfc_actual"].isna().sum())
    print(f"  [sfc_actual] {n_before} baris, {int(is_error.sum())} '#VALUE!' -> NaN, "
          f"{n_out} di luar rentang wajar (0,2-2,0 kg/kWh) -> NaN, total kosong "
          f"{n_missing} ({n_missing/n_before*100:.1f}%) -- KOSONG TOTAL sejak Sep 2025.")
    return df


def load_roh_only() -> pd.DataFrame:
    """Ambil KHUSUS kolom ROH (rencana P2B) dari sheet bulanan 'BEBAN UNIT 1'
    -- SPARSE (~12% jam), dipakai HANYA utk hitung deviasi ROH vs Redispatch,
    BUKAN sbg sumber beban aktual utama (lihat load_beban_redispatch utk itu)."""
    path = C.DATA_DIR / "data_risk_modelling_destinator.xlsx"
    xls = pd.ExcelFile(path)
    beban_sheets = [s for s in xls.sheet_names if not s.startswith("redispatch")]

    rows = []
    for sheet in beban_sheets:
        raw = pd.read_excel(path, sheet_name=sheet, header=None, skiprows=3)
        for cols in [(1, 2), (5, 6)]:
            sub = raw.iloc[:, list(cols)].copy()
            sub.columns = ["time", "roh_mw"]
            sub = sub.dropna(subset=["time"])
            rows.append(sub)
    df = pd.concat(rows, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])
    df["roh_mw"] = pd.to_numeric(df["roh_mw"], errors="coerce")
    df = df.sort_values("time").drop_duplicates(subset="time")
    df["time_hour"] = df["time"].dt.floor("h")
    hourly = df.groupby("time_hour", as_index=False)[["roh_mw"]].mean()
    hourly = hourly.rename(columns={"time_hour": "time"})
    print(f"  [roh_only] {len(df)} baris 30-menit -> {len(hourly)} baris per jam "
          f"({hourly['roh_mw'].notna().sum()} py nilai riil, sisanya NaN -- SPARSE, "
          f"HANYA dipakai utk deviasi vs redispatch)")
    return hourly


def load_beban_redispatch() -> pd.DataFrame:
    """Sheet 'redispatch <bulan>' -- PADAT PENUH (1.488 baris/bulan, 30-menit,
    TANPA masalah header berulang spt sheet ROH) di SEMUA 20 bulan. Kolom:
    BEBAN REDISPATCH (instruksi override P2B, sering 0 di awal 2025 lalu jadi
    nyaris selalu terisi sejak pertengahan 2025 -- P2B mulai aktif redispatch),
    BEBAN NETTO UNIT 1 (realisasi aktual, PADAT & DIPAKAI SBG SUMBER UTAMA
    beban aktual utk seluruh model -- menggantikan proksi sebelumnya)."""
    path = C.DATA_DIR / "data_risk_modelling_destinator.xlsx"
    xls = pd.ExcelFile(path)
    sheets = [s for s in xls.sheet_names if s.startswith("redispatch")]

    rows = []
    for sheet in sheets:
        raw = pd.read_excel(path, sheet_name=sheet, header=None, skiprows=3)
        sub = raw.iloc[:, [1, 2, 3, 15]].copy()
        sub.columns = ["time", "beban_redispatch_mw", "beban_netto_mw", "cf_harian_per_jam"]
        sub = sub.dropna(subset=["time"])
        rows.append(sub)
    df = pd.concat(rows, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])
    for c in ["beban_redispatch_mw", "beban_netto_mw", "cf_harian_per_jam"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values("time").drop_duplicates(subset="time")

    df["time_hour"] = df["time"].dt.floor("h")
    hourly = df.groupby("time_hour", as_index=False)[
        ["beban_redispatch_mw", "beban_netto_mw"]].mean()
    hourly = hourly.rename(columns={"time_hour": "time"})
    n_valid = hourly["beban_netto_mw"].notna().sum()
    print(f"  [beban_redispatch] {len(df)} baris 30-menit dari {len(sheets)} sheet -> "
          f"{len(hourly)} baris per jam, {n_valid} ({n_valid/len(hourly)*100:.1f}%) py "
          f"beban_netto_mw valid -- PADAT, dipakai sbg sumber beban aktual utama")
    return hourly


# ==============================================================================
# 3. COAL BLENDING (per hari, Unit 1) -- CV blend & rasio MRC/LRC
# ==============================================================================
_MONTH_MAP_ID = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "april": 4, "mei": 5, "juni": 6, "juli": 7,
    "agustus": 8, "sept": 9, "okt": 10, "nov": 11, "des": 12,
}


def load_coal_blending() -> pd.DataFrame:
    """Setiap sheet = 1 bulan, kolom: Tanggal, Kalori, Tonase, MRC%, LRC%, ...
    Kolom Tonase SELALU 0 di seluruh sheet (dead column di sumbernya) --
    diabaikan, BUKAN dipakai seolah data riil."""
    path = C.DATA_DIR / "data_coal_blending_destinator.xlsx"
    xls = pd.ExcelFile(path)
    rows = []
    for sheet in xls.sheet_names:
        # nama sheet spt "coal blend jan 25" / "coal blend april 26"
        parts = sheet.replace("coal blend", "").strip().split()
        mon_str, yr_str = parts[0], parts[1]
        month = _MONTH_MAP_ID[mon_str]
        year = 2000 + int(yr_str)
        raw = pd.read_excel(path, sheet_name=sheet, header=None, skiprows=3)
        sub = raw.iloc[:, [1, 2, 4, 5]].copy()
        sub.columns = ["day", "cv_blend_kcal_kg", "rasio_mrc", "rasio_lrc"]
        sub = sub.dropna(subset=["day"])
        sub["day"] = pd.to_numeric(sub["day"], errors="coerce")
        sub = sub.dropna(subset=["day"])
        sub["time"] = pd.to_datetime(dict(year=year, month=month, day=sub["day"].astype(int)),
                                      errors="coerce")
        rows.append(sub[["time", "cv_blend_kcal_kg", "rasio_mrc", "rasio_lrc"]])
    df = pd.concat(rows, ignore_index=True).dropna(subset=["time"])
    for c in ["cv_blend_kcal_kg", "rasio_mrc", "rasio_lrc"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values("time").drop_duplicates(subset="time").reset_index(drop=True)
    print(f"  [coal_blending] {len(df)} hari dari {len(xls.sheet_names)} sheet bulanan, "
          f"rentang {df['time'].min().date()} - {df['time'].max().date()}")
    return df


# ==============================================================================
# 4. STOK HOP (Hari Operasi Penuh, per hari) -- format pivot (tanggal = kolom)
# ==============================================================================
def load_stok_hop() -> pd.DataFrame:
    """Parse format pivot tak beraturan: blok per bulan, baris metrik
    'HOP FULL LOAD (INVENTORY)' & 'CHCR', kolom = tanggal. Bulan Nov & Des 2025
    KOSONG total di sumber -- diimputasi terpisah di impute_missing_hop()."""
    path = C.DATA_DIR / "data_stok_hari_operasi_batubara_destinator.xlsx"
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb["data harian operasi (HOP) BB"]
    all_rows = list(ws.iter_rows(values_only=True))

    records = []
    i = 0
    while i < len(all_rows):
        row = all_rows[i]
        is_month_marker = (len(row) > 1 and isinstance(row[1], type(all_rows[3][1])))
        if is_month_marker and i + 2 < len(all_rows):
            date_header_row = all_rows[i + 1]
            hop_row = all_rows[i + 2]
            if date_header_row[1] == "Hari Tanggal" and isinstance(hop_row[0], str) \
                    and "HOP" in hop_row[0]:
                dates = date_header_row[2:]
                vals = hop_row[2:]
                for d, v in zip(dates, vals):
                    if d is not None and hasattr(d, "year") and isinstance(v, (int, float)):
                        records.append({"time": pd.Timestamp(d), "hop_hari_inventory": v})
        i += 1
    df = pd.DataFrame(records).drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)
    df["hop_is_imputed"] = False
    print(f"  [stok_hop] {len(df)} hari terparse, rentang {df['time'].min().date()} - "
          f"{df['time'].max().date()}")
    return df


def impute_missing_hop(df_hop: pd.DataFrame) -> pd.DataFrame:
    """[STATED BY USER] Nov & Des 2025 kosong total di sumber -> diganti asumsi
    prediksi: proyeksi tren linear dari 3 bulan sebelumnya (Ags-Okt 2025),
    di-clip agar tak negatif. Baris hasil imputasi diberi flag
    hop_is_imputed=True eksplisit -- SELALU bisa dibedakan dari data riil di
    seluruh analisis downstream (dashboard, validasi, dst)."""
    ref = df_hop[(df_hop["time"] >= "2025-08-01") & (df_hop["time"] < "2025-11-01")].copy()
    ref = ref.sort_values("time")
    if len(ref) < 10:
        raise ValueError("Data referensi Ags-Okt 2025 terlalu sedikit utk proyeksi tren")
    x = (ref["time"] - ref["time"].min()).dt.days.values.astype(float)
    y = ref["hop_hari_inventory"].values.astype(float)
    slope, intercept = np.polyfit(x, y, 1)

    missing_dates = pd.date_range("2025-11-01", "2025-12-31", freq="D")
    x0 = ref["time"].min()
    x_missing = (missing_dates - x0).days.values.astype(float)
    y_pred = np.clip(intercept + slope * x_missing, a_min=0.5, a_max=None)
    df_imputed = pd.DataFrame({
        "time": missing_dates, "hop_hari_inventory": y_pred, "hop_is_imputed": True,
    })
    print(f"  [stok_hop] imputasi Nov-Des 2025: {len(df_imputed)} hari (tren linear dari "
          f"Ags-Okt 2025, slope={slope:.3f} hari/hari, clip>=0.5)")
    out = pd.concat([df_hop, df_imputed], ignore_index=True).sort_values("time").reset_index(drop=True)
    return out


# ==============================================================================
# 5. GABUNGKAN SEMUA -> tabel per jam
# ==============================================================================
def load_clean() -> pd.DataFrame:
    print("Memuat & membersihkan 6 sumber data DESTINATOR (Unit 1)...")
    coal_flow = load_coal_flow()
    sfc_actual = load_sfc_actual()
    roh = load_roh_only()
    redispatch = load_beban_redispatch()
    blending = load_coal_blending()
    hop = load_stok_hop()
    hop = impute_missing_hop(hop)

    df = coal_flow.merge(redispatch, on="time", how="left")
    df = df.merge(roh, on="time", how="left")
    df = df.merge(sfc_actual, on="time", how="left")

    df["date"] = df["time"].dt.floor("D")
    blending = blending.rename(columns={"time": "date"})
    hop = hop.rename(columns={"time": "date"})
    df = df.merge(blending, on="date", how="left")
    df = df.merge(hop, on="date", how="left")

    # ffill kolom harian (blending & stok) ke resolusi per jam -- nilai
    # berlaku sepanjang hari itu sampai update berikutnya
    for c in ["cv_blend_kcal_kg", "rasio_mrc", "rasio_lrc", "hop_hari_inventory", "hop_is_imputed"]:
        df[c] = df.groupby(df["date"])[c].transform("first")
        df[c] = df[c].ffill()

    # PENTING: sumber ROH/beban_netto SANGAT JARANG (~88% jam tak ada data --
    # diverifikasi: tiap sheet bulanan hanya memuat ~3-4 hari riil dari 30-31
    # hari, sisanya baris header berulang; BUKAN bug parsing). TIDAK di-ffill
    # (itu memalsukan kepercayaan diri pada data yg sebenarnya kosong) --
    # dibiarkan NaN, dgn flag jelas kapan tersedia.
    df["roh_data_available"] = df["roh_mw"].notna()
    # (catatan: dulu ada flag "redispatch_data_available" di sini -- dihapus krn
    # namanya menyesatkan (sebenarnya cek ketersediaan beban_netto_mw/REALISASI,
    # bukan kolom beban_redispatch_mw) DAN skr selalu True 100% (beban_netto_mw
    # padat penuh), jadi tak informatif lagi.
    df["sfc_data_available"] = df["sfc_actual"].notna()

    # DEVIASI ROH vs REDISPATCH -- INTI masalah yg DESTINATOR selesaikan
    # [KLARIFIKASI USER]: ROH (rencana P2B) sering meleset dari REDISPATCH
    # (realisasi operasional aktual, beban_netto_mw dari sheet redispatch yg
    # PADAT) -- bukan ROH vs "realisasi" sheet asal yg sama (itu SPARSE &
    # kurang bisa diandalkan). Dihitung HANYA dimana ROH tersedia (~12% jam).
    df["deviasi_roh_redispatch_pct"] = np.where(
        df["roh_mw"].abs() > 1e-6,
        (df["beban_netto_mw"] - df["roh_mw"]).abs() / df["roh_mw"].abs() * 100.0,
        np.nan,
    )

    df["hour"] = df["time"].dt.hour
    df["day_of_week"] = df["time"].dt.dayofweek  # 0=Senin
    df["is_weekend"] = df["day_of_week"].isin([5, 6])
    df["month"] = df["time"].dt.month
    df["is_holiday"] = df["time"].dt.date.isin(ALL_HOLIDAYS)

    df = df.drop(columns=["date"]).sort_values("time").reset_index(drop=True)
    print(f"Tabel gabungan akhir: {len(df)} baris x {len(df.columns)} kolom, "
          f"rentang {df['time'].min()} - {df['time'].max()}")
    return df


if __name__ == "__main__":
    d = load_clean()
    print("\n--- 5 baris pertama ---")
    print(d.head())
    print("\n--- info kelengkapan data ---")
    print(d.isna().mean().round(3) * 100)
