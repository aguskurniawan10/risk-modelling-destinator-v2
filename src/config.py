"""
config.py -- Satu sumber kebenaran untuk seluruh ambang batas, asumsi, dan
parameter model risiko DESTINATOR (PLTU Pelabuhanratu, Unit 1).

Mengikuti pola project PLTU Jeranjang: SEMUA angka yang bukan hasil hitungan
langsung dari data riil harus lewat sini, dengan sumber/alasan didokumentasikan
di komentar -- supaya mudah dikalibrasi ulang dan transparan mana yang data
riil vs asumsi.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"

RANDOM_STATE = 42


@dataclass
class ScopeConfig:
    """Cakupan model -- PLTU Pelabuhanratu (Pelabuhan Ratu), Unit 1.
    DITENTUKAN OLEH KETERSEDIAAN DATA, bukan pilihan bebas: file coal flow &
    coal blending HANYA punya data Unit 1; file risk_modelling (ROH/redispatch)
    punya data per-unit DAN agregat 3-unit, tapi supaya konsisten dgn 2 sumber
    data lain (flow & blending), model ini fokus ke UNIT 1 SAJA. Bisa
    diperluas ke 3 unit kalau data coal flow & blending Unit 2/3 tersedia."""
    plant: str = "PLTU Pelabuhanratu"
    unit: str = "Unit 1"
    n_units_total: int = 3  # total unit di PLTU Pelabuhanratu (350 MW x 3), referensi FKI DESTINATOR


@dataclass
class CFMConfig:
    """Coal Flow Maximum (CFM) -- ambang derating akibat keterbatasan coal
    flow, DIBERIKAN LANGSUNG OLEH PENGGUNA (bukan hasil kalibrasi statistik).
    """
    cfm_threshold_tph: float = 225.0  # [STATED BY USER] batas Coal Flow Maximum, T/h
    # dari histori 14.439 pembacaan valid (Jan'25-Agu'26): 8,8% jam mencapai/lewati
    # ambang ini -- event rate sehat utk label risiko biner (jauh dari kasus
    # 1-episode NOx di PLTU Jeranjang).


@dataclass
class SFCConfig:
    """Specific Fuel Consumption -- target & mekanisme penurunan target CV
    blending, SESUAI KETENTUAN YANG DIBERIKAN PENGGUNA."""
    sfc_target_kg_per_kwh: float = 0.7551  # [STATED BY USER]
    # "Target CV blending adalah yg tercapai SFC-nya" [STATED BY USER] --
    # artinya CV_blending_target BUKAN angka tetap, melainkan level CV yang
    # (berdasar hubungan empiris SFC~CV dari data riil blending+beban)
    # menghasilkan SFC = target di atas. Dihitung di sensitivity_fit.py,
    # BUKAN diberikan sbg konstanta di sini.


@dataclass
class FuelPriceConfig:
    """Harga batubara PER JENIS, SESUAI KETENTUAN YANG DIBERIKAN PENGGUNA --
    MRC & LRC kini py harga BERBEDA (menggantikan versi awal yg cuma py 1
    harga seragam & tak bisa menciptakan trade-off ekonomi riil)."""
    mrc_price_rp_per_kg: float = 1238.0  # [STATED BY USER]
    lrc_price_rp_per_kg: float = 1100.0  # [STATED BY USER]


@dataclass
class SafetyStockConfig:
    """Stok minimum (Hari Operasi Penuh, HOP) per jenis batubara,
    SESUAI KETENTUAN YANG DIBERIKAN PENGGUNA."""
    mrc_min_days: float = 5.0   # [STATED BY USER]
    lrc_min_days: float = 10.0  # [STATED BY USER]


@dataclass
class DataQualityConfig:
    """Penanganan data hilang/rusak, didokumentasikan eksplisit -- lihat
    src/data_loader.py utk implementasi."""
    # Kode error DCS yg ditemukan di data Total Coal Flow (bukan angka valid):
    dcs_error_codes: tuple = ("I/O Timeout", "No Data", "Arc Off-line", "Configure")
    # Stok HOP bulan Nov & Des 2025 KOSONG total di sumber data (dikonfirmasi:
    # tak ada blok bulan itu sama sekali di sheet) -- [STATED BY USER] diganti
    # dgn asumsi/prediksi, BUKAN dibiarkan kosong. Metode: proyeksi dari rerata
    # 3 bulan sebelum (Ags-Okt 2025) + tren linear, diberi flag is_imputed=True
    # supaya bisa dibedakan dari data riil di semua analisis downstream.
    hop_missing_months: tuple = ("2025-11", "2025-12")


@dataclass
class ValidationConfig:
    """Kriteria PENERIMAAN hasil validasi model -- sama persis dgn pola
    PLTU Jeranjang (Hosmer & Lemeshow, 2013 utk ambang ROC-AUC; Lewis, 1982
    utk skala MAPE)."""
    roc_auc_good: float = 0.70
    roc_auc_acceptable: float = 0.60
    pr_auc_skill_ratio_min: float = 1.5
    mape_pct_good: float = 10.0
    mape_pct_acceptable: float = 20.0
    mape_pct_reasonable: float = 50.0
    r2_good: float = 0.5
    r2_acceptable: float = 0.2
    min_test_events: int = 10


THRESH_CFM = CFMConfig()
SFC = SFCConfig()
FUEL_PRICE = FuelPriceConfig()
SAFETY_STOCK = SafetyStockConfig()
DQ = DataQualityConfig()
SCOPE = ScopeConfig()
VALID = ValidationConfig()
