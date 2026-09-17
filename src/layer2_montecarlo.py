"""
layer2_montecarlo.py -- Simulasi Monte Carlo utk skenario rasio blending
MRC:LRC, dibangun DI ATAS model Layer 1 yg sudah dilatih (konsisten dgn
prinsip PLTU Jeranjang: Layer 2 bukan mesin probabilitas terpisah, tapi
menjalankan Layer 1 pada kondisi ter-bootstrap + variabel keputusan
di-override).

ASUMSI BIAYA DAMPAK [PENTING -- BELUM ADA DI DATA YANG DIBERIKAN]: tidak ada
kolom harga batubara/biaya dampak di 5 sumber data DESTINATOR (beda dgn PLTU
Jeranjang yg py "Harga Batubara Rp/kcal"). Nilai impact_* di bawah HANYA
placeholder berskala wajar (mengikuti proporsi dampak Layer1 PLTU Jeranjang)
DAN WAJIB dikalibrasi ulang dgn angka riil PLTU Pelabuhanratu sebelum dipakai
sbg dasar keputusan finansial -- ditandai jelas di summary/dashboard.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from . import config as C
from .features import FEATURE_COLUMNS
from . import layer1_predictive as L1
from . import sensitivity_fit as SF

RASIO_MRC_GRID = tuple(round(x, 2) for x in np.arange(0.0, 1.01, 0.05))  # 0%-100% MRC, step 5pp
# (step 5pp, BUKAN 10pp -- SAMA PERSIS dgn step slider dashboard, spy KPI "live"
# bisa lookup langsung dari grid ini tanpa perlu simulasi terpisah & tanpa
# selisih akibat RNG state berbeda antara live-sim vs scanner grid)

# [ASUMSI -- perlu kalibrasi dgn data riil PLTU Pelabuhanratu]
IMPACT_CFM_EXCEED_RP = 150_000_000     # Rp per kejadian derating akibat CFM
IMPACT_STOCK_BELOW_SAFETY_RP = 300_000_000  # Rp per kejadian (risiko stockout + start-up ulang)
N_SCENARIOS_DEFAULT = 4000


def prepare_sim_pool(df: pd.DataFrame) -> pd.DataFrame:
    """Basis bootstrap Layer 2 -- KINI beban_netto_mw PADAT 100% (sheet
    redispatch), jadi TIDAK lagi dibatasi ke subset roh_data_available spt
    versi sebelumnya (yg cuma ~12% jam) -- seluruh 14.592 jam bisa dipakai."""
    return df[df["beban_netto_mw"].notna() & (df["beban_netto_mw"] > 10)].reset_index(drop=True)


def run_scenarios_for_ratio(sim_pool: pd.DataFrame, coal_gcv: pd.DataFrame, sfc_model: dict,
                             layer1_results: dict, rasio_mrc: float, n_scenarios: int,
                             rng: np.random.Generator, beban_override_mw: float | None = None) -> pd.DataFrame:
    idx = rng.integers(0, len(sim_pool), size=n_scenarios)
    sim = sim_pool.iloc[idx].reset_index(drop=True).copy()

    # Beban Gross [VARIABEL KEPUTUSAN, opsional]: jika diberikan, timpa
    # beban_netto_mw hasil bootstrap dgn nilai TETAP pilihan pengguna --
    # dipakai utk simulasi "bagaimana kalau beban di level X MW", bukan
    # cuma mengandalkan distribusi historis. Jika None (default), beban
    # tetap dari bootstrap sim_pool spt biasa.
    if beban_override_mw is not None:
        sim["beban_netto_mw"] = float(beban_override_mw)

    mrc_pool = coal_gcv.loc[coal_gcv["jenis"] == "MRC", "gcv_arb"].values
    lrc_pool = coal_gcv.loc[coal_gcv["jenis"] == "LRC", "gcv_arb"].values
    cv_mrc_sample = rng.choice(mrc_pool, size=n_scenarios, replace=True)
    cv_lrc_sample = rng.choice(lrc_pool, size=n_scenarios, replace=True)

    sim["rasio_mrc"] = rasio_mrc
    sim["rasio_lrc"] = 1.0 - rasio_mrc
    sim["cv_blend_kcal_kg"] = rasio_mrc * cv_mrc_sample + (1 - rasio_mrc) * cv_lrc_sample

    resid_sample = rng.choice(sfc_model["resid"], size=n_scenarios, replace=True)
    sfc_pred = (sfc_model["intercept"] + sfc_model["slope_cv"] * sim["cv_blend_kcal_kg"]
                + sfc_model["slope_beban"] * sim["beban_netto_mw"] + resid_sample)
    sfc_pred = np.clip(sfc_pred, 0.3, 1.8)
    sim["coal_flow_tph"] = sfc_pred * sim["beban_netto_mw"]

    # fitur turunan yg bergantung coal_flow_tph dihitung ulang scr sederhana
    # (level baru dipakai sbg proxy rolling/lag -- pola konsisten dgn
    # simplifikasi Layer 2 PLTU Jeranjang: konteks historis dipertahankan,
    # hanya titik keputusan yg disesuaikan)
    sim["coal_flow_roll_mean_3h"] = sim["coal_flow_tph"]
    sim["coal_flow_roll_max_3h"] = sim["coal_flow_tph"]
    sim["coal_flow_lag1"] = sim_pool.iloc[idx]["coal_flow_tph"].values
    sim["coal_flow_delta_1h"] = sim["coal_flow_tph"] - sim["coal_flow_lag1"]

    combined_safety = sim["rasio_mrc"] * C.SAFETY_STOCK.mrc_min_days + \
                       sim["rasio_lrc"] * C.SAFETY_STOCK.lrc_min_days
    sim["hop_below_combined_safety"] = (sim["hop_hari_inventory"] < combined_safety).astype(float)

    # boros_mrc_indicator dihitung ULANG dgn rasio_mrc skenario (bukan rasio
    # historis baris asal) -- beban_netto_mw tetap dari konteks ter-bootstrap.
    p33_beban = sim_pool["beban_netto_mw"].quantile(0.33)
    is_low_load = sim["beban_netto_mw"] < p33_beban
    sim["boros_mrc_indicator"] = np.where(is_low_load, sim["rasio_mrc"], 0.0)

    X = sim[FEATURE_COLUMNS]
    sim["prob_cfm_exceed"] = L1.predict_proba(layer1_results["y_cfm_exceed"], X, "xgb")
    sim["prob_stock_below_safety"] = L1.predict_proba(layer1_results["y_stock_below_safety"], X, "xgb")

    # Biaya bahan bakar riil [STATED BY USER: MRC Rp 1.238/kg, LRC Rp 1.100/kg]
    # -- KINI berbeda per jenis, jadi menciptakan trade-off ekonomi RIIL
    # (bukan corner solution spt versi awal yg cuma py 1 harga): rasio MRC
    # tinggi = CV lbh baik (turunkan coal flow & risiko) TAPI harga per kg
    # lbh mahal -- kedua efek berlawanan arah, dihitung per JAM konsumsi.
    price_per_kg = (sim["rasio_mrc"] * C.FUEL_PRICE.mrc_price_rp_per_kg
                     + sim["rasio_lrc"] * C.FUEL_PRICE.lrc_price_rp_per_kg)
    sim["fuel_cost_rp_per_hour"] = sim["coal_flow_tph"] * 1000.0 * price_per_kg
    sim["risk_cost_rp"] = (sim["prob_cfm_exceed"] * IMPACT_CFM_EXCEED_RP
                            + sim["prob_stock_below_safety"] * IMPACT_STOCK_BELOW_SAFETY_RP)
    sim["total_cost_rp"] = sim["risk_cost_rp"] + sim["fuel_cost_rp_per_hour"]
    sim["sfc_pred"] = sfc_pred

    # --- BPP (Biaya Pokok Produksi), Rp/kWh -- risiko baru [STATED BY USER] ---
    # BPP = Total Cost per jam / energi yg dihasilkan per jam (beban_MW x 1000 kWh).
    # Dihitung dari fuel_cost SAJA (BPP konvensional = komponen bahan bakar,
    # bukan termasuk risk_cost yg sifatnya probabilistik/kontinjensi) --
    # dipisah dari bpp_dgn_risiko yg memasukkan estimasi risk_cost jg, sbg
    # pembanding "BPP + kontinjensi risiko operasional".
    energi_kwh = sim["beban_netto_mw"] * 1000.0
    sim["bpp_rp_per_kwh"] = sim["fuel_cost_rp_per_hour"] / energi_kwh
    sim["bpp_dgn_risiko_rp_per_kwh"] = sim["total_cost_rp"] / energi_kwh
    return sim


def summarize_ratio(sim: pd.DataFrame, rasio_mrc: float) -> dict:
    cost = sim["total_cost_rp"].values
    risk_only = sim["risk_cost_rp"].values
    return {
        "rasio_mrc_pct": round(rasio_mrc * 100, 1),
        "rasio_lrc_pct": round((1 - rasio_mrc) * 100, 1),
        "mean_beban_mw": float(sim["beban_netto_mw"].mean()),
        "mean_cost_rp": float(np.mean(cost)),
        "mean_risk_cost_rp": float(np.mean(risk_only)),
        "mean_fuel_cost_rp_per_hour": float(sim["fuel_cost_rp_per_hour"].mean()),
        "p50_cost_rp": float(np.percentile(cost, 50)),
        "p95_cost_rp": float(np.percentile(cost, 95)),
        "cvar95_cost_rp": float(cost[cost >= np.percentile(cost, 95)].mean()),
        "prob_cfm_exceed": float(sim["prob_cfm_exceed"].mean()),
        "prob_stock_below_safety": float(sim["prob_stock_below_safety"].mean()),
        "mean_sfc_pred": float(sim["sfc_pred"].mean()),
        "mean_cv_blend": float(sim["cv_blend_kcal_kg"].mean()),
        "mean_bpp_rp_per_kwh": float(sim["bpp_rp_per_kwh"].mean()),
        "mean_bpp_dgn_risiko_rp_per_kwh": float(sim["bpp_dgn_risiko_rp_per_kwh"].mean()),
    }


def run_full_grid(df: pd.DataFrame, layer1_results: dict, n_scenarios: int = N_SCENARIOS_DEFAULT,
                   seed: int = C.RANDOM_STATE, beban_override_mw: float | None = None) -> pd.DataFrame:
    sim_pool = prepare_sim_pool(df)
    coal_gcv = SF.get_coal_type_gcv()
    sfc_model = SF.fit_sfc_model(df)
    rng = np.random.default_rng(seed)

    rows = []
    for rasio_mrc in RASIO_MRC_GRID:
        sim = run_scenarios_for_ratio(sim_pool, coal_gcv, sfc_model, layer1_results,
                                       rasio_mrc, n_scenarios, rng, beban_override_mw=beban_override_mw)
        rows.append(summarize_ratio(sim, rasio_mrc))
    return pd.DataFrame(rows)
