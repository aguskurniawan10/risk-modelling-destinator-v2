"""
layer3_optimization.py -- Cari rasio_mrc* yang meminimalkan CVaR95 Total Risk
Cost, dgn batasan safety stock (rasio_mrc tak boleh membuat stok agregat lebih
sering di bawah ambang aman gabungan).
"""
from __future__ import annotations
import pandas as pd

from . import config as C


def optimize_ratio(grid_df: pd.DataFrame, max_prob_stock: float = 0.5) -> dict:
    """r* = argmin CVaR95 Total Risk Cost, dgn batasan prob_stock_below_safety
    <= max_prob_stock (jangan pilih rasio yg scr sistematis membuat stok
    nyaris selalu di bawah aman)."""
    feasible = grid_df[grid_df["prob_stock_below_safety"] <= max_prob_stock]
    fully_feasible = len(feasible) > 0
    pool = feasible if fully_feasible else grid_df
    best = pool.loc[pool["cvar95_cost_rp"].idxmin()]

    return {
        "rasio_mrc_star_pct": float(best["rasio_mrc_pct"]),
        "rasio_lrc_star_pct": float(best["rasio_lrc_pct"]),
        "cvar95_cost_rp": float(best["cvar95_cost_rp"]),
        "mean_cost_rp": float(best["mean_cost_rp"]),
        "prob_cfm_exceed": float(best["prob_cfm_exceed"]),
        "prob_stock_below_safety": float(best["prob_stock_below_safety"]),
        "mean_sfc_pred": float(best["mean_sfc_pred"]),
        "mean_cv_blend": float(best["mean_cv_blend"]),
        "mean_bpp_rp_per_kwh": float(best["mean_bpp_rp_per_kwh"]),
        "mean_bpp_dgn_risiko_rp_per_kwh": float(best["mean_bpp_dgn_risiko_rp_per_kwh"]),
        "fully_feasible": fully_feasible,
    }


def recommend_action(opt_result: dict, sfc_model_summary: dict) -> str:
    gap_sfc = opt_result["mean_sfc_pred"] - C.SFC.sfc_target_kg_per_kwh
    gap_txt = (f"SUDAH mendekati target ({gap_sfc:+.3f} kg/kWh)" if abs(gap_sfc) < 0.02
               else f"masih ada gap {gap_sfc:+.3f} kg/kWh dari target -- TERUTAMA "
                    f"dipengaruhi level BEBAN, bukan pilihan blend (lihat catatan sensitivitas)")
    return (
        f"Rasio blending optimal: {opt_result['rasio_mrc_star_pct']:.0f}% MRC : "
        f"{opt_result['rasio_lrc_star_pct']:.0f}% LRC (CV blend rata-rata "
        f"~{opt_result['mean_cv_blend']:.0f} kcal/kg). Estimasi SFC pada kombinasi ini "
        f"~{opt_result['mean_sfc_pred']:.3f} kg/kWh -- {gap_txt}. "
        f"P(CFM exceed 2 jam ke depan)={opt_result['prob_cfm_exceed']*100:.1f}%, "
        f"P(stok di bawah aman 7 hari ke depan)={opt_result['prob_stock_below_safety']*100:.1f}%. "
        f"BPP (komponen bahan bakar) \u2248 Rp{opt_result['mean_bpp_rp_per_kwh']:.0f}/kWh; "
        f"BPP+kontinjensi risiko operasional \u2248 "
        f"Rp{opt_result['mean_bpp_dgn_risiko_rp_per_kwh']:.0f}/kWh."
    )
