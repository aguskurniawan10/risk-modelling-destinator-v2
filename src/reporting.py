"""
reporting.py -- Figure PNG siap pakai utk dashboard & PPT (gaya visual
konsisten dgn proyek PLTU Jeranjang: navy/teal, matplotlib).
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from . import config as C

NAVY = "#0B2E4E"
TEAL = "#0E7C86"
GOLD = "#D4A017"
RED = "#C0392B"
GREY = "#9AA5B1"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "axes.edgecolor": "#D9E2E8", "axes.grid": True,
    "grid.color": "#EDF2F5", "grid.linewidth": 0.8, "axes.titlesize": 13,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def _save(fig, name):
    path = C.FIGURE_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def fig_label_event_rate(label_summary: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    labels = {"y_cfm_exceed": "Coal Flow\n>= CFM (225 t/h)", "y_stock_below_safety": "Stok < Ambang\nAman Gabungan"}
    x = [labels.get(t, t) for t in label_summary["target"]]
    colors = [TEAL, NAVY]
    bars = ax.bar(x, label_summary["event_rate_pct"], color=colors, width=0.5)
    for b, v in zip(bars, label_summary["event_rate_pct"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.1f}%", ha="center", fontweight="bold")
    ax.set_ylabel("Probabilitas kejadian (%)")
    ax.set_title("Frekuensi Kejadian Risiko - Data Historis (20 bulan)")
    fig.tight_layout()
    return _save(fig, "01_label_event_rate")


def fig_cost_vs_ratio(grid_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = grid_df["rasio_mrc_pct"]
    ax.plot(x, grid_df["mean_cost_rp"] / 1e6, "o-", color=TEAL, label="Total Cost rata-rata")
    ax.plot(x, grid_df["cvar95_cost_rp"] / 1e6, "o-", color=RED, label="CVaR95 (5% skenario terburuk)")
    ax.set_xlabel("Rasio MRC dlm Blending (%)")
    ax.set_ylabel("Total Cost per Jam (Rp Juta)")
    ax.set_title("Layer 3 - Total Cost vs Rasio Blending MRC:LRC")
    ax.legend(frameon=False)
    fig.tight_layout()
    return _save(fig, "02_cost_vs_ratio")


def fig_prob_vs_ratio(grid_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = grid_df["rasio_mrc_pct"]
    ax.plot(x, grid_df["prob_cfm_exceed"] * 100, "o-", color=TEAL, label="P(CFM exceed)")
    ax.plot(x, grid_df["prob_stock_below_safety"] * 100, "o-", color=NAVY, label="P(Stok < aman)")
    ax.set_xlabel("Rasio MRC dlm Blending (%)")
    ax.set_ylabel("Probabilitas Risiko (%)")
    ax.set_title("Sensitivitas Probabilitas Risiko thd Rasio Blending")
    ax.legend(frameon=False)
    fig.tight_layout()
    return _save(fig, "03_prob_vs_ratio")


def fig_benchmarking(bench_df: pd.DataFrame, target: str):
    sub = bench_df[bench_df["target"] == target]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(sub))
    w = 0.35
    ax.bar(x - w / 2, sub["roc_auc"], w, label="ROC-AUC", color=NAVY)
    ax.bar(x + w / 2, sub["pr_auc"], w, label="PR-AUC", color=TEAL)
    ax.axhline(0.5, color=GREY, ls=":", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([p.split(". ")[1] if ". " in p else p for p in sub["pendekatan"]],
                        rotation=15, ha="right", fontsize=8)
    ax.set_title(f"Benchmarking - {target}")
    ax.legend(frameon=False)
    fig.tight_layout()
    return _save(fig, f"04_benchmarking_{target}")


def fig_walk_forward(wf_df: pd.DataFrame, target: str):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(wf_df))
    ax.plot(x, wf_df["roc_auc"], "o-", color=TEAL, label="ROC-AUC")
    ax.plot(x, wf_df["pr_auc"], "s-", color=GOLD, label="PR-AUC")
    ax.axhline(0.5, color=GREY, ls=":", lw=1)
    ax.set_xticks(list(x))
    ax.set_xticklabels(wf_df["test_month"], rotation=60, fontsize=7)
    ax.set_title(f"Walk-Forward Testing - {target} (retrain tiap bulan)")
    ax.legend(frameon=False)
    fig.tight_layout()
    return _save(fig, f"05_walk_forward_{target}")


def fig_calibration(calib_df: pd.DataFrame, target: str):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "--", color=GREY, label="Kalibrasi sempurna")
    ax.plot(calib_df["mean_predicted_prob"], calib_df["observed_frequency"], "o-", color=TEAL,
            label="Model (XGBoost)")
    ax.set_xlabel("Probabilitas rata-rata yang diprediksi")
    ax.set_ylabel("Frekuensi kejadian aktual")
    ax.set_title(f"Calibration Test - {target}")
    ax.legend(frameon=False)
    fig.tight_layout()
    return _save(fig, f"06_calibration_{target}")


def fig_sfc_vs_load(df: pd.DataFrame, sfc_model: dict):
    sub = df[df["roh_data_available"] & df["coal_flow_tph"].notna() & (df["beban_netto_mw"] > 10)].copy()
    sub["sfc_realized"] = sub["coal_flow_tph"] / sub["beban_netto_mw"]
    sub = sub[(sub["sfc_realized"] > 0.2) & (sub["sfc_realized"] < 2.0)]
    fig, ax = plt.subplots(figsize=(7, 4.8))
    sc = ax.scatter(sub["beban_netto_mw"], sub["sfc_realized"], c=sub["cv_blend_kcal_kg"],
                     cmap="viridis", s=10, alpha=0.6)
    ax.axhline(C.SFC.sfc_target_kg_per_kwh, color=RED, ls="--", lw=1.5,
               label=f"Target SFC {C.SFC.sfc_target_kg_per_kwh:.4f} kg/kWh")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("CV Blend (kcal/kg)")
    ax.set_xlabel("Beban Netto (MW)")
    ax.set_ylabel("SFC Realized (kg/kWh)")
    ax.set_title("SFC vs Beban (warna = CV Blend) -- efek beban jauh dominan")
    ax.legend(frameon=False)
    fig.tight_layout()
    return _save(fig, "07_sfc_vs_load")


def fig_risk_before_after(grid_df: pd.DataFrame, opt_rasio_mrc_pct: float):
    """Dumbbell chart: skenario agresif (0% MRC, kejar cost rendah tanpa
    model) vs rekomendasi model (rasio_mrc*)."""
    worst = grid_df.iloc[0]  # 0% MRC
    best = grid_df.loc[grid_df["rasio_mrc_pct"] == opt_rasio_mrc_pct].iloc[0]
    metrics = [
        ("P(CFM exceed) %", worst["prob_cfm_exceed"] * 100, best["prob_cfm_exceed"] * 100),
        ("P(Stok < aman) %", worst["prob_stock_below_safety"] * 100, best["prob_stock_below_safety"] * 100),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    for i, (label, w, b) in enumerate(metrics):
        ax.plot([w, b], [i, i], color=GREY, lw=2, zorder=1)
        ax.scatter([w], [i], color=RED, s=140, zorder=2, label="0% MRC (tanpa model)" if i == 0 else None)
        ax.scatter([b], [i], color="#1E7A46", s=140, zorder=2,
                   label=f"{opt_rasio_mrc_pct:.0f}% MRC (rekomendasi model)" if i == 0 else None)
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels([m[0] for m in metrics])
    ax.set_xlabel("Probabilitas kejadian risiko (%)")
    ax.set_title("Risk Map - 0% MRC vs Rekomendasi Model")
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    fig.tight_layout()
    return _save(fig, "08_risk_before_after")


def fig_cost_distribution(sim: pd.DataFrame, rasio_mrc_pct: float):
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    cost = sim["total_cost_rp"] / 1e6
    ax.hist(cost, bins=40, color=TEAL, alpha=0.85)
    p50, p95 = np.percentile(cost, 50), np.percentile(cost, 95)
    cvar95 = cost[cost >= p95].mean()
    ax.axvline(p50, color=NAVY, ls="--", lw=1.5, label=f"P50 = {p50:,.0f} Jt")
    ax.axvline(cvar95, color=RED, ls="-", lw=1.5, label=f"CVaR95 = {cvar95:,.0f} Jt")
    ax.set_xlabel("Total Cost per Jam (Rp Juta)")
    ax.set_ylabel("Jumlah skenario Monte Carlo")
    ax.set_title(f"Distribusi Total Cost pada Rasio MRC = {rasio_mrc_pct:.0f}%")
    ax.legend(frameon=False)
    fig.tight_layout()
    return _save(fig, "09_cost_distribution")


def fig_load_forecast_comparison(algo_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    algo_df = algo_df.sort_values("mape_pct")
    colors = [GOLD if b else TEAL for b in algo_df["is_best"]]
    bars = ax.barh(algo_df["algoritma"], algo_df["mape_pct"], color=colors)
    for b, v in zip(bars, algo_df["mape_pct"]):
        ax.text(v + 0.3, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center", fontsize=9)
    ax.axvline(5, color=RED, ls="--", lw=1.5, label="Target <5%")
    ax.set_xlabel("MAPE Test Set (%)")
    ax.set_title("Perbandingan Algoritma Load Forecast (H+168 jam)")
    ax.legend(frameon=False)
    fig.tight_layout()
    return _save(fig, "10_load_forecast_algo_comparison")


def fig_roh_redispatch_gap(df: pd.DataFrame, load_forecast_col: str = "beban_forecast_168h_mw"):
    """Bandingkan ROH (rencana), Realisasi Beban Netto (beban_netto_mw aktual),
    & Forecast model pd jendela waktu yg ROH-nya tersedia -- menunjukkan
    scr visual seberapa besar gap ROH-vs-Redispatch yg jadi alasan
    DESTINATOR dibangun, dibanding akurasi forecast model."""
    sub = df[df["roh_data_available"]].copy().sort_values("time")
    if len(sub) > 200:
        sub = sub.tail(200)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(sub["time"], sub["roh_mw"], "o-", color=GOLD, label="ROH (rencana P2B)", ms=3, lw=1)
    ax.plot(sub["time"], sub["beban_netto_mw"], "o-", color=NAVY, label="Realisasi Beban Netto (aktual)", ms=3, lw=1)
    if load_forecast_col in sub.columns:
        ax.plot(sub["time"], sub[load_forecast_col], "o-", color=TEAL,
                 label="Forecast Model (H+168 jam)", ms=3, lw=1, alpha=0.8)
    ax.set_ylabel("Beban (MW)")
    ax.set_title("Gap ROH vs Realisasi Beban Netto vs Forecast Model (jendela waktu py data ROH)")
    ax.legend(frameon=False, fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    return _save(fig, "11_roh_redispatch_forecast_gap")
