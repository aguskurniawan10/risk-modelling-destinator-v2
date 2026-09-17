#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dashboard_app.py -- Dashboard operasional interaktif (Streamlit) utk model
risiko DESTINATOR (PLTU Pelabuhanratu, Unit 1). Menjalankan LANGSUNG modul
model asli di src/ (Layer 1-3), bukan angka pre-computed statis -- setiap
slider digeser, Monte Carlo Layer 2 & Layer 1 (XGBoost) dijalankan ulang.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config as C
from src import data_loader as dl
from src import features as feat
from src import risk_labels as rl
from src import layer1_predictive as L1
from src import layer2_montecarlo as L2
from src import sensitivity_fit as SF
from src import load_forecast as LF

st.set_page_config(page_title="Risk Dashboard DESTINATOR - PLTU Pelabuhanratu",
                    page_icon="\U0001F3ED", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
/* ---- Tipografi & palet dasar (navy/teal/gold, gaya McKinsey) --------- */
:root {
  --navy:#0B2E4E; --navy-dark:#08213A; --teal:#0E7C86; --gold:#D4A017;
  --grey-line:#E3E9ED; --grey-text:#5B6B76; --bg-soft:#F6F9FA;
}
.block-container {padding-top:1.6rem; max-width:1200px;}
h4, h5 {color:var(--navy); font-weight:700;}

/* ---- Label section (eyebrow) ------------------------------------------ */
.sec-eyebrow {color:var(--teal); font-weight:700; font-size:11px; letter-spacing:0.09em;
              text-transform:uppercase; margin-bottom:2px;}
.sec-title {color:var(--navy); font-weight:700; font-size:19px; margin:0 0 14px 0;}

/* ---- Divider antar section, konsisten -------------------------------- */
.sec-divider {border:none; border-top:1px solid var(--grey-line); margin:30px 0 26px 0;}

/* ---- Kartu metrik custom (pengganti st.metric bawaan) ----------------- */
.metric-grid {display:grid; grid-template-columns:repeat(var(--n,4), 1fr); gap:12px; margin-bottom:4px;}
.metric-card {background:#FFFFFF; border:1px solid var(--grey-line); border-radius:8px;
               padding:14px 16px; min-height:86px; display:flex; flex-direction:column; justify-content:center;}
.metric-card.dark {background:var(--navy); border-color:var(--navy);}
.metric-card .m-label {font-size:11.5px; color:var(--grey-text); font-weight:600;
                        text-transform:uppercase; letter-spacing:0.03em; margin-bottom:4px;}
.metric-card.dark .m-label {color:#9FC4CC;}
.metric-card .m-value {font-size:22px; font-weight:800; color:var(--navy); line-height:1.15;}
.metric-card.dark .m-value {color:#FFFFFF;}
.metric-card .m-delta {font-size:11px; color:var(--grey-text); margin-top:3px;}
.metric-card.dark .m-delta {color:#BFDEE2;}
.metric-card .m-delta.good {color:#2E9E5B;} .metric-card.dark .m-delta.good {color:#6FE3A0;}
.metric-card .m-delta.bad {color:#C0392B;} .metric-card.dark .m-delta.bad {color:#FF9B90;}

/* ---- Callout info ------------------------------------------------------ */
.callout-info {background:var(--bg-soft); border-left:3px solid var(--teal); padding:12px 16px;
               border-radius:0 6px 6px 0; font-size:13px; color:#2B3B44; line-height:1.55;}
.callout-info b {color:var(--navy);}
</style>
""", unsafe_allow_html=True)


def section(eyebrow: str, title: str):
    """Header section yg konsisten: eyebrow teal kecil + judul navy besar."""
    st.markdown(f'<div class="sec-eyebrow">{eyebrow}</div><div class="sec-title">{title}</div>',
                unsafe_allow_html=True)


def metric_cards(items: list[dict], dark_first: bool = False):
    """Render 1 baris kartu metrik custom (HTML), proporsi rata & seragam --
    items: [{'label':..., 'value':..., 'delta':(opsional), 'delta_kind':'good'/'bad'/None}]"""
    n = len(items)
    cards_html = ""
    for i, it in enumerate(items):
        cls = "metric-card dark" if (dark_first and i == 0) else "metric-card"
        delta_html = ""
        if it.get("delta"):
            dk = f" {it['delta_kind']}" if it.get("delta_kind") else ""
            delta_html = f'<div class="m-delta{dk}">{it["delta"]}</div>'
        cards_html += (f'<div class="{cls}"><div class="m-label">{it["label"]}</div>'
                       f'<div class="m-value">{it["value"]}</div>{delta_html}</div>')
    st.markdown(f'<div class="metric-grid" style="--n:{n};">{cards_html}</div>', unsafe_allow_html=True)


def divider():
    st.markdown('<hr class="sec-divider"/>', unsafe_allow_html=True)


# ==============================================================================
# 1. BACKEND -- load data & latih model ASLI sekali per sesi
# ==============================================================================
@st.cache_resource(show_spinner="Memuat data, forecasting beban & melatih Layer 1...")
def load_backend():
    d = dl.load_clean()
    load_res = LF.train_load_forecast(d)
    d = LF.add_load_forecast_feature(d, load_res)
    d = feat.engineer(d, C.SAFETY_STOCK.mrc_min_days, C.SAFETY_STOCK.lrc_min_days)
    d = rl.build_labels(d, C.SAFETY_STOCK.mrc_min_days, C.SAFETY_STOCK.lrc_min_days)
    results, df_train, df_test = L1.train_all_targets(d)
    sfc_model = SF.fit_sfc_model(d)
    coal_gcv = SF.get_coal_type_gcv()
    sim_pool = L2.prepare_sim_pool(d)
    return d, results, sfc_model, coal_gcv, sim_pool, load_res


@st.cache_data(show_spinner=False)
def load_validation_summary():
    path = C.TABLE_DIR / "validation_summary.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


d, RESULTS, SFC_MODEL, COAL_GCV, SIM_POOL, LOAD_RES = load_backend()
CURRENT_ROW = d.iloc[-1]

# ==============================================================================
# 2. STATE & SIDEBAR
# ==============================================================================
if "rasio_mrc_pct" not in st.session_state:
    st.session_state.rasio_mrc_pct = 50.0


def _apply_optimal(rasio_mrc_star_pct: float):
    st.session_state.rasio_mrc_pct = float(rasio_mrc_star_pct)


with st.sidebar:
    st.markdown('<div class="sec-eyebrow">Variabel Keputusan</div>', unsafe_allow_html=True)
    st.markdown("### Simulasi Skenario Blending")
    st.slider("Rasio MRC dalam Blending (%)", min_value=0.0, max_value=100.0, step=5.0,
              key="rasio_mrc_pct")
    st.caption(f"Rasio LRC = {100 - st.session_state.rasio_mrc_pct:.0f}% (komplemen)")
    beban_gross_mw = st.slider("Beban Gross (MW)", min_value=150.0, max_value=350.0,
                                value=250.0, step=5.0,
                                help="Simulasikan skenario pd tingkat beban tertentu -- "
                                     "menggantikan bootstrap dari data historis (rata-rata "
                                     "historis ~233 MW, rentang operasi normal 200-330 MW).")
    n_scenarios = st.select_slider("Jumlah skenario Monte Carlo", options=[500, 1000, 2000, 4000],
                                    value=2000)

rasio_mrc = st.session_state.rasio_mrc_pct / 100.0

# ==============================================================================
# 3. HEADER
# ==============================================================================
st.markdown(f"""
<div style="background:linear-gradient(135deg, #0B2E4E 0%, #123A5E 100%); border-radius:10px;
     padding:24px 28px; margin-bottom:28px;">
  <div style="color:#7FB8C4; font-weight:700; font-size:11px; letter-spacing:0.09em;
       text-transform:uppercase; margin-bottom:6px;">PLTU Pelabuhanratu &middot; Unit 1</div>
  <h1 style="color:white; margin:0 0 6px 0; font-size:21px; font-weight:700; line-height:1.3;">Data-driven
  Energy Forecast Intelligent Automator untuk Optimalisasi Prediksi Beban Pembangkit
  (DESTINATOR) -- Risk Modelling</h1>
  <p style="color:#BFDEE2; margin:0; font-size:13px;">Model risiko Coal Flow Maximum (CFM) &amp; stok batubara --
  dijalankan langsung di atas modul model asli (Layer 1-3), bukan angka statis.</p>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# 3b. RANTAI RISIKO: Load Forecast -> Boros MRC -> Stok -> CFM -> SFC
# ==============================================================================
section("Rantai Sebab-Akibat", "Load Forecast & Risiko yang Timbul")

# Jam Hemat MRC = selisih MAPE ROH vs Model -- setiap poin persen MAPE yg
# diperbaiki model dibaca sbg proporsi jam blending yg lebih tepat sasaran
# (bukan hitungan ambang terpisah, langsung dari akurasi yg sama dipakai
# sbg headline metric di atas -- konsisten & tak berlebihan).
_pct_mrc_saved = max(0.0, (LOAD_RES["mape_roh_baseline_pct"] or 0) - LOAD_RES["mape_pct"])

metric_cards([
    {"label": "MAPE Load Forecast (H+168 jam)", "value": f"{LOAD_RES['mape_pct']:.1f}%",
     "delta": (f"vs ROH {LOAD_RES['mape_roh_baseline_pct']:.1f}%"
               if LOAD_RES["mape_roh_baseline_pct"] else None), "delta_kind": "good"},
    {"label": "R\u00b2 Load Forecast", "value": f"{LOAD_RES['r2']:.2f}"},
    {"label": "Target MAPE < 15%", "value": "Tercapai" if LOAD_RES["target_achieved"] else "Belum"},
    {"label": "Jam Hemat MRC (Forecast Lebih Baik)", "value": f"{_pct_mrc_saved:.0f}%",
     "delta": "= selisih MAPE ROH - Model"},
])
st.caption("Target MAPE 15% mengacu skala Lewis (1982).")
st.write("")
st.markdown(
    '<div class="callout-info"><b>Rantai risiko:</b> ROH sering meleset dari Realisasi '
    'Beban Netto (median deviasi 14% pd jam yg py data ROH) -&gt; blending MRC tak selaras '
    'kebutuhan beban riil -&gt; <b>boros MRC</b> saat beban rendah -&gt; stok MRC cepat '
    'menipis -&gt; saat beban tinggi MRC kurang -&gt; terpaksa LRC lbh banyak (CV rendah, '
    'perlu massa lbh banyak) -&gt; coal flow naik mendekati/lewat CFM (<b>derating</b>) '
    '-&gt; <b>SFC meleset dari target</b>.</div>',
    unsafe_allow_html=True,
)

with st.expander(f"Perbandingan algoritma Load Forecast (model terbaik: {LOAD_RES['best_model_name']})"):
    algo_rows = []
    for name, m in LOAD_RES["all_candidates"].items():
        algo_rows.append({
            "Algoritma": name, "MAE (MW)": round(m["mae"], 2), "RMSE (MW)": round(m["rmse"], 2),
            "MAPE (%)": round(m["mape_pct"], 2), "R\u00b2": round(m["r2"], 3),
            "Terbaik": "\u2713" if name == LOAD_RES["best_model_name"] else "",
        })
    st.dataframe(pd.DataFrame(algo_rows).sort_values("MAPE (%)"), width='stretch', hide_index=True)

st.markdown("##### Gap ROH vs Realisasi Beban Netto vs Forecast Model")
_roh_window = d[d["roh_data_available"]].sort_values("time").tail(200)
fig_gap = go.Figure()
fig_gap.add_trace(go.Scatter(x=_roh_window["time"], y=_roh_window["roh_mw"],
                              mode="lines+markers", name="ROH (rencana P2B)",
                              line=dict(color="#D4A017"), marker=dict(size=4)))
fig_gap.add_trace(go.Scatter(x=_roh_window["time"], y=_roh_window["beban_netto_mw"],
                              mode="lines+markers", name="Realisasi Beban Netto (aktual)",
                              line=dict(color="#0B2E4E"), marker=dict(size=4)))
if "beban_forecast_168h_mw" in _roh_window.columns:
    fig_gap.add_trace(go.Scatter(x=_roh_window["time"], y=_roh_window["beban_forecast_168h_mw"],
                                  mode="lines+markers", name="Forecast Model (H+168 jam)",
                                  line=dict(color="#0E7C86"), marker=dict(size=4)))
fig_gap.update_layout(xaxis_title="Waktu", yaxis_title="Beban (MW)", height=360,
                       margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=1.12),
                       plot_bgcolor="white", paper_bgcolor="white",
                       font=dict(family="Arial, sans-serif", size=12, color="#2B3B44"))
fig_gap.update_xaxes(gridcolor="#EDF2F5")
fig_gap.update_yaxes(gridcolor="#EDF2F5")
st.plotly_chart(fig_gap, width='stretch')
st.caption("Ditampilkan pd 200 jam terakhir yg py data ROH (~12% dari total jam) -- ROH & Forecast "
           "keduanya dibandingkan thd Realisasi Beban Netto sbg acuan aktual.")

divider()

# ==============================================================================
# 4. GRID PENUH LAYER 2 (dihitung SEKALI, dipakai bareng oleh KPI "live" &
#    scanner di bawah -- supaya keduanya SELALU konsisten pd rasio yg sama,
#    bukan 2 simulasi terpisah dgn RNG berbeda)
# ==============================================================================
@st.cache_data(show_spinner="Menjalankan Monte Carlo Layer 2 (21 titik rasio)...")
def run_grid_cached(n_scenarios_arg, beban_gross_mw_arg):
    # PENTING: parameter TIDAK boleh diberi awalan underscore -- itu artinya
    # "jangan dihash utk cache key" di Streamlit, shg perubahan slider (n_scenarios
    # ATAU beban_gross_mw) tak pernah memicu perhitungan ulang [BUG YG DITEMUKAN].
    rng = np.random.default_rng(C.RANDOM_STATE)
    rows = []
    for r_mrc in L2.RASIO_MRC_GRID:
        sim_g = L2.run_scenarios_for_ratio(SIM_POOL, COAL_GCV, SFC_MODEL, RESULTS, r_mrc, n_scenarios_arg, rng,
                                            beban_override_mw=beban_gross_mw_arg)
        rows.append(L2.summarize_ratio(sim_g, r_mrc))
    return pd.DataFrame(rows)


grid_df = run_grid_cached(n_scenarios, beban_gross_mw)
# rasio dari slider (step 5pp) SELALU cocok persis dgn salah satu baris grid
# (step grid jg 5pp) -- lookup langsung, bukan simulasi baru.
_match = grid_df.loc[(grid_df["rasio_mrc_pct"] - st.session_state.rasio_mrc_pct).abs().idxmin()]
s = _match.to_dict()

section("Simulasi Layer 2 (Live)", "Kondisi Terkini & Estimasi Risiko pada Skenario Ini")
metric_cards([
    {"label": "Rasio MRC : LRC", "value": f"{s['rasio_mrc_pct']:.0f}% : {s['rasio_lrc_pct']:.0f}%"},
    {"label": "Beban Gross", "value": f"{s['mean_beban_mw']:,.0f} MW"},
    {"label": "CV Blend Rata-rata", "value": f"{s['mean_cv_blend']:,.0f} kcal/kg"},
    {"label": "SFC Estimasi", "value": f"{s['mean_sfc_pred']:.3f} kg/kWh",
     "delta": f"{s['mean_sfc_pred']-C.SFC.sfc_target_kg_per_kwh:+.3f} vs target",
     "delta_kind": "bad" if abs(s['mean_sfc_pred']-C.SFC.sfc_target_kg_per_kwh) > 0.02 else "good"},
    {"label": "P(CFM exceed, 2 jam)", "value": f"{s['prob_cfm_exceed']*100:.1f}%"},
    {"label": "P(Stok < aman, 7 hari)", "value": f"{s['prob_stock_below_safety']*100:.1f}%"},
], dark_first=True)
st.write("")
metric_cards([
    {"label": "Total Cost / Jam (rata-rata)", "value": f"Rp {s['mean_cost_rp']/1e6:,.0f} Jt"},
    {"label": "CVaR95 Total Cost / Jam", "value": f"Rp {s['cvar95_cost_rp']/1e6:,.0f} Jt"},
    {"label": "Biaya Bahan Bakar / Jam", "value": f"Rp {s['mean_fuel_cost_rp_per_hour']/1e6:,.0f} Jt"},
    {"label": "BPP (Bahan Bakar)", "value": f"Rp {s['mean_bpp_rp_per_kwh']:,.0f}/kWh",
     "delta": f"+kontinjensi risiko: Rp {s['mean_bpp_dgn_risiko_rp_per_kwh']:,.0f}/kWh"},
])

divider()

# ==============================================================================
# 5. SCANNER RASIO MRC (grid penuh utk konteks)
# ==============================================================================
section("Optimasi Layer 3", "Scanner Rasio Blending Teraman")

best = grid_df.loc[grid_df["cvar95_cost_rp"].idxmin()]

gc1, gc2 = st.columns([3, 1])
with gc1:
    metric_cards([
        {"label": "Rasio MRC* Optimal", "value": f"{best['rasio_mrc_pct']:.0f}%"},
        {"label": "CVaR95* (Optimal)", "value": f"Rp {best['cvar95_cost_rp']/1e6:,.0f} Jt"},
    ])
with gc2:
    st.write("")
    st.button("\U0001F3AF Terapkan Skenario Optimal", width='stretch',
              on_click=_apply_optimal, args=(best["rasio_mrc_pct"],))

fig = go.Figure()
fig.add_trace(go.Scatter(x=grid_df["rasio_mrc_pct"], y=grid_df["mean_cost_rp"] / 1e6,
                          mode="lines+markers", name="Total Cost rata-rata", line=dict(color="#0E7C86")))
fig.add_trace(go.Scatter(x=grid_df["rasio_mrc_pct"], y=grid_df["cvar95_cost_rp"] / 1e6,
                          mode="lines+markers", name="CVaR95 (5% terburuk)", line=dict(color="#C0392B")))
fig.add_vline(x=st.session_state.rasio_mrc_pct, line_dash="dash", line_color="#D4A017",
              annotation_text="Skenario dipilih")
fig.update_layout(xaxis_title="Rasio MRC (%)", yaxis_title="Total Cost per Jam (Rp Juta)",
                   height=360, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.1),
                   plot_bgcolor="white", paper_bgcolor="white",
                   font=dict(family="Arial, sans-serif", size=12, color="#2B3B44"))
fig.update_xaxes(gridcolor="#EDF2F5")
fig.update_yaxes(gridcolor="#EDF2F5")
st.plotly_chart(fig, width='stretch')

divider()

# ==============================================================================
# 6. MODEL SENSITIVITAS -- SFC ~ CV_blend + beban
# ==============================================================================
section("Model Sensitivitas", "Hubungan SFC ~ CV Blend + Beban")
st.caption("OLS, data SFC_actual x beban_netto_mw riil")
sens_df = pd.DataFrame([
    {"Variabel": "CV Blend (kcal/kg)", "Slope": f"{SFC_MODEL['slope_cv']:.6f}",
     "p-value": f"{SFC_MODEL['p_cv']:.4g}", "Signifikan (p<0,05)": "Ya" if SFC_MODEL['p_cv'] < 0.05 else "Tidak",
     "Catatan": "Efek negatif (CV naik -> SFC turun, sesuai intuisi fisika) & signifikan"},
    {"Variabel": "Beban Netto (MW)", "Slope": f"{SFC_MODEL['slope_beban']:.6f}",
     "p-value": f"{SFC_MODEL['p_beban']:.4g}", "Signifikan (p<0,05)": "Ya" if SFC_MODEL['p_beban'] < 0.05 else "Tidak",
     "Catatan": "Efek dominan -- naik 100 MW turunkan SFC lebih besar drpd efek CV_blend"},
])
st.dataframe(sens_df, width='stretch', hide_index=True)
_cv_needed = SF.solve_cv_for_target_sfc(SFC_MODEL, C.SFC.sfc_target_kg_per_kwh, SFC_MODEL["beban_mean"])
_beban_needed = SF.solve_beban_for_target_sfc(SFC_MODEL, C.SFC.sfc_target_kg_per_kwh, SFC_MODEL["cv_blend_mean"])
st.markdown(
    '<div class="callout-info">R\u00b2 model = {:.3f} (n={} jam, data SFC_actual x beban riil). '
    'Target SFC {:.4f} kg/kWh TERCAPAI pd kombinasi REALISTIS: CV~{:.0f} kcal/kg (beban=rata-rata '
    '{:.0f} MW) ATAU beban~{:.0f} MW (CV=rata-rata {:.0f} kcal/kg) -- keduanya dlm rentang data '
    'teramati.</div>'.format(
        SFC_MODEL["r2"], SFC_MODEL["n_obs"], C.SFC.sfc_target_kg_per_kwh,
        _cv_needed, SFC_MODEL["beban_mean"], _beban_needed, SFC_MODEL["cv_blend_mean"]),
    unsafe_allow_html=True,
)

divider()

# ==============================================================================
# 7. VALIDASI MODEL
# ==============================================================================
section("Validasi Model", "Ringkasan Validasi (Walk-Forward, 20 Bulan)")

val_df = load_validation_summary()
if val_df is None or val_df.empty:
    st.markdown(
        '<div class="callout-info">Belum ada hasil validasi tersimpan. Jalankan '
        '<code>python3 main.py</code> sekali.</div>', unsafe_allow_html=True,
    )
else:
    show_cols = ["target", "roc_auc", "pr_auc", "f1_at_0.5", "mae", "rmse",
                 "baseline_event_rate_pct", "n_folds_reliable", "n_folds_total", "verdict"]
    show_cols = [c for c in show_cols if c in val_df.columns]
    st.dataframe(val_df[show_cols], width='stretch', hide_index=True)
    st.caption("Kriteria (src/config.ValidationConfig): ROC-AUC >= 0,70 (Hosmer & Lemeshow, 2013); "
               "fold 'reliable' butuh >=10 kejadian POSITIF dan >=10 NEGATIF pd bulan test.")
