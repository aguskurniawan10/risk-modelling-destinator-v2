"""
main.py -- Orkestrasi penuh model risiko DESTINATOR (PLTU Pelabuhanratu
Unit 1): Layer 1 (predictive) -> Layer 2 (Monte Carlo blending) -> Layer 3
(optimasi rasio) -> Validasi -> Ringkasan JSON/CSV siap pakai dashboard & PPT.

Jalankan: python3 main.py
"""
from __future__ import annotations
import json
import time
import pandas as pd

from src import config as C
from src import data_loader as dl
from src import features as feat
from src import risk_labels as rl
from src import layer1_predictive as L1
from src import layer2_montecarlo as L2
from src import layer3_optimization as L3
from src import sensitivity_fit as SF
from src import validation as V
from src import reporting as R
from src import load_forecast as LF


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def main():
    C.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    C.FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 1. Data & fitur --------------------------------------------------
    log("Memuat & menggabungkan 6 sumber data...")
    df = dl.load_clean()
    df.to_csv(C.DATA_DIR / "unified_hourly_unit1.csv", index=False)

    # ---- 1b. Load Forecast -- HARUS sebelum feature
    # engineering krn beban_forecast_168h_mw jadi salah satu fitur Layer 1 --
    # menguji apakah forecast beban terhubung ke risiko CFM & stok
    # [KLARIFIKASI USER].
    log("Melatih & membandingkan algoritma Load Forecast (horizon 168 jam / 7 hari)...")
    load_res = LF.train_load_forecast(df)
    log(f"Model TERBAIK: {load_res['best_model_name']}")
    log("Perbandingan seluruh algoritma:")
    algo_rows = []
    for name, m in load_res["all_candidates"].items():
        marker = " <-- TERBAIK" if name == load_res["best_model_name"] else ""
        log(f"  {name:18s} MAE={m['mae']:7.2f}  RMSE={m['rmse']:7.2f}  "
            f"MAPE={m['mape_pct']:6.2f}%  R2={m['r2']:.3f}{marker}")
        algo_rows.append({"algoritma": name, **m, "is_best": name == load_res["best_model_name"]})
    pd.DataFrame(algo_rows).to_csv(C.TABLE_DIR / "load_forecast_algo_comparison.csv", index=False)
    log(f"Load Forecast: MAE={load_res['mae']:.2f} MW  RMSE={load_res['rmse']:.2f} MW  "
        f"MAPE={load_res['mape_pct']:.2f}%  R2={load_res['r2']:.3f}  "
        f"(target <15%: {'TERCAPAI' if load_res['target_achieved'] else 'BELUM TERCAPAI'})")
    if load_res["mape_roh_baseline_pct"] is not None:
        log(f"Baseline ROH apa adanya: MAPE={load_res['mape_roh_baseline_pct']:.2f}% "
            f"(n={load_res['n_roh_baseline']}) -- model {'LEBIH BAIK' if load_res['mape_pct'] < load_res['mape_roh_baseline_pct'] else 'TIDAK LEBIH BAIK'} dari ROH")

    log("Walk-forward bulanan Load Forecast (pisahkan steady-state vs anomali)...")
    load_wf_df = LF.walk_forward_load_forecast(df, model_name=load_res["best_model_name"])
    load_wf_df.to_csv(C.TABLE_DIR / "load_forecast_walk_forward.csv", index=False)
    load_wf_summary = LF.summarize_steady_vs_anomaly(load_wf_df)
    log(f"  Steady-state ({load_wf_summary['steady_state_n_months']} bulan normal): "
        f"MAPE={load_wf_summary['steady_state_mape_pct']:.2f}%")
    log(f"  Anomali {LF.ANOMALY_MONTHS} ({load_wf_summary['anomaly_n_months']} bulan): "
        f"MAPE={load_wf_summary['anomaly_mape_pct']:.2f}% -- berhimpitan dgn periode data stok "
        f"HOP kosong & awal tren penurunan stok riil (BUKAN artefak imputasi, beban_netto_mw "
        f"dari sumber terpisah total)")
    pd.DataFrame([{
        "best_model_name": load_res["best_model_name"],
        "horizon_h": load_res["horizon_h"], "n_train": load_res["n_train"], "n_test": load_res["n_test"],
        "mae": load_res["mae"], "rmse": load_res["rmse"], "mape_pct": load_res["mape_pct"],
        "r2": load_res["r2"], "target_mape_pct": load_res["target_mape_pct"],
        "target_achieved": load_res["target_achieved"],
        "mape_roh_baseline_pct": load_res["mape_roh_baseline_pct"],
        "n_roh_baseline": load_res["n_roh_baseline"],
        "steady_state_mape_pct": load_wf_summary["steady_state_mape_pct"],
        "anomaly_mape_pct": load_wf_summary["anomaly_mape_pct"],
    }]).to_csv(C.TABLE_DIR / "load_forecast_metrics.csv", index=False)
    df = LF.add_load_forecast_feature(df, load_res)

    log("Feature engineering...")
    df = feat.engineer(df, C.SAFETY_STOCK.mrc_min_days, C.SAFETY_STOCK.lrc_min_days)

    log("Membangun label risiko...")
    df = rl.build_labels(df, C.SAFETY_STOCK.mrc_min_days, C.SAFETY_STOCK.lrc_min_days)

    label_summary = pd.DataFrame([
        {"target": t, "event_rate_pct": round(100 * df[t].mean(), 2), "n_event": int(df[t].sum())}
        for t in rl.TARGET_COLUMNS
    ])
    label_summary.to_csv(C.TABLE_DIR / "label_summary.csv", index=False)
    log("Label summary:\n" + label_summary.to_string(index=False))

    # ---- 2. Layer 1: predictive risk model ---------------------------------
    log("Melatih Layer 1 (Logistic Regression + XGBoost, time-based split)...")
    results, df_train, df_test = L1.train_all_targets(df)
    layer1_metrics = []
    for target, res in results.items():
        m = res["metrics_test"].copy()
        m.insert(0, "target", target)
        layer1_metrics.append(m)
        log(f"{target}: event rate train/test = {res['event_rate_train_pct']}/"
            f"{res['event_rate_test_pct']}%")
        log(res["metrics_test"][["model", "roc_auc", "pr_auc", "f1_at_0.5"]].to_string(index=False))
    layer1_metrics_df = pd.concat(layer1_metrics, ignore_index=True)
    layer1_metrics_df.to_csv(C.TABLE_DIR / "layer1_test_metrics.csv", index=False)

    # ---- 3. Validasi: benchmarking, walk-forward, kalibrasi, tornado ------
    log("Menjalankan benchmarking & walk-forward utk semua target...")
    bench_rows = []
    for target in rl.TARGET_COLUMNS:
        b = V.benchmark_models(df, target)
        b.insert(0, "target", target)
        bench_rows.append(b)
    bench_df = pd.concat(bench_rows, ignore_index=True)
    bench_df.to_csv(C.TABLE_DIR / "benchmarking.csv", index=False)

    wf_all = V.walk_forward_all_targets(df)
    for t, wdf in wf_all.items():
        wdf.to_csv(C.TABLE_DIR / f"walk_forward_{t}.csv", index=False)
    validation_summary_df = V.validation_summary_table(df, wf_all)
    validation_summary_df.to_csv(C.TABLE_DIR / "validation_summary.csv", index=False)
    log("Ringkasan Validasi Model:\n" + validation_summary_df.to_string(index=False))

    calib_tables = {}
    tornado_tables = {}
    for target in rl.TARGET_COLUMNS:
        calib_tables[target] = V.calibration_report(results[target], df_test, target)
        calib_tables[target].to_csv(C.TABLE_DIR / f"calibration_{target}.csv", index=False)
        tornado_tables[target] = V.tornado_table(df, results[target], target)
        tornado_tables[target].to_csv(C.TABLE_DIR / f"tornado_{target}.csv", index=False)

    # ---- 4. Sensitivity: SFC ~ CV_blend + beban ----------------------------
    log("Fit hubungan SFC ~ CV_blend + beban...")
    sfc_model = SF.fit_sfc_model(df)
    log(f"SFC model: intercept={sfc_model['intercept']:.4f} slope_cv={sfc_model['slope_cv']:.6f} "
        f"(p={sfc_model['p_cv']:.4g}) slope_beban={sfc_model['slope_beban']:.6f} "
        f"(p={sfc_model['p_beban']:.4g}) R2={sfc_model['r2']:.3f} n={sfc_model['n_obs']}")
    pd.DataFrame([{
        "n_obs": sfc_model["n_obs"], "intercept": sfc_model["intercept"],
        "slope_cv": sfc_model["slope_cv"], "p_cv": sfc_model["p_cv"],
        "slope_beban": sfc_model["slope_beban"], "p_beban": sfc_model["p_beban"],
        "r2": sfc_model["r2"], "resid_std": sfc_model["resid_std"],
    }]).to_csv(C.TABLE_DIR / "sfc_model.csv", index=False)

    # ---- 5. Layer 2: Monte Carlo grid rasio MRC:LRC -----------------------
    log(f"Menjalankan Monte Carlo Layer 2 ({L2.N_SCENARIOS_DEFAULT} skenario x "
        f"{len(L2.RASIO_MRC_GRID)} rasio)...")
    grid_df = L2.run_full_grid(df, results, n_scenarios=L2.N_SCENARIOS_DEFAULT)
    grid_df.to_csv(C.TABLE_DIR / "monte_carlo_grid.csv", index=False)
    log("Grid Monte Carlo:\n" + grid_df.round(3).to_string(index=False))

    # ---- 6. Layer 3: optimasi rasio blending -------------------------------
    log("Optimasi Layer 3...")
    opt_result = L3.optimize_ratio(grid_df)
    recommendation = L3.recommend_action(opt_result, sfc_model)
    log(recommendation)

    # ---- 7. Ringkasan JSON --------------------------------------------------
    current_row = df.iloc[-1]
    summary = {
        "plant": C.SCOPE.plant, "unit": C.SCOPE.unit,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cfm_threshold_tph": C.THRESH_CFM.cfm_threshold_tph,
        "sfc_target_kg_per_kwh": C.SFC.sfc_target_kg_per_kwh,
        "safety_stock_mrc_days": C.SAFETY_STOCK.mrc_min_days,
        "safety_stock_lrc_days": C.SAFETY_STOCK.lrc_min_days,
        "label_summary": label_summary.to_dict(orient="records"),
        "layer1_test_metrics": layer1_metrics_df.round(4).to_dict(orient="records"),
        "benchmarking": bench_df.round(4).to_dict(orient="records"),
        "validation_summary": validation_summary_df.astype(object).where(
            validation_summary_df.notna(), None).to_dict(orient="records"),
        "sfc_model": {k: v for k, v in sfc_model.items() if k != "resid"},
        "load_forecast": {
            "best_model_name": load_res["best_model_name"],
            "horizon_h": load_res["horizon_h"], "mae": load_res["mae"], "rmse": load_res["rmse"],
            "mape_pct": load_res["mape_pct"], "r2": load_res["r2"],
            "target_mape_pct": load_res["target_mape_pct"], "target_achieved": load_res["target_achieved"],
            "mape_roh_baseline_pct": load_res["mape_roh_baseline_pct"],
            "n_roh_baseline": load_res["n_roh_baseline"],
            "all_candidates": {k: {kk: vv for kk, vv in v.items()}
                                for k, v in load_res["all_candidates"].items()},
        },
        "monte_carlo_grid": grid_df.round(4).to_dict(orient="records"),
        "opt_result": opt_result,
        "recommendation": recommendation,
        "current_condition_example": {
            "time": str(current_row["time"]),
            "coal_flow_tph": round(float(current_row["coal_flow_tph"]), 1),
            "cv_blend_kcal_kg": round(float(current_row["cv_blend_kcal_kg"]), 1),
            "rasio_mrc_pct": round(float(current_row["rasio_mrc"]) * 100, 1),
            "rasio_lrc_pct": round(float(current_row["rasio_lrc"]) * 100, 1),
            "hop_hari_inventory": round(float(current_row["hop_hari_inventory"]), 2),
            "beban_netto_mw": round(float(current_row["beban_netto_mw"]), 1)
                if pd.notna(current_row["beban_netto_mw"]) else None,
            "beban_forecast_168h_mw": round(float(current_row["beban_forecast_168h_mw"]), 1)
                if pd.notna(current_row["beban_forecast_168h_mw"]) else None,
        },
        "data_notes": {
            "hop_missing_months_imputed": list(C.DQ.hop_missing_months),
            "roh_data_availability_pct": round(100 * df["roh_data_available"].mean(), 1),
            "sfc_actual_availability_pct": round(100 * df["sfc_data_available"].mean(), 1),
            "coal_gcv_source": "data_rekapitulasi_kedatangan_batubara_destinator.xlsx (884 shipment, "
                                "klasifikasi MRC/LRC dari nama supplier + fallback ambang GCV>=4400 kcal/kg)",
            "beban_source": "sheet redispatch (data_risk_modelling_destinator.xlsx) -- PADAT 100% "
                             "di seluruh 20 bulan, menggantikan proksi ROH/implied sebelumnya",
        },
    }
    with open(C.OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    log(f"Selesai tabel. Ringkasan tersimpan di {C.OUTPUT_DIR / 'summary.json'}")

    # ---- 8. Figures ---------------------------------------------------------
    log("Membuat figure...")
    R.fig_label_event_rate(label_summary)
    R.fig_cost_vs_ratio(grid_df)
    R.fig_prob_vs_ratio(grid_df)
    for target in rl.TARGET_COLUMNS:
        R.fig_benchmarking(bench_df, target)
        R.fig_walk_forward(wf_all[target], target)
        R.fig_calibration(calib_tables[target], target)
    R.fig_sfc_vs_load(df, sfc_model)

    # figure tambahan: before/after & distribusi cost pd rasio optimal
    import numpy as _np
    R.fig_risk_before_after(grid_df, opt_result["rasio_mrc_star_pct"])
    sim_pool = L2.prepare_sim_pool(df)
    coal_gcv = SF.get_coal_type_gcv()
    rng_opt = _np.random.default_rng(C.RANDOM_STATE)
    sim_opt = L2.run_scenarios_for_ratio(sim_pool, coal_gcv, sfc_model, results,
                                          opt_result["rasio_mrc_star_pct"] / 100.0,
                                          L2.N_SCENARIOS_DEFAULT, rng_opt)
    R.fig_cost_distribution(sim_opt, opt_result["rasio_mrc_star_pct"])

    algo_df = pd.read_csv(C.TABLE_DIR / "load_forecast_algo_comparison.csv")
    R.fig_load_forecast_comparison(algo_df)
    R.fig_roh_redispatch_gap(df)

    log(f"Selesai. {len(list(C.FIGURE_DIR.glob('*.png')))} figure tersimpan di {C.FIGURE_DIR}")


if __name__ == "__main__":
    main()
