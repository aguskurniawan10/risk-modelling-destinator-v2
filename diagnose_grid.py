"""
diagnose_grid.py -- Jalankan ini di komputer Anda (dgn XGBoost asli terinstal)
utk mendiagnosis kenapa CVaR95 vs Rasio MRC tampak datar/naik-turun random,
bukan turun mulus spt yg diharapkan.

Cara pakai: taruh file ini di folder DESTINATOR_RISK (sejajar dgn main.py),
lalu jalankan: python3 diagnose_grid.py
"""
import sys
sys.path.insert(0, ".")
import numpy as np
import pandas as pd

from src import config as C
from src import data_loader as dl
from src import load_forecast as LF
from src import features as feat
from src import risk_labels as rl
from src import layer1_predictive as L1
from src import layer2_montecarlo as L2
from src import sensitivity_fit as SF

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

print("=" * 70)
print("STEP 1: Memuat data & melatih model (spt main.py)")
print("=" * 70)
d = dl.load_clean()
load_res = LF.train_load_forecast(d)
print(f"Load forecast: model={load_res['best_model_name']}, MAPE={load_res['mape_pct']:.2f}%")
d = LF.add_load_forecast_feature(d, load_res)
d = feat.engineer(d, C.SAFETY_STOCK.mrc_min_days, C.SAFETY_STOCK.lrc_min_days)
d = rl.build_labels(d, C.SAFETY_STOCK.mrc_min_days, C.SAFETY_STOCK.lrc_min_days)

print("\n" + "=" * 70)
print("STEP 2: Melatih Layer 1 (Logistic Regression + XGBoost ASLI)")
print("=" * 70)
results, df_train, df_test = L1.train_all_targets(d)
for target, res in results.items():
    print(f"\n--- {target} ---")
    print(res["metrics_test"][["model", "roc_auc", "pr_auc"]].to_string(index=False))
    # cek feature importance XGBoost -- variabel apa yg paling dominan
    xgb_model = res["models"]["xgb"]
    if hasattr(xgb_model, "feature_importances_"):
        imp = pd.Series(xgb_model.feature_importances_, index=L1.FEATURE_COLUMNS)
        print("Top 8 feature importance (XGBoost):")
        print(imp.sort_values(ascending=False).head(8).to_string())

print("\n" + "=" * 70)
print("STEP 3: Cek prob_cfm_exceed & prob_stock_below_safety PER RASIO")
print("(inilah sumber paling mungkin dari kurva CVaR95 yg datar/random)")
print("=" * 70)
sfc_model = SF.fit_sfc_model(d)
coal_gcv = SF.get_coal_type_gcv()
sim_pool = L2.prepare_sim_pool(d)

rng = np.random.default_rng(C.RANDOM_STATE)
rows = []
for r in L2.RASIO_MRC_GRID:
    sim = L2.run_scenarios_for_ratio(sim_pool, coal_gcv, sfc_model, results, r, 2000, rng)
    row = L2.summarize_ratio(sim, r)
    # tambahan: rata-rata cv_blend & coal_flow yg DIPAKAI utk prediksi Layer 1
    row["mean_cv_blend_input"] = sim["cv_blend_kcal_kg"].mean()
    row["mean_coal_flow_input"] = sim["coal_flow_tph"].mean()
    rows.append(row)
gdf = pd.DataFrame(rows)

print(gdf[["rasio_mrc_pct", "mean_cv_blend_input", "mean_coal_flow_input",
           "mean_sfc_pred", "prob_cfm_exceed", "prob_stock_below_safety",
           "mean_cost_rp", "cvar95_cost_rp"]].to_string(index=False))

print("\n" + "=" * 70)
print("STEP 4: Kesimpulan otomatis")
print("=" * 70)
cvar_trend = gdf["cvar95_cost_rp"].corr(gdf["rasio_mrc_pct"])
prob_cfm_trend = gdf["prob_cfm_exceed"].corr(gdf["rasio_mrc_pct"])
prob_stock_trend = gdf["prob_stock_below_safety"].corr(gdf["rasio_mrc_pct"])
print(f"Korelasi CVaR95 vs Rasio MRC: {cvar_trend:.3f} (harusnya NEGATIF kuat, mis. < -0.7)")
print(f"Korelasi P(CFM exceed) vs Rasio MRC: {prob_cfm_trend:.3f}")
print(f"Korelasi P(Stok<aman) vs Rasio MRC: {prob_stock_trend:.3f} (harusnya NEGATIF kuat)")
if abs(cvar_trend) < 0.3:
    print("\n>>> DIAGNOSIS: CVaR95 TIDAK berkorelasi kuat dgn rasio MRC.")
    print(">>> Kemungkinan besar: model XGBoost/LogReg TIDAK menangkap hubungan")
    print(">>> antara cv_blend/coal_flow dgn probabilitas risiko sekuat yg diharapkan,")
    print(">>> ATAU rentang cv_blend/coal_flow di data training tidak cukup lebar")
    print(">>> utk model belajar pola yg jelas. Kirim tabel di STEP 3 & STEP 2 ini")
    print(">>> ke Claude utk didiagnosis lebih lanjut.")
else:
    print("\n>>> Tren sudah sesuai harapan. Kalau dashboard masih tampak flat,")
    print(">>> kemungkinan itu murni cache -- pastikan sudah reboot/redeploy total.")
