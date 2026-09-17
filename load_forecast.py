"""
load_forecast.py -- Forecasting beban (load): horizon
STLF 7 hari (168 jam), fitur lag temporal (load_lag1/24/48/168) + kalender
siklikal, MEMBANDINGKAN 5 algoritma (Linear Regression, Ridge, Random
Forest, Extra Trees, Gradient Boosting) & memilih yg terbaik -- mengikuti
metodologi FKI DESTINATOR (Hyndman & Athanasopoulos, 2018 utk struktur lag;
Weron, 2006 utk klasifikasi horizon STLF).

SUMBER BEBAN: beban_netto_mw dari sheet REDISPATCH -- PADAT 100% di seluruh
20 bulan.

RENTANG OPERASI NORMAL 200-330 MW + SMOOTHING 3 JAM [STATED BY USER, mirip
capping rule notebook referensi pengguna]: beban di luar rentang ini
(shutdown/overhaul ~0 MW: 7,4% jam [dikonfirmasi pengguna sbg overhaul
terjadwal], atau ekskursi ekstrem) DIKELUARKAN dari fitur histori & target
evaluasi -- bukan soal load forecasting, melainkan unit commitment/jadwal
pemeliharaan yg datanya terpisah & tak tersedia. Smoothing rolling-mean 3
jam meredam noise jangka pendek yg tak bermakna operasional.

TEMUAN SETELAH PERBANDINGAN 5 ALGORITMA + FITUR DIPERKAYA + SMOOTHING &
RENTANG OPERASI: MAPE turun dari ~18% (fitur lag+kalender polos, rentang
>50 MW) menjadi ~12-13% (walk-forward, target <15% TERCAPAI). Walk-forward
BULANAN jg mengungkap 2 rezim beban: STEADY-STATE (~13 bulan, MAPE lbh
rendah) vs ANOMALI Nov 2025-Jan 2026 (MAPE jauh lbh tinggi, ~3 bulan) --
persis berhimpitan dgn periode data stok HOP kosong & awal tren penurunan
stok riil [dikonfirmasi: beban_netto_mw dari sumber TERPISAH TOTAL dari
data stok, jadi ini BUKAN artefak imputasi -- beban rata-rata turun nyata
~22% (250->193 MW) mengindikasikan gangguan operasional riil pd unit].
Lihat walk_forward_load_forecast() & summarize_steady_vs_anomaly() utk
laporan terpisah steady-state vs anomali (BUKAN dirata-rata jadi 1 angka
yg menyamarkan perbedaan rezim ini).

KONEKSI KE RISIKO [KLARIFIKASI USER]: ROH meleset dari Realisasi Beban
Netto -> boros MRC -> stok menipis -> CFM -> SFC meleset. Forecast (168
jam, SAMA PERSIS dgn horizon label y_stock_below_safety) dipakai sbg fitur
tambahan Layer 1. Periode anomali beban di atas jg berhimpitan dgn awal
krisis stok -- indikasi bahwa degradasi akurasi forecast beban itu sendiri
bisa jadi SINYAL DINI gangguan operasional, bukan cuma soal akurasi model.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from . import config as C

HORIZON_LOAD_H = 168  # 7 hari -- SAMA PERSIS dgn horizon asli DESTINATOR (STLF) [STATED BY USER]
TARGET_MAPE_PCT = 15.0  # [STATED BY USER] target akurasi load forecast -- sesuai skala Lewis
# (1982), "Industrial and Business Forecasting Methods": MAPE <10% = Highly Accurate,
# 10-20% = Good Forecasting, 20-50% = Reasonable, >50% = Inaccurate. Target 15% berada
# tepat di pita "Good Forecasting" -- referensi yg SAMA dgn kriteria MAPE regresi
# lain di model ini (lihat src/config.py ValidationConfig).
MIN_OPERATING_MW = 200.0  # [STATED BY USER] rentang operasi normal Unit 1: 200-330 MW
MAX_OPERATING_MW = 330.0  # (mirip capping rule notebook referensi -- di luar ini dianggap
                           # shutdown/overhaul/anomali, bukan pola beban normal yg direplikasi)
SMOOTHING_WINDOW_H = 3  # rolling mean 3 jam (center) pd beban sblm jadi lag/target --
                         # turunkan MAPE dari ~18% ke ~12% (walk-forward) [STATED BY USER: smoothing]

# [TEMUAN, bukan asumsi] beban rata-rata turun tajam & nyata (~22%, dari sumber
# beban_netto_mw yg PADAT & TERPISAH TOTAL dari data stok HOP -- BUKAN artefak
# imputasi) mulai Nov 2025 s.d Feb 2026 -- persis berhimpitan dgn periode data
# stok HOP kosong & awal tren penurunan stok yg ditemukan terpisah. Walk-forward
# bulanan menunjukkan MAPE MELEDAK (18-46%) khusus di jendela ini, sementara
# bulan2 lain (steady-state) MAPE jauh lbh baik (9-18%). Dilaporkan TERPISAH,
# bukan dirata-ratakan begitu saja, spy tak menyamarkan performa steady-state
# yg sebenarnya cukup baik.
ANOMALY_MONTHS = ("2025-11", "2025-12", "2026-01")

LOAD_FEATURE_COLUMNS = [
    "load_lag1", "load_lag2", "load_lag3", "load_lag24", "load_lag48", "load_lag72", "load_lag168",
    "load_roll_mean_24h", "load_roll_std_24h", "load_roll_min_24h", "load_roll_max_24h",
    "load_roll_mean_72h", "load_roll_mean_168h",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend", "is_holiday",
]

CANDIDATE_MODELS = {
    "LinearRegression": lambda: LinearRegression(),
    "Ridge": lambda: Ridge(alpha=1.0),
    "RandomForest": lambda: RandomForestRegressor(
        n_estimators=300, max_depth=8, random_state=C.RANDOM_STATE, n_jobs=1),
    "ExtraTrees": lambda: ExtraTreesRegressor(
        n_estimators=300, max_depth=10, random_state=C.RANDOM_STATE, n_jobs=1),
    "GradientBoosting": lambda: GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=C.RANDOM_STATE),
}

try:
    from prophet import Prophet
    _PROPHET_AVAILABLE = True
except ImportError:
    _PROPHET_AVAILABLE = False


class _ProphetWrapper:
    """Wrapper API sklearn-style (fit/predict) di atas Facebook Prophet.
    BEDA PARADIGMA dari 4 model lain: Prophet TIDAK pakai fitur lag manual
    (load_lag1 dst) -- ia fit dekomposisi trend+seasonality (harian,
    mingguan) LANGSUNG dari histori (ds=waktu, y=beban), lalu forecast pd
    timestamp target scr langsung. Regressor kalender (is_weekend,
    is_holiday) ditambahkan sbg extra_regressor.

    CATATAN JUJUR: kode ini BELUM bisa diuji di sandbox pengembangan
    (tanpa akses internet, `pip install prophet` gagal) -- WAJIB dites
    ulang di environment Anda yg py Prophet terinstal sebelum dipakai sbg
    dasar keputusan. API dipastikan sesuai dokumentasi resmi Prophet
    (Taylor & Letham, 2018), tapi belum ada verifikasi hasil MAPE riil."""

    def __init__(self):
        self.model = None
        self._train_time = None

    def fit(self, X_with_time: pd.DataFrame, y: pd.Series):
        train_df = pd.DataFrame({
            "ds": X_with_time["_time"].values, "y": y.values,
            "is_weekend": X_with_time["is_weekend"].values.astype(float),
            "is_holiday": X_with_time["is_holiday"].values.astype(float),
        })
        self.model = Prophet(
            daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False,
            changepoint_prior_scale=0.05,
        )
        self.model.add_regressor("is_weekend")
        self.model.add_regressor("is_holiday")
        self.model.fit(train_df)
        return self

    def predict(self, X_with_time: pd.DataFrame) -> np.ndarray:
        future_df = pd.DataFrame({
            "ds": X_with_time["_time"].values,
            "is_weekend": X_with_time["is_weekend"].values.astype(float),
            "is_holiday": X_with_time["is_holiday"].values.astype(float),
        })
        fc = self.model.predict(future_df)
        return fc["yhat"].values


if _PROPHET_AVAILABLE:
    CANDIDATE_MODELS["Prophet"] = lambda: _ProphetWrapper()


def build_load_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.sort_values("time").reset_index(drop=True).copy()
    # Smoothing rolling-mean 3 jam SEBELUM jadi lag/target -- meredam noise
    # jangka pendek yg tak bermakna operasional, mirip semangat capping rule
    # notebook referensi (batasi ke rentang & pola beban yg wajar/dpt diulang).
    d["beban_smooth_mw"] = d["beban_netto_mw"].rolling(
        SMOOTHING_WINDOW_H, min_periods=1, center=True).mean()
    load = d["beban_smooth_mw"]
    d["load_lag1"] = load.shift(1)
    d["load_lag2"] = load.shift(2)
    d["load_lag3"] = load.shift(3)
    d["load_lag24"] = load.shift(24)
    d["load_lag48"] = load.shift(48)
    d["load_lag72"] = load.shift(72)
    d["load_lag168"] = load.shift(168)
    d["load_roll_mean_24h"] = load.rolling(24, min_periods=6).mean()
    d["load_roll_std_24h"] = load.rolling(24, min_periods=6).std()
    d["load_roll_min_24h"] = load.rolling(24, min_periods=6).min()
    d["load_roll_max_24h"] = load.rolling(24, min_periods=6).max()
    d["load_roll_mean_72h"] = load.rolling(72, min_periods=12).mean()
    d["load_roll_mean_168h"] = load.rolling(168, min_periods=24).mean()
    d["hour_sin"] = np.sin(2 * np.pi * d["hour"] / 24)
    d["hour_cos"] = np.cos(2 * np.pi * d["hour"] / 24)
    d["dow_sin"] = np.sin(2 * np.pi * d["day_of_week"] / 7)
    d["dow_cos"] = np.cos(2 * np.pi * d["day_of_week"] / 7)
    return d


def _in_operating_range(s: pd.Series) -> pd.Series:
    return (s >= MIN_OPERATING_MW) & (s <= MAX_OPERATING_MW)


def train_load_forecast(df: pd.DataFrame, test_frac: float = 0.2) -> dict:
    """Bandingkan seluruh algoritma di CANDIDATE_MODELS (termasuk Prophet
    bila terinstal), pilih MAPE terbaik pd test set. Evaluasi HANYA pd
    periode operasi (beban aktual & load_lag1 > 50 MW) -- periode shutdown
    dikeluarkan (lihat catatan modul)."""
    d = build_load_features(df)
    d["y_load_future"] = d["beban_netto_mw"].shift(-HORIZON_LOAD_H)

    valid = d.dropna(subset=LOAD_FEATURE_COLUMNS + ["y_load_future"]).reset_index(drop=True)
    valid_model = valid[_in_operating_range(valid["y_load_future"])
                         & _in_operating_range(valid["load_lag1"])].reset_index(drop=True)
    n = len(valid_model)
    cut = int(n * (1 - test_frac))
    train, test = valid_model.iloc[:cut], valid_model.iloc[cut:]

    # Prophet perlu "ds" = waktu TARGET (waktu t + horizon), bukan waktu t --
    # supaya dibandingkan adil dgn model lain yg memprediksi y_load_future.
    train_time_target = train["time"] + pd.Timedelta(hours=HORIZON_LOAD_H)
    test_time_target = test["time"] + pd.Timedelta(hours=HORIZON_LOAD_H)

    X_train_sklearn, y_train = train[LOAD_FEATURE_COLUMNS], train["y_load_future"]
    X_test_sklearn, y_test = test[LOAD_FEATURE_COLUMNS], test["y_load_future"]
    X_train_prophet = train[["is_weekend", "is_holiday"]].copy()
    X_train_prophet["_time"] = train_time_target.values
    X_test_prophet = test[["is_weekend", "is_holiday"]].copy()
    X_test_prophet["_time"] = test_time_target.values

    candidates = {}
    for name, make_model in CANDIDATE_MODELS.items():
        model = make_model()
        if name == "Prophet":
            model.fit(X_train_prophet, y_train)
            pred = model.predict(X_test_prophet)
        else:
            model.fit(X_train_sklearn, y_train)
            pred = model.predict(X_test_sklearn)
        mape = float(np.mean(np.abs((y_test - pred) / y_test)) * 100)
        candidates[name] = {
            "model": model, "pred": pred,
            "mae": mean_absolute_error(y_test, pred),
            "rmse": mean_squared_error(y_test, pred) ** 0.5,
            "mape_pct": mape, "r2": r2_score(y_test, pred),
        }
    best_name = min(candidates, key=lambda k: candidates[k]["mape_pct"])
    best = candidates[best_name]

    # Baseline ROH: [STATED BY USER] TIDAK boleh ikut difilter smoothing/rentang
    # operasi 200-330 MW yg dipakai model -- itu perlakuan khusus utk model,
    # ROH dinilai APA ADANYA pd rentang penuh (BUKAN dr subset `test` yg sudah
    # kena filter model) supaya perbandingan jujur, tak menguntungkan sepihak.
    # Batas jendela waktu tetap disamakan dgn periode test model (out-of-time),
    # cuma dibuang pembagi nyaris nol (shutdown) spy MAPE tak meledak jadi inf.
    test_time_window = (d["time"] >= test["time"].min()) & (d["time"] <= test["time"].max())
    roh_sub = d[test_time_window & d["roh_mw"].notna() & (d["roh_mw"] > 20) & (d["beban_netto_mw"] > 20)]
    if len(roh_sub) >= 5:
        mape_roh_baseline = float(np.mean(
            np.abs((roh_sub["beban_netto_mw"] - roh_sub["roh_mw"]) / roh_sub["beban_netto_mw"])) * 100)
        n_roh_baseline = len(roh_sub)
    else:
        mape_roh_baseline, n_roh_baseline = None, 0

    return {
        "model": best["model"], "best_model_name": best_name,
        "all_candidates": {k: {kk: vv for kk, vv in v.items() if kk not in ("model", "pred")}
                            for k, v in candidates.items()},
        "n_train": len(train), "n_test": len(test),
        "mae": best["mae"], "rmse": best["rmse"], "mape_pct": best["mape_pct"], "r2": best["r2"],
        "y_test": y_test.values, "pred_test": best["pred"], "test_time": test["time"].values,
        "horizon_h": HORIZON_LOAD_H,
        "mape_roh_baseline_pct": mape_roh_baseline, "n_roh_baseline": n_roh_baseline,
        "target_mape_pct": TARGET_MAPE_PCT,
        "target_achieved": best["mape_pct"] < TARGET_MAPE_PCT,
        "min_operating_mw": MIN_OPERATING_MW,
    }


def add_load_forecast_feature(df: pd.DataFrame, load_model: dict) -> pd.DataFrame:
    """Tambahkan kolom beban_forecast_168h_mw -- anti-leakage: prediksi di
    waktu t HANYA pakai lag features s.d waktu t."""
    d = build_load_features(df)
    model = load_model["model"]
    has_features = d[LOAD_FEATURE_COLUMNS].notna().all(axis=1)
    d["beban_forecast_168h_mw"] = np.nan
    sub = d.loc[has_features]
    if isinstance(model, _ProphetWrapper):
        X_prophet = sub[["is_weekend", "is_holiday"]].copy()
        X_prophet["_time"] = (sub["time"] + pd.Timedelta(hours=HORIZON_LOAD_H)).values
        d.loc[has_features, "beban_forecast_168h_mw"] = model.predict(X_prophet)
    else:
        d.loc[has_features, "beban_forecast_168h_mw"] = model.predict(sub[LOAD_FEATURE_COLUMNS])
    return df.merge(d[["time", "beban_forecast_168h_mw"]], on="time", how="left")


def walk_forward_load_forecast(df: pd.DataFrame, model_name: str | None = None) -> pd.DataFrame:
    """Walk-forward BULANAN (latih s.d bulan N, uji bulan N+1) -- dipakai utk
    memisahkan performa steady-state dari periode anomali (lihat ANOMALY_MONTHS),
    krn evaluasi static 80/20 split menyamarkan perbedaan ini dgn merata-ratakan
    semua periode jadi satu angka."""
    d = build_load_features(df)
    d["y_load_future"] = d["beban_netto_mw"].shift(-HORIZON_LOAD_H)
    valid = d.dropna(subset=LOAD_FEATURE_COLUMNS + ["y_load_future"]).reset_index(drop=True)
    valid = valid[_in_operating_range(valid["y_load_future"])
                  & _in_operating_range(valid["load_lag1"])].reset_index(drop=True)
    valid["ym"] = valid["time"].dt.to_period("M")
    months = sorted(valid["ym"].unique())
    model_name = model_name or "GradientBoosting"
    make_model = CANDIDATE_MODELS[model_name]

    rows = []
    for i in range(2, len(months) - 1):
        train = valid[valid["ym"].isin(months[: i + 1])]
        test = valid[valid["ym"] == months[i + 1]]
        if len(train) < 200 or len(test) < 20:
            continue
        model = make_model()
        model.fit(train[LOAD_FEATURE_COLUMNS], train["y_load_future"])
        pred = model.predict(test[LOAD_FEATURE_COLUMNS])
        mape = float(np.mean(np.abs((test["y_load_future"] - pred) / test["y_load_future"])) * 100)
        rows.append({
            "test_month": str(months[i + 1]), "n_test": len(test), "mape_pct": mape,
            "is_anomaly_period": str(months[i + 1]) in ANOMALY_MONTHS,
        })
    return pd.DataFrame(rows)


def summarize_steady_vs_anomaly(wf_df: pd.DataFrame) -> dict:
    steady = wf_df[~wf_df["is_anomaly_period"]]
    anomaly = wf_df[wf_df["is_anomaly_period"]]
    return {
        "steady_state_mape_pct": float(np.average(steady["mape_pct"], weights=steady["n_test"]))
                                  if len(steady) else None,
        "steady_state_n_months": int(len(steady)),
        "anomaly_mape_pct": float(np.average(anomaly["mape_pct"], weights=anomaly["n_test"]))
                             if len(anomaly) else None,
        "anomaly_n_months": int(len(anomaly)),
        "overall_mape_pct": float(np.average(wf_df["mape_pct"], weights=wf_df["n_test"])),
    }


if __name__ == "__main__":
    from . import data_loader as dl

    d = dl.load_clean()
    res = train_load_forecast(d)
    print(f"Load Forecast (horizon {res['horizon_h']}h / 7 hari):")
    print(f"  n_train={res['n_train']} n_test={res['n_test']}")
    print("  Perbandingan algoritma:")
    for name, m in res["all_candidates"].items():
        marker = " <-- TERBAIK" if name == res["best_model_name"] else ""
        print(f"    {name:18s} MAE={m['mae']:.2f}  RMSE={m['rmse']:.2f}  "
              f"MAPE={m['mape_pct']:.2f}%  R2={m['r2']:.3f}{marker}")
    print(f"  Target MAPE < 15%: {'TERCAPAI' if res['target_achieved'] else 'BELUM TERCAPAI'}")
    if res["mape_roh_baseline_pct"] is not None:
        print(f"  Baseline ROH apa adanya: MAPE={res['mape_roh_baseline_pct']:.2f}% "
              f"(n={res['n_roh_baseline']})")

