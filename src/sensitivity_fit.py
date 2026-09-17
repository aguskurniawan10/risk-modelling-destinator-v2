"""
sensitivity_fit.py -- Hubungan empiris SFC ~ CV_blend + beban (OLS, data
SFC_ACTUAL riil x beban_netto_mw riil dari sheet redispatch -- PADAT, n~5.638
jam, Jan-Agu 2025).

TEMUAN (setelah data SFC_actual & beban redispatch padat tersedia,
menggantikan versi awal yg py n~1.640 dari proksi ROH/beban sparse): efek
CV_blend thd SFC signifikan (p<0,001) DAN searah wajar (CV naik -> SFC turun,
sesuai intuisi fisika: kalori lbh tinggi = massa batubara lbh sedikit utk
energi yg sama). Target SFC 0,7551 kg/kWh TERCAPAI pd kombinasi realistis:
CV~4.082 kcal/kg (beban=rata-rata) ATAU beban~252 MW (CV=rata-rata) --
KEDUANYA dlm rentang data teramati, BUKAN ekstrapolasi liar spt versi
proksi sebelumnya (yg sempat menghasilkan CV~712 kcal/kg, tidak realistis,
akibat sample kecil & noise dari proksi tak langsung).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats as sstats

from . import config as C


def fit_sfc_model(df: pd.DataFrame) -> dict:
    """OLS: sfc_actual ~ cv_blend_kcal_kg + beban_netto_mw. SFC_ACTUAL riil
    (data DATA_SFC_DESTINATOR.xlsx, padat penuh Jan-Agu 2025) dipasangkan dgn
    beban_netto_mw dari sheet REDISPATCH (padat 100% di seluruh 20 bulan) --
    n jauh lebih besar & lebih andal drpd versi sebelumnya (proksi
    coal_flow/beban ROH yg cuma ~12% jam)."""
    sub = df[df["sfc_data_available"] & (df["beban_netto_mw"] > 10)].copy()
    sub = sub[(sub["sfc_actual"] > 0.2) & (sub["sfc_actual"] < 2.0)]

    X = np.column_stack([np.ones(len(sub)), sub["cv_blend_kcal_kg"], sub["beban_netto_mw"]])
    y = sub["sfc_actual"].values
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    dof = n - k
    sigma2 = float((resid ** 2).sum() / dof)
    se = np.sqrt(np.clip(np.diag(sigma2 * XtX_inv), 1e-12, None))
    t_stat = beta / se
    pvals = 2 * sstats.t.sf(np.abs(t_stat), dof)
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot

    return {
        "n_obs": n,
        "intercept": beta[0], "slope_cv": beta[1], "slope_beban": beta[2],
        "p_intercept": pvals[0], "p_cv": pvals[1], "p_beban": pvals[2],
        "r2": r2, "resid_std": float(np.std(resid)),
        "resid": resid,  # utk bootstrap noise di Layer 2
        "cv_blend_range": (float(sub["cv_blend_kcal_kg"].min()), float(sub["cv_blend_kcal_kg"].max())),
        "beban_range": (float(sub["beban_netto_mw"].min()), float(sub["beban_netto_mw"].max())),
        "beban_mean": float(sub["beban_netto_mw"].mean()),
        "beban_median": float(sub["beban_netto_mw"].median()),
        "cv_blend_mean": float(sub["cv_blend_kcal_kg"].mean()),
    }


def predict_sfc(model: dict, cv_blend: float, beban_mw: float) -> float:
    return model["intercept"] + model["slope_cv"] * cv_blend + model["slope_beban"] * beban_mw


def solve_beban_for_target_sfc(model: dict, target_sfc: float, cv_blend: float) -> float:
    """Beban yg dibutuhkan utk capai target_sfc pd CV_blend tertentu."""
    return (target_sfc - model["intercept"] - model["slope_cv"] * cv_blend) / model["slope_beban"]


def solve_cv_for_target_sfc(model: dict, target_sfc: float, beban_mw: float) -> float:
    """CV_blend yg dibutuhkan utk capai target_sfc pd beban tertentu -- KINI
    REALISTIS (dlm rentang data teramati) setelah model difit dgn data
    SFC_actual x beban_netto_mw padat, beda dgn versi proksi awal."""
    return (target_sfc - model["intercept"] - model["slope_beban"] * beban_mw) / model["slope_cv"]


def get_coal_type_gcv() -> pd.DataFrame:
    """Ekstrak GCV riil per shipment dari rekap kedatangan batubara,
    diklasifikasi MRC/LRC: dari nama supplier eksplisit ('MRC'/'LRC' di
    kolom Suppliers) jika ada, else fallback ambang GCV>=4400 kcal/kg
    (definisi standar MRC/LRC dari dokumen FKI DESTINATOR)."""
    path = C.DATA_DIR / "data_rekapitulasi_kedatangan_batubara_destinator.xlsx"
    df = pd.read_excel(path, sheet_name="PRATU")
    df = df.rename(columns={"GCV\nARB": "gcv_arb", "Suppliers": "supplier"})
    df["gcv_arb"] = pd.to_numeric(df["gcv_arb"], errors="coerce")
    df = df.dropna(subset=["gcv_arb"])

    def classify(row):
        s = str(row["supplier"]).upper()
        if "MRC" in s:
            return "MRC"
        if "LRC" in s:
            return "LRC"
        return "MRC" if row["gcv_arb"] >= 4400 else "LRC"

    df["jenis"] = df.apply(classify, axis=1)
    return df[["jenis", "gcv_arb", "supplier"]].reset_index(drop=True)
