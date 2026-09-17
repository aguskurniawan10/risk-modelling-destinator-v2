"""
metrics_utils.py
==============================================================================
SATU SUMBER KEBENARAN untuk seluruh metrik evaluasi yang dipakai di
Layer 1 (predictive), benchmarking, dan walk-forward testing.

Kenapa modul terpisah:
- Menjamin metrik yang dilaporkan di layer1_test_metrics.csv, benchmarking.csv,
  dan walk_forward.csv semuanya dihitung dengan definisi & rounding yang
  identik (tidak ada versi berbeda tersebar di beberapa file).

Dua kelompok metrik:
1. classification_metrics()  -> untuk target BINER (y_nox_exceed, dst).
   - ROC-AUC, PR-AUC          : diskriminasi & ranking (skala ambang).
   - Brier score (= MSE)      : akurasi probabilitas, MSE(proba, y).
   - RMSE                     : sqrt(Brier) -- skala sama dgn probabilitas.
   - MAE                      : rata-rata |proba - y|, lebih tahan outlier
                                  drpd Brier/RMSE.
   - F1 / Precision / Recall @0.5     : performa pada ambang keputusan baku.
   - F1 / Precision / Recall @ambang terbaik : ambang yang memaksimalkan F1
                                  pada data test itu sendiri -- dilaporkan
                                  TERPISAH dari F1@0.5 supaya tidak
                                  menyesatkan (ini "upper bound" in-sample,
                                  bukan ambang operasional yang direkomendasikan).
   CATATAN: MAPE SENGAJA tidak dipakai untuk target biner (0/1) -- MAPE
   membagi dgn y_true, sehingga tak terdefinisi/meledak setiap kali
   y_true = 0 (mayoritas baris pada kasus rare-event). Untuk MAPE yang
   valid secara statistik, lihat regression_metrics() di bawah, dipakai
   pada target regresi kontinu (lihat regression_forecast.py).

2. regression_metrics()      -> untuk target KONTINU (mis. NPHR aktual
   2 jam ke depan). MAE, RMSE, MAPE, R2 -- di sinilah MAPE bermakna scr
   statistik krn y_true (NPHR, kcal/kWh) tidak pernah nol.
==============================================================================
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    mean_absolute_error, mean_squared_error, f1_score, precision_score,
    recall_score, r2_score, precision_recall_curve,
)

from . import config as C


def _best_f1_threshold(y_true, proba) -> tuple[float, float, float, float]:
    """Cari ambang yang memaksimalkan F1 pada data ini sendiri.
    Dilaporkan sbg referensi upper-bound, BUKAN ambang operasional final
    (ambang operasional harus dipilih dari data validasi terpisah / cost
    trade-off, bukan dioptimasi di data yang sama dgn yang dilaporkan)."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1s = np.where(
        (precision + recall) > 0,
        2 * precision * recall / np.where((precision + recall) > 0, precision + recall, 1),
        0.0,
    )
    # precision_recall_curve mengembalikan len(thresholds) = len(precision) - 1
    if len(thresholds) == 0:
        return 0.5, f1_score(y_true, (np.asarray(proba) >= 0.5).astype(int), zero_division=0), \
            precision_score(y_true, (np.asarray(proba) >= 0.5).astype(int), zero_division=0), \
            recall_score(y_true, (np.asarray(proba) >= 0.5).astype(int), zero_division=0)
    best_idx = int(np.argmax(f1s[:-1])) if len(f1s) > 1 else 0
    best_thr = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    pred_best = (np.asarray(proba) >= best_thr).astype(int)
    return (
        best_thr,
        f1_score(y_true, pred_best, zero_division=0),
        precision_score(y_true, pred_best, zero_division=0),
        recall_score(y_true, pred_best, zero_division=0),
    )


def classification_metrics(y_true, proba, label: str | None = None) -> dict:
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    row: dict = {}
    if label is not None:
        row["pendekatan" if label else "label"] = label

    if pd.Series(y_true).nunique() > 1:
        row["roc_auc"] = roc_auc_score(y_true, proba)
        row["pr_auc"] = average_precision_score(y_true, proba)
    else:
        row["roc_auc"], row["pr_auc"] = np.nan, np.nan

    mse = mean_squared_error(y_true, proba)          # identik dgn Brier score
    row["brier_score"] = brier_score_loss(y_true, proba)
    row["mse"] = mse
    row["rmse"] = float(np.sqrt(mse))
    row["mae"] = mean_absolute_error(y_true, proba)

    pred_05 = (proba >= 0.5).astype(int)
    row["precision_at_0.5"] = precision_score(y_true, pred_05, zero_division=0)
    row["recall_at_0.5"] = recall_score(y_true, pred_05, zero_division=0)
    row["f1_at_0.5"] = f1_score(y_true, pred_05, zero_division=0)

    if pd.Series(y_true).nunique() > 1:
        best_thr, f1b, prb, reb = _best_f1_threshold(y_true, proba)
        row["best_threshold"] = best_thr
        row["precision_at_best_thr"] = prb
        row["recall_at_best_thr"] = reb
        row["f1_at_best_thr"] = f1b
    else:
        row["best_threshold"] = np.nan
        row["precision_at_best_thr"] = np.nan
        row["recall_at_best_thr"] = np.nan
        row["f1_at_best_thr"] = np.nan

    return row


def regression_metrics(y_true, y_pred, label: str | None = None) -> dict:
    """MAE / MSE / RMSE / MAPE / R2 untuk target KONTINU (bukan 0/1)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    row: dict = {}
    if label is not None:
        row["label"] = label
    mse = mean_squared_error(y_true, y_pred)
    row["mae"] = mean_absolute_error(y_true, y_pred)
    row["mse"] = mse
    row["rmse"] = float(np.sqrt(mse))
    # MAPE aman di sini krn y_true (NPHR, kcal/kWh) tidak pernah 0.
    row["mape_pct"] = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    row["r2"] = r2_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else np.nan
    return row


# ==============================================================================
# VERDICT ("Diterima" / "Cukup" / "Perlu Perbaikan") -- SATU fungsi dipakai
# bersama oleh dashboard teks (dashboard_output.py) & dashboard interaktif
# (dashboard_app.py) supaya label penerimaan konsisten di semua tempat.
# Ambang batas dikonfigurasi terpusat di config.ValidationConfig (VALID).
# ==============================================================================
def classification_verdict(roc_auc: float, pr_auc: float, baseline_event_rate: float) -> str:
    """Verdict klasifikasi biner berbasis ROC-AUC (diskriminasi absolut) DAN
    rasio skill PR-AUC thd baseline no-skill (= event rate itu sendiri) --
    ROC-AUC tinggi saja bisa menyesatkan pada rare-event, jadi keduanya
    dicek bersama."""
    V = C.VALID
    if roc_auc is None or (isinstance(roc_auc, float) and np.isnan(roc_auc)):
        return "Tak dapat dinilai (0 kejadian pada periode test)"
    skill_ratio = (pr_auc / baseline_event_rate) if baseline_event_rate and baseline_event_rate > 0 else np.nan
    skill_ok = np.isnan(skill_ratio) or skill_ratio >= V.pr_auc_skill_ratio_min
    if roc_auc >= V.roc_auc_good and skill_ok:
        return "Diterima (Baik)"
    if roc_auc >= V.roc_auc_acceptable:
        return "Cukup - perlu pemantauan"
    return "Perlu Perbaikan"


def regression_verdict(mape_pct: float, r2: float) -> str:
    """Verdict regresi kontinu berbasis skala MAPE Lewis (1982) + R2."""
    V = C.VALID
    if mape_pct is None or (isinstance(mape_pct, float) and np.isnan(mape_pct)):
        return "Tak dapat dinilai"
    r2_ok_good = np.isnan(r2) or r2 >= V.r2_good
    r2_ok_acceptable = np.isnan(r2) or r2 >= V.r2_acceptable
    if mape_pct <= V.mape_pct_good and r2_ok_good:
        return "Diterima (Baik)"
    if mape_pct <= V.mape_pct_acceptable and r2_ok_acceptable:
        return "Cukup - dapat dipakai dgn pemantauan"
    if mape_pct <= V.mape_pct_reasonable:
        return "Wajar - perlu perbaikan lanjutan"
    return "Perlu Perbaikan"
