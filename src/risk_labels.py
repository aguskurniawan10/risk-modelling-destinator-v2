"""
risk_labels.py -- Label risiko horizon ke depan, anti-leakage (label dari
data t+1..t+H, fitur dari data s.d t). Mengikuti pola risk_labels.py PLTU
Jeranjang.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from . import config as C

HORIZON_CFM_H = 2   # jam -- horizon deteksi dini risiko CFM (selaras konvensi "2 jam ke depan" PLTU Jeranjang)
HORIZON_STOCK_D = 7  # hari -- horizon risiko stok (stok berubah lambat/harian, bukan per jam)

TARGET_COLUMNS = ["y_cfm_exceed", "y_stock_below_safety"]


def build_labels(df: pd.DataFrame, mrc_min_days: float, lrc_min_days: float) -> pd.DataFrame:
    d = df.sort_values("time").reset_index(drop=True).copy()

    # --- y_cfm_exceed: coal flow akan >= ambang CFM dlm HORIZON_CFM_H jam ke depan ---
    future_max = d["coal_flow_tph"][::-1].rolling(HORIZON_CFM_H, min_periods=1).max()[::-1]
    future_max = future_max.shift(-1)  # geser 1 langkah: mulai dari t+1, bukan t
    d["y_cfm_exceed"] = (future_max >= C.THRESH_CFM.cfm_threshold_tph).astype(float)
    d.loc[future_max.isna(), "y_cfm_exceed"] = np.nan  # ujung data, horizon tak lengkap

    # --- y_stock_below_safety: stok gabungan akan di bawah ambang aman dlm
    # HORIZON_STOCK_D hari ke depan (resolusi harian -> HORIZON_STOCK_D*24 jam) ---
    combined_safety = d["rasio_mrc"] * mrc_min_days + d["rasio_lrc"] * lrc_min_days
    horizon_steps = HORIZON_STOCK_D * 24
    future_min_stock = d["hop_hari_inventory"][::-1].rolling(horizon_steps, min_periods=24).min()[::-1]
    future_min_stock = future_min_stock.shift(-1)
    d["y_stock_below_safety"] = (future_min_stock < combined_safety).astype(float)
    d.loc[future_min_stock.isna(), "y_stock_below_safety"] = np.nan

    # buang baris ujung yg horizonnya tak lengkap (2 target sekaligus)
    n_before = len(d)
    d = d.dropna(subset=TARGET_COLUMNS).reset_index(drop=True)
    print(f"  [risk_labels] {n_before} -> {len(d)} baris (buang ujung horizon tak lengkap)")
    for t in TARGET_COLUMNS:
        rate = d[t].mean() * 100
        print(f"    {t}: event rate = {rate:.2f}% ({int(d[t].sum())}/{len(d)})")
    return d


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from . import data_loader as dl
    from . import features as feat

    df = dl.load_clean()
    df = feat.engineer(df, C.SAFETY_STOCK.mrc_min_days, C.SAFETY_STOCK.lrc_min_days)
    df = build_labels(df, C.SAFETY_STOCK.mrc_min_days, C.SAFETY_STOCK.lrc_min_days)
    print(df[TARGET_COLUMNS].describe())
