"""
validation.py -- Metode validasi utk DESTINATOR risk model (Unit 1),
mengikuti pola persis PLTU Jeranjang: time-based split, backtesting,
benchmarking, calibration test, tornado chart, walk-forward bulanan.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from . import config as C
from .features import FEATURE_COLUMNS
from .risk_labels import TARGET_COLUMNS
from .layer1_predictive import time_based_split, calibration_table, train_one_target, predict_proba
from .metrics_utils import classification_metrics, classification_verdict, regression_verdict


# ---------------------------------------------------------------------------
# BACKTESTING
# ---------------------------------------------------------------------------
def backtest_period(df_test: pd.DataFrame, result: dict, target: str,
                     model_name: str = "xgb") -> pd.DataFrame:
    proba = predict_proba(result, df_test, model_name=model_name)
    out = pd.DataFrame({
        "time": df_test["time"].values,
        "y_true": df_test[target].values,
        "y_pred_proba": proba,
    })
    out["y_pred_label"] = (out["y_pred_proba"] >= 0.5).astype(int)
    out["hit"] = (out["y_pred_label"] == out["y_true"]).astype(int)
    return out


# ---------------------------------------------------------------------------
# BENCHMARKING -- 4 pendekatan: no-skill, rule-of-thumb, LogReg, XGBoost
# ---------------------------------------------------------------------------
_RULE_THUMB = {
    # rule-of-thumb operator: 1 variabel paling intuitif per target,
    # dibandingkan thd ambang yg sudah dikenal operator (bukan model)
    "y_cfm_exceed": lambda d: (d["coal_flow_roll_mean_3h"] >= 0.9 * C.THRESH_CFM.cfm_threshold_tph).astype(float),
    "y_stock_below_safety": lambda d: (d["hop_hari_inventory"] < 10.0).astype(float),  # ambang tunggal LRC (lbh konservatif)
}


def benchmark_models(df: pd.DataFrame, target: str, test_frac: float = 0.2) -> pd.DataFrame:
    df_train, df_test = time_based_split(df, test_frac)
    y_test = df_test[target]
    rows = []

    baseline_rate = df_train[target].mean()
    proba_baseline = np.full(len(df_test), baseline_rate)
    rows.append(_score_row("a. Fixed-ratio / no-skill baseline", y_test, proba_baseline))

    rule_score = _RULE_THUMB[target](df_test)
    rows.append(_score_row("b. Rule-of-thumb operator (1 variabel)", y_test, rule_score))

    res = train_one_target(df_train, df_test, target)
    rows.append(_score_row("c. Logistic regression (Layer 1)", y_test, predict_proba(res, df_test, "logreg")))
    rows.append(_score_row("d. XGBoost (Layer 1)", y_test, predict_proba(res, df_test, "xgb")))

    return pd.DataFrame(rows)


def _score_row(name, y_true, proba) -> dict:
    row = {"pendekatan": name}
    row.update(classification_metrics(y_true, proba))
    return row


# ---------------------------------------------------------------------------
# CALIBRATION TEST
# ---------------------------------------------------------------------------
def calibration_report(result: dict, df_test: pd.DataFrame, target: str,
                        model_name: str = "xgb", n_bins: int = 8) -> pd.DataFrame:
    proba = predict_proba(result, df_test, model_name)
    y_true = df_test[target].values
    return calibration_table(y_true, proba, n_bins=n_bins)


# ---------------------------------------------------------------------------
# SENSITIVITY: permutation importance + tornado
# ---------------------------------------------------------------------------
def permutation_importance_table(result: dict, df_test: pd.DataFrame, target: str,
                                  model_name: str = "xgb", n_repeats: int = 15) -> pd.DataFrame:
    model = result["models"][model_name]
    X_test = df_test[FEATURE_COLUMNS]
    y_test = df_test[target]
    if model_name == "logreg":
        X_test = pd.DataFrame(result["scaler"].transform(X_test), columns=FEATURE_COLUMNS)
    r = permutation_importance(model, X_test, y_test, n_repeats=n_repeats,
                                random_state=C.RANDOM_STATE, scoring="average_precision", n_jobs=2)
    return pd.DataFrame({
        "feature": FEATURE_COLUMNS, "importance_mean": r.importances_mean, "importance_std": r.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)


def tornado_table(df: pd.DataFrame, result: dict, target: str,
                   model_name: str = "xgb", swing: float = 1.0) -> pd.DataFrame:
    base_row = df[FEATURE_COLUMNS].mean().to_frame().T
    base_proba = predict_proba(result, base_row, model_name)[0]
    rows = []
    for feat_name in FEATURE_COLUMNS:
        std = df[feat_name].std()
        low_row, high_row = base_row.copy(), base_row.copy()
        low_row[feat_name] -= swing * std
        high_row[feat_name] += swing * std
        p_low = predict_proba(result, low_row, model_name)[0]
        p_high = predict_proba(result, high_row, model_name)[0]
        rows.append({"feature": feat_name, "proba_low": p_low, "proba_base": base_proba,
                      "proba_high": p_high, "swing_range": abs(p_high - p_low)})
    return pd.DataFrame(rows).sort_values("swing_range", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# WALK-FORWARD TESTING (bulanan, 20 bulan data -> hingga 19 fold)
# ---------------------------------------------------------------------------
def walk_forward_validation(df: pd.DataFrame, target: str, model_name: str = "xgb") -> pd.DataFrame:
    d = df.copy()
    d["year_month"] = d["time"].dt.to_period("M")
    months = sorted(d["year_month"].unique())

    rows = []
    for i in range(len(months) - 1):
        train_months = months[: i + 1]
        test_month = months[i + 1]
        df_train = d[d["year_month"].isin(train_months)]
        df_test = d[d["year_month"] == test_month]
        if df_train[target].nunique() < 2 or len(df_test) < 20:
            continue
        res = train_one_target(df_train, df_test, target)
        proba = predict_proba(res, df_test, model_name)
        y_test = df_test[target]
        row = {"train_upto": str(train_months[-1]), "test_month": str(test_month),
               "n_train": len(df_train), "n_test": len(df_test),
               "event_rate_test_pct": round(100 * y_test.mean(), 2)}
        row.update(classification_metrics(y_test, proba))
        rows.append(row)
    return pd.DataFrame(rows)


def walk_forward_summary(wf_df: pd.DataFrame, min_test_events: int = 10) -> dict:
    d = wf_df.copy()
    d["n_pos_test"] = (d["event_rate_test_pct"] / 100.0 * d["n_test"]).round().astype(int)
    d["n_neg_test"] = d["n_test"] - d["n_pos_test"]
    # "reliable" butuh KEDUA kelas terwakili memadai -- fold dgn event rate
    # 100% (tanpa kelas negatif) jg membuat ROC-AUC tak terdefinisi, sama
    # spt fold 0 kejadian positif; kalau hanya cek n_pos, fold all-positive
    # lolos filter tapi ROC-AUC-nya NaN & merusak rata-rata tertimbang.
    d["reliable"] = (d["n_pos_test"] >= min_test_events) & (d["n_neg_test"] >= min_test_events)
    rel = d[d["reliable"]]
    if len(rel) == 0:
        rel = d
    w = rel["n_test"]
    return {
        "n_folds_total": int(len(d)), "n_folds_reliable": int(len(rel)),
        "roc_auc_weighted_avg": float(np.average(rel["roc_auc"], weights=w)),
        "pr_auc_weighted_avg": float(np.average(rel["pr_auc"], weights=w)),
        "f1_at_0.5_weighted_avg": float(np.average(rel["f1_at_0.5"], weights=w)),
        "mae_weighted_avg": float(np.average(rel["mae"], weights=w)),
        "rmse_weighted_avg": float(np.average(rel["rmse"], weights=w)),
        "excluded_folds": d.loc[~d["reliable"], "test_month"].tolist(),
    }


def walk_forward_all_targets(df: pd.DataFrame, model_name: str = "xgb",
                              targets: list[str] | None = None) -> dict[str, pd.DataFrame]:
    targets = targets or TARGET_COLUMNS
    return {t: walk_forward_validation(df, t, model_name) for t in targets}


def validation_summary_table(df: pd.DataFrame, wf_all: dict[str, pd.DataFrame],
                              min_test_events: int | None = None) -> pd.DataFrame:
    min_test_events = min_test_events or C.VALID.min_test_events
    rows = []
    for target, wf_df in wf_all.items():
        if wf_df.empty:
            rows.append({"target": target, "n_folds_total": 0, "n_folds_reliable": 0,
                         "roc_auc": np.nan, "pr_auc": np.nan, "f1_at_0.5": np.nan,
                         "mae": np.nan, "rmse": np.nan, "baseline_event_rate_pct": np.nan,
                         "verdict": "Tak dapat dinilai (data tidak cukup)"})
            continue
        summ = walk_forward_summary(wf_df, min_test_events=min_test_events)
        baseline_rate = df[target].mean()
        verdict = classification_verdict(summ["roc_auc_weighted_avg"], summ["pr_auc_weighted_avg"], baseline_rate)
        rows.append({
            "target": target, "n_folds_total": summ["n_folds_total"], "n_folds_reliable": summ["n_folds_reliable"],
            "roc_auc": round(summ["roc_auc_weighted_avg"], 4), "pr_auc": round(summ["pr_auc_weighted_avg"], 4),
            "f1_at_0.5": round(summ["f1_at_0.5_weighted_avg"], 4), "mae": round(summ["mae_weighted_avg"], 4),
            "rmse": round(summ["rmse_weighted_avg"], 4), "baseline_event_rate_pct": round(100 * baseline_rate, 2),
            "verdict": verdict,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from . import data_loader as dl
    from . import features as feat
    from . import risk_labels as rl

    d = dl.load_clean()
    d = feat.engineer(d, C.SAFETY_STOCK.mrc_min_days, C.SAFETY_STOCK.lrc_min_days)
    d = rl.build_labels(d, C.SAFETY_STOCK.mrc_min_days, C.SAFETY_STOCK.lrc_min_days)

    for target in TARGET_COLUMNS:
        print("=" * 70, "\nBENCHMARKING:", target)
        print(benchmark_models(d, target).round(4)[["pendekatan", "roc_auc", "pr_auc"]])
        print("\nWALK-FORWARD:", target)
        wf = walk_forward_validation(d, target)
        print(wf.round(4)[["test_month", "n_test", "event_rate_test_pct", "roc_auc", "pr_auc"]])
