"""
layer1_predictive.py
==============================================================================
LAPISAN 1 - PREDICTIVE RISK MODEL (DESTINATOR, PLTU Pelabuhanratu Unit 1)
------------------------------------------------------------------------------
Memprediksi 2 probabilitas risiko (lihat risk_labels.py TARGET_COLUMNS):
    1. y_cfm_exceed          - coal flow akan >= 225 T/h (CFM) dlm 2 jam ke depan
    2. y_stock_below_safety  - stok HOP gabungan akan di bawah ambang aman
                                (tertimbang rasio MRC/LRC) dlm 7 hari ke depan

Metode identik dgn PLTU Jeranjang: Logistic Regression (dasar, transparan) vs
XGBoost (pembanding non-linear, early stopping) -- time-based split, bukan
random split.
==============================================================================
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier

from . import config as C
from .features import FEATURE_COLUMNS
from .risk_labels import TARGET_COLUMNS
from .metrics_utils import classification_metrics


def time_based_split(df: pd.DataFrame, test_frac: float = 0.2):
    n = len(df)
    cut = int(n * (1 - test_frac))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def _fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def _time_carve_validation(X_train: pd.DataFrame, y_train: pd.Series, val_frac: float = 0.15):
    n = len(X_train)
    cut = int(n * (1 - val_frac))
    if cut < 30 or (n - cut) < 20:
        return None
    X_fit, X_val = X_train.iloc[:cut], X_train.iloc[cut:]
    y_fit, y_val = y_train.iloc[:cut], y_train.iloc[cut:]
    if y_fit.nunique() < 2 or y_val.nunique() < 2:
        return None
    return X_fit, y_fit, X_val, y_val


def _fit_xgb_classifier(X_train: pd.DataFrame, y_train: pd.Series, pos_weight: float) -> XGBClassifier:
    carved = _time_carve_validation(X_train, y_train)
    base_kwargs = dict(
        n_estimators=500, max_depth=3, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5, reg_lambda=2.0,
        eval_metric="aucpr", scale_pos_weight=pos_weight,
        random_state=C.RANDOM_STATE, n_jobs=1, tree_method="hist", device="cpu",
        # n_jobs=1, tree_method & device di-set EKSPLISIT -- hindari non-determinisme
        # lintas platform (Windows lokal vs Linux Streamlit Cloud): multi-threading
        # histogram-based tree building XGBoost TIDAK dijamin bit-for-bit identik
        # antar platform/jumlah thread meski random_state & versi library sama persis
        # (perilaku resmi terdokumentasi XGBoost, bukan bug kode ini).
    )
    if carved is not None:
        X_fit, y_fit, X_val, y_val = carved
        try:
            xgb = XGBClassifier(early_stopping_rounds=30, **base_kwargs)
            xgb.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
            return xgb
        except TypeError:
            pass
    xgb = XGBClassifier(**{**base_kwargs, "n_estimators": 250, "learning_rate": 0.05})
    xgb.fit(X_train, y_train)
    return xgb


def train_one_target(df_train: pd.DataFrame, df_test: pd.DataFrame, target: str) -> dict:
    X_train, y_train = df_train[FEATURE_COLUMNS], df_train[target]
    X_test, y_test = df_test[FEATURE_COLUMNS], df_test[target]

    scaler = _fit_scaler(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    n_pos = y_train.sum()
    models = {}

    logreg = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=C.RANDOM_STATE)
    logreg.fit(X_train_s, y_train)
    models["logreg"] = logreg

    pos_weight = max((len(y_train) - n_pos) / max(n_pos, 1), 1.0)
    models["xgb"] = _fit_xgb_classifier(X_train, y_train, pos_weight)

    result = {
        "target": target, "scaler": scaler, "models": models,
        "n_train": len(df_train), "n_test": len(df_test),
        "event_rate_train_pct": round(100 * y_train.mean(), 2),
        "event_rate_test_pct": round(100 * y_test.mean(), 2),
    }

    metrics = []
    for name, model in models.items():
        proba = model.predict_proba(X_test_s if name == "logreg" else X_test)[:, 1]
        row = {"model": name}
        row.update(classification_metrics(y_test, proba))
        metrics.append(row)
    result["metrics_test"] = pd.DataFrame(metrics)
    return result


def train_all_targets(df: pd.DataFrame, test_frac: float = 0.2) -> dict:
    df_train, df_test = time_based_split(df, test_frac)
    results = {}
    for target in TARGET_COLUMNS:
        results[target] = train_one_target(df_train, df_test, target)
    return results, df_train, df_test


def predict_proba(result: dict, X: pd.DataFrame, model_name: str = "xgb") -> np.ndarray:
    model = result["models"][model_name]
    if model_name == "logreg":
        X = result["scaler"].transform(X[FEATURE_COLUMNS])
        return model.predict_proba(X)[:, 1]
    return model.predict_proba(X[FEATURE_COLUMNS])[:, 1]


def calibration_table(y_true, y_prob, n_bins: int = 10) -> pd.DataFrame:
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    return pd.DataFrame({"mean_predicted_prob": mean_pred, "observed_frequency": frac_pos})


def logreg_coefficients(result: dict) -> pd.DataFrame:
    logreg: LogisticRegression = result["models"]["logreg"]
    coefs = pd.DataFrame({"feature": FEATURE_COLUMNS, "coefficient": logreg.coef_[0]})
    coefs["abs_coefficient"] = coefs["coefficient"].abs()
    return coefs.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)


def xgb_feature_importance(result: dict) -> pd.DataFrame:
    xgb: XGBClassifier = result["models"]["xgb"]
    imp = pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": xgb.feature_importances_})
    return imp.sort_values("importance", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    from . import data_loader as dl
    from . import features as feat
    from . import risk_labels as rl

    d = dl.load_clean()
    d = feat.engineer(d, C.SAFETY_STOCK.mrc_min_days, C.SAFETY_STOCK.lrc_min_days)
    d = rl.build_labels(d, C.SAFETY_STOCK.mrc_min_days, C.SAFETY_STOCK.lrc_min_days)

    results, df_train, df_test = train_all_targets(d)
    for target, res in results.items():
        print("=" * 70)
        print(target, "| event rate train/test:",
              res["event_rate_train_pct"], "/", res["event_rate_test_pct"], "%")
        print(res["metrics_test"])
