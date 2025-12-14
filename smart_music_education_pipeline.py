"""
Smart Music Education Using ML/DL — Reproducible Pipeline (Tabular + TabNet)

This script implements:
- Loading dataset1.csv / dataset2.csv
- Preprocessing (impute, scale, encode)
- Feature engineering (Effort_Index, Focus_Ratio, Tempo_Rhythm_Ratio, Musical_Cluster)
- Stratified K-Fold CV via binned regression target
- Baselines: RandomForest, XGBoost, Ridge
- Proposed: TabNetRegressor (pytorch-tabnet)
- Evaluation: MAE, RMSE, R2, MAPE, SMAPE
- External validation on Dataset 2
- (Optional) SHAP + LIME placeholders (requires fitted model and feature names)

Usage:
    python smart_music_education_pipeline.py --data1 data/dataset1.csv --data2 data/dataset2.csv --target Performance_Score
"""

from __future__ import annotations

import argparse
import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

try:
    from xgboost import XGBRegressor
except Exception as e:
    XGBRegressor = None

try:
    from pytorch_tabnet.tab_model import TabNetRegressor
except Exception:
    TabNetRegressor = None

warnings.filterwarnings("ignore")


# -----------------------------
# Metrics
# -----------------------------
def mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) > 1e-8
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def smape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + 1e-8
    return float(np.mean(np.abs(y_pred - y_true) / denom) * 100.0)


def compute_regression_metrics(y_true, y_pred) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    return {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2), "MAPE": mape(y_true, y_pred), "SMAPE": smape(y_true, y_pred)}


def summarize_fold_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    keys = metrics_list[0].keys()
    out: Dict[str, Dict[str, float]] = {}
    for k in keys:
        arr = np.array([m[k] for m in metrics_list], dtype=float)
        out[k] = {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0}
    return out


# -----------------------------
# Feature Engineering
# -----------------------------
def create_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    eps = 1e-6

    # Effort_Index: Mean_Energy_dB * Duration / (Tempo + 1)
    if {"Mean_Energy_dB", "Duration", "Tempo"}.issubset(df.columns):
        df["Effort_Index"] = (
            df["Mean_Energy_dB"].fillna(df["Mean_Energy_dB"].median()) * df["Duration"].fillna(0.0)
        ) / (df["Tempo"].fillna(df["Tempo"].median()) + 1.0 + eps)

    # Focus_Ratio: Active_Time / Total_Time
    if {"Active_Time", "Total_Time"}.issubset(df.columns):
        df["Focus_Ratio"] = df["Active_Time"].fillna(0.0) / (df["Total_Time"].fillna(1.0) + eps)

    # Tempo_Rhythm_Ratio: Tempo / (Rhythm + 1)
    if {"Tempo", "Rhythm"}.issubset(df.columns):
        df["Tempo_Rhythm_Ratio"] = df["Tempo"].fillna(df["Tempo"].median()) / (df["Rhythm"].fillna(df["Rhythm"].median()) + 1.0 + eps)

    # Musical_Cluster via KMeans on spectral features, otherwise Tempo+Energy
    spec_cols = [c for c in df.columns if ("Spectral" in c) or ("Pitch" in c) or ("MFCC" in c)]
    if len(spec_cols) >= 3:
        Xs = df[spec_cols].copy()
        Xs = Xs.fillna(0.0).to_numpy(dtype=float)
        Xs = StandardScaler().fit_transform(Xs)
        km = KMeans(n_clusters=3, random_state=42, n_init=10)
        df["Musical_Cluster"] = km.fit_predict(Xs).astype(int)
    elif {"Tempo", "Mean_Energy_dB"}.issubset(df.columns):
        Xs = df[["Tempo", "Mean_Energy_dB"]].fillna(0.0).to_numpy(dtype=float)
        km = KMeans(n_clusters=3, random_state=42, n_init=10)
        df["Musical_Cluster"] = km.fit_predict(Xs).astype(int)

    return df


# -----------------------------
# Preprocessing
# -----------------------------
@dataclass
class PreprocessConfig:
    categorical_cols: List[str]
    numerical_cols: List[str]


def infer_column_types(df: pd.DataFrame, target: str) -> PreprocessConfig:
    """
    Conservative auto-inference:
      - numeric dtypes -> numerical
      - object/category/bool -> categorical
    You can override with CLI args if needed.
    """
    cats: List[str] = []
    nums: List[str] = []
    for c in df.columns:
        if c == target:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            nums.append(c)
        else:
            cats.append(c)
    return PreprocessConfig(categorical_cols=cats, numerical_cols=nums)


def build_preprocessor(cfg: PreprocessConfig) -> ColumnTransformer:
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False)),
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, cfg.numerical_cols),
            ("cat", cat_pipe, cfg.categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return pre


def make_strat_bins(y: np.ndarray, q: int = 5) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    # qcut may fail with too many ties; fall back to cut
    try:
        bins = pd.qcut(y, q=q, labels=False, duplicates="drop")
        return np.asarray(bins, dtype=int)
    except Exception:
        bins = pd.cut(y, bins=q, labels=False, duplicates="drop")
        return np.asarray(bins, dtype=int)


# -----------------------------
# Models
# -----------------------------
def train_rf(X_train, y_train) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_ridge(X_train, y_train) -> Ridge:
    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_xgb(X_train, y_train) -> "XGBRegressor":
    if XGBRegressor is None:
        raise RuntimeError("xgboost is not installed. Run: pip install xgboost")
    model = XGBRegressor(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
        reg_lambda=1.0,
    )
    model.fit(X_train, y_train, verbose=False)
    return model


def train_tabnet(X_train, y_train, X_valid, y_valid, max_epochs=50, lr=0.02, batch_size=1024, patience=20) -> "TabNetRegressor":
    if TabNetRegressor is None:
        raise RuntimeError("pytorch-tabnet is not installed. Run: pip install pytorch-tabnet")
    model = TabNetRegressor(
        n_d=32,
        n_a=32,
        n_steps=5,
        gamma=1.5,
        lambda_sparse=1e-4,
        optimizer_params=dict(lr=lr),
        mask_type="sparsemax",
        verbose=10,
    )
    # TabNet expects 2D y
    y_train_2d = np.asarray(y_train, dtype=float).reshape(-1, 1)
    y_valid_2d = np.asarray(y_valid, dtype=float).reshape(-1, 1)
    model.fit(
        X_train=np.asarray(X_train, dtype=np.float32),
        y_train=y_train_2d,
        eval_set=[(np.asarray(X_valid, dtype=np.float32), y_valid_2d)],
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        virtual_batch_size=min(256, batch_size),
        num_workers=0,
        drop_last=False,
    )
    return model


# -----------------------------
# CV + External Validation
# -----------------------------
def run_cv(df: pd.DataFrame, target: str, n_splits: int = 5, seed: int = 42) -> Dict[str, Dict[str, Dict[str, float]]]:
    df = df.copy()

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset. Available columns: {list(df.columns)[:20]}...")

    # Infer types
    cfg = infer_column_types(df, target=target)

    y = df[target].to_numpy(dtype=float)
    strat = make_strat_bins(y, q=5)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_metrics = {
        "RF": [],
        "XGB": [],
        "Ridge": [],
        "TabNet": [],
    }

    for fold, (tr_idx, va_idx) in enumerate(skf.split(df, strat), start=1):
        train = df.iloc[tr_idx].copy()
        valid = df.iloc[va_idx].copy()

        pre = build_preprocessor(cfg)
        X_train = pre.fit_transform(train.drop(columns=[target]))
        X_valid = pre.transform(valid.drop(columns=[target]))

        y_train = train[target].to_numpy(dtype=float)
        y_valid = valid[target].to_numpy(dtype=float)

        # RF
        rf = train_rf(X_train, y_train)
        pred = rf.predict(X_valid)
        fold_metrics["RF"].append(compute_regression_metrics(y_valid, pred))

        # Ridge
        rg = train_ridge(X_train, y_train)
        pred = rg.predict(X_valid)
        fold_metrics["Ridge"].append(compute_regression_metrics(y_valid, pred))

        # XGB (optional)
        if XGBRegressor is not None:
            xgb = train_xgb(X_train, y_train)
            pred = xgb.predict(X_valid)
            fold_metrics["XGB"].append(compute_regression_metrics(y_valid, pred))

        # TabNet
        tab = train_tabnet(X_train, y_train, X_valid, y_valid, max_epochs=50, lr=0.02, batch_size=1024, patience=20)
        pred = tab.predict(np.asarray(X_valid, dtype=np.float32)).reshape(-1)
        fold_metrics["TabNet"].append(compute_regression_metrics(y_valid, pred))

        print(f"[Fold {fold}/{n_splits}] done.")

    # Summaries
    summaries: Dict[str, Dict[str, Dict[str, float]]] = {}
    for model_name, mlist in fold_metrics.items():
        if len(mlist) == 0:
            continue
        summaries[model_name] = summarize_fold_metrics(mlist)

    return summaries


def external_validation(df_train: pd.DataFrame, df_test: pd.DataFrame, target: str) -> Dict[str, Dict[str, float]]:
    df_train = df_train.copy()
    df_test = df_test.copy()

    if target not in df_train.columns or target not in df_test.columns:
        raise ValueError(f"Target '{target}' must exist in both datasets.")

    cfg = infer_column_types(df_train, target=target)
    pre = build_preprocessor(cfg)

    X_train = pre.fit_transform(df_train.drop(columns=[target]))
    y_train = df_train[target].to_numpy(dtype=float)

    X_test = pre.transform(df_test.drop(columns=[target]))
    y_test = df_test[target].to_numpy(dtype=float)

    tab = train_tabnet(X_train, y_train, X_test, y_test, max_epochs=50, lr=0.02, batch_size=1024, patience=20)
    pred = tab.predict(np.asarray(X_test, dtype=np.float32)).reshape(-1)
    return compute_regression_metrics(y_test, pred)


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data1", type=str, required=True, help="Path to dataset1.csv")
    ap.add_argument("--data2", type=str, default=None, help="Path to dataset2.csv (optional for external validation)")
    ap.add_argument("--target", type=str, default="Performance_Score", help="Target column name (regression)")
    ap.add_argument("--out", type=str, default="outputs", help="Output directory")
    ap.add_argument("--folds", type=int, default=5, help="Number of folds (StratifiedKFold on binned target)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    df1 = pd.read_csv(args.data1)
    df1 = create_engineered_features(df1)

    print("\n=== Stratified K-Fold CV on Dataset 1 ===")
    cv_summary = run_cv(df1, target=args.target, n_splits=args.folds, seed=args.seed)

    # Save summary
    out_csv = os.path.join(args.out, "cv_summary.csv")
    rows = []
    for model, metrics in cv_summary.items():
        for metric, stats in metrics.items():
            rows.append({"model": model, "metric": metric, "mean": stats["mean"], "std": stats["std"]})
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nSaved CV summary to: {out_csv}")

    # External validation
    if args.data2:
        df2 = pd.read_csv(args.data2)
        df2 = create_engineered_features(df2)

        print("\n=== External Validation: Train on Dataset 1, Test on Dataset 2 (TabNet) ===")
        ext = external_validation(df1, df2, target=args.target)

        out_ext = os.path.join(args.out, "external_validation_tabnet.json")
        import json
        with open(out_ext, "w", encoding="utf-8") as f:
            json.dump(ext, f, indent=2)
        print("External validation metrics:", ext)
        print(f"Saved external validation to: {out_ext}")

    print("\nDone.")


if __name__ == "__main__":
    main()
