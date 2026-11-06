"""Scoring model behind the Live Transaction Risk Simulator.

The production experiment used XGBoost (needs the `xgboost` package and the raw 6 MB CSV with free-text
addresses). To keep the deployed app light, the simulator trains a scikit-learn HistGradientBoosting
*surrogate* at start-up on the six analyst-facing inputs, using the same imbalance weight (18.34 : 1) so
its probability scale is comparable to the XGBoost scores that the 0.40 / 0.80 policy was tuned on.

On the same 80/20 hold-out split it gives ~42% precision / ~41% recall at tau = 0.80
(XGBoost: 42.8% / 34.0%), so the decision zones behave similarly. It is a decision-support
approximation, not the exact production model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import HistGradientBoostingClassifier

from . import data

PAYMENT_METHODS = ["credit card", "debit card", "PayPal", "bank transfer"]
DEVICES = ["desktop", "mobile", "tablet"]
FEATURE_LABELS = {
    "amount": "Transaction Amount",
    "age": "Account Age",
    "hour": "Transaction Hour",
    "qty": "Quantity",
    "payment": "Payment Method",
    "device": "Device Used",
}


@dataclass
class ModelBundle:
    model: HistGradientBoostingClassifier
    columns: list[str]
    normal: dict  # "safe" reference value per factor, used for what-if explanations


def _design(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    X = pd.DataFrame({"amount": df["amount"], "age": df["age"], "hour": df["hour"], "qty": df["qty"]})
    for p in PAYMENT_METHODS:
        X[f"pay_{p}"] = (df["payment"] == p).astype(float)
    for d in DEVICES:
        X[f"dev_{d}"] = (df["device"] == d).astype(float)
    return X if columns is None else X[columns]


@st.cache_resource(show_spinner="Menyiapkan model simulator…")
def get_bundle() -> ModelBundle:
    raw = data.transactions_slim()
    df = pd.DataFrame({
        "amount": raw["Transaction Amount"], "age": raw["Account Age Days"], "hour": raw["Transaction Hour"],
        "qty": raw["Quantity"], "payment": raw["Payment Method"], "device": raw["Device Used"],
    })
    y = raw["Is Fraudulent"].to_numpy()
    X = _design(df)
    ratio = float((y == 0).sum() / (y == 1).sum())
    model = HistGradientBoostingClassifier(
        max_depth=4, max_iter=150, learning_rate=0.08, min_samples_leaf=40,
        class_weight={0: 1.0, 1: ratio}, random_state=42,
    ).fit(X, y)
    legit = df[y == 0]
    normal = {
        "amount": float(legit["amount"].median()), "age": float(legit["age"].median()), "hour": 12,
        "qty": int(round(legit["qty"].median())),
        "payment": legit["payment"].mode().iloc[0], "device": legit["device"].mode().iloc[0],
    }
    return ModelBundle(model, list(X.columns), normal)


def _row(inputs: dict) -> pd.DataFrame:
    return pd.DataFrame([inputs])


def predict(bundle: ModelBundle, inputs: dict) -> float:
    X = _design(_row(inputs), bundle.columns)
    return float(bundle.model.predict_proba(X)[0, 1])


def contributions(bundle: ModelBundle, inputs: dict) -> pd.DataFrame:
    """What-if attribution: probability(input) - probability(input with one factor replaced by a normal value)."""
    base = predict(bundle, inputs)
    rows = []
    for key, label in FEATURE_LABELS.items():
        alt = dict(inputs)
        alt[key] = bundle.normal[key]
        rows.append({"factor": label, "delta": base - predict(bundle, alt)})
    return pd.DataFrame(rows)


def batch_predict(bundle: ModelBundle, df: pd.DataFrame) -> np.ndarray:
    return bundle.model.predict_proba(_design(df, bundle.columns))[:, 1]
