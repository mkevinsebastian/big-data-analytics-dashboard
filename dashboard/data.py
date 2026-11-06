"""Cached data loaders. Every file lives in ./data and was produced by scripts/build_data.py."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from .config import DATA_DIR


@st.cache_data(show_spinner=False)
def load_json(name: str) -> dict:
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / name)


def kpi_summary() -> dict:
    return load_json("kpi_summary.json")


def ml_results() -> dict:
    return load_json("ml_evaluation_results.json")


def graph_results() -> dict:
    return load_json("graph_analytics_results.json")


def sweep() -> pd.DataFrame:
    return load_csv("threshold_sweep.csv")


def feature_importance() -> pd.DataFrame:
    return load_csv("feature_importance.csv")


def risk_profile() -> pd.DataFrame:
    return load_csv("risk_profile.csv")


def graph_nodes() -> pd.DataFrame:
    return load_csv("graph_nodes.csv")


def graph_edges() -> pd.DataFrame:
    return load_csv("graph_edges.csv")


def graph_communities() -> pd.DataFrame:
    return load_csv("graph_communities.csv")


def transactions_slim() -> pd.DataFrame:
    return load_csv("transactions_slim.csv")
