"""Pure business calculations (no Streamlit dependency, so they are easy to unit-test).

Everything is derived from `data/threshold_sweep.csv`, which holds the exact hold-out confusion matrix
(4,727 transactions) and the USD value of each outcome bucket for every threshold.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .config import CostAssumptions


@dataclass(frozen=True)
class Outcome:
    """Confusion matrix + financial outcome for one threshold (optionally scaled to a bigger population)."""

    tau: float
    tn: float
    fp: float
    fn: float
    tp: float
    precision: float
    recall: float
    f1: float
    accuracy: float
    fp_rate: float            # share of legitimate customers wrongly blocked ("friction rate")
    fraud_usd_caught: float
    fraud_loss: float         # False Negative cost (missed fraud value + chargeback penalties)
    friction_cost: float      # False Positive cost (lost GMV + handling)
    net_cost: float           # fraud_loss + friction_cost
    loss_prevented: float     # fraud value stopped + chargeback penalties avoided
    net_benefit: float        # loss_prevented - friction_cost

    def as_dict(self) -> dict:
        return asdict(self)


def row_at(sweep: pd.DataFrame, tau: float) -> pd.Series:
    """Return the sweep row for a threshold (thresholds are stored with 2-decimal precision)."""
    hit = sweep.loc[np.isclose(sweep["threshold"], round(float(tau), 2))]
    if hit.empty:
        raise ValueError(f"Threshold {tau} is outside the pre-computed grid")
    return hit.iloc[0]


def evaluate(row: pd.Series, cost: CostAssumptions, scale: float = 1.0) -> Outcome:
    tn, fp, fn, tp = (float(row[k]) * scale for k in ("tn", "fp", "fn", "tp"))
    total = tn + fp + fn + tp
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    fraud_loss = float(row["fraud_usd_missed"]) * scale + fn * cost.chargeback_fee
    friction = float(row["legit_usd_blocked"]) * scale * cost.lost_sale_rate + fp * cost.handling_cost
    prevented = float(row["fraud_usd_caught"]) * scale + tp * cost.chargeback_fee
    return Outcome(
        tau=float(row["threshold"]), tn=tn, fp=fp, fn=fn, tp=tp,
        precision=prec, recall=rec, f1=f1, accuracy=(tn + tp) / total if total else 0.0,
        fp_rate=fp / (fp + tn) if fp + tn else 0.0,
        fraud_usd_caught=float(row["fraud_usd_caught"]) * scale,
        fraud_loss=fraud_loss, friction_cost=friction, net_cost=fraud_loss + friction,
        loss_prevented=prevented, net_benefit=prevented - friction,
    )


def cost_curve(sweep: pd.DataFrame, cost: CostAssumptions) -> pd.DataFrame:
    """Net cost decomposition for every threshold on the grid."""
    rows = [evaluate(r, cost).as_dict() for _, r in sweep.iterrows()]
    return pd.DataFrame(rows)


def no_model_cost(sweep: pd.DataFrame, cost: CostAssumptions) -> float:
    """Loss if every transaction were approved (no fraud engine at all) on the hold-out set."""
    r = sweep.iloc[0]
    positives = float(r["tp"] + r["fn"])
    return float(r["fraud_usd_caught"] + r["fraud_usd_missed"]) + positives * cost.chargeback_fee


def reduction_pct(before: float, after: float) -> float:
    return (before - after) / before * 100.0 if before else 0.0


def decision_for(prob: float, review_from: float, block_from: float) -> str:
    if prob >= block_from:
        return "BLOCK"
    if prob >= review_from:
        return "REVIEW"
    return "APPROVE"
