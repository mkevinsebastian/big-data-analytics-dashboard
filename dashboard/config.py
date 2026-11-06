"""Central configuration: paths, palette, decision policy and default business assumptions."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# ----------------------------------------------------------------------------- #
# Decision policy (taken from the Assignment II report)
# ----------------------------------------------------------------------------- #
OPTIMAL_TAU = 0.80        # F1-optimal threshold found in threshold tuning
DEFAULT_TAU = 0.50        # standard (untuned) classifier threshold used as the "before" baseline
REVIEW_FROM = 0.40        # < 0.40  -> APPROVE
BLOCK_FROM = 0.80         # >= 0.80 -> BLOCK, in between -> MANUAL REVIEW / 2FA

# Insight-2 rule thresholds
HIGH_AMOUNT_USD = 1000.0
NEW_ACCOUNT_DAYS = 30
DAWN_LAST_HOUR = 5        # 00:00 - 05:59

# ----------------------------------------------------------------------------- #
# Palette (readable on both the light and the dark Streamlit theme)
# ----------------------------------------------------------------------------- #
C_FRAUD = "#EF4444"
C_LEGIT = "#3B82F6"
C_GOOD = "#10B981"
C_WARN = "#F59E0B"
C_ACCENT = "#6366F1"
C_MUTED = "#94A3B8"
C_RING = "#F59E0B"

# Business feature groups used in the feature-importance chart
STRONG_FEATURES = ("Transaction Amount", "Account Age", "Transaction Hour (Dini Hari)")
WEAK_FEATURES = ("Payment Method", "Device Used", "Address Mismatch")


@dataclass(frozen=True)
class CostAssumptions:
    """Business assumptions behind the 'Total Net Cost' (all editable in the sidebar).

    The Assignment II report only states the chargeback penalty range (USD 20 - 100 per case);
    the other two values are explicit planning assumptions, not measured facts.
    """

    chargeback_fee: float = 60.0   # extra penalty per missed fraud (False Negative), mid-range of 20-100
    lost_sale_rate: float = 0.50   # share of a wrongly blocked legit order that is lost (abandon / churn)
    handling_cost: float = 5.0     # customer-support cost per wrongly blocked legit order (False Positive)
