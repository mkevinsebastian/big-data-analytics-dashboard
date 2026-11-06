"""Headline numbers behind the three mandatory business insights. Everything is computed from ./data,
so the text on screen can never drift away from the underlying results."""
from __future__ import annotations

from . import config as C
from . import data
from .analytics import reduction_pct, row_at

RING_MIN_SIZE = 2          # a ring needs at least two linked transactions
MACRO_MIN_SIZE = 8         # "macro" community = 8+ linked transactions
MACRO_RATE_PCT = 30.0


def tuning() -> dict:
    sw = data.sweep()
    a, b = row_at(sw, C.DEFAULT_TAU), row_at(sw, C.OPTIMAL_TAU)
    return dict(
        fp_before=int(a.fp), fp_after=int(b.fp), fp_cut_pct=reduction_pct(a.fp, b.fp),
        prec_before=float(a.precision), prec_after=float(b.precision),
        rec_before=float(a.recall), rec_after=float(b.recall),
        f1_before=float(a.f1_score), f1_after=float(b.f1_score),
        fn_before=int(a.fn), fn_after=int(b.fn),
    )


def risk_profile() -> dict:
    p = data.risk_profile().set_index(["factor", "segment"])["fraud_rate_pct"]
    prof = data.risk_profile()
    combo = prof[prof.factor == "Kombinasi Risiko"].set_index("segment")
    return dict(
        amount_high=float(p[("Transaction Amount", "> $1.000")]),
        amount_low=float(p[("Transaction Amount", "< $100")]),
        new_account=float(p[("Account Age", "0 - 30 hari (akun baru)")]),
        dawn=float(p[("Transaction Hour", "Dini hari 00:00-05:59")]),
        all_three=float(combo.loc["Ketiga faktor sekaligus", "fraud_rate_pct"]),
        all_three_n=int(combo.loc["Ketiga faktor sekaligus", "transactions"]),
        no_risk=float(combo.loc["Tidak ada faktor risiko", "fraud_rate_pct"]),
        baseline=float(data.kpi_summary()["fraud_rate_pct"]),
    )


def graph() -> dict:
    comm, nodes = data.graph_communities(), data.graph_nodes()
    rings = comm[(comm.fraud_rate_pct == 100) & (comm["size"] >= RING_MIN_SIZE)]
    macro = comm[(comm["size"] >= MACRO_MIN_SIZE)].sort_values("fraud_rate_pct", ascending=False)
    ring_nodes = nodes[nodes.community_id.isin(rings.community_id)]
    missed = int((ring_nodes.oof_prob < C.BLOCK_FROM).sum())
    fraud_nodes = nodes[nodes.is_fraud == 1]
    return dict(
        rings=rings, n_rings=len(rings), ring_members=len(ring_nodes), ring_missed=missed,
        macro=macro, macro_high=macro[macro.fraud_rate_pct > MACRO_RATE_PCT],
        n_communities=len(comm), fraud_in_network=len(fraud_nodes),
        fraud_missed_in_network=int((fraud_nodes.oof_prob < C.BLOCK_FROM).sum()),
        ring_shared_subnet=int((rings.shared_subnets > 0).sum()),
        ring_shared_location=int((rings.shared_locations > 0).sum()),
        ring_same_state=int((rings.n_ship_states == 1).sum()),
    )
