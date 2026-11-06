"""Plotly figure builders. All figures use transparent backgrounds so Streamlit's light/dark theme shows through."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import config as C
from .theme import fmt_int, fmt_pct, fmt_usd

_LAYOUT = dict(margin=dict(l=10, r=10, t=20, b=40), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
               legend=dict(orientation="h", yanchor="top", y=-0.3, xanchor="center", x=0.5))


def _base(fig: go.Figure, title: str = "", height: int = 380) -> go.Figure:
    legend_items = sum(1 for t in fig.data if t.showlegend is not False and t.name)
    layout = dict(_LAYOUT)
    layout["margin"] = dict(l=10, r=10, t=20, b=110 if legend_items > 1 else 40)
    fig.update_layout(title=dict(text=title, font=dict(size=15)), height=height, **layout)
    return fig


# ------------------------------------------------------------------ confusion / cost matrix
def confusion_matrix_fig(tn, fp, fn, tp, title="Confusion Matrix", usd: dict | None = None) -> go.Figure:
    """2x2 matrix. If `usd` is given ({'tn','fp','fn','tp'} -> text) the dollar impact is shown in each cell."""
    z = [[0, 1], [1, 0]]  # 0 = correct decision, 1 = costly error
    counts = [[tn, fp], [fn, tp]]
    names = [["True Negative<br>Legit disetujui", "False Positive<br>Legit terblokir"],
             ["False Negative<br>Fraud lolos", "True Positive<br>Fraud tertangkap"]]
    keys = [["tn", "fp"], ["fn", "tp"]]
    text = []
    for i in range(2):
        row = []
        for j in range(2):
            extra = f"<br><i>{usd[keys[i][j]]}</i>" if usd else ""
            row.append(f"<b>{fmt_int(counts[i][j])}</b><br>{names[i][j]}{extra}")
        text.append(row)
    fig = go.Figure(go.Heatmap(
        z=z, x=["Prediksi: Aman", "Prediksi: Fraud"], y=["Aktual: Legit", "Aktual: Fraud"],
        text=text, texttemplate="%{text}", textfont=dict(size=13, color="white"),
        colorscale=[[0, "#0F9D77"], [1, "#D64550"]], showscale=False, hoverinfo="skip", xgap=4, ygap=4))
    fig.update_yaxes(autorange="reversed")
    return _base(fig, title, 360)


def policy_cost_fig(labels, fraud_loss, friction) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(name="Kerugian fraud lolos (FN)", x=labels, y=fraud_loss, marker_color=C.C_FRAUD,
                text=[fmt_usd(v) for v in fraud_loss], textposition="inside")
    fig.add_bar(name="Biaya gesekan pelanggan sah (FP)", x=labels, y=friction, marker_color=C.C_WARN,
                text=[fmt_usd(v) for v in friction], textposition="inside")
    totals = [a + b for a, b in zip(fraud_loss, friction)]
    fig.add_scatter(x=labels, y=totals, mode="text", text=[f"<b>{fmt_usd(t)}</b>" for t in totals],
                    textposition="top center", showlegend=False)
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title="Total net cost (USD)", rangemode="tozero")
    return _base(fig, "Trade-off biaya: fraud lolos vs pelanggan sah terblokir", 380)


# ------------------------------------------------------------------ threshold tuning
def threshold_curves_fig(sweep: pd.DataFrame, tau: float, f1_best: float) -> go.Figure:
    d = sweep[(sweep.threshold >= 0.10) & (sweep.threshold <= 0.90)]
    fig = go.Figure()
    for col, name, color in [("precision", "Precision", C.C_LEGIT), ("recall", "Recall", C.C_FRAUD),
                             ("f1_score", "F1-score", C.C_GOOD)]:
        fig.add_scatter(x=d.threshold, y=d[col], name=name, mode="lines", line=dict(color=color, width=3))
    fig.add_vline(x=tau, line_width=2, line_dash="dash", line_color=C.C_ACCENT,
                  annotation_text=f"τ = {tau:.2f}".replace(".", ","), annotation_position="top")
    fig.add_vline(x=f1_best, line_width=1, line_dash="dot", line_color=C.C_MUTED,
                  annotation_text="F1 optimal", annotation_position="bottom right")
    fig.update_xaxes(title="Threshold τ")
    fig.update_yaxes(title="Skor", tickformat=".0%", range=[0, 1.02])
    return _base(fig, "Precision – Recall – F1 terhadap threshold", 380)


def cost_curve_fig(curve: pd.DataFrame, tau: float, tau_min: float) -> go.Figure:
    d = curve[(curve.tau >= 0.10) & (curve.tau <= 0.90)]
    fig = go.Figure()
    fig.add_scatter(x=d.tau, y=d.fraud_loss, name="Fraud lolos (FN)", stackgroup="c", line=dict(width=0),
                    fillcolor="rgba(239,68,68,.55)")
    fig.add_scatter(x=d.tau, y=d.friction_cost, name="Gesekan pelanggan (FP)", stackgroup="c",
                    line=dict(width=0), fillcolor="rgba(245,158,11,.55)")
    fig.add_scatter(x=d.tau, y=d.net_cost, name="Total net cost", mode="lines", line=dict(color=C.C_ACCENT, width=3))
    fig.add_vline(x=tau, line_width=2, line_dash="dash", line_color=C.C_ACCENT)
    fig.add_scatter(x=[tau_min], y=[float(d.loc[np.isclose(d.tau, tau_min), "net_cost"].iloc[0])],
                    mode="markers+text", name="Biaya minimum", marker=dict(size=13, symbol="star", color=C.C_GOOD),
                    text=["min"], textposition="bottom center")
    fig.update_xaxes(title="Threshold τ")
    fig.update_yaxes(title="USD (hold-out 4.727 transaksi)", rangemode="tozero")
    return _base(fig, "Total net cost vs threshold", 380)


# ------------------------------------------------------------------ feature importance & EDA
def feature_importance_fig(imp: pd.DataFrame, model: str) -> go.Figure:
    d = imp[["feature", model]].sort_values(model)
    colors = [C.C_FRAUD if f in C.STRONG_FEATURES else (C.C_MUTED if f in C.WEAK_FEATURES else C.C_LEGIT)
              for f in d.feature]
    fig = go.Figure(go.Bar(x=d[model], y=d.feature, orientation="h", marker_color=colors,
                           text=[fmt_pct(v, 1) for v in d[model]], textposition="outside"))
    fig.update_xaxes(tickformat=".0%", title="Kontribusi terhadap keputusan model", range=[0, d[model].max() * 1.18])
    return _base(fig, f"Fraud risk drivers – {model}", 400)


def risk_profile_fig(profile: pd.DataFrame, baseline_pct: float) -> go.Figure:
    factors = ["Transaction Amount", "Account Age", "Transaction Hour", "Payment Method", "Device Used",
               "Address Mismatch"]
    d = profile[profile.factor.isin(factors)].copy()
    d["strong"] = d.factor.isin(["Transaction Amount", "Account Age", "Transaction Hour"])
    fig = make_subplots(rows=2, cols=3, subplot_titles=factors, vertical_spacing=0.22, horizontal_spacing=0.06,
                        shared_yaxes=True)
    for k, f in enumerate(factors):
        s = d[d.factor == f]
        fig.add_bar(x=s.segment, y=s.fraud_rate_pct, row=k // 3 + 1, col=k % 3 + 1,
                    marker_color=C.C_FRAUD if f in C.STRONG_FEATURES else C.C_MUTED, showlegend=False,
                    text=[fmt_pct(v, 1, True) for v in s.fraud_rate_pct], textposition="outside",
                    customdata=np.stack([s.transactions, s.frauds], axis=-1),
                    hovertemplate="%{x}<br>Fraud rate %{y:.2f}%<br>%{customdata[0]:,} transaksi, "
                                  "%{customdata[1]:,} fraud<extra></extra>")
        fig.add_hline(y=baseline_pct, line_dash="dot", line_color=C.C_ACCENT, row=k // 3 + 1, col=k % 3 + 1)
    fig.update_yaxes(title_text="Fraud rate (%)", col=1)
    fig.update_yaxes(range=[0, max(60, d.fraud_rate_pct.max() * 1.2)])
    fig.update_xaxes(tickfont=dict(size=10))
    fig.update_layout(height=560, title=dict(text="Fraud rate aktual per segmen (garis putus = baseline "
                                                  f"{baseline_pct:.2f}%)".replace(".", ","), font=dict(size=15)),
                      **{k: v for k, v in _LAYOUT.items() if k not in ("legend", "margin")},
                      margin=dict(l=10, r=10, t=60, b=40))
    return fig


def benchmark_fig(models: dict) -> go.Figure:
    order = ["Sklearn_LogisticRegression", "Sklearn_RandomForest", "XGBoost", "XGBoost_Tuned",
             "Spark_LogisticRegression", "Spark_RandomForest", "Spark_GBTClassifier"]
    rows = [models[k] for k in order if k in models]
    names = [r["model_name"] for r in rows]
    fig = go.Figure()
    for metric, label, color in [("precision", "Precision", C.C_LEGIT), ("recall", "Recall", C.C_FRAUD),
                                 ("f1_score", "F1", C.C_GOOD), ("pr_auc", "PR-AUC", C.C_ACCENT)]:
        fig.add_bar(name=label, x=names, y=[r[metric] for r in rows], marker_color=color)
    fig.update_layout(barmode="group")
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    fig.update_xaxes(tickangle=-25)
    return _base(fig, "Perbandingan model (hold-out 4.727 transaksi; Spark: 4.718)", 430)


def scalability_fig(bench: dict) -> go.Figure:
    fig = go.Figure()
    labels = {"Sklearn_LR_sec": "LogReg (sklearn)", "XGBoost_sec": "XGBoost", "Spark_LR_sec": "LogReg (Spark)",
              "Spark_RF_sec": "RandomForest (Spark)"}
    for scale, name, color in [("scale_v2_23k", "23 ribu baris", C.C_LEGIT), ("scale_v1_200k", "200 ribu baris", C.C_WARN)]:
        b = bench[scale]
        keys = [k for k in labels if k in b]
        fig.add_bar(name=name, x=[labels[k] for k in keys], y=[b[k] for k in keys], marker_color=color,
                    text=[f"{b[k]:.1f}s" for k in keys], textposition="outside")
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Waktu training (detik)")
    return _base(fig, "Skalabilitas: single-node vs Spark MLlib", 380)


# ------------------------------------------------------------------ graph
_REL_COLOR = {"IP Subnet /16": C.C_WARN, "Lokasi": "rgba(148,163,184,.7)", "IP Subnet /16 + Lokasi": C.C_ACCENT}


def network_fig(nodes: pd.DataFrame, edges: pd.DataFrame, ring_ids: set[int], title: str) -> go.Figure:
    fig = go.Figure()
    for rel, color in _REL_COLOR.items():
        e = edges[edges.relation == rel]
        if e.empty:
            continue
        pos = nodes.set_index("node_id")[["x", "y"]]
        xs, ys = [], []
        for s, t in zip(e.source, e.target):
            xs += [pos.at[s, "x"], pos.at[t, "x"], None]
            ys += [pos.at[s, "y"], pos.at[t, "y"], None]
        fig.add_scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=1.6), hoverinfo="skip",
                        name=f"Edge: {rel}")
    for flag, name, color in [(0, "Transaksi legit", C.C_LEGIT), (1, "Transaksi fraud", C.C_FRAUD)]:
        n = nodes[nodes.is_fraud == flag]
        fig.add_scatter(
            x=n.x, y=n.y, mode="markers", name=name,
            marker=dict(color=color, size=np.clip(6 + np.sqrt(n.amount) * 0.5, 7, 22), opacity=0.9,
                        line=dict(width=0.6, color="rgba(255,255,255,.7)")),
            customdata=np.stack([n.node_id, n.community_id, n.amount, n.hour, n.account_age_days, n.subnet,
                                 n.location, n.ship_state, n.oof_prob], axis=-1),
            hovertemplate="<b>%{customdata[0]}</b> · Komunitas %{customdata[1]}<br>Nominal $%{customdata[2]:,.2f}"
                          "<br>Jam %{customdata[3]}:00 · Umur akun %{customdata[4]} hari"
                          "<br>Subnet /16: %{customdata[5]}<br>Lokasi: %{customdata[6]} · State kirim: %{customdata[7]}"
                          "<br>Skor model tabular: %{customdata[8]:.2f}<extra></extra>")
    ring = nodes[nodes.community_id.isin(ring_ids)]
    if not ring.empty:
        fig.add_scatter(x=ring.x, y=ring.y, mode="markers", name="Anggota fraud ring (100%)", hoverinfo="skip",
                        marker=dict(symbol="circle-open", size=26, color=C.C_RING, line=dict(width=3)))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x")
    return _base(fig, title, 620)


def ring_detail_fig(g: pd.DataFrame) -> go.Figure:
    """Layered view of ONE community: Subnet hubs (top), transactions, Location hubs, shipping-state hubs."""
    g = g.sort_values(["subnet", "location", "node_id"]).reset_index(drop=True)
    tx_x = {n: i for i, n in enumerate(g.node_id)}
    layers = {"subnet": 1.0, "location": -1.0, "ship_state": -2.0}
    label = {"subnet": "IP Subnet /16", "location": "Lokasi pelanggan", "ship_state": "State pengiriman"}
    fig = go.Figure()
    hubs = []
    for col, y in layers.items():
        for val, grp in g.groupby(col):
            xs = [tx_x[n] for n in grp.node_id]
            hubs.append(dict(layer=col, val=val, x=float(np.mean(xs)), y=y, n=len(grp)))
            for n in grp.node_id:
                shared = len(grp) > 1
                fig.add_scatter(x=[float(np.mean(xs)), tx_x[n]], y=[y, 0], mode="lines", showlegend=False,
                                hoverinfo="skip",
                                line=dict(width=2.2 if shared else 0.8,
                                          color=C.C_WARN if shared else "rgba(148,163,184,.45)"))
    h = pd.DataFrame(hubs)
    for col, sym in [("subnet", "diamond"), ("location", "square"), ("ship_state", "triangle-down")]:
        s = h[h.layer == col]
        fig.add_scatter(x=s.x, y=s.y, mode="markers+text", name=label[col], text=s.val, textposition="top center" if col == "subnet" else "bottom center",
                        marker=dict(symbol=sym, size=np.where(s.n > 1, 20, 11), color=np.where(s.n > 1, C.C_RING, C.C_MUTED),
                                    line=dict(width=1, color="rgba(255,255,255,.7)")),
                        customdata=s.n, hovertemplate=f"{label[col]}: %{{text}}<br>%{{customdata}} transaksi terhubung<extra></extra>")
    for flag, name, color in [(0, "Transaksi legit", C.C_LEGIT), (1, "Transaksi fraud", C.C_FRAUD)]:
        n = g[g.is_fraud == flag]
        fig.add_scatter(x=[tx_x[i] for i in n.node_id], y=[0] * len(n), mode="markers+text", name=name,
                        text=[f"${a:,.0f}" for a in n.amount], textposition="top center",
                        marker=dict(color=color, size=18, line=dict(width=1, color="white")),
                        customdata=np.stack([n.node_id, n.amount, n.hour, n.account_age_days, n.oof_prob], axis=-1),
                        hovertemplate="<b>%{customdata[0]}</b><br>Nominal $%{customdata[1]:,.2f}"
                                      "<br>Jam %{customdata[2]}:00 · Umur akun %{customdata[3]} hari"
                                      "<br>Skor model tabular %{customdata[4]:.2f}<extra></extra>")
    fig.update_xaxes(visible=False, range=[-0.8, max(len(g) - 1, 1) + 0.8])
    fig.update_yaxes(visible=False, range=[-2.9, 1.8])
    return _base(fig, "Struktur komunitas: transaksi vs infrastruktur bersama", 460)


def baseline_fig(items: list[tuple[str, float, str]]) -> go.Figure:
    fig = go.Figure(go.Bar(x=[i[0] for i in items], y=[i[1] for i in items], marker_color=[i[2] for i in items],
                           text=[fmt_pct(i[1], 2, True) for i in items], textposition="outside"))
    fig.update_yaxes(title="Fraud rate (%)", range=[0, 118])
    fig.update_xaxes(tickangle=-12)
    return _base(fig, "Fraud rate: populasi vs jaringan vs ring", 380)


# ------------------------------------------------------------------ simulator
def gauge_fig(prob: float, review_from: float, block_from: float) -> go.Figure:
    color = C.C_GOOD if prob < review_from else (C.C_WARN if prob < block_from else C.C_FRAUD)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=prob * 100, number=dict(suffix="%", valueformat=".1f"),
        title=dict(text="Probabilitas fraud"),
        gauge=dict(axis=dict(range=[0, 100]), bar=dict(color=color, thickness=0.3),
                   steps=[dict(range=[0, review_from * 100], color="rgba(16,185,129,.25)"),
                          dict(range=[review_from * 100, block_from * 100], color="rgba(245,158,11,.25)"),
                          dict(range=[block_from * 100, 100], color="rgba(239,68,68,.25)")],
                   threshold=dict(line=dict(color=color, width=4), value=prob * 100))))
    fig.update_layout(height=290, margin=dict(l=20, r=20, t=60, b=10), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def contribution_fig(contrib: pd.DataFrame) -> go.Figure:
    d = contrib.sort_values("delta")
    fig = go.Figure(go.Bar(x=d.delta * 100, y=d.factor, orientation="h",
                           marker_color=[C.C_FRAUD if v > 0.005 else C.C_MUTED for v in d.delta],
                           text=[f"{v * 100:+.1f} pp".replace(".", ",") for v in d.delta], textposition="outside"))
    fig.update_xaxes(title="Selisih probabilitas jika faktor dinormalkan (poin persentase)")
    return _base(fig, "Kontribusi tiap faktor terhadap skor risiko", 330)
