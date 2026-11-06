"""Tab 2 - Machine Learning Decision Engine (Insight 1: threshold tuning, Insight 2: risk drivers)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from .. import charts, data, insights
from .. import config as C
from ..analytics import cost_curve, evaluate, reduction_pct, row_at
from ..theme import plot, callout, fmt_int, fmt_num, fmt_pct, fmt_usd, kpi_card


def render(cost: C.CostAssumptions) -> None:
    sw, kpi = data.sweep(), data.kpi_summary()
    tune, prof = insights.tuning(), insights.risk_profile()

    # ================================================================ Insight 1 - threshold tuning
    st.subheader("Insight 1 · Interactive threshold tuning")
    st.caption("Model: XGBoost (scale_pos_weight = 18,34). Seluruh angka memakai hold-out 20% "
               f"({fmt_int(kpi['test_transactions'])} transaksi, {fmt_int(kpi['test_frauds'])} fraud) dan "
               "identik dengan laporan Assignment II.")
    tau = st.slider("Ambang batas keputusan τ (transaksi diblokir jika probabilitas ≥ τ)",
                    min_value=0.10, max_value=0.90, value=C.OPTIMAL_TAU, step=0.01, format="%.2f", key="tau")
    cur = evaluate(row_at(sw, tau), cost)
    ref = evaluate(row_at(sw, C.DEFAULT_TAU), cost)
    best = evaluate(row_at(sw, C.OPTIMAL_TAU), cost)

    m = st.columns(5)
    m[0].markdown(kpi_card("Precision", fmt_pct(cur.precision), f"τ = 0,50: {fmt_pct(ref.precision)}", C.C_LEGIT),
                  unsafe_allow_html=True)
    m[1].markdown(kpi_card("Recall", fmt_pct(cur.recall), f"τ = 0,50: {fmt_pct(ref.recall)}", C.C_FRAUD),
                  unsafe_allow_html=True)
    m[2].markdown(kpi_card("F1-score", fmt_num(cur.f1, 4), f"τ = 0,50: {fmt_num(ref.f1, 4)}", C.C_GOOD),
                  unsafe_allow_html=True)
    m[3].markdown(kpi_card("False positive", fmt_int(cur.fp), f"τ = 0,50: {fmt_int(ref.fp)}", C.C_WARN),
                  unsafe_allow_html=True)
    m[4].markdown(kpi_card("Total net cost", fmt_usd(cur.net_cost),
                           f"FN {fmt_usd(cur.fraud_loss)} + FP {fmt_usd(cur.friction_cost)}", C.C_ACCENT),
                  unsafe_allow_html=True)
    st.write("")

    cl, cr = st.columns([1, 1])
    plot(charts.confusion_matrix_fig(cur.tn, cur.fp, cur.fn, cur.tp,
                                                f"Confusion matrix · τ = {fmt_num(tau)}"), cl)
    plot(charts.threshold_curves_fig(sw, tau, C.OPTIMAL_TAU), cr)

    fp_cut = reduction_pct(ref.fp, cur.fp)
    if abs(tau - C.OPTIMAL_TAU) < 1e-9:
        callout(f"Bukti empiris τ = 0,80: False Positive turun {fmt_num(fp_cut, 0)}% "
                f"({fmt_int(ref.fp)} → {fmt_int(cur.fp)} pelanggan sah) dan presisi naik dari "
                f"{fmt_pct(ref.precision)} menjadi {fmt_pct(cur.precision)}. F1 tertinggi seluruh grid "
                f"({fmt_num(cur.f1, 4)}). Harga yang dibayar: recall turun dari {fmt_pct(ref.recall, 1)} ke "
                f"{fmt_pct(cur.recall, 1)}.", C.C_GOOD)
    else:
        delta = "turun" if fp_cut >= 0 else "naik"
        callout(f"Pada τ = {fmt_num(tau)}: False Positive {delta} {fmt_num(abs(fp_cut), 0)}% dibanding τ = 0,50 "
                f"({fmt_int(ref.fp)} → {fmt_int(cur.fp)}), presisi {fmt_pct(cur.precision)}, recall "
                f"{fmt_pct(cur.recall, 1)}. Geser ke τ = 0,80 untuk melihat titik F1 optimal "
                f"(FP −{fmt_num(reduction_pct(ref.fp, best.fp), 0)}%, presisi {fmt_pct(best.precision)}).",
                C.C_ACCENT)

    curve = cost_curve(sw, cost)
    grid = curve[(curve.tau >= 0.10) & (curve.tau <= 0.90)]
    tau_min = float(grid.loc[grid.net_cost.idxmin(), "tau"])
    plot(charts.cost_curve_fig(curve, tau, tau_min), st)
    if abs(tau_min - C.OPTIMAL_TAU) <= 0.05:
        callout(f"Dengan asumsi biaya saat ini, net cost terendah ada di τ ≈ {fmt_num(tau_min)}, berdekatan dengan "
                f"τ F1-optimal 0,80. Titik ini bergeser bila biaya chargeback atau tingkat kehilangan penjualan "
                f"diubah di sidebar.", C.C_GOOD)
    else:
        callout(f"Dengan asumsi biaya saat ini, net cost terendah ada di τ ≈ {fmt_num(tau_min)} "
                f"({fmt_usd(float(grid.net_cost.min()))}), berbeda dari τ F1-optimal 0,80 "
                f"({fmt_usd(best.net_cost)}). Pilihan τ adalah keputusan bisnis: F1 menyeimbangkan precision "
                f"dan recall, sedangkan biaya bergantung pada asumsi di sidebar.", C.C_WARN)

    # ================================================================ Insight 2 - risk drivers
    st.divider()
    st.subheader("Insight 2 · Fraud risk drivers")
    imp = data.feature_importance()
    a, b = st.columns([1, 1])
    with a:
        model = st.radio("Model", ["XGBoost", "Random Forest"], horizontal=True, key="imp_model")
        plot(charts.feature_importance_fig(imp, model), st)
        st.caption("Merah = prediktor kuat · abu-abu = fitur tanpa daya beda · biru = fitur lain. "
                   "Kontribusi fitur turunan (mis. log amount, flag akun baru, hour sin/cos) dijumlahkan ke fitur bisnisnya. "
                   "Peringkat berbeda antar-model: XGBoost (gain) paling bertumpu pada Account Age, Random Forest pada "
                   "Transaction Amount. Ketiga prediktor kuat konsisten berada di atas Payment Method, Device Used, dan Address Mismatch.")
    with b:
        st.markdown("##### Fraud rate aktual per segmen")
        d = data.risk_profile()
        d = d[d.factor.isin(["Transaction Amount", "Account Age", "Transaction Hour"])]
        st.dataframe(
            d[["factor", "segment", "fraud_rate_pct"]].rename(
                columns={"factor": "Faktor", "segment": "Segmen", "fraud_rate_pct": "Fraud rate (%)"}),
            hide_index=True, width="stretch",
            column_config={"Fraud rate (%)": st.column_config.ProgressColumn(
                "Fraud rate (%)", min_value=0, max_value=60, format="%.2f%%")})
    plot(charts.risk_profile_fig(data.risk_profile(), prof["baseline"]), st)
    callout(f"Anomali terkonsentrasi: transaksi > $1.000 punya fraud rate {fmt_pct(prof['amount_high'], 1, True)} "
            f"(vs {fmt_pct(prof['amount_low'], 1, True)} untuk < $100), akun ≤ 30 hari {fmt_pct(prof['new_account'], 1, True)}, "
            f"dini hari {fmt_pct(prof['dawn'], 1, True)}. Bila ketiganya terjadi bersamaan "
            f"({fmt_int(prof['all_three_n'])} transaksi) fraud rate mencapai {fmt_pct(prof['all_three'], 1, True)}; "
            f"tanpa satu pun faktor risiko hanya {fmt_pct(prof['no_risk'], 1, True)}. Payment method, device, dan "
            f"address mismatch tidak membedakan fraud (≈ 5% di semua kategori). Rekomendasi: velocity check pada "
            f"transaksi akun baru bernilai besar atau dini hari.", C.C_FRAUD)

    # ================================================================ model benchmark
    with st.expander("Benchmark model: scikit-learn / XGBoost vs Spark MLlib"):
        res = data.ml_results()
        plot(charts.benchmark_fig(res["evaluation_metrics"]), st)
        rows = []
        for v in res["evaluation_metrics"].values():
            rows.append({"Model": v["model_name"], "Precision": v["precision"], "Recall": v["recall"],
                         "F1": v["f1_score"], "ROC-AUC": v["roc_auc"], "PR-AUC": v["pr_auc"],
                         "FP": v["confusion_matrix"]["fp"], "FN": v["confusion_matrix"]["fn"],
                         "Training (detik)": round(v["training_time_sec"], 2)})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        plot(charts.scalability_fig(res["scalability_benchmark"]), st)
        st.caption("Pada 23 ribu baris, single-node jauh lebih cepat (overhead Spark mendominasi). Keunggulan "
                   "Spark baru relevan pada volume data yang jauh lebih besar dan terdistribusi.")
