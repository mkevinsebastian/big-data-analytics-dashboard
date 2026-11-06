"""Tab 1 - Executive Overview & Financial Impact."""
from __future__ import annotations

import streamlit as st

from .. import charts, data, insights
from .. import config as C
from ..analytics import evaluate, no_model_cost, row_at
from ..theme import plot, callout, fmt_int, fmt_num, fmt_pct, fmt_usd, insight_card, kpi_card


def render(cost: C.CostAssumptions) -> None:
    kpi, sw = data.kpi_summary(), data.sweep()
    tune, prof, gr = insights.tuning(), insights.risk_profile(), insights.graph()

    scope = st.radio("Skala angka", ["Proyeksi seluruh dataset (23.634 transaksi)",
                                     "Hold-out test (4.727 transaksi, terukur)"], horizontal=True,
                     help="Model dievaluasi pada hold-out 20% (4.727 transaksi). Proyeksi mengalikan hasil "
                          "hold-out dengan 23.634 / 4.727 ≈ 5,0 untuk menggambarkan seluruh dataset.")
    projected = scope.startswith("Proyeksi")
    scale = kpi["total_transactions"] / kpi["test_transactions"] if projected else 1.0
    n_total = kpi["total_transactions"] if projected else kpi["test_transactions"]
    n_fraud = kpi["fraud_count"] if projected else kpi["test_frauds"]
    tag = " (proyeksi)" if projected else " (hold-out)"

    opt = evaluate(row_at(sw, C.OPTIMAL_TAU), cost, scale)
    base = evaluate(row_at(sw, C.DEFAULT_TAU), cost, scale)
    r_opt = row_at(sw, C.OPTIMAL_TAU)

    # ---------------------------------------------------------------- three mandatory insights
    c1, c2, c3 = st.columns(3)
    c1.markdown(insight_card(
        "INSIGHT 1 · FINANSIAL", f"τ = 0,80 memangkas False Positive {fmt_num(tune['fp_cut_pct'], 0)}%",
        f"Pelanggan sah yang salah diblokir turun dari {fmt_int(tune['fp_before'])} menjadi "
        f"{fmt_int(tune['fp_after'])}; presisi naik dari {fmt_pct(tune['prec_before'])} ke "
        f"{fmt_pct(tune['prec_after'])} (hold-out).", C.C_GOOD), unsafe_allow_html=True)
    c2.markdown(insight_card(
        "INSIGHT 2 · PROFIL RISIKO", "Fraud menumpuk pada akun baru, nominal besar, dini hari",
        f"Fraud rate {fmt_pct(prof['amount_high'], 1, True)} untuk transaksi > $1.000, "
        f"{fmt_pct(prof['new_account'], 1, True)} untuk akun ≤ 30 hari, {fmt_pct(prof['dawn'], 1, True)} "
        f"pukul 00:00–05:59, dibanding baseline {fmt_pct(prof['baseline'], 2, True)}. "
        "Fokuskan velocity check pada transaksi baru.", C.C_FRAUD), unsafe_allow_html=True)
    c3.markdown(insight_card(
        "INSIGHT 3 · GRAPH ANALYTICS", f"{gr['n_rings']} fraud ring 100% terungkap lewat jaringan",
        f"Pasangan transaksi fraud yang berbagi IP subnet atau lokasi. Model tabular melewatkan "
        f"{gr['ring_missed']} dari {gr['ring_members']} anggota ring pada τ = 0,80 saat dinilai satu per satu.",
        C.C_ACCENT), unsafe_allow_html=True)

    st.write("")
    # ---------------------------------------------------------------- KPI cards
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(kpi_card("Total transaksi diproses" + tag, fmt_int(n_total),
                         f"{fmt_int(n_fraud)} fraud aktual ({fmt_pct(kpi['fraud_rate_pct'], 2, True)}), "
                         f"rasio {fmt_num(kpi['imbalance_ratio'])} : 1", C.C_LEGIT), unsafe_allow_html=True)
    k2.markdown(kpi_card("Fraud terdeteksi" + tag, fmt_int(opt.tp),
                         f"recall {fmt_pct(opt.recall, 1)} · presisi {fmt_pct(opt.precision, 1)} pada τ = 0,80",
                         C.C_FRAUD), unsafe_allow_html=True)
    k3.markdown(kpi_card("Financial loss prevented" + tag, fmt_usd(opt.loss_prevented),
                         f"net benefit setelah biaya gesekan: {fmt_usd(opt.net_benefit)}", C.C_GOOD),
                unsafe_allow_html=True)
    k4.markdown(kpi_card("False positive friction rate", fmt_pct(opt.fp_rate, 2),
                         f"vs {fmt_pct(base.fp_rate, 2)} pada τ = 0,50 · "
                         f"{fmt_int(opt.fp)} pelanggan sah terblokir{tag}", C.C_WARN), unsafe_allow_html=True)

    st.write("")
    # ---------------------------------------------------------------- cost-benefit matrix
    st.subheader("Cost-benefit trade-off matrix")
    left, right = st.columns([1, 1])
    usd = {
        "tp": f"+{fmt_usd(opt.loss_prevented)} kerugian dicegah",
        "fn": f"−{fmt_usd(opt.fraud_loss)} kerugian fraud",
        "fp": f"−{fmt_usd(opt.friction_cost)} gesekan pelanggan",
        "tn": f"{fmt_usd(float(r_opt.legit_usd_approved) * scale)} GMV sah aman",
    }
    plot(charts.confusion_matrix_fig(opt.tn, opt.fp, opt.fn, opt.tp,
                                                  f"Dampak finansial per kuadran · τ = 0,80{tag}", usd), left)
    nm = no_model_cost(sw, cost) * scale
    plot(charts.policy_cost_fig(
        ["Tanpa model", "Model τ = 0,50", "Model τ = 0,80"],
        [nm, base.fraud_loss, opt.fraud_loss], [0, base.friction_cost, opt.friction_cost]), right)

    avg_fn = kpi["avg_fraud_amount"] + cost.chargeback_fee
    avg_fp = kpi["avg_legit_amount"] * cost.lost_sale_rate + cost.handling_cost
    callout(f"Ekonomi per kasus: satu fraud yang lolos (FN) rata-rata merugikan {fmt_usd(avg_fn)} "
            f"(nominal rata-rata {fmt_usd(kpi['avg_fraud_amount'])} + chargeback {fmt_usd(cost.chargeback_fee)}), "
            f"sedangkan satu pelanggan sah yang diblokir (FP) rata-rata menelan {fmt_usd(avg_fp)} "
            f"(GMV hilang {fmt_pct(cost.lost_sale_rate, 0)} + penanganan {fmt_usd(cost.handling_cost)}). "
            f"Satu FN setara ±{fmt_num(avg_fn / avg_fp, 1)} FP. Asumsi biaya dapat diubah di sidebar.",
            C.C_WARN)

    # ---------------------------------------------------------------- executive summary
    st.subheader("Ringkasan eksekutif")
    saving = base.net_cost - opt.net_cost
    st.markdown(
        f"""
- **Efektivitas.** Dengan ambang τ = 0,80, sistem menangkap **{fmt_int(opt.tp)} fraud{tag}** bernilai
  **{fmt_usd(opt.fraud_usd_caught)}** dan mencegah kerugian total **{fmt_usd(opt.loss_prevented)}**
  (termasuk denda chargeback yang dihindari).
- **Pengalaman pelanggan.** Dibanding ambang standar 0,50, pelanggan sah yang salah diblokir turun
  **{fmt_num(tune['fp_cut_pct'], 0)}%** ({fmt_int(tune['fp_before'])} → {fmt_int(tune['fp_after'])} pada hold-out).
  Presisi naik {fmt_pct(tune['prec_before'])} → {fmt_pct(tune['prec_after'])}: porsi pelanggan sah di antara transaksi
  yang diblokir turun dari {fmt_pct(1 - tune['prec_before'], 0)} menjadi {fmt_pct(1 - tune['prec_after'], 0)}.
- **Dampak biaya.** Total net cost{tag} berubah dari {fmt_usd(base.net_cost)} (τ = 0,50) menjadi
  {fmt_usd(opt.net_cost)} (τ = 0,80): {'penghematan' if saving >= 0 else 'kenaikan'} {fmt_usd(abs(saving))}
  pada asumsi biaya saat ini.
- **Yang masih kurang.** Recall turun dari {fmt_pct(tune['rec_before'], 1)} ke {fmt_pct(tune['rec_after'], 1)}, jadi
  sekitar dua pertiga fraud masih lolos pada τ = 0,80. Perlu lapisan kontrol tambahan: velocity check untuk akun
  baru, nominal besar, dan dini hari, serta investigasi graph untuk ring terorganisir.
- **Catatan data.** Dataset bersifat simulasi (fitur seperti payment method, device, dan address mismatch tidak
  membedakan fraud), sehingga angka absolut adalah ilustrasi metodologi, bukan prediksi bisnis nyata.
""")
