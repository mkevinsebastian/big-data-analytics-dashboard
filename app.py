"""Fraud Detection Dashboard - entry point.  Run with:  streamlit run app.py"""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Fraud Detection Dashboard · BINUS MTI", page_icon="🛡️", layout="wide",
                   initial_sidebar_state="expanded")

from dashboard import config as C  # noqa: E402  (must come after set_page_config)
from dashboard import data  # noqa: E402
from dashboard.theme import fmt_int, hero, inject_css  # noqa: E402
from dashboard.views import graph_view, ml_engine, overview, simulator  # noqa: E402


def sidebar() -> C.CostAssumptions:
    meta = data.ml_results()["metadata"]
    kpi = data.kpi_summary()
    with st.sidebar:
        st.markdown("### 🛡️ Fraud Detection")
        st.caption(f"{meta['course']}\n\nFinal Project (AOL) · BINUS Graduate Program, Magister Teknik Informatika")
        st.markdown(f"**{meta['student_name']}** · NIM {meta['nim']}")
        st.divider()
        st.markdown("#### Asumsi biaya")
        st.caption("Dipakai untuk Total Net Cost dan Financial Loss Prevented. Ini asumsi perencanaan, bukan data terukur.")
        fee = st.number_input("Denda chargeback per fraud lolos (USD)", 0.0, 500.0, C.CostAssumptions.chargeback_fee, 5.0,
                              help="Laporan menyebut kisaran USD 20–100 per kasus.")
        lost = st.slider("Penjualan hilang saat pelanggan sah diblokir", 0, 100, int(C.CostAssumptions.lost_sale_rate * 100),
                         5, format="%d%%", help="Porsi nilai pesanan yang batal / pelanggan pindah ke kompetitor.")
        handling = st.number_input("Biaya penanganan per pelanggan sah terblokir (USD)", 0.0, 100.0,
                                   C.CostAssumptions.handling_cost, 1.0)
        st.divider()
        st.markdown("#### Sumber data")
        st.caption(f"Dataset: {fmt_int(kpi['total_transactions'])} transaksi e-commerce "
                   f"(rasio imbalance {str(kpi['imbalance_ratio']).replace('.', ',')} : 1).\n\n"
                   "Hasil eksperimen: Assignment II (ML + Graph Analytics). Hasil dijalankan pada "
                   f"{meta['generated_at']}.")
        st.caption("Tema terang/gelap: menu ⋮ di kanan atas → Settings → Theme.")
    return C.CostAssumptions(chargeback_fee=fee, lost_sale_rate=lost / 100.0, handling_cost=handling)


def main() -> None:
    inject_css()
    cost = sidebar()
    hero("Executive & Operational Fraud Detection Dashboard",
         "Menerjemahkan hasil Big Data, Machine Learning, dan Graph Analytics menjadi keputusan bisnis.")
    t1, t2, t3, t4 = st.tabs(["📊 Executive Overview", "🤖 ML Decision Engine", "🕸️ Graph & Fraud Rings",
                              "🧪 Risk Simulator"])
    with t1:
        overview.render(cost)
    with t2:
        ml_engine.render(cost)
    with t3:
        graph_view.render()
    with t4:
        simulator.render()


main()
