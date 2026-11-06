"""Tab 3 - Graph Analytics & Fraud Rings Investigation (Insight 3)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from .. import charts, data, insights
from .. import config as C
from ..theme import plot, callout, fmt_int, fmt_num, fmt_pct, kpi_card

VIEW_OPTIONS = ["Fraud ring 100%", "Komunitas makro (≥ 8 transaksi)", "Komunitas yang berisi fraud", "Semua komunitas"]


def _repack(nodes: pd.DataFrame) -> pd.DataFrame:
    """Re-arrange the selected communities on a compact grid (positions inside a community are kept)."""
    nodes = nodes.copy()
    ids = list(nodes.community_id.unique())
    rel = {}
    radius = 0.4
    for cid in ids:
        g = nodes[nodes.community_id == cid]
        cx, cy = g.x.mean(), g.y.mean()
        rel[cid] = (cx, cy)
        radius = max(radius, float(np.hypot(g.x - cx, g.y - cy).max()))
    cols = int(np.ceil(np.sqrt(len(ids))))
    step = 2 * radius + 0.7
    ox = nodes.community_id.map(lambda c: (ids.index(c) % cols) * step)
    oy = nodes.community_id.map(lambda c: -(ids.index(c) // cols) * step)
    nodes["x"] = nodes.x - nodes.community_id.map(lambda c: rel[c][0]) + ox
    nodes["y"] = nodes.y - nodes.community_id.map(lambda c: rel[c][1]) + oy
    return nodes


def render() -> None:
    gres, kpi = data.graph_results(), data.kpi_summary()
    comm, nodes, edges = data.graph_communities(), data.graph_nodes(), data.graph_edges()
    gr = insights.graph()
    net, het = gres["transaction_risk_network"], gres["heterogeneous_graph"]
    ring_ids = set(gr["rings"].community_id)

    # ------------------------------------------------------------------ KPIs
    k = st.columns(5)
    k[0].markdown(kpi_card("Heterogeneous graph", fmt_int(het["total_nodes"]),
                           f"simpul · {fmt_int(het['total_edges'])} relasi (transaksi, subnet, lokasi, state)",
                           C.C_LEGIT), unsafe_allow_html=True)
    k[1].markdown(kpi_card("Jaringan risiko transaksi", fmt_int(net["nodes"]),
                           f"transaksi · {fmt_int(net['edges'])} edge · modularity Q {fmt_num(net['modularity_q'], 3)}",
                           C.C_ACCENT), unsafe_allow_html=True)
    k[2].markdown(kpi_card("Komunitas terdeteksi", fmt_int(net["total_communities"]),
                           f"greedy modularity · {fmt_int(net['connected_components'])} komponen terhubung",
                           C.C_WARN), unsafe_allow_html=True)
    k[3].markdown(kpi_card("Fraud ring (100%)", fmt_int(gr["n_rings"]),
                           f"{fmt_int(gr['ring_members'])} transaksi fraud saling terhubung", C.C_FRAUD),
                  unsafe_allow_html=True)
    k[4].markdown(kpi_card("Lolos dari model tabular", f"{gr['ring_missed']} / {gr['ring_members']}",
                           "anggota ring di bawah τ = 0,80 saat dinilai sendiri-sendiri", C.C_GOOD),
                  unsafe_allow_html=True)
    st.write("")

    # ------------------------------------------------------------------ explorer
    st.subheader("Syndicate & network explorer")
    mode = st.radio("Tampilan", ["Peta jaringan", "Detail satu komunitas"], horizontal=True, key="graph_mode")

    if mode == "Peta jaringan":
        view = st.selectbox("Komunitas yang ditampilkan", VIEW_OPTIONS, index=1)
        if view == VIEW_OPTIONS[0]:
            keep = comm[comm.community_id.isin(ring_ids)].community_id
        elif view == VIEW_OPTIONS[1]:
            keep = comm[comm["size"] >= insights.MACRO_MIN_SIZE].community_id
        elif view == VIEW_OPTIONS[2]:
            keep = comm[comm.fraud_count > 0].community_id
        else:
            keep = comm.community_id
        sub = nodes[nodes.community_id.isin(keep)]
        if view == VIEW_OPTIONS[3]:
            sub_edges = edges[edges.source.isin(sub.node_id) & edges.target.isin(sub.node_id)]
        else:
            sub = _repack(sub)
            sub_edges = edges[edges.same_community & edges.source.isin(sub.node_id)]
        plot(charts.network_fig(sub, sub_edges, ring_ids,
                               f"{view}: {fmt_int(sub.community_id.nunique())} komunitas, {fmt_int(len(sub))} transaksi, "
                               f"{fmt_int(len(sub_edges))} edge"), st)
        st.caption("Titik = transaksi (ukuran = nominal). Garis = dua transaksi berbagi IP subnet /16 atau lokasi "
                   "pelanggan. Lingkaran kuning = anggota fraud ring 100%. Arahkan kursor untuk detail, "
                   "drag untuk zoom.")
    else:
        pool = comm[(comm.fraud_count >= 2) | (comm.community_id.isin(ring_ids)) | (comm["size"] >= insights.MACRO_MIN_SIZE)]
        pool = pool.sort_values(["fraud_rate_pct", "size"], ascending=False)
        labels = {int(r.community_id): f"Komunitas #{int(r.community_id)} · {int(r['size'])} transaksi · "
                                       f"{fmt_num(r.fraud_rate_pct)}% fraud" + (" · FRAUD RING" if r.community_id in ring_ids else "")
                  for _, r in pool.iterrows()}
        cid = st.selectbox("Pilih komunitas", list(labels), format_func=labels.get)
        g = nodes[nodes.community_id == cid]
        plot(charts.ring_detail_fig(g), st)
        c = comm[comm.community_id == cid].iloc[0]
        shared_sub = g.subnet.value_counts()
        shared_loc = g.location.value_counts()
        callout("Infrastruktur bersama: "
                f"subnet /16 {', '.join(shared_sub[shared_sub > 1].index) or 'tidak ada'} · "
                f"lokasi {', '.join(shared_loc[shared_loc > 1].index) or 'tidak ada'} · "
                f"{int(c.n_ship_states)} state pengiriman berbeda ({', '.join(sorted(g.ship_state.unique()))}).",
                C.C_RING)
        t = g[["node_id", "is_fraud", "amount", "hour", "account_age_days", "subnet", "location", "ship_state", "oof_prob"]].copy()
        t["is_fraud"] = t.is_fraud.map({1: "Fraud", 0: "Legit"})
        t["Model tabular @0,80"] = np.where(t.oof_prob >= C.BLOCK_FROM, "Terblokir", "Lolos")
        st.dataframe(
            t.rename(columns={"node_id": "Transaksi", "is_fraud": "Label", "amount": "Nominal ($)", "hour": "Jam",
                              "account_age_days": "Umur akun (hari)", "subnet": "Subnet /16", "location": "Lokasi",
                              "ship_state": "State kirim", "oof_prob": "Skor model"}),
            hide_index=True, width="stretch",
            column_config={"Skor model": st.column_config.ProgressColumn("Skor model", min_value=0, max_value=1, format="%.2f")})

    # ------------------------------------------------------------------ breakdown
    st.divider()
    st.subheader("Fraud rings breakdown")
    show_cols = ["community_id", "size", "fraud_count", "fraud_rate_pct", "avg_amount", "top_subnet", "top_location",
                 "n_ship_states", "avg_account_age", "dawn_share_pct", "fraud_missed_by_ml"]
    names = {"community_id": "Komunitas", "size": "Ukuran", "fraud_count": "Fraud", "fraud_rate_pct": "Fraud rate",
             "avg_amount": "Rata-rata ($)", "top_subnet": "Subnet bersama", "top_location": "Lokasi bersama",
             "n_ship_states": "State kirim berbeda", "avg_account_age": "Umur akun rata-rata (hari)",
             "dawn_share_pct": "Dini hari (%)", "fraud_missed_by_ml": "Lolos model tabular"}
    cfg = {"Fraud rate": st.column_config.ProgressColumn("Fraud rate", min_value=0, max_value=100, format="%.2f%%")}
    t1, t2 = st.tabs(["Mikro: fraud ring 100%", "Makro: komunitas besar berisiko tinggi"])
    with t1:
        st.dataframe(gr["rings"][show_cols].rename(columns=names), hide_index=True, width="stretch", column_config=cfg)
    with t2:
        st.dataframe(gr["macro"][show_cols].head(10).rename(columns=names), hide_index=True, width="stretch", column_config=cfg)

    macro_top = gr["macro"].iloc[0]
    b1, b2 = st.columns([1, 1])
    plot(charts.baseline_fig([
        ("Populasi (23.634 tx)", kpi["fraud_rate_pct"], C.C_MUTED),
        ("Sampel jaringan (1.500 tx)", kpi["network_sample_fraud_rate_pct"], C.C_LEGIT),
        (f"Komunitas makro #{int(macro_top.community_id)}", float(macro_top.fraud_rate_pct), C.C_WARN),
        ("Fraud ring mikro", 100.0, C.C_FRAUD)]), b1)
    with b2:
        callout(f"Cara membaca: komunitas makro terbesar (#{int(macro_top.community_id)}, {int(macro_top['size'])} transaksi) "
                f"punya fraud rate {fmt_pct(macro_top.fraud_rate_pct, 2, True)}, jauh di atas baseline populasi "
                f"{fmt_pct(kpi['fraud_rate_pct'], 2, True)}. Namun sampel jaringan sengaja disaring ke transaksi yang "
                f"menyentuh subnet atau lokasi yang pernah terkait fraud, sehingga baseline sampelnya sendiri "
                f"{fmt_pct(kpi['network_sample_fraud_rate_pct'], 2, True)}. Pembanding yang adil untuk komunitas makro "
                f"adalah angka terakhir ini. Sinyal yang benar-benar menonjol adalah {gr['n_rings']} ring mikro 100%.",
                C.C_WARN)
        callout(f"Nilai tambah graph (Insight 3): saat dinilai sendiri-sendiri oleh model tabular (skor out-of-fold, "
                f"τ = 0,80), {gr['ring_missed']} dari {gr['ring_members']} transaksi fraud dalam ring lolos. "
                f"Hubungan antar-transaksi hanya terlihat lewat graph. Di seluruh jaringan, "
                f"{gr['fraud_missed_in_network']} dari {gr['fraud_in_network']} fraud lolos dari model tabular.",
                C.C_ACCENT)

    # ------------------------------------------------------------------ interpretation
    st.subheader("Interpretasi investigasi")
    st.markdown(f"""
- **Pola ring.** Kelima ring berupa pasangan transaksi fraud. {gr['ring_shared_location']} pasang berbagi lokasi pelanggan
  dan {gr['ring_shared_subnet']} pasang berbagi IP subnet /16 (proxy atau blok ISP yang sama).
- **Hub pengiriman.** State pengiriman anggota tiap ring *berbeda* ({gr['ring_same_state']} dari {gr['n_rings']} ring yang
  mengirim ke state yang sama). Data ini tidak menunjukkan satu alamat drop-off bersama. Yang terlihat di data hanya
  infrastruktur akun/IP yang dipakai bersama, dengan tujuan pengiriman berbeda.
- **Hipotesis untuk analis.** Pola ini konsisten dengan satu pelaku yang membuka beberapa akun lewat proxy atau lokasi
  yang sama. Verifikasi lanjutan: device fingerprint, alamat lengkap (bukan hanya state), dan riwayat akun terkait.
- **Sentralitas.** Rata-rata PageRank transaksi fraud hanya {fmt_num(gres['centrality_metrics']['pagerank_disparity_ratio'])}×
  dan degree {fmt_num(gres['centrality_metrics']['degree_disparity_ratio'])}× dari transaksi legit. Fraud berada di klaster
  kecil yang terpisah, bukan di hub besar, sehingga community detection lebih efektif daripada memburu simpul yang
  paling terhubung.
- **Batasan.** Dataset bersifat simulasi dan jaringan dibangun dari 1.500 transaksi pertama pada kohort berisiko.
  Ring berukuran 2 transaksi adalah petunjuk investigasi, bukan bukti sindikat.
""")

    ring_nodes = nodes[nodes.community_id.isin(ring_ids)].sort_values(["community_id", "node_id"]).copy()
    with st.expander("Anggota fraud ring dan skor model tabular"):
        ring_nodes["Model tabular @0,80"] = np.where(ring_nodes.oof_prob >= C.BLOCK_FROM, "Terblokir", "Lolos")
        st.dataframe(ring_nodes[["community_id", "node_id", "amount", "hour", "account_age_days", "subnet", "location",
                                 "ship_state", "oof_prob", "Model tabular @0,80"]]
                     .rename(columns={"community_id": "Ring", "node_id": "Transaksi", "amount": "Nominal ($)",
                                      "hour": "Jam", "account_age_days": "Umur akun", "subnet": "Subnet /16",
                                      "location": "Lokasi", "ship_state": "State kirim", "oof_prob": "Skor model"}),
                     hide_index=True, width="stretch")
