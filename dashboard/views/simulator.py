"""Tab 4 - Live Transaction Risk Simulator for fraud analysts / operations."""
from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from .. import charts, data, model
from .. import config as C
from ..analytics import decision_for
from ..theme import plot, callout, fmt_num, fmt_pct, fmt_usd, show

DECISIONS = {
    "APPROVE": dict(icon="🟢", title="APPROVE", sub="Risiko rendah (< 0,40)", color=C.C_GOOD,
                    actions=["Proses pesanan seperti biasa.",
                             "Tidak perlu gesekan tambahan bagi pelanggan.",
                             "Tetap masukkan ke monitoring pasif (pola velocity harian)."]),
    "REVIEW": dict(icon="🟡", title="MANUAL REVIEW / 2FA CHALLENGE", sub="Risiko sedang (0,40 – 0,79)",
                   color=C.C_WARN,
                   actions=["Minta verifikasi tambahan (OTP / 2FA) sebelum pembayaran diselesaikan.",
                            "Tahan fulfilment sampai analis menyelesaikan review (target SLA singkat).",
                            "Periksa riwayat akun: jumlah transaksi dalam 24 jam terakhir dan perubahan alamat."]),
    "BLOCK": dict(icon="🔴", title="BLOCK / REJECT", sub="Risiko tinggi (≥ 0,80)", color=C.C_FRAUD,
                  actions=["Tolak atau tahan transaksi, dan kirim ke antrean investigasi fraud.",
                           "Cek akun lain dengan IP subnet atau lokasi yang sama di tab Graph Analytics.",
                           "Pertimbangkan pembatasan sementara pada akun dan device terkait.",
                           "Sediakan jalur banding bagi pelanggan sah agar friksi tetap rendah."]),
}

PRESETS = {
    "Pelanggan normal": dict(amount=150.0, age=180, hour=14, qty=2, payment="credit card", device="desktop"),
    "Akun baru, dini hari": dict(amount=250.0, age=7, hour=3, qty=2, payment="PayPal", device="mobile"),
    "Profil sindikat": dict(amount=1800.0, age=5, hour=2, qty=4, payment="bank transfer", device="mobile"),
}


def _apply_preset(name: str) -> None:
    for k, v in PRESETS[name].items():
        st.session_state[f"sim_{k}"] = v


def _risk_factors(inp: dict, rates: dict) -> list[tuple[str, str]]:
    f = []
    if inp["amount"] > C.HIGH_AMOUNT_USD:
        f.append(("Nominal > $1.000", f"fraud rate historis {fmt_pct(rates['amt_high'], 1, True)}"))
    elif inp["amount"] >= 500:
        f.append(("Nominal $500 – $1.000", f"fraud rate historis {fmt_pct(rates['amt_mid'], 1, True)}"))
    if inp["age"] <= C.NEW_ACCOUNT_DAYS:
        f.append(("Akun baru (≤ 30 hari)", f"fraud rate historis {fmt_pct(rates['new'], 1, True)}"))
    if 0 <= inp["hour"] <= C.DAWN_LAST_HOUR:
        f.append(("Waktu dini hari (00:00–05:59)", f"fraud rate historis {fmt_pct(rates['dawn'], 1, True)}"))
    if len(f) >= 2:
        f.append(("Kombinasi ≥ 2 sinyal risiko", "pola velocity transaksi baru; risiko berlipat"))
    return f


def render() -> None:
    bundle = model.get_bundle()
    prof = data.risk_profile().set_index(["factor", "segment"])["fraud_rate_pct"]
    rates = dict(amt_high=prof[("Transaction Amount", "> $1.000")], amt_mid=prof[("Transaction Amount", "$500 - $1.000")],
                 new=prof[("Account Age", "0 - 30 hari (akun baru)")], dawn=prof[("Transaction Hour", "Dini hari 00:00-05:59")])

    st.caption("Alat bantu analis: masukkan atribut transaksi, lalu skor risiko dihitung real-time oleh model "
               "gradient boosting (surrogate XGBoost, lihat README) dan diterjemahkan menjadi keputusan bisnis.")

    st.markdown("**Skenario cepat**")
    pc = st.columns(len(PRESETS))
    for col, name in zip(pc, PRESETS):
        col.button(name, key=f"preset_{name}", on_click=_apply_preset, args=(name,), width="stretch")

    defaults = PRESETS["Pelanggan normal"]
    for k, v in defaults.items():
        st.session_state.setdefault(f"sim_{k}", v)

    f1, f2, f3 = st.columns(3)
    amount = f1.number_input("Transaction Amount (USD)", min_value=1.0, max_value=10000.0, step=10.0, key="sim_amount")
    age = f2.number_input("Account Age (hari)", min_value=0, max_value=3650, step=1, key="sim_age")
    hour = f3.slider("Transaction Hour (0–23)", 0, 23, key="sim_hour")
    g1, g2, g3 = st.columns(3)
    qty = g1.number_input("Quantity", min_value=1, max_value=5, step=1, key="sim_qty",
                          help="Model dilatih pada kuantitas 1–5.")
    payment = g2.selectbox("Payment Method", model.PAYMENT_METHODS, key="sim_payment")
    device = g3.selectbox("Device Used", model.DEVICES, key="sim_device")

    if st.button("Analyze Transaction Risk", type="primary", width="stretch"):
        inp = dict(amount=float(amount), age=int(age), hour=int(hour), qty=int(qty), payment=payment, device=device)
        prob = model.predict(bundle, inp)
        contrib = model.contributions(bundle, inp)
        st.session_state["sim_result"] = dict(inp=inp, prob=prob, contrib=contrib,
                                             decision=decision_for(prob, C.REVIEW_FROM, C.BLOCK_FROM))
        hist = st.session_state.setdefault("sim_history", [])
        hist.insert(0, dict(Nominal=inp["amount"], Umur_akun=inp["age"], Jam=inp["hour"], Qty=inp["qty"],
                            Payment=payment, Device=device, Probabilitas=prob,
                            Keputusan=DECISIONS[st.session_state["sim_result"]["decision"]]["title"]))
        del hist[10:]

    res = st.session_state.get("sim_result")
    if not res:
        callout("Pilih skenario atau isi form, lalu klik Analyze Transaction Risk.", C.C_ACCENT)
        return

    d = DECISIONS[res["decision"]]
    st.divider()
    a, b = st.columns([1, 1.25])
    plot(charts.gauge_fig(res["prob"], C.REVIEW_FROM, C.BLOCK_FROM), a)
    with b:
        show(f'<div class="decision" style="--accent:{d["color"]}"><div class="big">{d["icon"]} {html.escape(d["title"])}</div>'
             f'<div>{html.escape(d["sub"])} · probabilitas fraud <b>{fmt_pct(res["prob"], 1)}</b></div></div>')
        factors = _risk_factors(res["inp"], rates)
        st.markdown("**Risk factors triggered**")
        if factors:
            for name, detail in factors:
                st.markdown(f"- 🚩 **{name}**: {detail}")
        else:
            st.markdown("- Tidak ada faktor risiko dari Insight 2 yang terpicu (nominal ≤ $500, akun > 30 hari, bukan dini hari).")
        st.markdown("**Tindakan yang disarankan**")
        for act in d["actions"]:
            st.markdown(f"- {act}")

    plot(charts.contribution_fig(res["contrib"]), st)
    st.caption("Bar merah = faktor yang menaikkan skor dibanding nilai normal (nominal median, umur akun median, jam 12:00). "
               "Payment Method dan Device Used mendekati 0 karena memang tidak membedakan fraud pada data ini.")

    hist = st.session_state.get("sim_history", [])
    if hist:
        with st.expander(f"Riwayat analisis sesi ini ({len(hist)})"):
            h = pd.DataFrame(hist)
            h["Nominal"] = h["Nominal"].map(fmt_usd)
            h["Probabilitas"] = h["Probabilitas"].map(lambda p: fmt_pct(p, 1))
            st.dataframe(h, hide_index=True, width="stretch")
