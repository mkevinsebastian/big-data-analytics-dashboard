"""
Build the pre-computed artefacts used by the Streamlit dashboard.

Run ONCE (offline) whenever the source experiment changes:

    pip install -r requirements-dev.txt
    python scripts/build_data.py --source "../Fraud_Detection_Project"

The dashboard itself (app.py) never needs xgboost / networkx / the 6 MB raw CSV;
it only reads the small files written to ./data. This keeps Streamlit Cloud
deployments fast and the repository light.

Outputs (./data):
    ml_evaluation_results.json   copy of the Assignment II ML metrics (unchanged)
    graph_analytics_results.json copy of the Assignment II graph metrics (unchanged)
    threshold_sweep.csv          XGBoost hold-out confusion matrix + USD amounts per threshold
    feature_importance.csv       XGBoost / Random Forest importance grouped by business feature
    risk_profile.csv             observed fraud rate per segment (EDA)
    kpi_summary.json             headline dataset numbers
    transactions_slim.csv        6 simulator features + label (no PII, no free-text addresses)
    graph_nodes.csv / graph_edges.csv / graph_communities.csv   transaction risk network
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import xgboost as xgb
from networkx.algorithms.community import greedy_modularity_communities
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

DATASET_REL = "dataset/Fraudulent_E-Commerce_Transaction_Data_2.csv"
RANDOM_STATE = 42


# --------------------------------------------------------------------------- #
# Shared helpers (mirror ml_pipeline_assignment2.py / graph_analytics_assignment2.py)
# --------------------------------------------------------------------------- #
def parse_state(addr: str) -> str:
    if not isinstance(addr, str) or not addr.strip():
        return "UNK"
    lines = addr.strip().split("\n")
    last = lines[-1] if len(lines) > 1 else lines[0]
    m = re.search(r",\s*([A-Z]{2})\s+\d{5}", last)
    if m:
        return m.group(1)
    m = re.search(r"\b([A-Z]{2})\b", last)
    return m.group(1) if m else "UNK"


def preprocess(df: pd.DataFrame):
    """Same feature engineering as the Assignment II pipeline."""
    df = df.copy()
    median_age = float(df.loc[df["Customer Age"] >= 18, "Customer Age"].median())
    df["customer_age_clean"] = df["Customer Age"].apply(lambda x: median_age if x < 18 else float(x))
    p99 = float(df["Transaction Amount"].quantile(0.99))
    df["amount_capped"] = np.minimum(df["Transaction Amount"], p99)
    df["address_mismatch"] = (df["Shipping Address"] != df["Billing Address"]).astype(int)
    df["state_mismatch"] = (df["Shipping Address"].map(parse_state) != df["Billing Address"].map(parse_state)).astype(int)
    df["log_amount"] = np.log1p(df["Transaction Amount"])
    df["amount_to_qty_ratio"] = df["Transaction Amount"] / np.maximum(df["Quantity"], 1)
    df["is_high_amount"] = (df["Transaction Amount"] > 1000.0).astype(int)
    df["is_new_account"] = (df["Account Age Days"] <= 30).astype(int)
    df["is_dawn_tx"] = ((df["Transaction Hour"] >= 0) & (df["Transaction Hour"] <= 5)).astype(int)
    df["hour_sin"] = np.sin(2.0 * np.pi * df["Transaction Hour"] / 24.0)
    df["hour_cos"] = np.cos(2.0 * np.pi * df["Transaction Hour"] / 24.0)
    enc = pd.get_dummies(df, columns=["Payment Method", "Product Category", "Device Used"],
                         drop_first=True, dtype=float)
    ignore = ["Transaction ID", "Customer ID", "Transaction Date", "Customer Location", "IP Address",
              "Shipping Address", "Billing Address", "shipping_state", "shipping_zip", "billing_state",
              "billing_zip", "Customer Age", "Is Fraudulent"]
    feats = [c for c in enc.columns if c not in ignore]
    return enc, feats


def make_xgb(imbalance: float) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                             scale_pos_weight=imbalance, random_state=RANDOM_STATE,
                             eval_metric="aucpr", n_jobs=-1)


def business_group(col: str) -> str:
    """Collapse engineered / one-hot columns into the business feature they came from."""
    if col in ("amount_capped", "log_amount", "amount_to_qty_ratio", "is_high_amount", "Transaction Amount"):
        return "Transaction Amount"
    if col in ("Account Age Days", "is_new_account"):
        return "Account Age"
    if col in ("Transaction Hour", "hour_sin", "hour_cos", "is_dawn_tx"):
        return "Transaction Hour (Dini Hari)"
    if col.startswith("Payment Method_"):
        return "Payment Method"
    if col.startswith("Device Used_"):
        return "Device Used"
    if col.startswith("Product Category_"):
        return "Product Category"
    if col in ("address_mismatch", "state_mismatch"):
        return "Address Mismatch"
    if col == "Quantity":
        return "Quantity"
    if col == "customer_age_clean":
        return "Customer Age"
    return col


# --------------------------------------------------------------------------- #
# Build steps
# --------------------------------------------------------------------------- #
def build_ml(df_raw: pd.DataFrame, out: Path, ml_json: dict):
    enc, feats = preprocess(df_raw)
    X, y = enc[feats].values, enc["Is Fraudulent"].values
    idx = np.arange(len(enc))
    imbalance = float((y == 0).sum() / (y == 1).sum())

    X_tr, X_te, y_tr, y_te, i_tr, i_te = train_test_split(
        X, y, idx, test_size=0.20, random_state=RANDOM_STATE, stratify=y)
    model = make_xgb(imbalance).fit(X_tr, y_tr)
    prob = model.predict_proba(X_te)[:, 1]
    amount = df_raw["Transaction Amount"].values[i_te]

    # --- verify we reproduce the published Assignment II numbers -----------------
    ref_cm = ml_json["evaluation_metrics"]["XGBoost_Tuned"]["confusion_matrix"]
    pred = prob >= 0.80
    cm = dict(tp=int(((pred) & (y_te == 1)).sum()), fp=int(((pred) & (y_te == 0)).sum()))
    print(f"[verify] tau=0.80  reproduced TP={cm['tp']} FP={cm['fp']} | report TP={ref_cm['tp']} FP={ref_cm['fp']}")
    ref_cm5 = ml_json["evaluation_metrics"]["XGBoost"]["confusion_matrix"]
    p5 = prob >= 0.50
    print(f"[verify] tau=0.50  reproduced TP={int((p5 & (y_te == 1)).sum())} FP={int((p5 & (y_te == 0)).sum())} "
          f"| report TP={ref_cm5['tp']} FP={ref_cm5['fp']}")

    # --- threshold sweep with USD amounts ----------------------------------------
    rows = []
    for th in np.round(np.arange(0.05, 0.9501, 0.01), 2):
        p = prob >= th
        pos, neg = y_te == 1, y_te == 0
        tp, fp = int((p & pos).sum()), int((p & neg).sum())
        fn, tn = int((~p & pos).sum()), int((~p & neg).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append(dict(threshold=th, tn=tn, fp=fp, fn=fn, tp=tp,
                         precision=prec, recall=rec, f1_score=f1,
                         fraud_usd_caught=float(amount[p & pos].sum()),
                         fraud_usd_missed=float(amount[~p & pos].sum()),
                         legit_usd_blocked=float(amount[p & neg].sum()),
                         legit_usd_approved=float(amount[~p & neg].sum())))
    pd.DataFrame(rows).to_csv(out / "threshold_sweep.csv", index=False)

    # --- feature importance grouped by business feature --------------------------
    rf_imp = None
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100, max_depth=12, class_weight="balanced",
                                random_state=RANDOM_STATE, n_jobs=-1).fit(X_tr, y_tr)
    rf_imp = rf.feature_importances_
    imp = pd.DataFrame({"column": feats, "XGBoost": model.feature_importances_, "Random Forest": rf_imp})
    imp["feature"] = imp["column"].map(business_group)
    grouped = imp.groupby("feature")[["XGBoost", "Random Forest"]].sum().reset_index()
    grouped.to_csv(out / "feature_importance.csv", index=False)
    print(grouped.sort_values("XGBoost", ascending=False).round(4).to_string(index=False))

    # --- out-of-fold scores for every transaction (used by graph section) --------
    oof = cross_val_predict(make_xgb(imbalance), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
                            method="predict_proba", n_jobs=1)[:, 1]
    return oof, imbalance, len(i_te), int(y_te.sum())


def build_risk_profile(df: pd.DataFrame, out: Path):
    rows = []

    def add(group, segment, mask):
        sub = df[mask]
        rows.append(dict(factor=group, segment=segment, transactions=len(sub),
                         frauds=int(sub["Is Fraudulent"].sum()),
                         fraud_rate_pct=round(sub["Is Fraudulent"].mean() * 100, 2)))

    a = df["Transaction Amount"]
    for lab, m in [("< $100", a < 100), ("$100 - $500", (a >= 100) & (a < 500)),
                   ("$500 - $1.000", (a >= 500) & (a <= 1000)), ("> $1.000", a > 1000)]:
        add("Transaction Amount", lab, m)
    g = df["Account Age Days"]
    for lab, m in [("0 - 30 hari (akun baru)", g <= 30), ("31 - 90 hari", (g > 30) & (g <= 90)),
                   ("91 - 365 hari", g > 90)]:
        add("Account Age", lab, m)
    h = df["Transaction Hour"]
    for lab, m in [("Dini hari 00:00-05:59", h <= 5), ("Pagi 06:00-11:59", (h >= 6) & (h <= 11)),
                   ("Siang/Sore 12:00-17:59", (h >= 12) & (h <= 17)), ("Malam 18:00-23:59", h >= 18)]:
        add("Transaction Hour", lab, m)
    for col, name in [("Payment Method", "Payment Method"), ("Device Used", "Device Used")]:
        for v in sorted(df[col].unique()):
            add(name, str(v), df[col] == v)
    mism = df["Shipping Address"] != df["Billing Address"]
    add("Address Mismatch", "Shipping != Billing", mism)
    add("Address Mismatch", "Shipping == Billing", ~mism)
    # compound profile (Insight 2)
    risky = (a > 1000) | (g <= 30) | (h <= 5)
    all3 = (a > 1000) & (g <= 30) & (h <= 5)
    add("Kombinasi Risiko", "Tidak ada faktor risiko", ~risky)
    add("Kombinasi Risiko", "Minimal 1 faktor", risky)
    add("Kombinasi Risiko", "Ketiga faktor sekaligus", all3)
    pd.DataFrame(rows).to_csv(out / "risk_profile.csv", index=False)


def build_graph(df: pd.DataFrame, oof: np.ndarray, out: Path):
    """Re-run the exact Assignment II graph construction and export every community (not only 10)."""
    df = df.copy()
    df["oof_prob"] = oof
    df["ip_subnet_16"] = df["IP Address"].apply(lambda x: ".".join(str(x).split(".")[:2]))
    df["ship_state"] = df["Shipping Address"].apply(parse_state)

    fraud_subnets = set(df.loc[df["Is Fraudulent"] == 1, "ip_subnet_16"])
    fraud_locs = set(df.loc[df["Is Fraudulent"] == 1, "Customer Location"])
    risk = df[df["ip_subnet_16"].isin(fraud_subnets) | df["Customer Location"].isin(fraud_locs)]
    sample = risk.head(1500).copy()
    sample["node_id"] = "TX_" + sample["Transaction ID"].astype(str).str[:8]

    G = nx.Graph()
    for _, r in sample.iterrows():
        G.add_node(r["node_id"], is_fraud=int(r["Is Fraudulent"]))
    reasons: dict[tuple, set] = {}
    for key, tag in (("ip_subnet_16", "IP Subnet /16"), ("Customer Location", "Lokasi")):
        for _, grp in sample.groupby(key):
            ids = list(grp["node_id"])
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    G.add_edge(ids[i], ids[j])
                    reasons.setdefault(tuple(sorted((ids[i], ids[j]))), set()).add(tag)

    active = G.subgraph([n for n, d in G.degree() if d > 0]).copy()
    comms = list(greedy_modularity_communities(active))
    comm_of = {n: k + 1 for k, c in enumerate(comms) for n in c}

    # layout: one spring layout per connected component, packed on a grid
    comps = sorted(nx.connected_components(active), key=len, reverse=True)
    cols = int(np.ceil(np.sqrt(len(comps))))
    pos = {}
    for k, comp in enumerate(comps):
        sub = active.subgraph(comp)
        lay = nx.spring_layout(sub, seed=RANDOM_STATE, k=0.9 / np.sqrt(max(len(comp), 1)) + 0.05) \
            if len(comp) > 2 else {n: np.array([i * 0.4 - 0.2, 0.0]) for i, n in enumerate(comp)}
        scale = 0.35 + 0.12 * np.sqrt(len(comp))
        cx, cy = (k % cols) * 3.0, -(k // cols) * 3.0
        for n, p in lay.items():
            pos[n] = (cx + float(p[0]) * scale, cy + float(p[1]) * scale)

    nodes = sample[sample["node_id"].isin(comm_of)].copy()
    nodes["community_id"] = nodes["node_id"].map(comm_of)
    nodes["x"] = nodes["node_id"].map(lambda n: pos[n][0])
    nodes["y"] = nodes["node_id"].map(lambda n: pos[n][1])
    nodes = nodes.rename(columns={"Is Fraudulent": "is_fraud", "Transaction Amount": "amount",
                                  "Transaction Hour": "hour", "Account Age Days": "account_age_days",
                                  "Customer Location": "location", "ip_subnet_16": "subnet"})
    keep = ["node_id", "community_id", "is_fraud", "amount", "hour", "account_age_days", "subnet",
            "location", "ship_state", "oof_prob", "x", "y"]
    nodes = nodes[keep]
    nodes.to_csv(out / "graph_nodes.csv", index=False)

    edges = pd.DataFrame([dict(source=a, target=b, relation=" + ".join(sorted(r)))
                          for (a, b), r in reasons.items() if a in comm_of and b in comm_of])
    edges["community_id"] = edges["source"].map(comm_of)
    edges["same_community"] = edges["source"].map(comm_of) == edges["target"].map(comm_of)
    edges.to_csv(out / "graph_edges.csv", index=False)

    # community profile
    recs = []
    for cid, g in nodes.groupby("community_id"):
        shared_sub = g["subnet"].value_counts()
        shared_loc = g["location"].value_counts()
        recs.append(dict(
            community_id=int(cid), size=len(g), fraud_count=int(g["is_fraud"].sum()),
            legit_count=int((1 - g["is_fraud"]).sum()),
            fraud_rate_pct=round(g["is_fraud"].mean() * 100, 2),
            avg_amount=round(float(g["amount"].mean()), 2),
            total_amount=round(float(g["amount"].sum()), 2),
            shared_subnets=int((shared_sub > 1).sum()), shared_locations=int((shared_loc > 1).sum()),
            top_subnet=shared_sub.index[0] if shared_sub.iloc[0] > 1 else "-",
            top_location=shared_loc.index[0] if shared_loc.iloc[0] > 1 else "-",
            top_ship_state=g["ship_state"].value_counts().index[0],
            n_ship_states=int(g["ship_state"].nunique()),
            avg_account_age=round(float(g["account_age_days"].mean()), 1),
            dawn_share_pct=round(float((g["hour"] <= 5).mean() * 100), 1),
            fraud_missed_by_ml=int(((g["is_fraud"] == 1) & (g["oof_prob"] < 0.80)).sum()),
        ))
    comm = pd.DataFrame(recs).sort_values(["fraud_rate_pct", "size"], ascending=False)
    comm.to_csv(out / "graph_communities.csv", index=False)

    rings = comm[(comm.fraud_rate_pct == 100) & (comm["size"] >= 2)]
    print(f"[graph] communities={len(comm)}  100% rings={len(rings)}  >30%={int((comm.fraud_rate_pct > 30).sum())}")
    fr = nodes[nodes.is_fraud == 1]
    print(f"[graph] frauds in network={len(fr)}; missed by tabular ML @0.80 (out-of-fold)="
          f"{int((fr.oof_prob < 0.8).sum())} ({(fr.oof_prob < 0.8).mean():.1%})")
    ring_nodes = nodes[nodes.community_id.isin(rings.community_id)]
    print(f"[graph] ring members={len(ring_nodes)}, missed by ML="
          f"{int((ring_nodes.oof_prob < 0.8).sum())}")
    return len(sample), int(sample["Is Fraudulent"].sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="../Fraud_Detection_Project",
                    help="folder that contains dataset/ and the Assignment II *.json results")
    args = ap.parse_args()
    src = Path(args.source).resolve()
    out = Path(__file__).resolve().parent.parent / "data"
    out.mkdir(exist_ok=True)

    df = pd.read_csv(src / DATASET_REL)
    ml_json = json.loads((src / "ml_evaluation_results.json").read_text(encoding="utf-8"))
    shutil.copy(src / "ml_evaluation_results.json", out / "ml_evaluation_results.json")
    shutil.copy(src / "graph_analytics_results.json", out / "graph_analytics_results.json")

    oof, imbalance, n_test, pos_test = build_ml(df, out, ml_json)
    build_risk_profile(df, out)
    n_sample, f_sample = build_graph(df, oof, out)

    slim = df[["Transaction Amount", "Account Age Days", "Transaction Hour", "Quantity",
               "Payment Method", "Device Used", "Is Fraudulent"]]
    slim.to_csv(out / "transactions_slim.csv", index=False)

    kpi = dict(total_transactions=len(df), fraud_count=int(df["Is Fraudulent"].sum()),
               fraud_rate_pct=round(float(df["Is Fraudulent"].mean() * 100), 2),
               imbalance_ratio=round(imbalance, 2), test_transactions=n_test, test_frauds=pos_test,
               avg_fraud_amount=round(float(df.loc[df["Is Fraudulent"] == 1, "Transaction Amount"].mean()), 2),
               avg_legit_amount=round(float(df.loc[df["Is Fraudulent"] == 0, "Transaction Amount"].mean()), 2),
               network_sample_size=n_sample, network_sample_frauds=f_sample,
               network_sample_fraud_rate_pct=round(f_sample / n_sample * 100, 2))
    (out / "kpi_summary.json").write_text(json.dumps(kpi, indent=2), encoding="utf-8")
    print("[done]", kpi)


if __name__ == "__main__":
    main()
