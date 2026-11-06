# Executive & Operational Fraud Detection Dashboard

Final Project (AOL) · **COMP8035041 – Big Data Analytics** · BINUS Graduate Program, Magister Teknik Informatika
Kevin Sebastian · NIM 2702746012

Dashboard Streamlit interaktif yang menerjemahkan hasil Assignment II (Big Data, Machine Learning, Graph Analytics
pada 23.634 transaksi e-commerce) menjadi keputusan bisnis untuk eksekutif dan analis fraud.

## Empat tab

| Tab | Isi |
|---|---|
| **Executive Overview** | KPI (total transaksi, fraud terdeteksi, financial loss prevented, false-positive friction rate), cost-benefit trade-off matrix, ringkasan eksekutif, tiga insight bisnis |
| **ML Decision Engine** | Slider threshold τ 0,10–0,90 (confusion matrix, precision, recall, F1, total net cost), fraud risk drivers (feature importance + fraud rate per segmen), benchmark sklearn/XGBoost vs Spark MLlib |
| **Graph & Fraud Rings** | Peta jaringan interaktif (IP subnet /16 dan lokasi), detail per komunitas (transaksi, subnet, lokasi, state pengiriman), tabel fraud ring 100% dan komunitas makro, interpretasi investigasi |
| **Risk Simulator** | Form transaksi, tombol **Analyze Transaction Risk**, probabilitas fraud, keputusan APPROVE (< 0,40) / MANUAL REVIEW – 2FA (0,40–0,79) / BLOCK (≥ 0,80), risk factors triggered, kontribusi tiap faktor |

## Tiga insight bisnis

1. **Finansial & threshold tuning.** τ = 0,80 memangkas False Positive 82% (615 → 111) dan menaikkan presisi dari 17,89% ke 42,78% pada hold-out 4.727 transaksi.
2. **Profil risiko.** Fraud rate 53,2% untuk nominal > $1.000, 23,0% untuk akun ≤ 30 hari, 11,1% untuk dini hari (baseline 5,17%). Fokus mitigasi: velocity check transaksi baru.
3. **Sindikat dari graph.** 5 fraud ring dengan fraud rate 100% terungkap lewat keterhubungan transaksi. Model tabular melewatkan 6 dari 10 anggota ring pada τ = 0,80 saat dinilai satu per satu.

## Menjalankan secara lokal

```bash
git clone https://github.com/mkevinsebastian/big-data-analytics-dashboard.git
cd big-data-analytics-dashboard
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```

Aplikasi terbuka di <http://localhost:8501>. Tema terang/gelap mengikuti preferensi sistem dan dapat diganti lewat menu ⋮ → Settings → Theme.

## Deploy ke Streamlit Community Cloud

1. Push repositori ini ke GitHub (publik, atau privat dengan akses yang diberikan ke Streamlit).
2. Buka <https://share.streamlit.io> dan login dengan akun GitHub.
3. Klik **Create app** → **Deploy a public app from GitHub**.
4. Isi: Repository `mkevinsebastian/big-data-analytics-dashboard`, Branch `main`, Main file path `app.py`.
5. Di **Advanced settings**, pilih Python 3.11 atau 3.12, lalu klik **Deploy**.
6. Salin URL `https://<nama-app>.streamlit.app` untuk diserahkan kepada dosen.

Tidak ada secret atau database. Seluruh data berada di folder `data/` (± 1,1 MB) sehingga build cepat.

## Struktur proyek

```
app.py                     entry point (sidebar, tabs)
dashboard/
  config.py                path, palet, kebijakan keputusan (0,40 / 0,80), asumsi biaya default
  analytics.py             perhitungan biaya & confusion matrix per threshold (tanpa Streamlit)
  insights.py              angka headline untuk tiga insight (dihitung dari data/)
  data.py                  loader CSV/JSON ber-cache
  charts.py                semua figure Plotly
  theme.py                 CSS, komponen kartu, format angka Indonesia
  model.py                 model simulator (HistGradientBoosting)
  views/                   overview.py · ml_engine.py · graph_view.py · simulator.py
data/                      artefak hasil eksperimen (lihat bawah)
scripts/build_data.py      membangun ulang data/ dari folder Fraud_Detection_Project
requirements.txt           dependensi aplikasi
requirements-dev.txt       tambahan untuk build_data.py (xgboost, networkx)
```

## Sumber data dan cara membangun ulang

`data/` dihasilkan dari folder `Fraud_Detection_Project/` (Assignment II):

| File | Asal |
|---|---|
| `ml_evaluation_results.json`, `graph_analytics_results.json` | salinan langsung hasil Assignment II |
| `threshold_sweep.csv` | XGBoost dilatih ulang dengan konfigurasi dan seed yang sama; confusion matrix per threshold plus nilai USD tiap kuadran |
| `feature_importance.csv` | XGBoost dan Random Forest, dikelompokkan ke fitur bisnis |
| `risk_profile.csv`, `kpi_summary.json` | EDA pada dataset V2 |
| `graph_nodes.csv`, `graph_edges.csv`, `graph_communities.csv` | konstruksi graph Assignment II diulang; seluruh 339 komunitas diekspor (JSON asli hanya memuat 10) |
| `transactions_slim.csv` | 6 fitur simulator + label; tanpa ID, IP, atau alamat |

```bash
pip install -r requirements-dev.txt
python scripts/build_data.py --source "../Fraud_Detection_Project"
```

Skrip memverifikasi hasil retraining terhadap laporan: τ = 0,80 menghasilkan TP = 83 dan FP = 111, τ = 0,50 menghasilkan TP = 134 dan FP = 615 (identik).

## Catatan metodologi dan batasan

- **Dataset simulasi.** Payment method, device, dan address mismatch tidak membedakan fraud. Angka absolut adalah ilustrasi metodologi, bukan prediksi bisnis nyata.
- **Asumsi biaya.** Denda chargeback ($60), penjualan hilang saat pelanggan sah diblokir (50%), dan biaya penanganan ($5) adalah asumsi perencanaan yang bisa diubah di sidebar. Laporan hanya menyebut kisaran chargeback $20–100. Titik biaya minimum bergeser mengikuti asumsi ini.
- **Skala angka.** Evaluasi model memakai hold-out 20% (4.727 transaksi). Tab 1 menawarkan proyeksi ke 23.634 transaksi (× ≈ 5,0) atau tampilan hold-out apa adanya.
- **Graph.** Jaringan dibangun dari 1.500 transaksi pertama pada kohort yang menyentuh subnet atau lokasi terkait fraud, sehingga baseline sampel (33,5%) jauh di atas populasi (5,17%). Kelima fraud ring berukuran 2 transaksi, berbagi lokasi pelanggan (4 ring) atau IP subnet (1 ring), dan **tidak** berbagi state pengiriman. Ring adalah petunjuk investigasi, bukan bukti sindikat.
- **Simulator.** Memakai surrogate scikit-learn (HistGradientBoosting dengan bobot kelas 18,34 : 1) agar `xgboost` dan CSV mentah tidak perlu ikut dideploy. Pada hold-out yang sama, precision/recall di τ = 0,80 sekitar 42%/41% (XGBoost asli: 42,8%/34,0%). Skor "out-of-fold" pada tab Graph berasal dari XGBoost 5-fold.
