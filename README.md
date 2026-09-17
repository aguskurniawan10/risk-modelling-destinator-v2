# Risk Modelling DESTINATOR -- PLTU Pelabuhanratu Unit 1

Model risiko 2-layer (Predictive + Monte Carlo/Optimization) untuk risiko
**Coal Flow Maximum (CFM) exceedance** dan **stok batubara di bawah ambang
aman**, dibangun dari 5 sumber data mentah DESTINATOR (load forecasting +
coal blending calculator, PT PLN Indonesia Power UBP Jawa Barat 2
Pelabuhanratu).

## Cara menjalankan

```bash
pip install -r requirements.txt
python3 main.py                    # jalankan seluruh pipeline, hasilkan output/
streamlit run dashboard_app.py     # dashboard interaktif
```

## Cakupan & keterbatasan (WAJIB dibaca)

- **Unit 1 saja** -- ditentukan oleh ketersediaan data (file coal flow &
  coal blending hanya punya data Unit 1).
- **CFM = 225 t/h** [ditentukan pengguna]. Event rate historis 11,2% --
  sehat, tersebar di semua 20 bulan data (bukan 1 episode).
- **Target SFC = 0,7551 kg/kWh** [ditentukan pengguna]. Model SFC~CV+beban
  (OLS, n=5.638 jam, data SFC_actual riil x beban_netto_mw riil dari sheet
  redispatch): kedua efek signifikan & searah wajar (CV naik -> SFC turun;
  beban naik -> SFC turun, efek beban lbh besar). Target SFC TERCAPAI pd
  kombinasi realistis: CV~4.082 kcal/kg (beban=rata-rata) ATAU beban~252 MW
  (CV=rata-rata) -- lihat `src/sensitivity_fit.py`.
- **Safety stock MRC=5 hari, LRC=10 hari** [ditentukan pengguna].
- **Harga batubara Rp 1.020/kg** [ditentukan pengguna] -- HANYA 1 angka
  (bukan per jenis MRC/LRC), jadi komponen biaya bahan bakar TIDAK
  menciptakan trade-off ekonomi antara MRC vs LRC. Rekomendasi Layer 3
  saat ini (100% MRC) TIDAK memperhitungkan kemungkinan MRC lebih mahal
  atau lebih terbatas pasokannya di dunia nyata.
- **Data stok HOP Nov-Des 2025 KOSONG di sumber asli** [dikonfirmasi
  pengguna] -- diimputasi via tren linear dari Ags-Okt 2025, ditandai
  `hop_is_imputed=True` di semua tabel turunan.
- **Beban aktual (beban_netto_mw)** diambil dari sheet REDISPATCH
  (data_risk_modelling_destinator.xlsx) -- PADAT 100% di seluruh 20 bulan
  (bukan proksi ROH/implied sparse spt versi awal).
- **SFC_actual** dari DATA_SFC_DESTINATOR.xlsx -- padat 100% Jan-Agu 2025,
  KOSONG TOTAL sejak Sep 2025 (ditemukan saat parsing, bukan asumsi).
- **Load Forecast (ala DESTINATOR)**: horizon H+24 jam (bukan H+168/7 hari
  spt DESTINATOR asli -- dicoba semua horizon 1/6/24/48/168 jam, SEMUA
  MAPE jauh dari target <5% krn beban level UNIT TUNGGAL didorong perintah
  dispatch P2B yg diskret, beda dari beban AGREGAT SISTEM yg DESTINATOR
  asli forecast). Model (MAPE ~18%) MENGALAHKAN baseline ROH apa adanya
  (MAPE ~22%), TAPI belum capai target 5% -- lihat `src/load_forecast.py`.
- **Rantai risiko** [klarifikasi pengguna]: ROH meleset dari Redispatch ->
  blending MRC tak selaras kebutuhan riil -> boros MRC saat beban rendah ->
  stok MRC menipis -> saat beban tinggi MRC kurang -> LRC lbh banyak (CV
  rendah) -> coal flow naik -> CFM -> SFC meleset dari target. Fitur
  `beban_forecast_24h_mw`, `boros_mrc_indicator`, dan
  `deviasi_roh_redispatch_pct` ditambahkan ke Layer 1 utk menguji rantai
  ini -- y_stock_below_safety walk-forward ROC-AUC NAIK dari 0,77 ke 0,86
  setelah fitur ini ditambahkan.
- **CV MRC & LRC** diambil dari data riil rekap kedatangan batubara (884
  shipment): MRC rata-rata ~4.604 kcal/kg, LRC rata-rata ~4.096 kcal/kg.

## Struktur kode (`src/`)

| File | Isi |
|---|---|
| `config.py` | Semua ambang batas & asumsi (ditandai `[STATED BY USER]` bila dari pengguna) |
| `data_loader.py` | Harmonisasi 6 sumber data mentah jadi 1 tabel per jam |
| `load_forecast.py` | Forecasting beban H+24 jam ala DESTINATOR (Gradient Boosting, lag features) |
| `features.py` | Feature engineering (rolling, lag, kalender, blending, boros MRC, load forecast) |
| `risk_labels.py` | Label `y_cfm_exceed` (2 jam ke depan) & `y_stock_below_safety` (7 hari ke depan) |
| `layer1_predictive.py` | Logistic Regression + XGBoost, time-based split |
| `sensitivity_fit.py` | Model SFC ~ CV_blend + beban (OLS, data riil padat) |
| `layer2_montecarlo.py` | Simulasi Monte Carlo per rasio blending MRC:LRC |
| `layer3_optimization.py` | Cari rasio_mrc* yg minimalkan CVaR95 Total Cost |
| `validation.py` | Backtesting, benchmarking, kalibrasi, tornado, walk-forward |
| `reporting.py` | Figure PNG siap pakai |

## Hasil validasi (walk-forward, 20 bulan, XGBoost)

| Target | ROC-AUC | Verdict |
|---|---|---|
| y_cfm_exceed | 0,90 (17/19 fold reliable) | Diterima (Baik) |
| y_stock_below_safety | 0,86 (6/17 fold reliable) | Diterima (Baik) |

Load Forecast (H+24 jam, Gradient Boosting): MAPE ~18% -- mengalahkan
baseline ROH apa adanya (~22%), TAPI belum capai target <5% (lihat
keterbatasan di atas).

Catatan pengujian: kode ini diuji end-to-end di sandbox tanpa akses
internet (tidak bisa `pip install xgboost`) memakai model pengganti
sementara (RandomForest) HANYA utk verifikasi wiring -- jalankan
`python3 main.py` dgn xgboost asli utk angka final.
