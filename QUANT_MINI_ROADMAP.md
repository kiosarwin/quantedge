# Roadmap Quant Mini

> Tujuan dokumen ini adalah menjelaskan jalur transisi dari bot rule-based + ML helper saat ini menuju sistem keputusan tunggal yang lebih mendekati **quant mini**: satu layer keputusan utama, dengan beberapa lapisan risiko sebagai pengaman.
>
> Dokumen pendamping:
> - [`QUANT_MINI_CHECKLIST.md`](./QUANT_MINI_CHECKLIST.md)
> - [`QUANT_MINI_DATA_SCHEMA.md`](./QUANT_MINI_DATA_SCHEMA.md)
> - [`QUANT_MINI_BOOTSTRAP_TO_LIVE.md`](./QUANT_MINI_BOOTSTRAP_TO_LIVE.md)

## 1) Definisi Target

Target akhir bukan satu model monolitik yang mengendalikan semua keputusan tanpa batas. Targetnya adalah:

- **satu decision layer utama** yang menggabungkan statistik kaya
- **risk layer non-negotiable** yang tetap dapat menolak trade
- **feature layer** yang menghasilkan sinyal, konteks, dan atribut perdagangan
- **audit layer** yang menjelaskan alasan di balik keputusan

Dengan kata lain:

- **ML/AI** menjadi lapisan keputusan utama
- **rule engine** menjadi penyedia fitur dan guardrail
- **risk manager** menjadi pagar keras

Sampai migrasi penuh selesai, sleeve alpha operasional tetap dibaca sebagai `trend_following`, `reversal`, dan `compression_breakout`; `neutral` adalah state routing residual, bukan alpha.

## 2) Prinsip Desain

1. **Satu keputusan, banyak bukti**
   - Keputusan trade harus berasal dari gabungan fitur, bukan dari satu sinyal tunggal.

2. **Safety over autonomy**
   - AI dapat mengusulkan, mengurutkan, dan menolak trade.
   - AI tidak boleh menaikkan leverage atau exposure di luar pagar risiko.

3. **Train on reality**
   - Data live, paper, shadow, dan out-of-sample harus dipisahkan secara eksplisit.
   - Hasil in-sample tidak boleh dicampur dengan evaluasi final.

4. **Minimal complexity first**
   - Setiap fitur baru harus memiliki justifikasi statistik atau operasional.
   - Hindari fitur yang tampak canggih tetapi tidak meningkatkan stabilitas.

5. **Explainability by design**
   - Setiap trade harus dapat ditelusuri ke fitur utama, regime, sleeve, EV, dan risiko.

## 3) Arsitektur Target

### Lapisan yang tetap rule-based

- kill switch
- max drawdown
- max open trades
- exposure per symbol / direction
- daily loss cap
- minimum notional / minimum risk
- exchange / execution failure handling
- paper-vs-live scope control

### Lapisan yang bisa naik ke AI

- ranking kandidat trade
- probabilitas menang per setup
- penentuan sleeve terbaik
- dynamic sizing
- regime confidence
- exit profile selection
- veto berbasis statistik, bukan rule statis

### Lapisan yang harus tetap audit-friendly

- reason code per rejection
- feature attribution per trade
- regime summary
- sleeve summary
- false-positive / false-negative analysis
- drift monitoring

## 4) Statistik yang Harus Masuk ke Single Brain

### Market context

- regime
- dispersion
- volatility bucket
- trend strength
- session / waktu UTC
- correlation with BTC / market beta

### Microstructure / perp context

- funding rate
- basis / premium
- open interest change
- order book imbalance
- liquidity sweep / absorption
- volume confirmation

### Strategy context

- sleeve
- direction
- score total
- score per komponen
- threshold buffer
- cohort / setup family
- exit profile

### Learning context

- historical win rate per sleeve
- historical win rate per regime
- rolling profit factor
- drawdown state
- recent streak
- shadow-vs-real flag
- sample age

## 5) Fase Migrasi

### Fase 0 — Rule Engine + Statistik Dasar

Status sekarang:

- strategi inti masih ditentukan oleh router/scorer
- ML sudah ada, tetapi masih harus dibatasi agar tidak jadi bottleneck
- risk layer sudah aktif sebagai pagar keras

Tujuan fase ini:

- data terkumpul rapi
- label trade bersih
- rejection reason bisa diaudit
- tidak ada gate statistik yang terlalu agresif

### Fase 1 — ML sebagai Advisor

AI boleh:

- memberi `p_win`
- memberi ranking kandidat
- memberi sizing penalty / bonus ringan
- memberi regime confidence

AI belum boleh:

- memveto secara agresif
- mengubah semua keputusan risk
- mengambil alih exit logic penuh

### Fase 2 — ML sebagai Gate Terkontrol

AI boleh:

- memveto trade dengan budget terbatas
- menurunkan ukuran posisi pada sinyal lemah
- memilih sleeve terbaik secara probabilistik

Syarat masuk fase ini:

- sample real cukup
- OOS/walk-forward stabil
- veto rate kecil dan terukur
- tidak ada drift besar per regime

### Fase 3 — ML sebagai Decision Brain Utama

AI boleh:

- mengurutkan kandidat secara menyeluruh
- memilih entry / no-entry
- memilih sizing dan exit profile
- menggunakan statistik multi-source sebagai input utama

Rule engine pada fase ini menjadi:

- feature generator
- guardrail
- fallback mode saat data tidak cukup

### Fase 4 — Production Quant Mini

Karakter sistem:

- keputusan utama berbasis model statistik
- eksplisit terhadap uncertainty
- adaptif terhadap regime shift
- tidak mudah overtrade
- tetap punya risk kill-switch yang keras

## 6) Kriteria Naik Fase

### Dari Advisor ke Gate Terkontrol

Wajib lolos:

- minimum trade sample memadai
- win rate stabil secara rolling
- profit factor tidak runtuh saat OOS
- max drawdown tetap terkontrol
- veto rate ML rendah

### Dari Gate Terkontrol ke Decision Brain

Wajib lolos:

- hasil walk-forward konsisten
- performa per regime tidak rapuh
- evaluasi sleeve x regime jelas
- confidence calibration layak
- model tidak hanya menang di satu periode

## 7) Data Schema Minimum

Trade record ideal harus menyimpan:

- timestamp open/close
- symbol
- direction
- regime
- sleeve
- exit profile
- total score
- feature scores
- EV result
- ML p_win
- ML threshold
- veto reason
- size multiplier breakdown
- dispersion value / state
- funding / OI / spread context
- outcome PnL
- MFE / MAE

## 8) Metrik yang Harus Dipantau

- trade count
- expectancy
- hit rate
- profit factor
- Sharpe / Sortino
- max drawdown
- turnover
- fee drag
- veto rate ML
- reject rate per stage
- win rate per sleeve
- win rate per regime
- calibration error
- OOS vs in-sample gap
- drift per periode

## 9) Guardrail yang Tidak Boleh Hilang

- risk cap tetap lebih tinggi dari model
- ML tidak boleh memaksa trade saat market tidak layak
- veto keras harus punya budget dan observability
- model harus bisa dimatikan tanpa mematikan bot
- semua keputusan harus bisa diaudit ulang

## 10) Rekomendasi Urutan Implementasi

1. rapikan data schema dan label trade
2. tambah monitoring calibration dan veto rate
3. jadikan ML ranking + sizing advisor yang stabil
4. tambahkan controlled veto budget per regime
5. pindahkan pemilihan sleeve ke decision layer
6. pindahkan exit profile selection ke model
7. evaluasi apakah rule router masih perlu sebagai decision layer atau hanya feature layer

## 11) Kesimpulan Praktis

Bot ini paling aman berkembang jika:

- sekarang tetap dijalankan sebagai **rule engine yang dibantu ML**
- kemudian ML naik menjadi **decision layer terkontrol**
- lalu baru menjadi **single brain** jika data dan OOS benar-benar membuktikan stabilitas

Jalur ini lebih lambat, tetapi jauh lebih aman untuk bootstrap ke live.
