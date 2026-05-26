# Ninja Trader — Panduan Lengkap (Bahasa Indonesia)

> Session handoff: baca `session.md` dulu; itu ringkasan state terbaru repo, VM, dan bot.

## Apa Itu Ninja Trader?

Ninja Trader adalah bot trading kripto otomatis yang beroperasi di Binance Futures (USDT-margined perpetual). Bot ini bekerja 24 jam penuh — memindai pasangan kripto, menilai sinyal menggunakan indikator institusional, mengelola posisi secara dinamis, dan **belajar dari riwayat tradingnya sendiri** untuk terus berkembang.

Persona performa internal bot ini dinamakan **Jim** — terinspirasi dari Jim Simons, matematikawan legendaris pendiri Renaissance Technologies yang menggunakan data dan algoritma untuk mengalahkan pasar.

---

## Cara Kerja Secara Umum

Bayangkan bot ini seperti seorang trader profesional yang tidak pernah tidur:

1. Setiap 60 detik, bot **memindai** seluruh pasangan futures di Binance
2. Bot **menilai** setiap pasangan menggunakan 7 indikator berbeda
3. Jika semua syarat terpenuhi, bot **membuka posisi** secara otomatis
4. Bot **memantau** posisi aktif setiap saat dan keluar di harga yang tepat
5. Setiap trade **dicatat** dan digunakan untuk memperbaiki keputusan berikutnya

```
Scan Pasar → Nilai Sinyal → Cek Risiko → Buka Posisi → Kelola Exit
                                                             ↓
                                                    Catat Data → Belajar
```

---

## Komponen Utama

| Komponen | File | Fungsi |
|---|---|---|
| Scanner | `scanner/scanner.py` | Menyaring pasangan berdasarkan volume dan blacklist |
| Market Data | `data/market_data.py` | Mengambil data OHLCV, funding rate, OI, order book |
| Scorer | `scoring/scorer.py` | Menilai kualitas sinyal trading |
| Risk Manager | `risk/risk_manager.py` | Memantau ekuitas, drawdown, batas posisi |
| Kelly Sizer | `risk/kelly_sizer.py` | Menghitung ukuran posisi optimal |
| Executor | `execution/executor.py` | Menempatkan order di Binance |
| Trade Manager | `execution/trade_manager.py` | Memantau posisi aktif dan eksekusi exit |
| Learner | `learning/learner.py` | Menyimpan riwayat trade dan menyesuaikan bobot sinyal |
| EV Model | `models/ev_model.py` | Menghitung expected value sebelum masuk trade |
| Adaptive Brain | `models/adaptive_brain.py` | Overlay sizing aktif, veto pair-health, pembelajaran online |
| ML Engine | `models/ml_engine.py` | Interface prediktor pasif dan helper kesiapan live trading |
| Fund Manager | `models/fund_manager.py` | Laporan performa (Sharpe, win rate, profit factor) |
| Shadow Engine | `backtest/shadow_engine.py` | Trade bayangan tanpa modal untuk mempercepat pengumpulan data |
| Dataset Logger | `data/dataset_logger.py` | Menulis dataset 85 kolom ke file parquet untuk ML |
| Telegram | `notifications/telegram.py` | Notifikasi trade, heartbeat, laporan performa |

---

## Strategi Bot Saat Ini

Bagian ini menjelaskan strategi operasional terbaru bot. Jika ada perbedaan antara bagian lama handbook dan catatan runtime, gunakan `session.md` dan `config/config.yaml` sebagai sumber kebenaran terbaru.

### Gambaran Besar Strategi

Bot ini adalah sistem trading directional untuk crypto perpetual futures. Ia bukan bot satu indikator. Entry hanya boleh terjadi setelah kandidat melewati beberapa layer validasi:

1. Scanner memilih pair futures yang likuid.
2. Market data mengambil OHLCV, funding, open interest, order book, dan konteks pasar luas.
3. Scorer memberi nilai numerik ke setiap pair.
4. Strategy router menentukan tesis strategi: trend, reversal, breakout, atau neutral.
5. Setup passport dibuat sebagai kontrak tesis trade.
6. Admission policy mengecek apakah kandidat boleh diuji di paper/live.
7. Cohort policy dan lifecycle mengecek kesehatan pola historis.
8. Adaptive Brain, ML, Fund Manager, Edge Detector, dan session logic memodulasi size.
9. Risk Manager mengecek exposure, sector, direction, correlation, drawdown, dan daily loss.
10. Trade Manager mengelola entry, TP, trailing, early cut, passport monitor, dan close.

Prinsip penting: **score tinggi belum tentu open posisi**. Trade hanya dibuka kalau setup, regime, sleeve, risk, dan policy semuanya lolos.

### Mode Runtime Terbaru

- Mode: `paper`.
- Starting equity paper: `$1000`.
- Base max open trades: `2`.
- Paper validation bisa melebar ke `5` posisi.
- EV gate dipakai sebagai observasi/telemetry, bukan hard gate utama saat sample collection.
- Cooldown dan loss-streak halt dinonaktifkan.
- Loss-streak guard tetap aktif, tetapi hanya memperketat kualitas setup dan menurunkan size saat bot sedang rugi.
- Cohort history sudah di-reset fresh setelah cohort policy baru, jadi bot belajar ulang dari data baru.

### Strategy Sleeves

Router mengelompokkan kandidat ke beberapa sleeve strategi:

| Sleeve | Fungsi | Status |
|---|---|---|
| `trend_following` | Mengikuti momentum/continuation, long atau short | Sleeve utama |
| `reversal` | Reversal setelah sweep, distribution, atau exhaustion | Aktif bersyarat |
| `compression_breakout` | Breakout dari compression/accumulation | Eksperimental |
| `neutral` | Tidak ada tesis strategi yang valid | Bukan alpha; hanya telemetry/shadow |

`neutral` tidak boleh dianggap sebagai strategi trading utama. Jika score pair cukup tinggi tetapi router memberi `neutral`, biasanya trade tetap ditolak karena tesisnya belum jelas.

### Setup Utama

#### 1. Trend Continuation

Ini setup utama saat market berada di `trending_expansion`.

Syarat umumnya:

- arah jelas, long atau short;
- trend strength cukup;
- struktur harga mendukung;
- volume dan open interest menunjukkan participation;
- tidak ada microstructure contradiction;
- directional alignment cukup;
- dispersion tidak terlalu merusak reliabilitas ranking.

Contoh cohort/setup key:

```text
trend_continuation|trending_expansion|long
```

Expected path untuk setup ini adalah `impulse_continuation`. Jika trade tidak mengikuti path ini dan justru menghabiskan MAE tanpa MFE yang layak, passport monitor bisa menutup posisi lebih awal.

#### 2. MTF Price Action Continuation

Setup continuation berbasis multi-timeframe.

Kondisi yang dicari:

- higher timeframe trend mendukung;
- pullback tidak merusak struktur;
- timeframe utama menunjukkan continuation break;
- volume/OI participation mendukung;
- diblok saat stress paper atau dispersion high karena masih experimental.

Expected path: `mtf_impulse_continuation`.

#### 3. VWAP Pullback Continuation

Setup trend pullback ke area VWAP lalu reclaim/continuation.

Logikanya:

- VWAP dipakai sebagai benchmark eksekusi institusional;
- trend harus searah;
- pullback masih sehat;
- reclaim/continuation terlihat;
- volume/OI participation mendukung.

Expected path: `vwap_reclaim_continuation`.

#### 4. Liquidity Sweep Reversal

Setup reversal berbasis liquidity raid/reclaim.

Kondisi yang dicari:

- harga mengambil liquidity atau sweep extreme;
- harga lalu reclaim/close balik;
- kualitas setup tinggi;
- cohort dipisahkan dari reversal generik.

Expected path: `snapback_then_follow_through`.

#### 5. Dedicated Short Setups

Bot punya jalur short khusus untuk pola seperti:

- Phase D atau post-distribution breakdown;
- bearish liquidity sweep;
- smart-money short setup.

Ini penting karena short tidak hanya berasal dari trend turun. Ada setup short yang muncul dari distribution atau liquidity raid.

### Scoring

Score dihitung dari banyak komponen:

- trend strength;
- volume confirmation;
- structure quality;
- open interest;
- funding sentiment;
- order book quality;
- volatility;
- higher timeframe alignment;
- smart-money context;
- broad market context.

Bobot inti saat ini:

| Komponen | Bobot |
|---|---:|
| `trend_strength` | 20 |
| `structure_quality` | 20 |
| `volume_confirmation` | 15 |
| `open_interest` | 15 |
| `funding_sentiment` | 10 |
| `order_book` | 10 |
| `volatility` | 10 |

Score adalah ranking awal, bukan izin open. Setelah score, kandidat masih bisa ditolak oleh router, paper scope, cohort policy, lifecycle, risk, correlation, atau slot budget.

### Market Regime

Regime utama:

| Regime | Arti | Bias Normal |
|---|---|---|
| `trending_expansion` | Momentum/expansion | Trend following |
| `accumulation_compression` | Compression sebelum breakout | Compression breakout |
| `distribution` | Pelemahan/distribution | Reversal/short |
| `chaos` | Noise tinggi/tidak stabil | Biasanya diblok |

Jika regime `chaos`, bot biasanya tidak trade walaupun ada pergerakan besar.

### Smart Money Layer

Smart-money phase membaca kombinasi:

- open interest;
- funding;
- volume buildup;
- taker flow;
- long/short imbalance;
- liquidity sweep;
- distribution/accumulation clues.

Phase yang mungkin muncul:

- `trending`;
- `accumulation`;
- `distribution`;
- `liquidity_sweep`;
- `neutral`;
- `chaos`.

Smart money bukan entry signal mandiri. Ia memperkuat atau melemahkan tesis router.

### Market Context Layer

Bot membaca konteks pasar luas agar pair-level signal tidak diperlakukan terisolasi.

Field utama:

- BTC trend;
- ETH/BTC trend;
- BTC dominance, jika feed tersedia;
- TOTAL crypto market, jika feed tersedia;
- risk-on/risk-off state;
- alt rotation state;
- confidence.

Dampaknya saat ini lebih banyak berupa soft multiplier, bukan hard gate. Contoh: long alt bisa dipenalti saat broad market risk-off, tetapi market context tidak otomatis memblok semua trade sebelum attribution membuktikan sinyalnya stabil.

### Setup Passport

Setiap trade valid punya setup passport. Ini adalah kontrak tesis trade yang disimpan bersama posisi.

Field penting:

- symbol, asset, sector;
- direction;
- setup type;
- sleeve;
- regime;
- smart-money phase;
- quality score;
- alignment;
- participation;
- funding/crowding state;
- expected path;
- invalidation;
- hold profile;
- market context saat entry.

Passport dipakai untuk dua hal:

1. pembelajaran cohort setelah trade close;
2. exit lebih awal jika tesis setup gagal sebelum TP1.

### Cohort dan Lifecycle

Cohort adalah kelompok statistik trade yang punya setup/konteks sama.

Contoh:

```text
trend_continuation|trending_expansion|long
```

Untuk setiap cohort, bot menghitung:

- jumlah trade;
- win rate;
- profit factor;
- expectancy;
- drawdown;
- lifecycle status.

Status lifecycle:

- `RESEARCH`;
- `PAPER_VALIDATION`;
- `SMALL_LIVE`;
- `ACTIVE`;
- `DISABLED`.

Setelah reset fresh, cohort aktif mulai dari `0`. Cohort baru akan terbentuk setelah trade baru closed dan masuk learner/history baru.

### Admission Policy

Agar kandidat bisa open, umumnya harus lolos:

- direction allowed;
- regime allowed;
- sleeve allowed;
- score threshold;
- setup quality;
- paper validation scope;
- duplicate-symbol check;
- cohort policy;
- lifecycle status;
- risk budget;
- sector cap;
- direction cap;
- correlation filter;
- exposure cap.

Paper validation lebih longgar dari live agar sample terkumpul, tetapi tetap bukan bebas. Ada score floor, high-conviction path, positive-EV probe untuk short liquidity sweep tertentu, dan stress tightening saat daily loss/loss streak memburuk.

### Risk dan Sizing

Parameter penting saat ini:

| Parameter | Nilai |
|---|---:|
| Base risk per trade | `1.5%` |
| Max risk per trade | `2.5%` |
| Daily loss cap | `5.0%` |
| Max drawdown | `20.0%` |
| Max sector risk | `2.5%` |
| Max direction risk | `5.5%` |
| Default leverage | `6x` |
| Max leverage | `12x` |

Size tidak hanya dihitung oleh Risk Manager. Size juga dimodulasi oleh:

- Adaptive Brain;
- Fund Manager;
- Kelly sizing;
- cohort confidence;
- Edge Detector;
- session factor;
- strategy sleeve multiplier;
- loss-streak guard;
- sector/direction exposure.

Akibatnya, score tinggi tetap bisa mendapat size kecil jika konteks atau risk tidak mendukung.

### Exit Logic

Trade Manager mengelola beberapa exit layer:

1. Take profit bertahap: TP1, TP2, lalu trailing runner.
2. Breakeven: stop bisa digeser ke breakeven setelah MFE tertentu.
3. ATR trailing: trailing disesuaikan per exit profile.
4. Early cut: loser bisa dipotong sebelum full SL jika MAE besar dan MFE lemah.
5. Passport monitor: posisi bisa ditutup sebelum TP1 jika tesis setup gagal.

Exit profile berbeda per sleeve:

| Sleeve | Karakter Exit |
|---|---|
| `trend_following` | Target lebih jauh, runner lebih besar |
| `compression_breakout` | Target sedang, trailing cukup aktif |
| `reversal` | Target lebih cepat, hold lebih pendek |

### Adaptive Brain dan ML

`AdaptiveBrain` aktif sebagai overlay, bukan otak utama. Ia membantu menilai pair health, menyesuaikan size, membaca edge/decay, dan memberi tekanan ke sleeve yang lebih sehat.

`MLPredictor` masih pasif/helper. ML belum menjadi pengambil keputusan utama karena sample live/paper yang bersih belum cukup.

### Position Rotation

Jika slot penuh, bot bisa mengganti posisi lemah dengan kandidat yang jauh lebih kuat.

Syaratnya ketat:

- kandidat baru harus punya score jauh lebih tinggi;
- kualitas minimal terpenuhi;
- p_win minimal terpenuhi;
- EV tidak terlalu buruk;
- posisi lama cukup tua;
- posisi yang sudah TP1/protected tidak mudah diganggu;
- jumlah rotasi per cycle dan per hari dibatasi.

Tujuannya mencegah slot penuh oleh posisi medioker tanpa membuat bot overtrade.

### Ringkasan Strategi

Strategi bot ini adalah **multi-layer systematic futures strategy** yang mencari continuation, reversal, dan breakout di crypto perpetual futures. Setiap entry divalidasi lewat score, regime, smart money, market context, setup passport, cohort learning, lifecycle status, dan risk guard.

Pada fase runtime saat ini, tujuan utama adalah **paper sample collection yang bersih** setelah cohort policy baru. Karena history lama sudah di-reset, bot sedang membangun ulang statistik setup dari trade fresh yang sesuai arsitektur terbaru.

---

## Pipeline Penilaian Sinyal (6 Langkah)

Sebelum bot membuka posisi, sinyal harus **lolos semua 6 tahap** berikut:

---

### Langkah 1 — Membangun Feature Vector

Bot mengumpulkan data dari **4 timeframe sekaligus**:
- **1 jam** — timeframe utama untuk sinyal
- **4 jam** — konfirmasi tren jangka menengah
- **15 menit** — detail struktur pasar
- **5 menit** — titik entry presisi

Data yang dikumpulkan:
- **ATR** — volatilitas pasar
- **ADX** — kekuatan tren
- **EMA 21/55/200** — arah tren
- **RSI** — momentum
- **Volume ratio** — konfirmasi volume
- **Open Interest (OI)** — minat institusional
- **Funding rate** — sentimen pasar derivatif
- **Order book** — tekanan beli/jual
- **Likuidasi** — tekanan paksa

---

### Langkah 2 — Klasifikasi Regime Pasar

Bot mengidentifikasi kondisi pasar saat ini ke dalam 4 kategori:

| Regime | Kondisi Pasar | Skor Minimum |
|---|---|---|
| `trending_expansion` | ADX > 25, range melebar, tren kuat | **55** |
| `accumulation_compression` | ATR rendah, OI naik, pasar sideways ketat | **65** |
| `distribution` | Struktur bearish, OI turun | **DIBLOKIR** |
| `chaos` | Volatilitas ekstrem, tidak ada struktur jelas | **DIBLOKIR** |

> Saat regime `distribution` atau `chaos`, bot **tidak akan membuka trade apapun** — terlalu berisiko.

Threshold berbeda karena `accumulation_compression` lebih sulit dibaca — butuh sinyal lebih kuat.

---

### Langkah 3 — Smart Money Gate

Bot mendeteksi apakah institusi (whale) sedang berakumulasi atau distribusi menggunakan pola OI + harga + volume.

- Jika tidak ada bias directional yang jelas → **trade diblokir**
- Jika smart money score < 60 → **trade diblokir**

Tujuannya: jangan trading melawan institusi besar.

---

### Langkah 4 — EV Model Gate (Expected Value)

Bot menghitung apakah trade ini **menguntungkan secara matematis** setelah dikurangi:
- Taker fee Binance: 0.04% per leg
- Estimasi slippage: 0.10%

Jika net EV < 0.05% → **trade diblokir**.

> Ini seperti memastikan setiap taruhan punya nilai positif — tidak ada gunanya trading jika secara statistik rugi setelah biaya.

---

### Langkah 5 — Penilaian Skor (0–100)

Jika semua gate lolos, bot menghitung skor akhir dari 7 komponen:

| Komponen | Bobot | Penjelasan |
|---|---|---|
| Kekuatan tren | 20 | Seberapa kuat tren saat ini |
| Kualitas struktur | 20 | Break of Structure (BOS) dan liquidity sweep |
| Konfirmasi volume | 15 | Apakah volume mendukung arah |
| Open interest | 15 | Apakah OI mendukung arah |
| Order book | 10 | Ketidakseimbangan bid/ask |
| Sentimen funding | 10 | Apakah funding rate mendukung arah |
| Volatilitas | 10 | Kondisi volatilitas optimal |

**Total skor minimum untuk trade: 60** (atau 65 untuk regime accumulation).

Bobot ini **tidak tetap** — Learner menyesuaikannya setelah setiap trade berdasarkan mana yang lebih sering benar.

---

### Langkah 6 — Konfirmasi Sinyal

Sinyal harus muncul **2 kali berturut-turut** (2 scan × 60 detik = 2 menit) sebelum order ditempatkan.

Jika sinyal tidak dikonfirmasi dalam 5 menit → **sinyal kedaluwarsa**.

> Ini mencegah bot masuk di sinyal palsu yang hanya muncul sesaat.

---

## Manajemen Posisi

### Entry (Masuk Posisi)

- Order **limit** lebih diutamakan (offset 0.05% dari mid price)
- Timeout 60 detik — jika tidak terisi, order dibatalkan
- Posisi diisi dalam **2 level**: 60% pertama, 40% sisanya

### Risiko Per Trade

| Parameter | Nilai |
|---|---|
| Risiko default | 1.5% ekuitas (baseline paper sekarang mulai dari $70) |
| Batas maksimum | 2.5% (batas config) |
| Leverage default | 6x |
| Leverage maksimum | 10x |
| Minimal notional | $5 (minimum Binance) |

Ukuran posisi dihitung menggunakan **fractional Kelly** (quarter-Kelly) — formula matematis untuk sizing optimal berdasarkan win rate dan rasio win/loss.

### Level Exit

| Level | Pemicu | Aksi |
|---|---|---|
| **Stop Loss (SL)** | Harga bergerak 1.5× ATR melawan posisi | Tutup 100% posisi |
| **TP1** | Harga mencapai 1.5R keuntungan | Tutup 50%, pindah SL ke breakeven, mulai trailing |
| **TP2** | Harga mencapai 2.0R keuntungan | Tutup 30% sisa, lanjutkan trailing |
| **Trailing Stop** | ATR × 1.5 ratchet (hanya bergerak searah profit) | Tutup sisa posisi |
| **Max Hold** | 48 jam sejak entry | Tutup semua — jangan tahan terlalu lama |

**Contoh konkret** (long $70, entry $100, SL $98.50):
- R distance = $1.50
- TP1 = $102.25 → tutup 50%, SL naik ke $100
- TP2 = $103.00 → tutup 30% sisa
- Trailing stop mengikuti harga sambil mengunci profit

---

## MFE dan MAE — Apa Itu?

Bot melacak dua metrik penting untuk setiap trade:

**MFE (Maximum Favorable Excursion)** = seberapa jauh harga bergerak ke arah yang menguntungkan sebelum berbalik, diukur dalam satuan R.

**MAE (Maximum Adverse Excursion)** = seberapa jauh harga bergerak melawan posisi di titik terburuk, diukur dalam satuan R.

**Mengapa penting?**

Misalnya setelah 30 trade:
- Rata-rata MFE losers = 1.8R → artinya trade yang rugi sebenarnya sempat untung 1.8R sebelum berbalik. TP terlalu jauh, atau bot terlambat keluar.
- Rata-rata MAE winners = 0.7R → artinya trade yang untung sempat turun 0.7R sebelum naik. SL mungkin terlalu ketat.

Data ini memandu penyesuaian TP dan SL di masa depan.

---

## Guard Risiko (Perlindungan Modal)

Bot memiliki beberapa lapisan perlindungan otomatis:

| Guard | Batas | Aksi |
|---|---|---|
| Batas rugi harian | -5% ekuitas (-$25) | Berhenti trading hari itu |
| Maksimum drawdown | -20% ekuitas | Berhenti trading |
| Kekalahan berturut-turut | 3 kali berturut-turut | Jeda sementara |
| Volatilitas ekstrem | ATR × 3.0 | Jeda sementara |
| Minimum R:R ratio | 2.0 | Blokir trade |
| Maksimum posisi terbuka | 2 posisi | Blokir entry baru |

---

## Sistem Pembelajaran (Jim Belajar dari Pengalaman)

Bot ini tidak statis — Jim terus belajar dan berkembang melalui 3 fase:

### Fase 1 — Menggunakan Asumsi Awal (0–19 trade tertutup)

Bot menggunakan asumsi konservatif yang sudah diprogram:
- Win rate diasumsikan: 52%
- Rata-rata keuntungan per win: 2.5% ekuitas
- Rata-rata kerugian per loss: 1.0% ekuitas

Jim belum punya cukup data nyata, jadi menggunakan "tebakan terdidik".

### Fase 2 — Data Nyata Aktif (20+ trade) ✅ Jim sudah di sini

- Win rate menggunakan **data aktual** Jim
- Kelly sizing menyesuaikan dengan distribusi outcome nyata
- Bobot sinyal bergeser ke komponen yang lebih sering benar

Jim tidak lagi menebak — dia menggunakan rekam jejaknya sendiri.

### Fase 3 — Decay Per Sinyal (50+ trade, masa depan)

- Setiap komponen sinyal dinilai secara individual berdasarkan prediktivitasnya
- Komponen yang tidak terbukti prediktif mendapat bobot lebih rendah

State pembelajaran tersimpan di: `models/learning_state_futures.json`

---

## Dataset ML (Parquet) — Fondasi Masa Depan

Setiap evaluasi sinyal menulis satu baris ke file `data/trades/trade_log_futures.parquet` dengan **85 kolom**.

### Apa yang dicatat?

| Kategori | Contoh Data |
|---|---|
| Identitas | ID record, jenis (trade/no-trade), sumber |
| Sinyal saat keputusan | Skor total, 7 komponen skor, gate mana yang gagal |
| Konteks pasar | Regime, momentum, OI change, funding, smart money |
| EV model | Expected value, probabilitas menang, Kelly fraction |
| Setup posisi | Entry, SL, TP1, TP2, ukuran, leverage |
| Dinamika trade | MFE, MAE, durasi hold, waktu ke puncak MFE |
| Outcome | PnL, R:R tercapai, jenis exit, menang/kalah |

### Near-Miss No-Trade — Keunggulan Utama

Bot juga mencatat sinyal yang **hampir diambil** tapi ditolak (skor ≥ 55 tapi belum memenuhi threshold). Ini memungkinkan analisis di masa depan:

> "Dari semua sinyal yang Jim tolak di skor 62-64 pada regime accumulation — berapa persen yang sebenarnya akan menang?"

Jika jawabannya "sebagian besar menang" → threshold terlalu tinggi, Jim melewatkan peluang bagus.

Tanpa data near-miss, kita hanya tahu **apa yang berhasil**. Dengan data ini, kita juga tahu **apa yang salah ditolak**.

---

## Shadow Engine — Trade Bayangan

Shadow Engine menjalankan "trade hantu" secara paralel dengan threshold lebih rendah, tanpa menggunakan modal sungguhan.

Tujuannya: mempercepat pengumpulan data ML. Daripada menunggu 3 trade nyata per minggu, shadow engine bisa menjalankan puluhan simulasi dan mengisi parquet lebih cepat.

State tersimpan di: `models/shadow_state.json` (tidak hilang saat restart)

---

## Deployment & Infrastruktur

**Server:** Google Cloud Platform (GCP) e2-micro  
**Process manager:** `systemd` service `ninja-watchdog`  
**Watchdog** memantau bot setiap 30 detik — restart otomatis jika crash

### Mode Operasi

| Mode | Deskripsi |
|---|---|
| `paper` | Ekuitas virtual $70, API testnet Binance, tidak ada order nyata |
| `live` | Modal nyata, API mainnet Binance, `testnet: false` wajib |
| `backtest` | Replay data historis OHLCV |

**Proteksi penting:**
- Mode `paper` **memaksa** `testnet: true` di kode — tidak bisa diubah lewat config
- Mode `live` **menolak start** jika `testnet: true` atau API key kosong

### Auto-Live Transition

Ketika 7 kriteria kesiapan terpenuhi (win rate, drawdown, jumlah trade, dll.), bot bisa **otomatis beralih** dari paper ke live trading. Membutuhkan API key mainnet yang sudah diset di `.env`.

### File Konfigurasi

```bash
# .env — rahasia, jangan di-commit ke git
BINANCE_API_KEY=xxx
BINANCE_API_SECRET=xxx
TELEGRAM_TOKEN=xxx
TELEGRAM_CHAT_ID=xxx
```

### Menjalankan Bot

```bash
python -m src                              # jalankan dengan config default
python -m src --mode paper                 # paksa mode paper
python -m src --config path/ke/config.yaml # config kustom
```

---

## Notifikasi Telegram

| Pesan | Kapan Dikirim |
|---|---|
| Trade dibuka | Saat order terisi |
| Trade ditutup | Saat exit — dengan PnL, alasan, MFE/MAE |
| Heartbeat | Setiap 1 menit pada konfigurasi saat ini — ekuitas, drawdown, daily PnL, posisi aktif |
| Laporan Fund Manager | Setiap 10 trade tertutup — Sharpe, win rate, profit factor, bonus Jim |
| Live readiness | Saat semua 7 kriteria terpenuhi |

---

## Peta File Penting

```
src/
  config.yaml                    ← SEMUA pengaturan ada di sini
  main.py                        ← Orkestrator utama bot
  data/
    client.py                    ← Koneksi ke Binance API
    market_data.py               ← Ambil data OHLCV + derivatif
    dataset_logger.py            ← Tulis dataset parquet ML
  analysis/
    indicators.py                ← ATR, ADX, EMA, RSI, volume
    structure.py                 ← BOS, liquidity sweep, swing point
    smart_money.py               ← Deteksi institusional via OI
    feature_engine.py            ← Feature vector terpadu
    regime.py                    ← Klasifikasi regime pasar
    spot_context.py              ← Basis spot/futures, Coinbase premium
  scoring/scorer.py              ← Pipeline sinyal lengkap
  risk/
    risk_manager.py              ← Ekuitas, drawdown, batas posisi
    kelly_sizer.py               ← Sizing posisi Kelly
  execution/
    executor.py                  ← Penempatan order
    trade_manager.py             ← Lifecycle posisi + MFE/MAE
  learning/learner.py            ← Penyesuaian bobot + log trade
  models/
    ev_model.py                  ← Kalkulasi expected value
    ml_engine.py                 ← Cek kesiapan live, prediksi ML
    fund_manager.py              ← Metrik performa
  backtest/
    engine.py                    ← Backtester historis
    shadow_engine.py             ← Ghost trading paralel
    reporter.py                  ← Tabel hasil backtest
  notifications/telegram.py     ← Alert trade, heartbeat

models/
  learning_state_futures.json   ← Bobot learner + log trade (persisten)
  shadow_state.json             ← State shadow engine (persisten)
data/trades/
  trade_log_futures.parquet     ← Dataset training ML
logs/
  futures_trader.log            ← File log berputar
```

---

## Pengaturan Utama (config.yaml)

### Trading
```yaml
trading:
  mode: paper                      # paper | live | backtest
  paper_starting_equity: 1000      # Ekuitas virtual paper sample collection saat ini
  min_score_threshold: 42          # Skor minimum untuk trade
  max_open_trades: 2               # Maksimum 2 posisi sekaligus
  regime_thresholds:
    trending_expansion: 50         # Lebih permisif di tren kuat
    accumulation_compression: 50   # Lebih ketat di sideways
    distribution: 52               # Short reversal lebih berisiko
```

### Risiko
```yaml
risk:
  risk_per_trade_pct: 1.5          # 1.5% ekuitas per trade
  max_risk_per_trade_pct: 2.5      # Hard cap 2.5% per trade
  daily_loss_cap_pct: 5.0          # Berhenti jika rugi $25/hari
  max_drawdown_pct: 20.0           # Berhenti jika drawdown mencapai batas konfigurasi
  min_rr_ratio: 2.0                # Minimum R:R 2:1
  default_leverage: 6              # 6x leverage default
```

### Exit
```yaml
exit:
  tp1_r_multiple: 1.5              # TP1 di 1.5R
  tp2_r_multiple: 2.5              # TP2 di 2.5R
  tp1_size_pct: 0.40               # Tutup 40% di TP1
  tp2_size_pct: 0.30               # Tutup 30% di TP2
  trail_size_pct: 0.30             # Tutup 30% sisanya di trailing
  breakeven_trigger_r: 0.8         # Pindah ke breakeven lebih cepat
  trailing_atr_multiplier: 1.3     # Trailing stop = ATR × 1.3
  max_hold_duration_s: 172800      # Maksimum hold 48 jam
```

---

## Panduan Penyesuaian (Tuning)

| Tujuan | Kunci Config | Lokasi |
|---|---|---|
| Bot lebih sering trade | Turunkan `min_score_threshold` | `trading` |
| Hindari pasar choppy | Naikkan threshold `accumulation_compression` | `trading.regime_thresholds` |
| Risiko lebih besar per trade | Naikkan `risk_per_trade_pct` | `risk` |
| Ambil profit lebih cepat | Turunkan `tp1_r_multiple` | `exit` |
| Trailing stop lebih ketat | Turunkan `trailing_atr_multiplier` | `exit` |
| Butuh tren lebih kuat | Naikkan `adx_trending_threshold` | `regime` |
| Filter pair volume rendah | Naikkan `min_24h_volume_usdt` | `filters` |

---

## Memantau Bot

### Cek status
```bash
sudo systemctl status ninja-watchdog
```

### Lihat log real-time
```bash
tail -f ~/ninja_trader/logs/futures_trader.log
```

### Lihat 80 baris log terakhir
```bash
tail -80 ~/ninja_trader/logs/futures_trader.log
```

### Restart bot
```bash
sudo systemctl restart ninja-watchdog
```

---

## Glosarium

| Istilah | Arti |
|---|---|
| **R** | Satuan risiko. 1R = jarak dari entry ke stop loss. TP1 = 1.5R artinya profit 1.5× risiko |
| **MFE** | Maximum Favorable Excursion — seberapa jauh harga menguntungkan di titik terbaik |
| **MAE** | Maximum Adverse Excursion — seberapa jauh harga merugikan di titik terburuk |
| **EV** | Expected Value — rata-rata hasil yang diharapkan secara matematis |
| **Kelly** | Formula matematis untuk sizing posisi optimal |
| **Regime** | Kondisi pasar saat ini (trending, sideways, distribusi, chaos) |
| **BOS** | Break of Structure — harga menembus level swing sebelumnya |
| **OI** | Open Interest — total posisi terbuka di futures |
| **Funding Rate** | Biaya yang dibayar long ke short (atau sebaliknya) di perpetual futures |
| **Smart Money** | Institusi besar / whale yang menggerakkan pasar |
| **Shadow Trade** | Trade simulasi tanpa modal nyata untuk mengumpulkan data lebih cepat |
| **Near-miss** | Sinyal yang hampir diambil tapi ditolak — dicatat untuk analisis ML |
| **Parquet** | Format file database kolumnar, efisien untuk data ML besar |
| **Drawdown** | Penurunan ekuitas dari titik tertinggi |
| **Sharpe Ratio** | Metrik return dibagi risiko — semakin tinggi semakin baik |
| **Profit Factor** | Total keuntungan dibagi total kerugian — harus > 1.0 |
