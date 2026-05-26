# Ninja Trader Strategy Stack

> Last updated: 2026-05-26.
>
> Kebenaran operasional terbaru tetap ada di [`session.md`](./session.md). Dokumen ini adalah peta belajar: bagaimana stack strategi bekerja, layer mana yang mengambil keputusan, dan apa yang perlu dibaca saat audit runtime.

## 1. Gambaran Besar

Ninja Trader adalah sistem trading crypto perpetual futures dua arah. Ini bukan bot satu sinyal. Entry hanya boleh terjadi setelah kandidat melewati beberapa layer:

1. Scanner memilih pair futures yang likuid.
2. Market data membangun snapshot pair dan konteks pasar luas.
3. Scorer menghitung score pair.
4. Strategy router memilih sleeve strategi.
5. Setup passport dibuat sebagai kontrak tesis trade.
6. Paper/live admission policy menentukan apakah kandidat boleh diuji.
7. Cohort policy dan strategy lifecycle mengecek kesehatan pola historis.
8. ML, fund manager, adaptive brain, session, dan edge detector memodulasi veto/size.
9. Risk guard, sector guard, direction guard, dan correlation guard melindungi akun.
10. Jika slot penuh, position rotation boleh mengganti posisi lemah dengan kandidat jauh lebih kuat.
11. Trade manager mengelola TP, trailing, early cut, passport monitor, close, dan persistence.

Prinsip penting: signal bagus belum tentu boleh dieksekusi. Eksekusi hanya terjadi kalau tesis, konteks, risk budget, dan policy semuanya cukup sehat.

## 2. Postur Saat Ini

Status runtime/paper terbaru yang perlu diingat:

- Mode: `paper`.
- Paper baseline: `$1000`.
- Base `max_open_trades`: `2`, dilebarkan oleh paper validation menjadi `5`.
- `min_score_threshold`: `42`, tetapi paper validation bisa menurunkan floor ke `35` pada jalur tertentu.
- `trend_long_only`: `false`, jadi trend-following long dan short sama-sama aktif.
- `ev_model.gate_enabled`: `false`; EV tetap dihitung dan dicatat, tetapi bukan hard gate admission saat ini.
- `AdaptiveBrain`: aktif sebagai overlay sizing/pair-health.
- `MLPredictor`: masih helper pasif, bukan otak utama.
- `position_rotation.enabled`: `true`, tetapi hanya boleh bekerja di bawah constraint ketat.
- `passport_monitor.enabled`: `true`, bisa menutup trade pre-TP1 kalau kontrak setup gagal.
- Market context aktif dan sudah mengalir ke scoring, dataset, learner, shadow ML, attribution, bootstrap audit, dan Telegram report.
- Sector rotation aktif: dihitung sekali per scan dari snapshot yang sudah di-fetch, memberi soft multiplier direction-aware, dan dipersist ke setup passport/dataset.
- Scanner sector mapping sudah dinamis dari Binance market metadata; live VM check terakhir: `eligible 56`, `other []`.
- Binance Alpha punya treatment risiko khusus: config Alpha aktif, geometry/stop-distance guard, exposure isolation, dan conditional score/size penalty untuk setup kualitas rendah, partisipasi lemah, atau volatilitas ekstrem.

Catatan runtime terakhir: VM berjalan di `screen -S ninja-trader-vm`, pid terakhir dicek `448557`, heartbeat `equity=$986.54`, scanner `56` eligible pairs, dan log menunjukkan `Sector rotation top: l2 rotating_in`.

## 3. Cara Membaca Stack Ini

Baca stack dari atas ke bawah:

- `Scanner` menjawab: pair mana yang layak dilihat?
- `MarketDataService` menjawab: data pair dan market backdrop apa yang tersedia?
- `Scorer` menjawab: seberapa kuat pair ini secara angka?
- `StrategyRouter` menjawab: tesis strategi apa yang valid?
- `SetupPassport` menjawab: trade ini seharusnya berjalan seperti apa?
- `Admission policy` menjawab: boleh diuji di paper/live atau tidak?
- `Cohort/Lifecycle` menjawab: pola seperti ini sehat berdasarkan history atau belum?
- `Overlays` menjawab: size perlu dinaikkan, diturunkan, atau diblok?
- `Risk/Correlation` menjawab: akun boleh menanggung exposure ini atau tidak?
- `TradeManager` menjawab: setelah open, kapan exit dan bagaimana state disimpan?

Kalau ada trade ditolak, jangan berhenti di score. Cari stage penolakan terakhir di log.

## 4. Strategy Sleeves

Router mengelompokkan kandidat ke sleeve strategi.

| Sleeve | Fungsi | Status saat ini | Status alpha |
|---|---|---|---|
| `trend_following` | Momentum/continuation long atau short | Sleeve utama | Kandidat alpha utama |
| `reversal` | Sweep, distribution, exhaustion reversal | Aktif bersyarat | Kandidat alpha kondisional |
| `compression_breakout` | Breakout dari compression/accumulation | Aktif eksperimen | Research sleeve |
| `neutral` | Fallback saat router tidak bisa memvalidasi sleeve | Untuk telemetry/shadow | Bukan alpha |

`neutral` tidak boleh diperlakukan sebagai edge. Ia berarti kandidat belum bisa dinaikkan menjadi tesis strategi yang jelas. Paper-only probe boleh mengamati neutral di kondisi tertentu, tetapi neutral tidak boleh dipromosikan ke live tanpa bukti.

## 5. Alur Admission Runtime

Urutan runtime secara ringkas:

1. Scanner mencari pair eligible.
2. `MarketDataService.fetch_snapshots()` mengambil OHLCV, order book, funding, OI, dan konteks luas.
3. Market context dihitung sekali per cycle dan ditempel ke tiap snapshot.
4. Sector rotation dihitung sekali per cycle dari snapshot yang sama dan ditempel ke tiap snapshot.
5. `Scorer.score_many()` menghitung score dan metadata signal.
6. Regime, smart-money, dan EV flags dievaluasi.
7. Score threshold dan pengecualian paper validation dicek.
8. Paper scope allowlist dicek.
9. Duplicate open symbol dan signal confirmation dicek.
10. Cohort policy mengecek symbol, sector, sleeve, dan context health.
11. Edge detector mengecek memory pair/regime.
12. Strategy lifecycle memberi status `RESEARCH`, `PAPER_VALIDATION`, `SMALL_LIVE`, `ACTIVE`, `DEGRADED`, atau `DISABLED`.
13. ML gate bisa veto hanya jika sample cukup; di paper saat ini hard veto sangat ditunda.
14. Candidate ranking memilih kandidat terkuat.
15. Risk guard mengecek slot, drawdown, daily loss, aggregate risk, sector risk, dan direction risk.
16. Kalau penuh karena `max_open_trades`, position rotation bisa dipertimbangkan.
17. `RiskManager.calculate_setup()` membuat stop, target, size, leverage, dan R distance.
18. Fund manager, adaptive brain, session modulator, cohort sizing, dan edge sizing menyesuaikan size.
19. Correlation filter mengecek exposure yang terlalu mirip.
20. Trade dibuka, state disimpan, dataset/logger/Telegram/shadow diperbarui.

## 6. Market Context Layer

File utama: [`src/models/market_context.py`](./src/models/market_context.py)

Tujuan: membuat pair-level signal sadar terhadap konteks pasar crypto yang lebih luas, bukan hanya candle pair itu sendiri.

Field utama:

- `btc_trend`: `up`, `down`, `flat`, atau `unknown`.
- `eth_btc_trend`: kekuatan ETH relatif terhadap BTC.
- `btc_d_trend`: tren BTC dominance, optional sampai feed tersedia.
- `total_trend`: tren total crypto market, optional sampai feed tersedia.
- `risk_on_state`: contoh `risk_on_alts`, `risk_on_btc`, `risk_off`, `neutral`, `mixed`, `unknown`.
- `rotation_state`: contoh `alts_outperforming`, `btc_outperforming`, `broad_alt_rotation`, `btc_dominance_bid`, `mixed_rotation`, `unknown`.
- `confidence`: 0.0 sampai 1.0, berdasarkan ketersediaan input dan kekuatan tren.

Config saat ini:

```yaml
market_context:
  enabled: true
  ttl_seconds: 300
  lookback_bars: 48
  trend_threshold_pct: 1.0
  btc_symbol: "BTC/USDT:USDT"
  eth_symbol: "ETH/USDT:USDT"
  btc_d_symbol: ""
  total_symbol: ""
```

Dampak ke sistem:

- Di scoring: soft multiplier, bukan hard gate.
- Di setup passport: snapshot context entry disimpan untuk trade baru.
- Di learner: closed trade punya field market context.
- Di shadow ML: shadow trade log dipromosikan ke schema yang sama.
- Di dataset logger: trade dan near-miss row punya kolom context.
- Di attribution/reporting: report bisa dibaca per risk state, rotation state, dan BTC x ETH/BTC trend.
- Di Telegram/bootstrap audit: ringkasan performa by market context sudah muncul.

Aturan belajar: market context boleh membuat score lebih selektif, tetapi jangan dijadikan hard gate sebelum attribution membuktikan sinyalnya stabil.

## 6A. Sector Rotation Layer

File utama: [`src/models/sector_rotation.py`](./src/models/sector_rotation.py)

Tujuan: membaca perpindahan uang lintas sektor dari snapshot yang sudah ada, tanpa menambah call exchange baru. Layer ini menjawab sektor mana yang sedang relatif kuat/lemah terhadap BTC pada scan berjalan.

Field utama per sektor:

- `state`: `rotating_in`, `rotating_out`, `firming`, `weakening`, `neutral`, `mixed`, atau `unknown`.
- `rank`: ranking sektor dalam cycle.
- `sector_return_pct`: rata-rata return sektor.
- `relative_btc_pct`: return sektor dikurangi return BTC.
- `breadth`: persentase pair sektor yang naik.
- `volume_accel`: akselerasi volume sektor.
- `oi_change_pct`: rata-rata perubahan OI.
- `confidence`: 0.0 sampai 1.0.

Dampak ke sistem:

- Di scoring: soft multiplier direction-aware. `rotating_in` mendukung long dan menekan short; `rotating_out` mendukung short dan menekan long.
- Di setup passport dan dataset: field `sector_rotation_*` disimpan untuk attribution dan postmortem.
- Di runtime log: cycle mencetak sektor teratas, misalnya `Sector rotation top: l2 rotating_in`.

Aturan belajar: sector rotation bukan hard gate. Ia adalah ranking/context overlay untuk menangkap perpindahan uang besar tanpa mematikan setup pair yang kuat.

## 7. Pair-Level Scoring

File: [`src/scoring/scorer.py`](./src/scoring/scorer.py)

Score menggabungkan:

- trend strength
- volume confirmation
- structure quality
- open interest
- funding sentiment
- order book quality
- volatility
- higher timeframe alignment
- smart-money bonus/penalty
- spot context jika tersedia
- broad market context multiplier
- sector rotation multiplier

Default weight:

```yaml
trend_strength: 20
volume_confirmation: 15
structure_quality: 20
open_interest: 15
funding_sentiment: 10
order_book: 10
volatility: 10
```

Score adalah ranking awal, bukan izin open. Score tinggi masih bisa ditolak oleh strategy scope, cohort policy, lifecycle, risk, correlation, atau slot budget.


## 8. Regime dan Smart Money

Regime utama:

| Regime | Makna | Aksi normal |
|---|---|---|
| `trending_expansion` | Momentum/expansion | Kandidat trend following |
| `accumulation_compression` | Compression/buildup | Kandidat compression breakout |
| `distribution` | Pelemahan/distribution | Kandidat reversal, terutama short |
| `chaos` | Noise tinggi/tidak stabil | Biasanya diblok |

Smart-money phase:

- `trending`
- `accumulation`
- `distribution`
- `liquidity_sweep`
- `neutral`
- `chaos`

Smart money membaca OI, stagnasi harga, funding, volume buildup, long/short ratio, dan taker flow. Ia bukan entry signal mandiri. Struktur dan route tetap harus valid.

## 9. Strategy Router dan Setup Passport

Router: [`src/models/strategy_router.py`](./src/models/strategy_router.py)
Passport: [`src/models/strategy_passport.py`](./src/models/strategy_passport.py)

Router memilih sleeve berdasarkan:

- regime
- direction
- trend/structure score
- smart-money phase
- dedicated short setup
- dispersion
- feature-vector directional alignment
- volume/OI participation
- funding/crowding state
- microstructure contradiction

Config penting:

```yaml
strategy:
  trend_min_score: 58
  trend_min_structure: 52
  trend_long_only: false
  trend_require_participation: true
  trend_min_alignment: 0.54
  breakout_min_alignment: 0.57
  reversal_min_alignment: 0.45
  enable_compression_breakout: true
  compression_min_structure: 55
  reversal_max_volatility: 80
```

Setup passport adalah kontrak trade. Ia menjawab:

- aset/sector apa?
- setup type apa?
- sleeve apa?
- expected path apa?
- invalidation apa?
- quality/alignment/participation seberapa kuat?
- market context saat entry apa?

Field penting:

- `symbol`, `asset`, `sector`
- `direction`, `setup_type`, `sleeve`
- `regime`, `sm_phase`
- `quality_score`, `alignment`, `participation_score`
- `crowding_state`, `funding_state`, `microstructure_state`
- `expected_path`, `invalidation`, `hold_profile`
- `market_context`, `market_risk_on_state`, `market_rotation_state`
- `market_btc_trend`, `market_eth_btc_trend`, `market_btc_d_trend`, `market_total_trend`

Kenapa penting: trade tidak lagi hanya menyimpan label sleeve. Trade menyimpan tesis yang bisa diaudit, dipelajari, dan dipakai untuk exit jika gagal.

## 10. Dedicated Short Setups

File: [`src/analysis/short_strategies.py`](./src/analysis/short_strategies.py)

Dedicated short setup berasal dari pola yang dulu penting di `kiosarwin/Futures`, terutama `IMMINENT_DUMP + PHASE_D` dan `IMMINENT_DUMP + LIQ_SWEEP`.

`phase_d`:

- price break di bawah support distribusi
- support memakai low-percentile lookback
- volume spike atau EMA 9/21 bearish cross mengonfirmasi
- EMA stack bearish atau melemah

`liq_sweep`:

- ada equal-high liquidity pool
- wick menembus pool
- candle close kembali di bawah pool
- upper wick dan bearish body menunjukkan rejection

Short setup confidence tinggi bisa masuk `reversal` tanpa harus selalu `regime == distribution`, karena tesisnya sering berupa transisi dari distribution menuju markdown.

## 11. EV Model dan Paper Validation

EV model: [`src/models/ev_model.py`](./src/models/ev_model.py)

Postur saat ini:

```yaml
ev_model:
  gate_enabled: false
```

Maknanya:

- EV tetap dihitung.
- `p_win`, EV net, confidence, fee/slippage, dan trade count tetap dicatat.
- EV belum menjadi hard blocker saat paper edge discovery.
- Negative EV tetap penting sebagai telemetry, bukan untuk diabaikan.

Paper validation adalah harness eksperimen, bukan live policy.

Efek utama:

- long dan short boleh diuji
- `trending_expansion`, `accumulation_compression`, dan `distribution` boleh diuji
- `trend_following`, `reversal`, dan `compression_breakout` boleh diuji
- open trades bisa dilebarkan sampai 5
- threshold bisa direlaksasi sampai floor tertentu
- ML hard gate ditunda
- positive-EV probe lane tersedia secara terbatas

Positive-EV probe saat ini narrow:

- paper only
- short only
- regime `trending_expansion` atau `distribution`
- smart-money phase `liquidity_sweep`
- EV net harus positif dan cukup besar
- p_win harus di atas floor
- score boleh di bawah threshold hanya dalam batas tertentu
- smart-money score harus kuat

Tujuannya adalah mengambil sample sweep short yang secara EV menarik tetapi mungkin tertahan threshold score normal.

## 12. Cohort Policy dan Strategy Lifecycle

Cohort policy: [`src/models/cohort_policy.py`](./src/models/cohort_policy.py)
Lifecycle: [`src/models/strategy_lifecycle.py`](./src/models/strategy_lifecycle.py)

Cohort policy mengevaluasi kesehatan pola historis berdasarkan:

- symbol
- sector
- direction
- regime
- smart-money phase
- strategy sleeve
- session/hour/day
- volatility/trend bucket
- market risk/rotation context
- attribution report

Ia bisa block, reduce size, memberi ranking bonus, memberi threshold relief, dan memberi priority symbol ke scanner.

Lifecycle stage:

| Stage | Makna | Izin trading |
|---|---|---|
| `RESEARCH` | sample terlalu kecil | paper only |
| `PAPER_VALIDATION` | observasi mulai cukup | paper only |
| `SMALL_LIVE` | edge mulai tervalidasi | live kecil bisa dipertimbangkan |
| `ACTIVE` | cohort paling sehat | live eligible |
| `DEGRADED` | edge melemah | reduce/block |
| `DISABLED` | edge rusak/terlalu riskan | block |

Aturan penting: satu win hoki tidak boleh menjadi primary edge. Primary edge butuh sample minimum, profit factor, expectancy, win rate, dan outlier dependence yang sehat.

## 13. ML, Edge Detector, Fund Manager, Adaptive Brain

ML predictor: [`src/models/ml_engine.py`](./src/models/ml_engine.py)

- Helper layer, bukan otak utama.
- Bisa train dari real + shadow records.
- Hard gate ditunda sampai sample cukup.
- Confidence bisa memengaruhi sizing.
- Akurasi tinggi dengan sample kecil harus dicurigai.

Edge detector: [`src/models/edge_detector.py`](./src/models/edge_detector.py)

- Memory layer sebelum EV.
- Mode saat ini soft-gate.
- Bisa memberi status, confidence, action, size multiplier.
- Tidak boleh jadi hard authority sebelum sample floor cukup.

Fund manager: [`src/models/fund_manager.py`](./src/models/fund_manager.py)

- Mengatur risk appetite dan sizing.
- Membaca equity, drawdown, daily PnL, open risk, streak, trade log, dan confidence.
- Bisa veto, reduce, atau modest size-up.
- Tidak menggantikan router/risk manager.

Adaptive brain: [`src/models/adaptive_brain.py`](./src/models/adaptive_brain.py)

- Thompson bandit
- online logistic model
- regime HMM
- alpha decay tracker
- volatility targeter

Ia bisa block pair yang edge-nya decay dan mengubah size, tetapi tidak membuat tesis trade sendiri.

## 14. Risk Manager, Sector Guard, dan Correlation

Risk manager: [`src/risk/risk_manager.py`](./src/risk/risk_manager.py)
Correlation filter: [`src/risk/correlation_filter.py`](./src/risk/correlation_filter.py)

Risk manager bertugas:

- max open trades
- daily loss cap
- max drawdown
- cooldown/loss streak lock jika aktif
- aggregate risk
- symbol risk
- sector risk
- direction risk
- stop, target, size, leverage, R distance

Nilai penting saat ini:

```yaml
risk_per_trade_pct: 1.5
max_risk_per_trade_pct: 2.5
daily_loss_cap_pct: 5.0
max_drawdown_pct: 20.0
min_rr_ratio: 1.8
max_leverage: 12
default_leverage: 6
min_risk_usd: 0.50
max_symbol_risk_pct: 2.5
max_sector_risk_pct: 2.5
max_direction_risk_pct: 5.50
```

Correlation filter mencegah banyak trade yang terlihat berbeda tetapi sebenarnya beta bet yang sama. Contoh: tiga alt long yang korelasinya tinggi bisa berperilaku seperti satu posisi long besar.

## 15. Position Rotation

Config: `position_rotation`

Tujuan: saat slot penuh, bot boleh mengganti open trade lemah dengan kandidat baru yang jauh lebih kuat.

Config utama:

```yaml
position_rotation:
  enabled: true
  max_rotations_per_cycle: 1
  max_rotations_per_day: 3
  min_candidate_score: 72.0
  min_score_advantage: 14.0
  min_candidate_quality: 58.0
  min_candidate_p_win: 0.38
  min_candidate_ev_net_pct: -0.75
  positive_ev_probe_min_candidate_score: 35.0
  positive_ev_probe_min_score_advantage: 0.0
  positive_ev_probe_min_candidate_quality: 0.0
  min_victim_age_s: 900
  protect_after_tp1: true
  protect_mfe_r: 0.80
  protect_unrealized_gain_r: 0.30
  max_victim_adverse_r: 0.90
```

Rotation hanya boleh mulai jika:

1. Kandidat sudah melewati admission sampai fase ready-open.
2. Risk guard block spesifik adalah `max_open_trades`.
3. Kandidat memenuhi floor score, quality, p_win, dan EV.
4. Budget rotasi cycle/day masih tersedia.
5. Ada victim yang eligible dan tidak protected.
6. Candidate advantage cukup besar dibanding victim strength.

Trade protected jika TP1 sudah kena, MFE kuat, unrealized gain berarti, terlalu muda, atau sudah terlalu adverse. Rotation bukan panic close; exit adverse harus lewat stop, early cut, atau passport monitor.

## 16. Execution, Trade Management, dan Exit

Executor: [`src/execution/executor.py`](./src/execution/executor.py)
Trade manager: [`src/execution/trade_manager.py`](./src/execution/trade_manager.py)

Execution menangani entry, stop, TP1, TP2, paper fills, market close, dan cancel-all.

Trade manager menangani:

- open trade state
- TP1 partial
- TP2 partial
- TP3/trailing remainder
- breakeven move
- trailing stop
- early adverse cut
- max hold
- passport failure exits
- manual close
- rotation close
- persistence di `data/open_trades.json`

Exit profile per sleeve:

| Sleeve | TP1 | TP2 | TP3/trailing | Max hold | Tesis |
|---|---:|---:|---:|---|---|
| `trend_following` | 2.0R 30% | 3.5R 30% | 5.0R 40% | 4 hari | trend perlu ruang dan runner |
| `compression_breakout` | 1.5R 35% | 2.8R 30% | 4.0R 35% | 2 hari | breakout harus cepat expand |
| `reversal` | 1.2R 40% | 2.0R 30% | 3.0R 30% | 1 hari | reversal lebih cepat dan rapuh |

## 17. Passport Monitor dan Early Cut

Passport monitor menutup trade pre-TP1 kalau kontrak setup gagal.

```yaml
passport_monitor:
  enabled: true
  min_age_s: 600
  trend_failure_mae_r: 0.70
  trend_mfe_ceiling_r: 0.20
  breakout_failure_mae_r: 0.45
  breakout_mfe_ceiling_r: 0.25
  reversal_failure_mae_r: 0.50
  reversal_mfe_ceiling_r: 0.20
```

Interpretasi:

- Trend continuation harus mulai menunjukkan impulse sebelum menghabiskan terlalu banyak adverse R.
- Breakout tidak boleh re-enter range lama dan diam di sana.
- Sweep reversal harus snap back relatif cepat.
- Setelah TP1 kena, passport monitor berhenti dan trade management normal mengambil alih.

Early adverse cut:

```yaml
early_cut:
  enabled: true
  min_age_s: 420
  max_age_s: 5400
  mae_r_threshold: 0.55
  mfe_r_ceiling: 0.15
```

Tujuannya mengubah sebagian full loser menjadi loser yang lebih kecil tanpa mengganggu trade yang sudah menunjukkan MFE.

## 18. Shadow Engine, Dataset, dan Attribution

Shadow engine: [`src/backtest/shadow_engine.py`](./src/backtest/shadow_engine.py)
Dataset logger: [`src/data/dataset_logger.py`](./src/data/dataset_logger.py)
Attribution: [`src/reporting/attribution.py`](./src/reporting/attribution.py)
Bootstrap report: [`src/reporting/bootstrap.py`](./src/reporting/bootstrap.py)

Shadow engine menjawab:

- Apakah gate menolak trade yang seharusnya menang?
- Apakah score-positive tetapi EV-negative memang buruk?
- Apakah cohort policy terlalu ketat?
- Apakah paper admission kehilangan edge?

Dataset dan attribution sekarang mencatat:

- strategy sleeve
- setup type
- setup quality score
- setup passport
- sector
- dispersion
- regime
- smart-money phase
- EV metrics
- lifecycle status
- cohort key
- session/hour/day
- market risk state
- market rotation state
- BTC trend
- ETH/BTC trend
- BTC.D trend
- TOTAL trend
- short setup label/confidence
- fees dan slippage estimates

Alasannya sederhana: sistem harus belajar footprint mana yang benar-benar bekerja. `Score tinggi` saja tidak cukup.

## 19. Contoh Praktis

### Contoh A: Trend Short Saat Risk-Off

Input:

- BTC down
- ETH/BTC down
- pair `trending_expansion`
- direction short
- structure bearish
- volume/OI kuat
- smart money netral atau bearish

Kemungkinan hasil:

- market context mendukung short
- router memilih `trend_following`
- passport `trend_continuation`
- expected path `impulse_continuation`
- trend exit profile aktif

Failure mode: tidak ada impulse, MAE membesar, passport monitor bisa close sebelum full stop.

### Contoh B: Alt Long Saat BTC Dominance Bid

Input:

- BTC up atau flat
- ETH/BTC down
- BTC dominance up
- pair long score cukup bagus

Kemungkinan hasil:

- market context menekan alt long
- score bisa turun di bawah threshold
- kalau masih lolos, size cenderung konservatif
- cohort/lifecycle tetap menentukan admission akhir

### Contoh C: Liquidity Sweep Short di Bawah Threshold

Input:

- direction short
- smart-money phase `liquidity_sweep`
- EV net positif
- p_win di atas floor
- score di bawah threshold tetapi masih dalam batas probe

Kemungkinan hasil:

- positive-EV probe bisa admit di paper
- sleeve bisa reversal atau neutral/reversal sesuai allowlist probe
- constraint paper-only tetap berlaku

### Contoh D: Slot Penuh Dengan Kandidat Baru Kuat

Input:

- open trades 5/5
- kandidat baru score tinggi
- setup quality tinggi
- p_win/EV memenuhi floor rotasi
- victim sudah cukup umur, belum TP1, MFE rendah, flat/slightly negative

Kemungkinan hasil:

- risk guard return `max_open_trades`
- rotation selector mencari victim eligible
- kalau advantage cukup, victim ditutup lewat normal close path
- kandidat baru dibuka

## 20. Cara Membaca Log Runtime

| Log/stage | Arti |
|---|---|
| `G-REG` | regime gate gagal |
| `G-SM` | smart-money gate gagal |
| `G-EV` | EV gate gagal; saat EV hard gate off, konteksnya perlu dicek lagi |
| `SCORE` | score di bawah threshold setelah adjustment |
| `PAPER_SCOPE` | tidak masuk allowlist paper validation |
| `COHORT` | cohort policy block |
| `LIFE` | lifecycle block atau masih paper-only |
| `RISK` | risk guard block, sering karena max open trades |
| `CORR` | correlation filter block |
| `FM` | fund manager veto |
| `BRAIN` | adaptive brain block |
| `rotation skipped` | slot penuh, tetapi kandidat/victim/budget tidak layak rotasi |
| `passport_failed_*` | kontrak setup gagal sebelum TP1 |

Saat menganalisis missed trade, urutan yang benar:

1. Lihat top candidate dan score.
2. Lihat gate flags `R/S/X`.
3. Lihat rejection stage.
4. Lihat sleeve dan paper scope.
5. Lihat cohort/lifecycle.
6. Lihat risk/correlation/rotation.
7. Baru simpulkan bottleneck.

## 21. Yang Perlu Dimonitor Selama Paper Validation

Metrik prioritas:

- expectancy by `strategy_sleeve`
- expectancy by `setup_type`
- expectancy by `direction`
- expectancy by sector
- expectancy by `risk_on_state`
- expectancy by `rotation_state`
- expectancy by `BTC trend x ETH/BTC trend`
- PnL by short setup label (`phase_d`, `liq_sweep`)
- win rate dan profit factor by cohort
- stop-hit rate
- TP1-hit rate
- average R result
- fee/slippage drag
- jumlah passport exits
- hasil rotated-out trade vs replacement trade
- shadow trades yang mengalahkan real admissions

Sebuah sleeve tidak valid hanya karena berhasil open trade. Ia valid kalau expectancy, PF, stop-hit rate, dan outlier dependence sehat pada sample yang cukup.

## 22. Jangan Dilakukan

- Jangan anggap `neutral` sebagai alpha.
- Jangan promosi cohort dari satu win hoki.
- Jangan re-enable EV hard gate tanpa review sample depth dan false rejects.
- Jangan ubah position rotation menjadi panic close.
- Jangan lebarkan paper validation lalu menganggap hasilnya live-ready.
- Jangan abaikan fee/slippage, terutama pada akun kecil.
- Jangan asumsikan restored old trades punya passport lengkap.
- Jangan jadikan broad market context hard gate sebelum attribution cukup.
- Jangan deploy ke VM tanpa verifikasi local/VM yang relevan.

## 23. Gap Saat Ini

Known gaps / next checks:

- Perlu trade baru post-upgrade untuk membuktikan `setup.setup_passport.market_context` persist di `open_trades.json`.
- 4 open trade terakhir masih pre-context-passport, jadi wajar belum punya market context di passport.
- `BTC.D` dan `TOTAL` masih blank sampai feed index yang reliabel tersedia.
- Market context masih soft dan perlu attribution lebih panjang.
- Position rotation perlu observasi forward: apakah replacement benar-benar meningkatkan expectancy net fees/churn?
- Compression breakout masih eksperimen.
- Exit logic bisa dibuat lebih sleeve-native jika data sudah cukup.
- EV hard gate tetap off sampai ada bukti bahwa ia memperbaiki admission, bukan men-starve exploration.

## 24. Quick Validation Checklist

Sebelum mengambil keputusan baru:

```bash
git status --short --branch
./venv/bin/python -c "from src.main import load_config, normalize_config; cfg=normalize_config(load_config('config/config.yaml')); print(cfg['trading']['paper_starting_equity'], cfg['risk']['min_risk_usd'], cfg['risk']['max_direction_risk_pct'], cfg['trading']['max_open_trades'], cfg['trading']['min_score_threshold'])"
ssh gcp-futures 'cd /home/kiosarwin/ninja_trader && screen -ls'
ssh gcp-futures 'pgrep -af "python -m src.main|python -m src|venv/bin/python -m src"'
ssh gcp-futures 'cd /home/kiosarwin/ninja_trader && tail -n 120 logs/boot.log'
```

Untuk cek open trade passport:

```bash
ssh gcp-futures 'cd /home/kiosarwin/ninja_trader && ./venv/bin/python -c "import json; data=json.load(open(\"data/open_trades.json\")); trades=data.get(\"trades\", []); print(\"open\", len(trades)); [print((t.get(\"setup\") or {}).get(\"symbol\"), bool(((t.get(\"setup\") or {}).get(\"setup_passport\"))), bool((((t.get(\"setup\") or {}).get(\"setup_passport\") or {}).get(\"market_context\")))) for t in trades]"'
```

Focused tests yang sering relevan untuk area ini:

```bash
./venv/bin/pytest -q tests/test_main_closed_trade_metadata.py tests/test_trade_manager_exit_profiles.py::test_trade_manager_persists_setup_passport tests/test_market_context.py tests/test_market_data_service.py
./venv/bin/python -m compileall src tests
git diff --check
```

## 25. Reference Map

Core files:

- [`config/config.yaml`](./config/config.yaml) - knobs dan gates.
- [`src/main.py`](./src/main.py) - admission loop dan orchestration.
- [`src/data/market_data.py`](./src/data/market_data.py) - snapshots dan market context attach.
- [`src/models/market_context.py`](./src/models/market_context.py) - classifier BTC/ETH broad context.
- [`src/scoring/scorer.py`](./src/scoring/scorer.py) - scoring dan bridge ke router.
- [`src/models/strategy_router.py`](./src/models/strategy_router.py) - sleeve selection dan passport creation.
- [`src/models/strategy_passport.py`](./src/models/strategy_passport.py) - kontrak setup yang persist.
- [`src/models/cohort_policy.py`](./src/models/cohort_policy.py) - cohort allow/block/bonus.
- [`src/models/strategy_lifecycle.py`](./src/models/strategy_lifecycle.py) - lifecycle edge/cohort.
- [`src/models/fund_manager.py`](./src/models/fund_manager.py) - sizing dan veto.
- [`src/models/adaptive_brain.py`](./src/models/adaptive_brain.py) - adaptive overlay.
- [`src/risk/risk_manager.py`](./src/risk/risk_manager.py) - sizing dan hard risk state.
- [`src/risk/correlation_filter.py`](./src/risk/correlation_filter.py) - compounding exposure guard.
- [`src/execution/trade_manager.py`](./src/execution/trade_manager.py) - lifecycle open trade dan exits.
- [`src/backtest/shadow_engine.py`](./src/backtest/shadow_engine.py) - counterfactual shadow trades.
- [`src/data/dataset_logger.py`](./src/data/dataset_logger.py) - training/near-miss rows.
- [`src/reporting/attribution.py`](./src/reporting/attribution.py) - performance slicing.
- [`src/reporting/bootstrap.py`](./src/reporting/bootstrap.py) - bootstrap audit lines.
- [`src/notifications/telegram.py`](./src/notifications/telegram.py) - heartbeat/reporting Telegram.

Related docs:

- [`session.md`](./session.md)
- [`QUANT_OPERATING_MODEL.md`](./QUANT_OPERATING_MODEL.md)
- [`QUANT_MINI_ROADMAP.md`](./QUANT_MINI_ROADMAP.md)
- [`QUANT_MINI_BOOTSTRAP_TO_LIVE.md`](./QUANT_MINI_BOOTSTRAP_TO_LIVE.md)

## 26. Research References

| Komponen | Referensi | Kenapa relevan |
|---|---|---|
| Trend following | Moskowitz, Ooi, Pedersen, *Time Series Momentum* | Mendukung momentum futures sebagai tesis inti. |
| Momentum crash risk | Daniel, Moskowitz, *Momentum Crashes* | Menjelaskan kenapa momentum perlu de-risking saat kondisi ekstrem. |
| Momentum dan reversal | Vayanos, Woolley, *An Institutional Theory of Momentum and Reversal* | Membantu melihat trend/reversal sebagai efek flow institusional. |
| Information diffusion | Andrei, Cujean, *Information percolation, momentum and reversal* | Menjelaskan delayed reaction dan reversal behavior. |

## 27. Ringkasan Belajar

Kalau hanya mau mengingat satu model mental:

- `Scorer` mengukur kekuatan kandidat.
- `Router` menentukan tesis trade.
- `Passport` menyimpan kontrak tesis itu.
- `Market context` memberi backdrop BTC/alt rotation.
- `Cohort/lifecycle` menentukan apakah pola sudah sehat.
- `Risk/correlation` menentukan apakah akun boleh menanggung exposure.
- `TradeManager` menjaga trade tetap sesuai kontrak setelah open.
- `Attribution` membuktikan mana yang benar-benar punya edge.

Tujuan paper phase sekarang bukan sekadar banyak entry. Tujuannya mengumpulkan sample yang cukup jelas sehingga edge bisa dipisahkan dari noise, fees, outlier, dan kebetulan.
