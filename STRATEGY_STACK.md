# Ninja Trader Strategy Stack

> Catatan handoff: `session.md` memuat state runtime terkini. Gunakan itu sebagai sumber kebenaran operasional, bukan konteks lama yang disusun ulang.

## Kerangka Strategi

Bot ini memperlakukan trading crypto perpetual directional sebagai stack yang sadar-regime:

1. `trend_following`
2. `compression_breakout`
3. `reversal`
4. fallback `neutral`

Basis riset yang dipakai:

- Momentum time-series / trend following merupakan salah satu edge directional paling konsisten pada instrumen futures yang likuid.
- Kualitas sinyal momentum crypto menurun ketika dispersion lintas aset menjadi ekstrem.
- Reversal cenderung lebih menarik pada kondisi crowding, liquidation sweep, dan distribution.
- Funding, basis, order flow, dan OI paling tepat diposisikan sebagai filter dan overlay sizing, bukan alpha mandiri.

## Pemetaan Komponen ke Literatur

Dokumen ini sengaja memisahkan antara:

- **basis akademik utama**: komponen yang punya dukungan literatur langsung
- **overlay implementasi**: komponen yang dipakai sebagai penguat keputusan atau guardrail
- **heuristik operasional**: komponen yang berguna secara praktis, tetapi tetap perlu diuji lebih jauh secara out-of-sample

Pemetaan ringkasnya:

- `trend_following` ditopang terutama oleh literatur time-series momentum.
- `reversal` ditopang oleh literatur momentum dan reversal yang menjelaskan mengapa momentum dapat berbalik pada rezim tertentu.
- overlay dispersion dan de-risking saat kondisi ekstrem dipakai sebagai adaptasi praktis dari literatur momentum-crash dan regime shift.
- funding, basis, dan open interest diperlakukan sebagai variabel kondisi pasar yang membantu sizing dan filtering pada pasar perpetual.
- order flow dan liquidity sweep dipakai sebagai heuristik mikrostruktur yang perlu terus divalidasi dengan data.

## Sleeve Strategi

### `trend_following`

Digunakan ketika:

- regime = `trending_expansion`
- kekuatan tren tinggi
- kualitas struktur tinggi
- smart money berada di `trending`, `accumulation`, atau `neutral`

Efek pada sistem:

- menaikkan score
- memperbesar ukuran posisi
- sedikit menurunkan threshold
- mengecil ketika dispersion naik

### `compression_breakout`

Digunakan ketika:

- regime = `accumulation_compression`
- struktur cukup kuat untuk membenarkan continuation breakout
- smart money berada di `accumulation`, `liquidity_sweep`, atau `neutral`

Efek pada sistem:

- memberi tambahan score kecil
- ukuran posisi sedikit lebih kecil daripada sleeve trend

### `reversal`

Digunakan ketika:

- smart money mendeteksi `liquidity_sweep`
- atau regime = `distribution` dengan konteks smart money yang cocok untuk reversal

Efek pada sistem:

- memberi tambahan score sedang
- ukuran default lebih kecil
- bisa memperoleh prioritas relatif lebih tinggi saat dispersion ekstrem

## Logika Dispersion

Cross-sectional dispersion dihitung dari standar deviasi kekuatan momentum seluruh simbol pada saat pengambilan keputusan.

- `normal`: sleeve trend dapat dipakai secara normal
- `warn`: sleeve trend dide-risk sebagian
- `high`: sleeve trend dide-risk secara material, reversal memperoleh prioritas relatif lebih tinggi

Ini konsisten dengan temuan akademik bahwa reliabilitas sinyal momentum menurun pada ekor distribusi dispersion.

## Metrik Evaluasi

Evaluasi ke depan harus melaporkan:

- expectancy per `strategy_sleeve`
- expectancy per `direction`
- expectancy per `regime`
- expectancy per `strategy_sleeve x regime`
- turnover dan fee drag per sleeve
- hit rate, profit factor, Sharpe, max drawdown, dan skew

## Status Implementasi

Sudah diimplementasikan:

- `strategy_router` merutekan ke `trend_following`, `compression_breakout`, `reversal`, dan `neutral`
- overlay score dan sizing yang aware terhadap dispersion telah diterapkan di scorer
- attribution untuk sleeve, exit profile, regime, dan side telah direkam di reporting
- profile exit per sleeve telah dipakai di live dan backtest untuk sizing TP, trailing, dan parameter hold time
- laporan attribution telah mengelompokkan trade berdasarkan sleeve, regime, side, dan kombinasi sleeve x regime/side

Parsial:

- `compression_breakout` tersedia, tetapi masih digate oleh config
- short reversal tersedia, tetapi masih dibatasi threshold dan flag config
- mesin exit masih menggunakan satu alur utama, walaupun parameternya sudah berbeda per sleeve
- reporting sudah mencakup grouping dispersion, tetapi belum ada faktor dispersion level universe-wide yang eksplisit di ringkasan backtester

Belum diimplementasikan:

- dedicated sleeve-specific exit logic
- market-neutral funding carry sleeve
- explicit cross-sectional dispersion factor di laporan backtester

Ringkasan cakupan vs target:

- cakupan yang sudah ada: routing, overlay scoring, metadata attribution, dan laporan grouping
- gap target yang tersisa: exit yang benar-benar native per sleeve, carry sleeve, dan reporting dispersion yang lebih eksplisit

## Referensi

| Komponen | Paper | Implikasi |
|---|---|---|
| `trend_following` | Tobias J. Moskowitz, Yao Hua Ooi, Lasse Heje Pedersen, *Time Series Momentum* (SSRN). `https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2089463` | Dasar utama untuk tren time-series di futures likuid. |
| `trend_following` dan de-risking saat ekstrem | Kent Daniel, Tobias J. Moskowitz, *Momentum Crashes* (NBER W20439). `https://www.nber.org/papers/w20439` | Menjelaskan kenapa momentum perlu dikurangi risikonya saat state panic / rebound. |
| `reversal` | Dimitri Vayanos, Paul Woolley, *An Institutional Theory of Momentum and Reversal* (NBER W14523). `https://www.nber.org/papers/w14523` | Menjelaskan momentum dan reversal dalam kerangka flow institusional yang sama. |
| `reversal` dan difusi informasi | D. Andrei, J. Cujean, *Information percolation, momentum and reversal* (JFE). `https://www.sciencedirect.com/science/article/pii/S0304405X16302380` | Reversal dapat muncul saat informasi meresap tidak serempak ke pasar. |
| Perpetual futures, funding, basis | Songrun He, Asaf Manela, Omri Ross, Victor von Wachter, *Fundamentals of Perpetual Futures* (arXiv 2212.06888). `https://arxiv.org/abs/2212.06888` | Dasar teoritis untuk memahami mekanisme funding dan deviasi spot-perp. |
| Perpetual futures pricing | Damien Ackerer, Julien Hugonnier, Urban Jermann, *Perpetual Futures Pricing* (NBER W32936). `https://www.nber.org/papers/w32936` | Menguatkan pemahaman teoritis tentang anchoring harga dan funding specification. |
| Open interest sebagai variabel informasi | Harrison Hong, Motohiro Yogo, *What Does Futures Market Interest Tell Us about the Macroeconomy and Asset Prices?* (NBER W16712). `https://www.nber.org/papers/w16712` | Mendukung open interest sebagai variabel informasi pasar, bukan alpha tunggal. |
| Open interest, volume, liquidations | Ioannis Giagkiozis, Emilio Said, *Reconciling Open Interest with Traded Volume in Perpetual Swaps* (arXiv 2310.14973). `https://arxiv.org/abs/2310.14973` | Relevan untuk membaca OI, volume, dan likuidasi pada perpetual crypto. |

Catatan:

- Bagian `smart_money`, `liquidity_sweep`, dan sebagian aturan entry/exit masih merupakan heuristik implementasi.
- Komponen ini masuk akal secara trading, tetapi tetap perlu dibuktikan dengan walk-forward, out-of-sample, dan evaluasi per sleeve sebelum dipakai live penuh.
