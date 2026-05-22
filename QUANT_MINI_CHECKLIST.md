# Checklist Implementasi Quant Mini

> Dokumen ini memecah `QUANT_MINI_ROADMAP.md` menjadi daftar kerja yang dapat dieksekusi. Urutan prioritasnya adalah stabilitas data, kualitas keputusan, lalu perluasan otonomi AI.

## A. Fondasi Data

- [ ] Pastikan setiap trade memiliki `open_ts` dan `close_ts`
- [ ] Pastikan setiap trade menyimpan `regime`, `sleeve`, dan `exit_profile`
- [ ] Pastikan `p_win`, `ML threshold`, dan alasan veto tersimpan
- [ ] Pastikan `dispersion_value` dan `dispersion_state` tersimpan secara konsisten
- [ ] Pastikan funding, OI, dan spread context ikut terekam
- [ ] Pisahkan label real, paper, dan shadow secara eksplisit

## B. Kualitas Statistik

- [ ] Tambahkan tracking calibration untuk ML / AdaptiveBrain
- [ ] Tambahkan veto rate per hari dan per regime
- [ ] Tambahkan gap OOS vs in-sample
- [ ] Tambahkan rolling PF, hit rate, dan expectancy per sleeve
- [ ] Tambahkan drift monitoring per periode

## C. ML Sebagai Advisor

- [ ] ML / AdaptiveBrain memberi `p_win` yang stabil
- [ ] ML / AdaptiveBrain memberi ranking kandidat trade
- [ ] ML / AdaptiveBrain memberi sizing bonus/penalty yang konservatif
- [ ] ML / AdaptiveBrain tidak memveto trade terlalu dini
- [ ] ML / AdaptiveBrain tetap dapat dimatikan tanpa mematikan bot

## D. ML Sebagai Gate Terkontrol

- [ ] Aktifkan veto budget harian
- [ ] Aktifkan veto budget per regime
- [ ] Batasi hard-lock hanya pada regime yang terbukti buruk
- [ ] Jaga veto rate tetap rendah dan terukur
- [ ] Pastikan rejection reason tetap auditable

## E. Decision Brain

- [ ] Pindahkan ranking sleeve ke decision layer utama
- [ ] Pindahkan sizing ke kombinasi statistik + risk cap
- [ ] Pindahkan pemilihan exit profile ke model
- [ ] Tambahkan confidence calibration per setup
- [ ] Validasi keputusan dengan walk-forward

## F. Produksi

- [ ] Tetapkan kill-switch yang tetap rule-based
- [ ] Pastikan semua keputusan bisa direplay dari log
- [ ] Pastikan bot tetap jalan saat ML gagal load
- [ ] Pastikan fallback rule engine tetap aman
- [ ] Buat laporan ringkas untuk review harian
