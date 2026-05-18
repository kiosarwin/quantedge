# Bootstrap ke Live

> Dokumen ini menjelaskan urutan aman dari fase bootstrap menuju live. Fokusnya bukan agresivitas, melainkan memastikan model tidak menjadi bottleneck dan tidak mengambil alih terlalu cepat.

## Fase 1 — Bootstrap Aman

Tujuan:

- kumpulkan data
- hindari overfitting
- menjaga bot tetap aktif trading

Aturan:

- ML hanya berperan sebagai advisor
- veto keras dibatasi atau dimatikan
- sizing dapat turun, tetapi tidak mematikan flow
- risk manager tetap menjadi pagar keras

Lulus jika:

- trade sample mencukupi
- rejection reason terdokumentasi dengan rapi
- bot hidup stabil
- tidak ada drift ekstrem pada sleeve utama

## Fase 2 — Paper Validasi

Tujuan:

- buktikan edge di lingkungan terkendali
- cek performa per sleeve dan regime

Aturan:

- boleh ada veto terbatas
- boleh ada soft-lock regime
- observability harus lengkap
- OOS mulai menjadi acuan utama

Lulus jika:

- win rate dan PF stabil
- drawdown wajar
- calibration ML tidak liar
- veto rate tidak membunuh trade flow

## Fase 3 — Controlled Live

Tujuan:

- masuk live dengan risiko terkontrol
- mempertahankan fleksibilitas AI

Aturan:

- ML boleh memveto dengan budget
- hard cap risiko tetap rule-based
- exit profile dan sizing sudah diuji
- fallback manual/rule tetap tersedia

Lulus jika:

- live hasil mendekati OOS
- model tidak drift tajam
- bot tidak undertrade karena gate terlalu ketat

## Fase 4 — Decision Brain

Tujuan:

- AI menjadi pengambil keputusan utama

Syarat:

- data multi-regime cukup panjang
- feature schema stabil
- model calibration baik
- audit trail lengkap

Implikasi:

- router rule-based turun menjadi feature generator
- ML memilih ranking, sizing, dan exit profile
- risk layer tetap non-negotiable

## Stop Conditions

Hentikan peningkatan otonomi jika:

- veto rate naik tajam
- OOS divergen dari in-sample
- trade count turun karena gate terlalu ketat
- max drawdown memburuk
- model mulai menang hanya di satu regime
