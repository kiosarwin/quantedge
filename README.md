# Ninja Trader

Bot trading futures yang dibangun sebagai stack kuantitatif bertahap: rule engine, scoring, risk guard, ML helper, lalu roadmap menuju `quant mini`.

## Pintu Masuk Untuk Semua Model AI

Jika Anda adalah model AI apa pun — `Codex`, `Claude`, `Gemini`, atau lainnya — maka `README.md` ini adalah pintu masuk utama proyek.

### Aturan wajib sebelum mengerjakan apa pun

1. Baca [`session.md`](./session.md) terlebih dahulu.
2. Baca [`AGENTS.md`](./AGENTS.md) sebagai aturan kerja lokal.
3. Baca dokumen yang relevan dengan tugas sebelum mengubah kode.
4. Jangan mulai refactor besar tanpa alasan yang jelas.
5. Prioritaskan perubahan kecil, terukur, dan bisa diverifikasi.

### Urutan baca yang disarankan

1. [`session.md`](./session.md) — state runtime dan konteks terbaru
2. [`AGENTS.md`](./AGENTS.md) — aturan kerja lokal yang wajib dipatuhi
3. [`QUANT_OPERATING_MODEL.md`](./QUANT_OPERATING_MODEL.md) — model operasi cohort dan trade admission
4. [`STRATEGY_STACK.md`](./STRATEGY_STACK.md) — struktur strategi dan basis riset
5. [`QUANT_MINI_ROADMAP.md`](./QUANT_MINI_ROADMAP.md) — target arsitektur jangka menengah
6. [`QUANT_MINI_BOOTSTRAP_TO_LIVE.md`](./QUANT_MINI_BOOTSTRAP_TO_LIVE.md) — urutan aman bootstrap ke live
7. [`QUANT_MINI_CHECKLIST.md`](./QUANT_MINI_CHECKLIST.md) — daftar implementasi
8. [`QUANT_MINI_DATA_SCHEMA.md`](./QUANT_MINI_DATA_SCHEMA.md) — schema data minimum untuk audit dan ML

### Prinsip kerja

- Gunakan perubahan paling kecil yang menyelesaikan masalah.
- Jangan refactor area yang tidak terkait task.
- Jaga agar risk layer tetap lebih kuat daripada model.
- Jangan biarkan ML menjadi bottleneck atau gate yang terlalu agresif.
- Pastikan setiap perubahan bisa diaudit dan diverifikasi.
- Jika ada konflik antar dokumen, ikuti instruksi yang paling spesifik dan paling baru.

### Urutan eksekusi yang aman

1. Pahami state sekarang dari `session.md`
2. Identifikasi file yang benar-benar relevan
3. Buat perubahan minimal
4. Verifikasi dengan test atau compile check yang paling spesifik
5. Lapor balik secara faktual

### Hal yang biasanya tidak boleh diubah tanpa alasan kuat

- logic risk guard dan kill switch
- schema trade log yang dipakai learning/reporting
- fallback paper/live behavior
- struktur cohort dan regime gating
- jalur deploy ke VM

## Gambaran Sistem

Arsitektur bot saat ini terdiri dari:

- **scoring layer** untuk menilai setup
- **strategy router** untuk memilih sleeve; alpha utama saat ini adalah `trend_following` dan `reversal`, sedangkan `neutral` adalah fallback residual dan tidak diperlakukan sebagai alpha
- **EV / p(win) layer** untuk menyaring edge
- **risk manager** untuk posisi, stop, dan exposure
- **ML layer** untuk prediksi, sizing, dan gating terkontrol
- **reporting layer** untuk attribution, audit, dan review

Target evolusinya adalah sistem `quant mini` dengan satu decision layer utama, tetapi tetap memakai pagar risiko keras.

## Dokumen Penting

- [`STRATEGY_STACK.md`](./STRATEGY_STACK.md) — strategi, sleeve, referensi riset
- [`QUANT_OPERATING_MODEL.md`](./QUANT_OPERATING_MODEL.md) — urutan admission trade dan lifecycle cohort
- [`QUANT_MINI_ROADMAP.md`](./QUANT_MINI_ROADMAP.md) — arah arsitektur jangka menengah
- [`QUANT_MINI_BOOTSTRAP_TO_LIVE.md`](./QUANT_MINI_BOOTSTRAP_TO_LIVE.md) — fase aman menuju live
- [`QUANT_MINI_CHECKLIST.md`](./QUANT_MINI_CHECKLIST.md) — backlog implementasi
- [`QUANT_MINI_DATA_SCHEMA.md`](./QUANT_MINI_DATA_SCHEMA.md) — field minimum untuk audit dan training
- [`runbook.md`](./runbook.md) — operasi harian
- [`CHANGELOG.md`](./CHANGELOG.md) — histori perubahan

## Jalur Kerja Kode

- `src/main.py` — orkestrasi utama bot
- `src/scoring/scorer.py` — scoring dan router
- `src/models/ml_engine.py` — ML predictor dan readiness
- `src/models/ev_model.py` — EV gate dan probabilitas
- `src/risk/risk_manager.py` — risk, sizing, dan exit setup
- `src/execution/trade_manager.py` — lifecycle trade
- `src/reporting/` — reporting dan attribution

## Verifikasi

Kalau mengubah logic, prioritaskan verifikasi yang paling dekat dengan area yang diubah:

- unit test yang spesifik
- `python -m compileall` untuk file yang disentuh
- test regresi untuk paper validation atau risk flow

Jangan menambah test besar yang tidak terkait perubahan.

## Deploy

- Repo ini punya alur kerja lokal + VM.
- Gunakan perubahan yang kecil, lalu sync ke VM hanya setelah verifikasi lulus.
- Untuk state runtime yang aktif, `session.md` adalah sumber kebenaran terbaru.

## Catatan

- Dokumen ini dibuat sebagai panduan lintas agent.
- Bila instruksi di dokumen lain bertentangan, ikuti instruksi yang paling spesifik dan paling baru.
